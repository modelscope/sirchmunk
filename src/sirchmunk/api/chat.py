# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified API endpoints for chat and search functionality
Provides WebSocket endpoint for real-time chat conversations with integrated search
"""
import logging
import platform
import re
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Request
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel
import json
import asyncio
import uuid
from datetime import datetime
import random
import os
import threading

import openai

from sirchmunk.search import AgenticSearch
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.api.components.history_storage import HistoryStorage
from sirchmunk.api.components.monitor_tracker import llm_usage_tracker
from sirchmunk.api.security import (
    is_path_allowed,
    verify_ws_token,
    validate_user_path,
    file_browser_limiter,
    audit_logger,
)

logger = logging.getLogger(__name__)

# Maximum number of full-pipeline retries for transient LLM errors in RAG
# search.  Individual LLM calls already retry internally (see OpenAIChat);
# this is a coarser-grained retry around the entire search pipeline.
_RAG_PIPELINE_MAX_RETRIES = 1
_RAG_PIPELINE_RETRY_DELAY = 2.0  # seconds


def _is_transient_llm_error(exc: Exception) -> bool:
    """Return True if *exc* is a transient LLM/network error worth retrying."""
    return isinstance(exc, (
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,   # all 5xx
        openai.RateLimitError,        # 429
        openai.NotFoundError,         # 404 — transient on some providers
        ConnectionError,
        TimeoutError,
    ))


def _classify_error(exc: Exception) -> str:
    """Return a human-readable error class for user-facing messages."""
    if isinstance(exc, openai.AuthenticationError):
        return "LLM authentication failed — check LLM_API_KEY"
    if isinstance(exc, openai.PermissionDeniedError):
        return "LLM permission denied — check API key permissions"
    if isinstance(exc, openai.BadRequestError):
        return "LLM rejected the request — check LLM_MODEL_NAME"
    if isinstance(exc, openai.NotFoundError):
        return "LLM endpoint returned 404 — check LLM_BASE_URL and LLM_MODEL_NAME"
    if isinstance(exc, openai.RateLimitError):
        return "LLM rate limit exceeded — wait and retry"
    if isinstance(exc, openai.InternalServerError):
        return "LLM server error (5xx) — provider-side issue"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "LLM connection/timeout error — check network"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "Network error — check connectivity"
    return str(exc)


def _resolve_rag_paths(kb_name: str) -> Tuple[List[str], str]:
    """Resolve RAG search paths from frontend kb_name or SIRCHMUNK_SEARCH_PATHS.

    User-provided paths are validated against the whitelist and filesystem
    constraints.  Invalid entries are silently dropped.
    """
    def _parse(s: str) -> List[str]:
        return [p.strip() for p in (s or "").split(",") if p.strip()]

    if kb_name and _parse(kb_name):
        raw_paths = _parse(kb_name)
        validated: List[str] = []
        for rp in raw_paths:
            ok, result = validate_user_path(rp)
            if ok:
                validated.append(result)
            else:
                logger.debug("Rejected user path %r: %s", rp, result)
        if validated:
            return validated, ", ".join(os.path.basename(p) for p in validated)

    # Fallback to environment variable
    env_paths = os.getenv("SIRCHMUNK_SEARCH_PATHS", "")
    paths = _parse(env_paths)
    display = ", ".join(paths) if paths else ""
    return paths, display


# ---------------------------------------------------------------------------
# Multi-turn conversation support
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY_MAX_TURNS = 10
_DEFAULT_HISTORY_MAX_TOKENS = 32000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for mixed CJK / Latin text (~3.5 chars per token)."""
    return max(1, int(len(text) / 3.5))


def _build_chat_history(
    session_id: str,
    *,
    max_turns: int = 0,
    max_tokens: int = 0,
) -> List[Dict[str, str]]:
    """Build an OpenAI-format message list from session history.

    Reads from the in-memory ``chat_sessions`` cache (populated on each
    WebSocket turn).  Falls back to ``history_storage`` on cache miss
    (e.g. after a server restart mid-session).

    Returns the most recent messages that fit within *max_turns* and
    *max_tokens* (whichever is more restrictive), ordered oldest-first.
    The *current* user message (not yet appended) is excluded.
    """
    if not max_turns:
        max_turns = int(os.getenv("CHAT_HISTORY_MAX_TURNS", str(_DEFAULT_HISTORY_MAX_TURNS)))
    if not max_tokens:
        max_tokens = int(os.getenv("CHAT_HISTORY_MAX_TOKENS", str(_DEFAULT_HISTORY_MAX_TOKENS)))

    raw: List[Dict[str, str]] = []
    session = chat_sessions.get(session_id)
    if session and session.get("messages"):
        raw = [{"role": m["role"], "content": m["content"]} for m in session["messages"]]
    else:
        raw = history_storage.get_recent_messages(session_id, limit=max_turns * 2)

    if not raw:
        return []

    # Traverse from newest to oldest, accumulating within budget
    kept: List[Dict[str, str]] = []
    budget = max_tokens
    for msg in reversed(raw[-(max_turns * 2):]):
        cost = _estimate_tokens(msg["content"])
        if budget - cost < 0 and kept:
            break
        budget -= cost
        kept.append(msg)

    kept.reverse()
    return kept


async def _rewrite_query_with_context(
    message: str,
    history: List[Dict[str, str]],
    llm: OpenAIChat,
) -> str:
    """Rewrite a context-dependent query into a self-contained search query.

    Uses a single lightweight LLM call (stream=False).  If the message
    is already self-contained the LLM returns it unchanged.
    """
    from sirchmunk.llm.prompts import QUERY_REWRITE

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
    )
    prompt = QUERY_REWRITE.format(history=history_text, message=message)
    resp = await llm.achat(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    rewritten = (resp.content or message).strip()
    if not rewritten:
        return message
    if rewritten != message:
        logger.info("[multi-turn] Query rewritten: '%s' → '%s'", message[:60], rewritten[:60])
    return rewritten


async def _filter_relevant_history(
    message: str,
    history: List[Dict[str, str]],
    llm: OpenAIChat,
) -> List[Dict[str, str]]:
    """Return *history* if topically relevant to *message*, else empty list.

    A single lightweight LLM call decides whether the conversation history
    shares the same topic as the current query.  When the user switches to
    an entirely new topic, injecting stale history would confuse the model,
    so we discard it.
    """
    if not history:
        return []

    from sirchmunk.llm.prompts import HISTORY_RELEVANCE_CHECK

    history_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:200]}" for m in history[-4:]
    )
    prompt = HISTORY_RELEVANCE_CHECK.format(history=history_text, message=message)
    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        text = (resp.content or "").strip()
        match = re.search(r'"relevant"\s*:\s*(true|false)', text, re.IGNORECASE)
        if match and match.group(1).lower() == "false":
            logger.info(
                "[multi-turn] History topic diverged from query — discarding %d turn(s)",
                len(history),
            )
            return []
    except Exception as exc:
        logger.debug("[multi-turn] History relevance check failed, keeping history: %s", exc)

    return history


# Tkinter availability is checked lazily to avoid initialising
# the macOS Cocoa/AppKit framework at module-import time.
# Eagerly importing tkinter in a headless server causes the macOS
# crash-reporter dialog to appear when the process receives SIGINT.
_tkinter_checked = False
TKINTER_AVAILABLE = False
tk = None
filedialog = None
messagebox = None


def _ensure_tkinter():
    """Lazily import tkinter on first use and cache the result."""
    global _tkinter_checked, TKINTER_AVAILABLE, tk, filedialog, messagebox
    if _tkinter_checked:
        return TKINTER_AVAILABLE
    _tkinter_checked = True
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd, messagebox as _mb
        tk = _tk
        filedialog = _fd
        messagebox = _mb
        TKINTER_AVAILABLE = True
    except ImportError:
        TKINTER_AVAILABLE = False
    return TKINTER_AVAILABLE

router = APIRouter(prefix="/api/v1", tags=["chat", "search"])

# Initialize persistent history storage
history_storage = HistoryStorage()

# In-memory cache for active sessions (for backward compatibility)
chat_sessions = {}

# Active WebSocket connections
_MAX_WS_CONNECTIONS = int(os.getenv("SIRCHMUNK_MAX_WS_CONNECTIONS", "100"))


class ChatConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> bool:
        """Accept a WebSocket connection if capacity allows."""
        if len(self.active_connections) >= _MAX_WS_CONNECTIONS:
            await websocket.close(code=1013, reason="Server at capacity")
            return False
        await websocket.accept()
        self.active_connections.append(websocket)
        return True

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

# Unified log callback management
class WebSocketLogger:
    """
    WebSocket-aware logger that wraps websocket communications.
    
    Provides logger-style methods (info, warning, etc.) similar to loguru,
    with support for flush and end parameters for streaming output.
    Compatible with sirchmunk.utils.log_utils.AsyncLogger interface.
    """
    
    def __init__(self, websocket: WebSocket, manager: Optional[ChatConnectionManager] = None, log_type: str = "log", task_id: Optional[str] = None):
        """
        Initialize WebSocket logger.
        
        Args:
            websocket: WebSocket connection to send logs to
            manager: Optional ConnectionManager for routing messages
            log_type: Type of log message ("log" or "search_log")
            task_id: Optional task ID for grouping related log messages
        """
        self.websocket = websocket
        self.manager = manager
        self.log_type = log_type
        self.task_id = task_id or str(uuid.uuid4())  # Generate unique task ID
    
    async def _send_log(self, level: str, message: str, flush: bool = False, end: str = "\n"):
        """
        Send log message through WebSocket.
        
        Args:
            level: Log level (info, warning, error, etc.)
            message: Message content
            flush: If True, force immediate output (adds small delay for streaming)
            end: String appended after message (default: "\n")
        """
        # Append end character to message
        full_message = message + end if end else message
        
        # Determine if this is a streaming message (no timestamp prefix should be added on frontend)
        # Streaming condition: message should be appended to current line (end is not a newline)
        # This indicates it's part of a multi-chunk streaming output (like LLM responses)
        is_streaming = end != "\n"
        
        # Prepare log message
        log_data = {
            "type": self.log_type,
            "level": level,
            "message": full_message,
            "timestamp": datetime.now().isoformat(),
            "is_streaming": is_streaming,  # Flag for frontend to know if this is streaming output
            "task_id": self.task_id,  # Task ID for grouping related messages
            "flush": flush,  # Include flush flag for frontend handling
        }
        
        # Send through WebSocket
        if self.manager:
            await self.manager.send_personal_message(json.dumps(log_data), self.websocket)
        else:
            await self.websocket.send_text(json.dumps(log_data))
        
        # If flush is requested, add small delay for proper streaming
        if flush:
            await asyncio.sleep(0.01)  # Very short delay for streaming (reduced from 0.05s)
        else:
            await asyncio.sleep(0.05)  # Standard delay (reduced from 0.1s)
    
    async def log(self, level: str, message: str, flush: bool = False, end: str = "\n"):
        """Log a message at the specified level"""
        await self._send_log(level, message, flush=flush, end=end)
    
    async def debug(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a debug message"""
        await self._send_log("debug", message, flush=flush, end=end)
    
    async def info(self, message: str, flush: bool = False, end: str = "\n"):
        """Log an info message"""
        await self._send_log("info", message, flush=flush, end=end)
    
    async def warning(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a warning message"""
        await self._send_log("warning", message, flush=flush, end=end)
    
    async def error(self, message: str, flush: bool = False, end: str = "\n"):
        """Log an error message"""
        await self._send_log("error", message, flush=flush, end=end)
    
    async def success(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a success message"""
        await self._send_log("success", message, flush=flush, end=end)
    
    async def critical(self, message: str, flush: bool = False, end: str = "\n"):
        """Log a critical message"""
        await self._send_log("critical", message, flush=flush, end=end)


class LogCallbackManager:
    """
    Centralized management for all log callback functions.
    
    Creates callback functions and logger instances that are compatible with
    sirchmunk.utils.log_utils.AsyncLogger interface, supporting flush and end parameters.
    """

    @staticmethod
    async def create_search_log_callback(websocket: WebSocket, manager: ChatConnectionManager, task_id: Optional[str] = None):
        """
        Create search log callback for chat WebSocket.
        
        Returns a callback function compatible with log_utils signature:
        async def callback(level: str, message: str, end: str, flush: bool)
        
        NOTE: The signature MUST match log_utils.LogCallback exactly:
              (level: str, message: str, end: str, flush: bool) -> None
        
        Args:
            websocket: WebSocket connection
            manager: Connection manager for routing
            task_id: Optional task ID for grouping related messages (auto-generated if not provided)
            
        Returns:
            Async callback function
        """
        # Generate unique task ID for this search session
        if task_id is None:
            task_id = f"search_{uuid.uuid4().hex[:8]}"
        
        logger = WebSocketLogger(websocket, manager, log_type="search_log", task_id=task_id)
        
        # Track recent messages for deduplication (message -> timestamp)
        recent_messages: Dict[str, float] = {}
        DEDUP_WINDOW_SEC = 0.5  # Messages within this window are considered duplicates

        # CRITICAL: This callback signature MUST match log_utils.LogCallback
        # Signature: (level: str, message: str, end: str, flush: bool) -> None
        async def search_log_callback(level: str, message: str, end: str, flush: bool):
            """
            Log callback compatible with log_utils.LogCallback type.
            
            Args:
                level: Log level (info, warning, error, etc.)
                message: Message content (WITHOUT end character appended)
                end: String to append after message
                flush: Whether to flush immediately
            """
            import time
            nonlocal recent_messages
            
            # Create unique key for this message (include level and message content)
            msg_key = f"{level}:{message}"
            current_time = time.time()
            
            # Check for duplicate within time window
            if msg_key in recent_messages:
                last_time = recent_messages[msg_key]
                if current_time - last_time < DEDUP_WINDOW_SEC:
                    # Skip duplicate message within dedup window
                    return
            
            # Clean up old entries (older than 2x window)
            cutoff = current_time - (DEDUP_WINDOW_SEC * 2)
            recent_messages = {k: v for k, v in recent_messages.items() if v > cutoff}
            
            # Record this message
            recent_messages[msg_key] = current_time
            
            await logger._send_log(level, message, flush=flush, end=end)
        
        return search_log_callback

    @staticmethod
    def create_logger(websocket: WebSocket, manager: Optional[ChatConnectionManager] = None, log_type: str = "log", task_id: Optional[str] = None) -> WebSocketLogger:
        """
        Create a WebSocketLogger instance with logger-style methods.
        
        This provides a logger interface similar to create_logger from log_utils,
        allowing usage like: await logger.info("message", flush=True, end="")
        
        Args:
            websocket: WebSocket connection
            manager: Optional ConnectionManager for routing messages
            log_type: Type of log message ("log" or "search_log")
            task_id: Optional task ID for grouping related messages (auto-generated if not provided)
            
        Returns:
            WebSocketLogger instance
            
        Example:
            logger = LogCallbackManager.create_logger(websocket, manager, "search_log")
            await logger.info("Processing started")
            await logger.info("Loading", flush=True, end=" -> ")
            await logger.success("Done!", flush=True)
        """
        if task_id is None:
            task_id = f"logger_{uuid.uuid4().hex[:8]}"
        return WebSocketLogger(websocket, manager, log_type, task_id)

manager = ChatConnectionManager()

# Search-related models and functions
class SearchRequest(BaseModel):
    query: str
    paths: Union[str, List[str]]  # Expects absolute file/directory paths from user's local filesystem
    mode: Optional[str] = "FAST"
    max_depth: Optional[int] = 5
    top_k_files: Optional[int] = 3


def get_envs() -> Dict[str, Any]:
    """Get LLM configuration from os.environ (backed by .env)."""
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY", "")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-5.2")

    logger.debug("LLM config loaded: base_url=%s, model=%s", base_url, model_name)

    return dict(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )


_chat_search_instance: Optional[AgenticSearch] = None
_chat_search_config: Optional[tuple] = None
_chat_search_lock = threading.Lock()


def get_search_instance(log_callback=None):
    """Get or create a cached AgenticSearch instance.

    Uses double-checked locking to ensure thread-safe singleton creation
    while keeping the fast path (reuse) lock-free.

    The heavy resources (embedding model, knowledge storage) are
    initialised only once.  Subsequent calls reuse the singleton and
    merely swap the per-request ``log_callback`` via
    ``update_log_callback``.

    The instance is automatically recreated when the LLM configuration
    (env vars) changes, e.g. after a settings update in the WebUI.

    Args:
        log_callback: Optional callback for streaming search logs

    Returns:
        Configured AgenticSearch instance
    """
    global _chat_search_instance, _chat_search_config

    try:
        envs = get_envs()
    except Exception as e:
        logger.warning("LLM configuration incomplete, check LLM_BASE_URL/LLM_API_KEY/LLM_MODEL_NAME")
        envs = {
            "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("LLM_API_KEY", ""),
            "model_name": os.getenv("LLM_MODEL_NAME", "gpt-5.2"),
        }

    current_config = (envs["api_key"], envs["base_url"], envs["model_name"])

    # Fast path (lock-free): reuse existing instance when config unchanged
    if _chat_search_instance is not None and current_config == _chat_search_config:
        _chat_search_instance.update_log_callback(log_callback)
        return _chat_search_instance

    # Slow path: acquire lock and double-check before creating
    with _chat_search_lock:
        if _chat_search_instance is not None and current_config == _chat_search_config:
            _chat_search_instance.update_log_callback(log_callback)
            return _chat_search_instance

        llm = OpenAIChat(
            base_url=envs["base_url"],
            api_key=envs["api_key"],
            model=envs["model_name"],
            log_callback=log_callback,
        )

        enable_cluster_reuse = os.getenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "true").lower() == "true"
        cluster_sim_threshold = float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.85"))
        cluster_sim_top_k = int(os.getenv("CLUSTER_SIM_TOP_K", "3"))

        _chat_search_instance = AgenticSearch(
            llm=llm,
            log_callback=log_callback,
            reuse_knowledge=enable_cluster_reuse,
            cluster_sim_threshold=cluster_sim_threshold,
            cluster_sim_top_k=cluster_sim_top_k,
            work_path=os.getenv("SIRCHMUNK_WORK_PATH") or None,
        )
        _chat_search_config = current_config
        return _chat_search_instance


_COOLDOWN_SECONDS = 1.0
_DIALOG_LOCK = threading.Lock()
_LAST_CLOSE_TIME = 0
_ROOT_INSTANCE = None


def _get_bg_root():
    """
    Retrieves the global root window.
    Initializes it only once (Singleton pattern) to prevent lag.
    """
    global _ROOT_INSTANCE

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Tkinter must be executed on the Main Thread.")

    if _ROOT_INSTANCE is None or not _ROOT_INSTANCE.winfo_exists():
        _ROOT_INSTANCE = tk.Tk()
        _ROOT_INSTANCE.title("File Picker")
        _ROOT_INSTANCE.attributes("-alpha", 0.0)
        _ROOT_INSTANCE.withdraw()

    return _ROOT_INSTANCE


def open_file_dialog(dialog_type: str = "files", multiple: bool = True) -> List[str]:
    """
    Opens a native file picker dialog using tkinter.
    """
    global _LAST_CLOSE_TIME

    if not _DIALOG_LOCK.acquire(blocking=False):
        return []

    selected_paths = []

    try:
        if time.time() - _LAST_CLOSE_TIME < _COOLDOWN_SECONDS:
            return []

        root = _get_bg_root()
        root.deiconify()
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()

        if platform.system() == "Darwin":
            root.update_idletasks()
        else:
            root.update()

        kwargs = {"parent": root, "title": "Select File(s)"}

        # Set file types filter
        if dialog_type == "files":
            filetypes = [
                ("All Files", "*.*"),
                ("PDF Documents", "*.pdf"),
                ("Word Documents", "*.docx *.doc"),
                ("Excel Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.svg"),
                ("Text Files", "*.txt *.md *.json *.xml"),
            ]

            if multiple:
                res = filedialog.askopenfilenames(filetypes=filetypes, **kwargs)
                selected_paths = list(res) if res else []
            else:
                res = filedialog.askopenfilename(filetypes=filetypes, **kwargs)
                selected_paths = [res] if res else []

        elif dialog_type == "directory":
            kwargs["title"] = "Select Directory"
            res = filedialog.askdirectory(**kwargs)
            selected_paths = [res] if res else []

    except Exception as e:
        logger.warning("File dialog error")
        selected_paths = []

    finally:
        if _ROOT_INSTANCE is not None and _ROOT_INSTANCE.winfo_exists():
            _ROOT_INSTANCE.attributes("-topmost", False)
            _ROOT_INSTANCE.withdraw()
            _ROOT_INSTANCE.update()

        _LAST_CLOSE_TIME = time.time()
        _DIALOG_LOCK.release()

    return selected_paths


async def _perform_web_search(query: str, websocket: WebSocket, manager: ChatConnectionManager) -> Dict[str, Any]:
    """
    Mock web search functionality
    TODO: Replace with actual web search implementation
    """
    await manager.send_personal_message(json.dumps({
        "type": "search_log",
        "level": "info",
        "message": "🌐 Starting web search...",
        "timestamp": datetime.now().isoformat()
    }), websocket)
    
    # Simulate web search delay
    await asyncio.sleep(random.uniform(0.5, 1.0))
    
    await manager.send_personal_message(json.dumps({
        "type": "search_log",
        "level": "info",
        "message": f"🔎 Searching web for: {query}",
        "timestamp": datetime.now().isoformat()
    }), websocket)
    
    await asyncio.sleep(random.uniform(0.5, 1.0))
    
    # Mock web search results
    web_results = {
        "sources": [
            {
                "url": "https://example.com/article1",
                "title": "Comprehensive Guide to " + query[:30],
                "snippet": "This article provides detailed information about the subject matter...",
                "relevance_score": 0.95
            },
            {
                "url": "https://example.com/article2", 
                "title": "Advanced Concepts and Applications",
                "snippet": "Exploring advanced techniques and real-world applications...",
                "relevance_score": 0.87
            },
            {
                "url": "https://example.com/article3",
                "title": "Latest Research and Findings",
                "snippet": "Recent discoveries and innovations in this field...",
                "relevance_score": 0.82
            }
        ],
        "summary": f"Found 3 relevant web sources for '{query}'. The sources cover comprehensive guides, advanced concepts, and latest research."
    }
    
    await manager.send_personal_message(json.dumps({
        "type": "search_log",
        "level": "success",
        "message": f"✅ Web search completed: found {len(web_results['sources'])} sources",
        "timestamp": datetime.now().isoformat()
    }), websocket)
    
    return web_results

async def _chat_only(
    message: str,
    websocket: WebSocket,
    manager: ChatConnectionManager,
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Mode 1: Pure chat mode (no RAG, no web search).

    When *history* is provided the LLM receives prior conversation turns
    so it can maintain context across messages.
    """
    try:
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "generating",
            "message": "💬 Generating response..."
        }), websocket)

        llm_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)

        envs: Dict[str, Any] = get_envs()
        llm = OpenAIChat(
            api_key=envs["api_key"],
            base_url=envs["base_url"],
            model=envs["model_name"],
            log_callback=llm_log_callback
        )

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."},
            *(history or []),
            {"role": "user", "content": message},
        ]

        llm_response = await llm.achat(messages=messages, stream=True)

        usage_data = llm_response.usage if llm_response.usage else {}
        llm_usage_tracker.record_usage(
            model=llm_response.model or envs["model_name"],
            usage=usage_data,
        )

        return llm_response.content, {}
    
    except Exception as e:
        # Send error message to frontend
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": f"LLM chat failed: {str(e)}"
        }), websocket)
        
        # Re-raise to be caught by outer handler
        raise


async def _run_rag_search(
    message: str,
    paths: List[str],
    search_mode: str,
    search_log_callback,
) -> tuple:
    """Execute a single RAG search attempt.

    Returns:
        ``(answer_text, llm_usages, references)`` where *references* is a
        list of evidence dicts extracted from the SearchContext cluster.
    """
    search_engine = get_search_instance(log_callback=search_log_callback)

    result = await search_engine.search(
        query=message,
        paths=paths,
        mode=search_mode,
        top_k_files=3,
        return_context=True,
    )

    references: List[Dict[str, Any]] = []
    if hasattr(result, "answer"):
        answer = result.answer or ""
        cluster = result.cluster
        if cluster and getattr(cluster, "evidences", None):
            for ev in cluster.evidences:
                if not ev.is_found:
                    continue
                raw_snippets = ev.snippets[:3] if ev.snippets else []
                snippets = []
                for s in raw_snippets:
                    if isinstance(s, dict):
                        snippets.append(s.get("snippet", str(s)))
                    else:
                        snippets.append(str(s))
                references.append({
                    "file": str(ev.file_or_url),
                    "summary": ev.summary or "",
                    "snippets": snippets,
                })
    else:
        answer = result if isinstance(result, str) else str(result)

    return answer, list(search_engine.llm_usages), references


async def _chat_rag(
    message: str,
    kb_name: str,
    websocket: WebSocket,
    manager: ChatConnectionManager,
    search_mode: str = "FAST",
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Mode 2: Chat + RAG (enable_rag=True, enable_web_search=False).

    When *history* is provided the search query is rewritten to be
    self-contained (resolving pronouns / omitted subjects), and the
    final answer is generated with conversation context.
    """
    sources = {}
    paths, paths_display = _resolve_rag_paths(kb_name)
    if not paths:
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": "No search paths specified for RAG search."
        }), websocket)
        response = "Please specify search paths for RAG search."
        return response, sources

    # Multi-turn: rewrite the query so it is self-contained for retrieval
    search_query = message
    if history:
        envs: Dict[str, Any] = get_envs()
        rewrite_llm = OpenAIChat(
            api_key=envs["api_key"], base_url=envs["base_url"], model=envs["model_name"],
        )
        search_query = await _rewrite_query_with_context(message, history, rewrite_llm)

    last_error: Optional[Exception] = None

    for attempt in range(_RAG_PIPELINE_MAX_RETRIES + 1):
        try:
            search_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)
            await search_log_callback("info", f"📂 Parsed search paths: {paths}", "\n", False)

            logger.info("[MODE 2] RAG search with query: %s, paths: %s", search_query, paths)

            search_result, llm_usages, references = await _run_rag_search(
                search_query, paths, search_mode, search_log_callback,
            )

            search_engine = get_search_instance()
            for usage in llm_usages:
                llm_usage_tracker.record_usage(
                    model=search_engine.llm._model,
                    usage=usage,
                )

            await manager.send_personal_message(json.dumps({
                "type": "search_complete",
                "message": "✅ Knowledge base search completed"
            }), websocket)

            sources["rag"] = [{
                "kb_name": paths_display,
                "content": f"Retrieved content from {paths_display}",
                "relevance_score": 0.92,
            }]
            if references:
                sources["references"] = references
            return search_result, sources

        except Exception as e:
            last_error = e
            friendly = _classify_error(e)

            if _is_transient_llm_error(e) and attempt < _RAG_PIPELINE_MAX_RETRIES:
                logger.warning(
                    "[MODE 2] Transient error on attempt %d/%d (%s), retrying in %.1fs",
                    attempt + 1, _RAG_PIPELINE_MAX_RETRIES + 1, friendly,
                    _RAG_PIPELINE_RETRY_DELAY,
                )
                await manager.send_personal_message(json.dumps({
                    "type": "status",
                    "stage": "retrying",
                    "message": f"⚠️ {friendly}, retrying..."
                }), websocket)
                await asyncio.sleep(_RAG_PIPELINE_RETRY_DELAY)
                continue

            # Permanent error or final retry exhausted — report and fall back
            logger.error("[MODE 2] RAG search failed: %s (%s)", friendly, e)

            await manager.send_personal_message(json.dumps({
                "type": "search_error",
                "message": f"❌ RAG search failed: {friendly}"
            }), websocket)
            await manager.send_personal_message(json.dumps({
                "type": "status",
                "stage": "fallback",
                "message": "⚠️ Falling back to pure chat mode..."
            }), websocket)

            response, sources = await _chat_only(message, websocket, manager, history=history)
            return response, sources

    # Should not be reached, but handle defensively
    response, sources = await _chat_only(message, websocket, manager, history=history)
    return response, sources


async def _chat_web_search(
    message: str,
    websocket: WebSocket,
    manager: ChatConnectionManager,
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Mode 3: Chat + Web Search (enable_rag=False, enable_web_search=True)."""
    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "web_search",
        "message": "🌐 Searching the web..."
    }), websocket)

    web_results = await _perform_web_search(message, websocket, manager)

    if not web_results or not web_results.get("sources"):
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "fallback",
            "message": "⚠️ Web search did not return results, falling back to pure chat mode..."
        }), websocket)
        logger.info("[MODE 3] Web search failed, falling back to chat only")
        return await _chat_only(message, websocket, manager, history=history)

    web_context = "\n\nBased on web search results:\n"
    for source in web_results["sources"]:
        web_context += f"- {source['title']}: {source['snippet']}\n"

    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "generating",
        "message": "💬 Generating response with web context..."
    }), websocket)

    envs: Dict[str, Any] = get_envs()
    llm_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)
    llm = OpenAIChat(
        api_key=envs["api_key"], base_url=envs["base_url"],
        model=envs["model_name"], log_callback=llm_log_callback,
    )

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the provided web search results to answer the user's question accurately."},
        *(history or []),
        {"role": "user", "content": f"{message}\n\nWeb search context:\n{web_context}"},
    ]

    llm_response = await llm.achat(messages=messages, stream=True)

    usage_data = llm_response.usage if llm_response.usage else {}
    llm_usage_tracker.record_usage(
        model=llm_response.model or envs["model_name"], usage=usage_data,
    )

    return llm_response.content, {"web": web_results["sources"]}


async def _chat_rag_web_search(
    message: str,
    kb_name: str,
    websocket: WebSocket,
    manager: ChatConnectionManager,
    search_mode: str = "FAST",
    *,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Mode 4: Chat + RAG + Web Search (enable_rag=True, enable_web_search=True)."""
    sources = {}
    paths, paths_display = _resolve_rag_paths(kb_name)
    if not paths:
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": "No search paths specified for RAG search."
        }), websocket)
        response = "Please specify search paths for RAG search."
        return response, sources

    # Multi-turn: rewrite query for retrieval
    search_query = message
    if history:
        envs_rw: Dict[str, Any] = get_envs()
        rewrite_llm = OpenAIChat(
            api_key=envs_rw["api_key"], base_url=envs_rw["base_url"], model=envs_rw["model_name"],
        )
        search_query = await _rewrite_query_with_context(message, history, rewrite_llm)

    rag_result = None

    for attempt in range(_RAG_PIPELINE_MAX_RETRIES + 1):
        try:
            search_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)
            await search_log_callback("info", f"📂 RAG search paths: {paths}", "\n", False)

            logger.info("[MODE 4] RAG search with query: %s, paths: %s", search_query, paths)

            rag_result, llm_usages, references = await _run_rag_search(
                search_query, paths, search_mode, search_log_callback,
            )

            search_engine = get_search_instance()
            for usage in llm_usages:
                llm_usage_tracker.record_usage(
                    model=search_engine.llm._model,
                    usage=usage,
                )

            await manager.send_personal_message(json.dumps({
                "type": "search_complete",
                "message": "✅ Knowledge base search completed"
            }), websocket)

            sources["rag"] = [{
                "kb_name": paths_display,
                "content": f"Retrieved from {paths_display}",
                "relevance_score": 0.92,
            }]
            if references:
                sources["references"] = references
            break  # success

        except Exception as e:
            friendly = _classify_error(e)

            if _is_transient_llm_error(e) and attempt < _RAG_PIPELINE_MAX_RETRIES:
                logger.warning(
                    "[MODE 4] Transient RAG error on attempt %d/%d (%s), retrying",
                    attempt + 1, _RAG_PIPELINE_MAX_RETRIES + 1, friendly,
                )
                await manager.send_personal_message(json.dumps({
                    "type": "status",
                    "stage": "retrying",
                    "message": f"⚠️ {friendly}, retrying..."
                }), websocket)
                await asyncio.sleep(_RAG_PIPELINE_RETRY_DELAY)
                continue

            logger.error("[MODE 4] RAG search failed: %s (%s)", friendly, e)
            await manager.send_personal_message(json.dumps({
                "type": "search_error",
                "message": f"⚠️ RAG search failed: {friendly}, continuing with web search..."
            }), websocket)
            rag_result = f"[RAG search unavailable: {friendly}]"
            sources["rag"] = [{"error": friendly}]
            break
    
    # Step 2: Perform web search
    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "web_search",
        "message": "🌐 Step 2/2: Searching the web..."
    }), websocket)

    # TODO: add llm usage
    web_results = await _perform_web_search(message, websocket, manager)
    sources["web"] = web_results["sources"]
    
    # Combine results
    web_context = "\n\n## Additional Web Sources:\n"
    for source in web_results["sources"]:
        web_context += f"- [{source['title']}]({source['url']})\n"
    
    # If RAG succeeded, use it as primary response; otherwise use web search only
    if rag_result and "[RAG search unavailable" not in rag_result:
        response = rag_result + web_context
    else:
        response = f"Based on web search results:\n{web_context}"
    
    return response, sources


# WebSocket endpoint for chat with integrated search
@router.websocket("/chat")
async def chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat conversations with integrated search
    
    Supports 4 modes:
    1. Pure chat: enable_rag=False, enable_web_search=False
    2. Chat + RAG: enable_rag=True, enable_web_search=False
    3. Chat + Web Search: enable_rag=False, enable_web_search=True (mock)
    4. Chat + RAG + Web Search: enable_rag=True, enable_web_search=True (RAG real, web mock)
    """
    # Auth check
    if not verify_ws_token(websocket):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    # Connection limit check
    connected = await manager.connect(websocket)
    if not connected:
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            message = request_data.get("message", "")
            session_id = request_data.get("session_id")
            history = request_data.get("history", [])
            kb_name = request_data.get("kb_name", "")
            enable_rag = request_data.get("enable_rag", False)
            enable_web_search = request_data.get("enable_web_search", False)
            search_mode = request_data.get("search_mode", "FAST")

            logger.debug("Chat request: rag=%s, mode=%s", enable_rag, search_mode)
            
            # Generate or use existing session ID
            if not session_id:
                session_id = f"chat_{uuid.uuid4().hex[:8]}"
            
            # Send session ID to client
            await manager.send_personal_message(json.dumps({
                "type": "session",
                "session_id": session_id
            }), websocket)
            
            # Store session data (in-memory + persistent)
            if session_id not in chat_sessions:
                chat_sessions[session_id] = {
                    "session_id": session_id,
                    "title": f"Chat Session",
                    "messages": [],
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "settings": {
                        "kb_name": kb_name,
                        "enable_rag": enable_rag,
                        "enable_web_search": enable_web_search,
                        "search_mode": search_mode,
                    }
                }
                # Save new session to persistent storage
                history_storage.save_session(chat_sessions[session_id])
            
            # Update session with new message
            session = chat_sessions[session_id]
            user_message = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            session["messages"].append(user_message)
            session["updated_at"] = datetime.now().isoformat()
            
            # Save user message to persistent storage
            history_storage.save_message(session_id, user_message)
            
            # ============================================================
            # Build conversation history for multi-turn support
            # ============================================================
            chat_history = _build_chat_history(session_id)

            if chat_history:
                envs_for_filter: Dict[str, Any] = get_envs()
                filter_llm = OpenAIChat(
                    api_key=envs_for_filter["api_key"],
                    base_url=envs_for_filter["base_url"],
                    model=envs_for_filter["model_name"],
                )
                chat_history = await _filter_relevant_history(message, chat_history, filter_llm)

            # ============================================================
            # Route to appropriate chat mode based on feature flags
            # ============================================================
            response = ""
            sources = {}

            if enable_rag and enable_web_search:
                logger.info("[MODE 4] Chat + RAG + Web Search")
                response, sources = await _chat_rag_web_search(
                    message, kb_name, websocket, manager,
                    search_mode=search_mode, history=chat_history,
                )

            elif enable_rag and not enable_web_search:
                logger.info("[MODE 2] Chat + RAG")
                response, sources = await _chat_rag(
                    message, kb_name, websocket, manager,
                    search_mode=search_mode, history=chat_history,
                )

            elif not enable_rag and enable_web_search:
                logger.info("[MODE 3] Chat + Web Search only")
                response, sources = await _chat_web_search(
                    message, websocket, manager, history=chat_history,
                )

            else:
                logger.info("[MODE 1] Pure chat mode")
                response, sources = await _chat_only(
                    message, websocket, manager, history=chat_history,
                )
            
            # ============================================================
            # Stream response to client
            # ============================================================
            words = response.split()
            
            for i, word in enumerate(words):
                await manager.send_personal_message(json.dumps({
                    "type": "stream",
                    "content": word + " "
                }), websocket)
                
                # Add small delay for realistic streaming
                if i % 3 == 0:  # Every 3 words
                    await asyncio.sleep(0.05)
            
            # Send sources if available
            if sources:
                await manager.send_personal_message(json.dumps({
                    "type": "sources",
                    **sources
                }), websocket)
            
            # Send final result
            await manager.send_personal_message(json.dumps({
                "type": "result",
                "content": response.strip(),
                "session_id": session_id
            }), websocket)
            
            # Store assistant response in session
            assistant_message = {
                "role": "assistant",
                "content": response.strip(),
                "sources": sources if sources else None,
                "timestamp": datetime.now().isoformat()
            }
            session["messages"].append(assistant_message)
            
            # Save assistant message to persistent storage
            history_storage.save_message(session_id, assistant_message)
            
            # Update session in persistent storage
            history_storage.save_session(session)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error occurred")
        logger.exception("WebSocket exception details")
        try:
            await manager.send_personal_message(json.dumps({
                "type": "error",
                "message": "An internal error occurred. Please try again."
            }), websocket)
        except:
            pass
        manager.disconnect(websocket)


# File picker endpoints
@router.post("/file-picker")
async def open_file_picker(request: Dict[str, Any]):
    """
    Open native file picker dialog using tkinter
    Returns real absolute paths from user's local filesystem
    """
    if not _ensure_tkinter():
        return {
            "success": False,
            "error": "Tkinter not available on this system",
            "data": []
        }
    
    dialog_type = request.get("type", "files")  # "files" or "directory"
    multiple = request.get("multiple", True)
    
    try:
        # Get absolute paths from user's local filesystem
        selected_paths = open_file_dialog(dialog_type, multiple)
        
        # Convert to absolute paths and validate they exist
        validated_paths = []
        for path in selected_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                validated_paths.append(abs_path)
        
        return {
            "success": True,
            "data": {
                "paths": validated_paths,
                "count": len(validated_paths),
                "type": dialog_type,
                "multiple": multiple
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to open file picker: {str(e)}",
            "data": []
        }

@router.get("/file-picker/status")
async def get_file_picker_status(request: Request):
    """Check if file picker is available on this system"""
    avail = _ensure_tkinter()
    client_host = request.client.host if request.client else ""
    is_local_client = client_host in ("127.0.0.1", "::1", "localhost")
    effective_tkinter = avail and is_local_client
    return {
        "success": True,
        "data": {
            "tkinter_available": effective_tkinter,
            "deployment_mode": "local" if effective_tkinter else "remote",
            "upload_enabled": True,
            "server_browser": True,
            "supported_types": ["files", "directory"],
            "features": {
                "multiple_files": effective_tkinter,
                "directory_selection": True,
                "absolute_paths": True
            }
        }
    }


@router.get("/file-browser")
async def browse_files(request: Request, path: str = "/", show_hidden: bool = False):
    """List files and directories at the given path (headless-safe alternative to Tkinter)"""
    client_ip = request.client.host if request.client else "unknown"

    # P1.2: Rate limiting
    if not file_browser_limiter.is_allowed(client_ip):
        return {"success": False, "error": "Too many requests, please try again later"}

    try:
        # P0.1: Remote mode requires SIRCHMUNK_ALLOWED_PATHS
        is_remote = client_ip not in ("127.0.0.1", "::1", "localhost")
        if is_remote:
            env_raw = os.getenv("SIRCHMUNK_ALLOWED_PATHS", "").strip()
            if not env_raw:
                audit_logger.log(client_ip=client_ip, action="browse", path=path, result="denied_no_config")
                return {"success": False, "error": "File browser is not available. Please contact the administrator."}

        abs_path = os.path.abspath(path)
        if not is_path_allowed(abs_path):
            logger.warning("File browser access denied: %s from %s", abs_path, client_ip)
            audit_logger.log(client_ip=client_ip, action="browse", path=path, result="denied")
            return {"success": False, "error": "Access denied"}
        if not os.path.exists(abs_path):
            return {"success": False, "error": "The specified path is not accessible"}
        if not os.path.isdir(abs_path):
            return {"success": False, "error": "The specified path is not accessible"}

        items = []
        for entry in os.scandir(abs_path):
            if not show_hidden and entry.name.startswith('.'):
                continue
            try:
                stat = entry.stat()
                items.append({
                    "name": entry.name,
                    "path": entry.path,
                    "is_dir": entry.is_dir(),
                    "size": stat.st_size if not entry.is_dir() else None,
                    "modified": stat.st_mtime,
                })
            except (PermissionError, OSError):
                continue

        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))

        audit_logger.log(client_ip=client_ip, action="browse", path=path, result="success")
        return {
            "success": True,
            "data": {
                "current_path": abs_path,
                "parent_path": os.path.dirname(abs_path),
                "items": items,
            }
        }
    except PermissionError:
        logger.warning("Permission denied for path %s from %s", abs_path, client_ip)
        audit_logger.log(client_ip=client_ip, action="browse", path=path, result="permission_denied")
        return {"success": False, "error": "Access denied"}
    except Exception as e:
        logger.exception("Unexpected error in file browser")
        audit_logger.log(client_ip=client_ip, action="browse", path=path, result="error")
        return {"success": False, "error": "An error occurred"}

# Chat session management endpoints
@router.get("/chat/sessions")
async def get_chat_sessions(limit: int = 20, offset: int = 0):
    """Get list of chat sessions"""
    sessions_list = list(chat_sessions.values())
    # Sort by updated_at (most recent first)
    sessions_list.sort(key=lambda x: x["updated_at"], reverse=True)
    
    # Apply pagination
    paginated_sessions = sessions_list[offset:offset + limit]
    
    # Format for response
    formatted_sessions = []
    for session in paginated_sessions:
        last_message = ""
        if session["messages"]:
            last_msg = session["messages"][-1]
            last_message = last_msg["content"][:100] + "..." if len(last_msg["content"]) > 100 else last_msg["content"]
        
        formatted_sessions.append({
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "message_count": len(session["messages"]),
            "last_message": last_message,
            "created_at": int(datetime.fromisoformat(session["created_at"]).timestamp()),
            "updated_at": int(datetime.fromisoformat(session["updated_at"]).timestamp()),
            "topics": ["AI", "Learning"]  # Mock topics
        })
    
    return {
        "success": True,
        "data": formatted_sessions,
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(sessions_list)
        }
    }

@router.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get specific chat session details"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session = chat_sessions[session_id]
    
    return {
        "success": True,
        "data": {
            "session_id": session["session_id"],
            "title": session.get("title", "Chat Session"),
            "messages": session["messages"],
            "settings": session.get("settings", {}),
            "created_at": session["created_at"],
            "updated_at": session["updated_at"]
        }
    }

@router.post("/chat/sessions/{session_id}/load")
async def load_chat_session(session_id: str):
    """Load chat session for continuation"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session = chat_sessions[session_id]
    
    return {
        "success": True,
        "message": f"Chat session loaded successfully",
        "data": {
            "session_id": session_id,
            "title": session.get("title", "Chat Session"),
            "message_count": len(session["messages"]),
            "loaded_at": datetime.now().isoformat()
        }
    }

# Search suggestions endpoint — returns files matching the query text
@router.get("/search/suggestions")
async def get_search_suggestions(query: str, kb_name: str = "", limit: int = 8):
    """Get file-name suggestions matching the typed query.

    Performs a fast filename-only search (via ``rga --files`` + regex
    filter) against the paths listed in *kb_name* (comma-separated).
    No LLM calls are involved.
    """
    if not query or len(query.strip()) < 2:
        return {"success": True, "data": [], "query": query}

    raw_paths = [p.strip() for p in kb_name.split(",") if p.strip()] if kb_name else []
    if not raw_paths:
        return {"success": True, "data": [], "query": query}

    try:
        from sirchmunk.retrieve.text_retriever import GrepRetriever
        from sirchmunk.search import AgenticSearch
        from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH
        import re as _re

        paths = AgenticSearch.validate_search_paths(
            raw_paths, require_exists=True,
        )
        if not paths:
            return {"success": True, "data": [], "query": query}

        retriever = GrepRetriever(
            work_path=os.getenv("SIRCHMUNK_WORK_PATH") or DEFAULT_SIRCHMUNK_WORK_PATH
        )
        escaped = _re.escape(query.strip())
        results = await retriever.retrieve_by_filename(
            patterns=[escaped],
            path=paths,
            max_depth=8,
        )

        def _human_size(path: str) -> str:
            try:
                size = os.path.getsize(path)
            except OSError:
                return ""
            for unit in ("B", "KB", "MB", "GB"):
                if size < 1024:
                    return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"

        suggestions = []
        for r in results[:limit]:
            filename = r.get("filename", "")
            filepath = r.get("path", "")
            ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else ""

            # Compute highlight range within the display name
            hl_start, hl_end = -1, -1
            try:
                match = _re.search(escaped, filename, _re.IGNORECASE)
                if match:
                    hl_start, hl_end = match.start(), match.end()
            except _re.error:
                pass

            suggestions.append({
                "filename": filepath,
                "display_name": filename,
                "type": ext,
                "size": _human_size(filepath),
                "kb_name": kb_name,
                "highlight_start": hl_start,
                "highlight_end": hl_end,
            })

        return {
            "success": True,
            "data": suggestions,
            "query": query,
            "total_matches": len(results),
        }
    except Exception as e:
        logger.warning(f"Suggestions search failed: {e}")
        return {"success": True, "data": [], "query": query}

@router.get("/search/knowledge-bases")
async def get_knowledge_bases():
    """Get list of available knowledge bases for search"""
    # Return empty list since we're using direct file paths now
    return {
        "success": True,
        "data": []
    }