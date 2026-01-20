# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Unified API endpoints for chat and search functionality
Provides WebSocket endpoint for real-time chat conversations with integrated search
"""
import platform

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import json
import asyncio
import uuid
from datetime import datetime
import random
import os
import threading
from sirchmunk.search import AgenticSearch
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.utils.constants import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
from api.components.history_storage import HistoryStorage
from api.components.settings_storage import SettingsStorage


# Try to import tkinter for file dialogs
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

router = APIRouter(prefix="/api/v1", tags=["chat", "search"])

# Initialize persistent history storage
history_storage = HistoryStorage()

# Initialize settings storage for LLM configuration (with error handling)
try:
    settings_storage = SettingsStorage()
except Exception as e:
    print(f"[WARNING] Failed to initialize SettingsStorage: {e}")
    settings_storage = None

# In-memory cache for active sessions (for backward compatibility)
chat_sessions = {}

# Active WebSocket connections
class ChatConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

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
    search_paths: Union[str, List[str]]  # Expects absolute file/directory paths from user's local filesystem
    mode: Optional[str] = "DEEP"
    max_depth: Optional[int] = 5
    top_k_files: Optional[int] = 3


def get_envs() -> Dict[str, Any]:
    """
    Get LLM configuration from settings storage or environment variables.
    """
    # Try to get from settings storage first (if available)
    if settings_storage is not None:
        base_url = settings_storage.get_env_variable("LLM_BASE_URL", "")
        api_key = settings_storage.get_env_variable("LLM_API_KEY", "")
        model_name = settings_storage.get_env_variable("LLM_MODEL_NAME", "")
    else:
        base_url = ""
        api_key = ""
        model_name = ""

    # Fallback to environment variables if not in settings
    if not base_url:
        base_url = LLM_BASE_URL
    if not api_key:
        api_key = LLM_API_KEY
    if not model_name:
        model_name = LLM_MODEL_NAME

    print(f"[ENV CONFIG] base_url={base_url}, model_name={model_name}, api_key={'***' if api_key else '(not set)'}")

    return dict(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )


def get_search_instance(log_callback=None):
    """
    Get configured search instance with optional log callback.
    
    Creates OpenAIChat instance with settings from DuckDB (priority) or environment variables (fallback).
    
    Args:
        log_callback: Optional callback for logging
        
    Returns:
        Configured AgenticSearch instance
    """
    # Get LLM configuration from settings storage (priority) or env variables (fallback)
    try:
        envs = get_envs()
        
        # Create OpenAI LLM instance with retrieved configuration
        llm = OpenAIChat(
            base_url=envs["base_url"],
            api_key=envs["api_key"],
            model=envs["model_name"],
            log_callback=log_callback,
        )
        
        # Create and return AgenticSearch instance with configured LLM
        return AgenticSearch(llm=llm, log_callback=log_callback)
    
    except Exception as e:
        # If settings retrieval fails, fall back to default AgenticSearch initialization
        print(f"[WARNING] Please config ENVs: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME. Error: {e}")
        return AgenticSearch(log_callback=log_callback)


_DIALOG_LOCK = threading.Lock()
_ROOT_INSTANCE = None

def _get_bg_root():
    """
    Retrieves or creates the global, hidden Tk root window.
    """
    global _ROOT_INSTANCE

    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Tkinter must be executed on the Main Thread.")

    if _ROOT_INSTANCE is None or not _ROOT_INSTANCE.winfo_exists():
        _ROOT_INSTANCE = tk.Tk()
        _ROOT_INSTANCE.withdraw()  # Completely hide the main window

        _ROOT_INSTANCE.attributes("-alpha", 0.0)
        _ROOT_INSTANCE.attributes("-topmost", True)

        if platform.system() == "Darwin":
            _ROOT_INSTANCE.update_idletasks()

    return _ROOT_INSTANCE


def open_file_dialog(dialog_type: str = "files", multiple: bool = True) -> List[str]:
    """
    Open native file dialog (Win/Mac/Linux).
    Robust version using a singleton root window with extended file type support.
    """
    if not _DIALOG_LOCK.acquire(blocking=False):
        return []
    
    selected_paths = []
    
    try:
        # Get singleton root
        root = _get_bg_root()

        # Lift window to ensure the dialog appears on top
        root.lift()
        root.attributes("-topmost", True)
        # Force update to ensure focus is correct in the Python process
        root.update()

        kwargs = {"parent": root, "title": "Select File(s)"}

        # Handle file dialog types
        if dialog_type == "files":
            filetypes = [
                ("All Files", "*.*"),
                ("PDF Documents", "*.pdf"),
                ("Word Documents", "*.docx *.doc"),
                ("PowerPoint Presentations", "*.pptx *.ppt"),
                ("Excel Spreadsheets", "*.xlsx *.xls *.csv"),
                ("Markdown & Text", "*.md *.markdown *.txt"),
                ("JSON & Data", "*.json *.xml *.yaml *.yml"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.svg *.webp"),
                ("Python Files", "*.py *.pyw"),
                ("Web Files", "*.html *.htm *.css *.js"),
                ("Archives", "*.zip *.tar *.gz *.rar *.7z")
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

        root.attributes("-topmost", False)
        root.update()
            
    except Exception as e:
        print(f"Dialog Error: {e}")
        selected_paths = []
    
    finally:
        # Release the lock
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
    manager: ChatConnectionManager
) -> tuple[str, Dict[str, Any]]:
    """
    Mode 1: Pure chat mode (no RAG, no web search)
    Direct LLM chat without any retrieval augmentation
    """
    try:
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "generating",
            "message": "💬 Generating response..."
        }), websocket)
        
        # Create log callback for streaming LLM output
        llm_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)
        
        # Initialize OpenAI client with log callback for streaming
        envs: Dict[str, Any] = get_envs()
        llm = OpenAIChat(
            api_key=envs["api_key"],
            base_url=envs["base_url"],
            model=envs["model_name"],
            log_callback=llm_log_callback
        )
        
        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."},
            {"role": "user", "content": message}
        ]
        
        # Generate response with streaming
        response = await llm.achat(messages=messages, stream=True)
        
        sources = {}
        
        return response, sources
    
    except Exception as e:
        # Send error message to frontend
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": f"LLM chat failed: {str(e)}"
        }), websocket)
        
        # Re-raise to be caught by outer handler
        raise


async def _chat_rag(
    message: str,
    kb_name: str,
    websocket: WebSocket,
    manager: ChatConnectionManager
) -> tuple[str, Dict[str, Any]]:
    """
    Mode 2: Chat + RAG (enable_rag=True, enable_web_search=False)
    LLM chat with knowledge base retrieval
    """
    sources = {}
    if not kb_name:
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": "No search paths specified for RAG search."
        }), websocket)
        response = "Please specify search paths for RAG search."
        return response, sources
    
    try:
        # Create log callback for streaming search logs
        search_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)

        # Send RAG start signal
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "rag",
            "message": f"🔍 Searching knowledge base: {kb_name}"
        }), websocket)

        # Create search instance with log callback
        search_engine = get_search_instance(log_callback=search_log_callback)
        
        search_paths = [path.strip() for path in kb_name.split(",")]
        await search_log_callback("info", f"📂 Parsed search paths: {search_paths}", "\n", False)

        # Execute RAG search
        print(f"[MODE 2] RAG search with query: {message}, paths: {search_paths}")
        
        search_result = await search_engine.search(
            query=message,
            search_paths=search_paths,
            max_depth=5,
            top_k_files=3,
            verbose=True
        )
        
        # Send search completion
        await manager.send_personal_message(json.dumps({
            "type": "search_complete",
            "message": "✅ Knowledge base search completed"
        }), websocket)
        
        # Use search result as response
        response = search_result
        
        # Add RAG sources
        sources["rag"] = [
            {
                "kb_name": kb_name,
                "content": f"Retrieved content from {kb_name}",
                "relevance_score": 0.92
            }
        ]
        
    except Exception as e:
        # Send search error
        await manager.send_personal_message(json.dumps({
            "type": "search_error",
            "message": f"❌ RAG search failed: {str(e)}"
        }), websocket)
        
        # Fallback to chat only
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "fallback",
            "message": "⚠️ RAG mode did not find relevant results, falling back to pure chat mode..."
        }), websocket)
        
        print(f"[MODE 2] RAG search failed, falling back to chat only: {str(e)}")
        response, sources = await _chat_only(message, websocket, manager)
    
    return response, sources


async def _chat_web_search(
    message: str,
    websocket: WebSocket,
    manager: ChatConnectionManager
) -> tuple[str, Dict[str, Any]]:
    """
    Mode 3: Chat + Web Search (enable_rag=False, enable_web_search=True)
    LLM chat with web search augmentation (currently mock)
    """
    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "web_search",
        "message": "🌐 Searching the web..."
    }), websocket)
    
    # Perform mock web search
    web_results = await _perform_web_search(message, websocket, manager)
    
    # Check if web search returned valid results
    if not web_results or not web_results.get("sources"):
        # Fallback to chat only
        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "fallback",
            "message": "⚠️ Web search did not return results, falling back to pure chat mode..."
        }), websocket)
        
        print(f"[MODE 3] Web search failed, falling back to chat only")
        response, sources = await _chat_only(message, websocket, manager)
        return response, sources
    
    # Generate response enhanced with web search results
    web_context = "\n\nBased on web search results:\n"
    for source in web_results["sources"]:
        web_context += f"- {source['title']}: {source['snippet']}\n"
    
    # Use LLM to generate response with web context
    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "generating",
        "message": "💬 Generating response with web context..."
    }), websocket)

    envs: Dict[str, Any] = get_envs()
    llm_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)
    llm = OpenAIChat(
        api_key=envs["api_key"],
        base_url=envs["base_url"],
        model=envs["model_name"],
        log_callback=llm_log_callback
    )
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the provided web search results to answer the user's question accurately."},
        {"role": "user", "content": f"{message}\n\nWeb search context:\n{web_context}"}
    ]
    
    response = await llm.achat(messages=messages, stream=True)
    sources = {"web": web_results["sources"]}
    
    return response, sources


async def _chat_rag_web_search(
    message: str,
    kb_name: str,
    websocket: WebSocket,
    manager: ChatConnectionManager
) -> tuple[str, Dict[str, Any]]:
    """
    Mode 4: Chat + RAG + Web Search (enable_rag=True, enable_web_search=True)
    LLM chat with both knowledge base retrieval and web search
    """
    sources = {}
    if not kb_name:
        await manager.send_personal_message(json.dumps({
            "type": "error",
            "message": "No search paths specified for RAG search."
        }), websocket)
        response = "Please specify search paths for RAG search."
        return response, sources

    # Step 1: Perform RAG search
    try:
        search_log_callback = await LogCallbackManager.create_search_log_callback(websocket, manager)

        await manager.send_personal_message(json.dumps({
            "type": "status",
            "stage": "rag",
            "message": f"🔍 Step 1/2: Searching knowledge base: {kb_name}"
        }), websocket)

        search_engine = get_search_instance(log_callback=search_log_callback)
        search_paths = [path.strip() for path in kb_name.split(",")]
        await search_log_callback("info", f"📂 RAG search paths: {search_paths}", "\n", False)

        print(f"[MODE 4] RAG search with query: {message}, paths: {search_paths}")
        
        rag_result = await search_engine.search(
            query=message,
            search_paths=search_paths,
            max_depth=5,
            top_k_files=3,
            verbose=True
        )
        
        await manager.send_personal_message(json.dumps({
            "type": "search_complete",
            "message": "✅ Knowledge base search completed"
        }), websocket)
        
        sources["rag"] = [
            {
                "kb_name": kb_name,
                "content": f"Retrieved from {kb_name}",
                "relevance_score": 0.92
            }
        ]
        
    except Exception as e:
        await manager.send_personal_message(json.dumps({
            "type": "search_error",
            "message": f"⚠️ RAG search failed: {str(e)}, continuing with web search..."
        }), websocket)
        rag_result = f"[RAG search unavailable: {str(e)}]"
        sources["rag"] = [{"error": str(e)}]
    
    # Step 2: Perform web search
    await manager.send_personal_message(json.dumps({
        "type": "status",
        "stage": "web_search",
        "message": "🌐 Step 2/2: Searching the web..."
    }), websocket)
    
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
    await manager.connect(websocket)
    
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

            print(f"\n{'='*60}")
            print(f"[CHAT REQUEST] Message: {message[:50]}...")
            print(f"[CHAT REQUEST] KB: {kb_name}, RAG: {enable_rag}, Web: {enable_web_search}")
            print(f"{'='*60}\n")
            
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
                        "enable_web_search": enable_web_search
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
            # Route to appropriate chat mode based on feature flags
            # ============================================================
            response = ""
            sources = {}
            
            if enable_rag and enable_web_search:
                # Mode 4: Chat + RAG + Web Search
                print(f"[MODE 4] Chat + RAG + Web Search")
                response, sources = await _chat_rag_web_search(
                    message, kb_name, websocket, manager
                )
                
            elif enable_rag and not enable_web_search:
                # Mode 2: Chat + RAG
                print(f"[MODE 2] Chat + RAG")
                response, sources = await _chat_rag(
                    message, kb_name, websocket, manager
                )
                    
            elif not enable_rag and enable_web_search:
                # Mode 3: Chat + Web Search only
                print(f"[MODE 3] Chat + Web Search only")
                response, sources = await _chat_web_search(
                    message, websocket, manager
                )
                
            else:
                # Mode 1: Pure chat (no RAG, no web search)
                print(f"[MODE 1] Pure chat mode")
                response, sources = await _chat_only(
                    message, websocket, manager
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
        print(f"[ERROR] WebSocket error: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            await manager.send_personal_message(json.dumps({
                "type": "error",
                "message": f"An error occurred: {str(e)}"
            }), websocket)
        except:
            pass
        manager.disconnect(websocket)

# REST API endpoints for search functionality
# @router.post("/search")
# async def search_files(request: SearchRequest):
#     """Search files using Sirchmunk search engine"""
#     try:
#         # Create log callback using LogCallbackManager
#         rest_log_callback = await LogCallbackManager.create_rest_log_callback()
#
#         # Get search instance with log callback
#         search_engine = get_search_instance(log_callback=rest_log_callback)
#
#         # Convert search_paths to appropriate format
#         if isinstance(request.search_paths, str):
#             search_paths = request.search_paths
#         else:
#             search_paths = request.search_paths
#
#         # Perform search
#         result = await search_engine.search(
#             query=request.query,
#             search_paths=search_paths,
#             max_depth=request.max_depth,
#             top_k_files=request.top_k_files
#         )
#
#         return {
#             "success": True,
#             "data": {
#                 "query": request.query,
#                 "search_paths": request.search_paths,
#                 "result": result
#             }
#         }
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
#
# @router.websocket("/search/ws")
# async def search_websocket(websocket: WebSocket):
#     """WebSocket endpoint for real-time search with streaming logs"""
#     await websocket.accept()
#
#     try:
#         while True:
#             # Receive search request from client
#             data = await websocket.receive_text()
#             request_data = json.loads(data)
#
#             query = request_data.get("query", "")
#             search_paths = request_data.get("search_paths", [])
#             max_depth = request_data.get("max_depth", 5)
#             top_k_files = request_data.get("top_k_files", 3)
#
#             if not query or not search_paths:
#                 await websocket.send_text(json.dumps({
#                     "type": "error",
#                     "message": "Query and search_paths are required"
#                 }))
#                 continue
#
#             # Create log callback for streaming
#             log_callback = await LogCallbackManager.create_websocket_log_callback(websocket)
#
#             try:
#                 # Create search instance with log callback
#                 search_engine = get_search_instance(log_callback=log_callback)
#
#                 # Send start signal
#                 await websocket.send_text(json.dumps({
#                     "type": "start",
#                     "query": query,
#                     "search_paths": search_paths
#                 }))
#
#                 # Perform search with streaming logs
#                 result = await search_engine.search(
#                     query=query,
#                     search_paths=search_paths,
#                     max_depth=max_depth,
#                     top_k_files=top_k_files,
#                     verbose=True
#                 )
#
#                 # Send final result
#                 await websocket.send_text(json.dumps({
#                     "type": "result",
#                     "success": True,
#                     "data": {
#                         "query": query,
#                         "search_paths": search_paths,
#                         "result": result
#                     }
#                 }))
#
#             except Exception as e:
#                 await websocket.send_text(json.dumps({
#                     "type": "error",
#                     "message": f"Search failed: {str(e)}"
#                 }))
#
#     except WebSocketDisconnect:
#         pass
#     except Exception as e:
#         try:
#             await websocket.send_text(json.dumps({
#                 "type": "error",
#                 "message": f"WebSocket error: {str(e)}"
#             }))
#         except:
#             pass

# File picker endpoints
@router.post("/file-picker")
async def open_file_picker(request: Dict[str, Any]):
    """
    Open native file picker dialog using tkinter
    Returns real absolute paths from user's local filesystem
    """
    if not TKINTER_AVAILABLE:
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
async def get_file_picker_status():
    """Check if file picker is available on this system"""
    return {
        "success": True,
        "data": {
            "tkinter_available": TKINTER_AVAILABLE,
            "supported_types": ["files", "directory"] if TKINTER_AVAILABLE else [],
            "features": {
                "multiple_files": TKINTER_AVAILABLE,
                "directory_selection": TKINTER_AVAILABLE,
                "absolute_paths": TKINTER_AVAILABLE
            }
        }
    }

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

# Legacy search endpoints for backward compatibility
@router.get("/search/{kb_name}/suggestions")
async def get_search_suggestions(kb_name: str, query: str, limit: int = 8):
    """Get search suggestions - kept for backward compatibility"""
    # For now, return empty suggestions since we're using real file search
    if not query or len(query.strip()) < 2:
        return {
            "success": True,
            "data": [],
            "query": query
        }

    return {
        "success": True,
        "data": [],
        "query": query,
        "total_matches": 0
    }

@router.get("/search/knowledge-bases")
async def get_knowledge_bases():
    """Get list of available knowledge bases for search"""
    # Return empty list since we're using direct file paths now
    return {
        "success": True,
        "data": []
    }