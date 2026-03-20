# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Search API endpoints for CLI and programmatic access.

Provides HTTP endpoints for executing AgenticSearch queries,
designed for CLI client mode and external integrations.

Endpoints
---------
POST /api/v1/search          Synchronous search (JSON response).
POST /api/v1/search/stream   Streaming search via Server-Sent Events (SSE).
GET  /api/v1/search/status   Service health / configuration check.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.search import AgenticSearch
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


# ===================================================================
#  Request / Response models
# ===================================================================

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query or question")
    paths: Optional[Union[str, List[str]]] = Field(
        default=None,
        description=(
            "Directory or file path(s) to search: a single string or a list of strings. "
            "Omit this field, or pass null, empty string '', empty list [], or only "
            "whitespace / empty entries, to use SIRCHMUNK_SEARCH_PATHS from the server "
            "environment (~/.sirchmunk/.env when loaded). Explicit non-empty paths always "
            "take priority over the environment default (then AgenticSearch falls back to "
            "cwd if env is also unset)."
        ),
    )
    mode: Literal["DEEP", "FAST", "FILENAME_ONLY"] = Field(
        default="FAST",
        description="Search mode: FAST (greedy search, 2-5s), DEEP (comprehensive analysis, 10-30s), or FILENAME_ONLY (file discovery, <1s)"
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum directory depth to search"
    )
    top_k_files: Optional[int] = Field(
        default=None,
        description="Number of top files to return"
    )
    max_loops: Optional[int] = Field(
        default=None,
        description="Maximum ReAct iterations (DEEP mode)"
    )
    max_token_budget: Optional[int] = Field(
        default=None,
        description="LLM token budget (DEEP mode)"
    )
    enable_dir_scan: bool = Field(
        default=True,
        description="Enable directory scanning (DEEP mode)"
    )
    include_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to include (glob)"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to exclude (glob)"
    )
    return_context: bool = Field(
        default=False,
        description="Return full SearchContext with KnowledgeCluster, answer, and telemetry"
    )


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    success: bool
    data: dict
    error: Optional[str] = None


# ===================================================================
#  Concurrency control
# ===================================================================

_MAX_CONCURRENT_SEARCHES = int(os.getenv("SIRCHMUNK_MAX_CONCURRENT_SEARCHES", "3"))
_search_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_SEARCHES)


# ===================================================================
#  Cached singleton (for non-streaming endpoint)
# ===================================================================

_search_instance: Optional[AgenticSearch] = None
_search_config: Optional[tuple] = None


def _read_llm_config() -> tuple:
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-5.2")
    return api_key, base_url, model_name


def _get_search_instance() -> AgenticSearch:
    """Get or create AgenticSearch singleton (for non-streaming use)."""
    global _search_instance, _search_config

    api_key, base_url, model_name = _read_llm_config()
    current_config = (api_key, base_url, model_name)

    if _search_instance is not None and current_config == _search_config:
        return _search_instance

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_API_KEY is not configured. Set it in your environment or .env file."
        )

    llm = OpenAIChat(base_url=base_url, api_key=api_key, model=model_name)

    enable_cluster_reuse = os.getenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "true").lower() == "true"
    cluster_sim_threshold = float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.85"))
    cluster_sim_top_k = int(os.getenv("CLUSTER_SIM_TOP_K", "3"))

    _search_instance = AgenticSearch(
        llm=llm,
        work_path=DEFAULT_SIRCHMUNK_WORK_PATH,
        verbose=False,
        reuse_knowledge=enable_cluster_reuse,
        cluster_sim_threshold=cluster_sim_threshold,
        cluster_sim_top_k=cluster_sim_top_k,
    )
    _search_config = current_config
    logger.info("AgenticSearch instance created for API")
    return _search_instance


# ===================================================================
#  Per-request search factory (for SSE streaming — concurrent-safe)
# ===================================================================

def _create_search_instance(log_callback=None) -> AgenticSearch:
    """Create an ``AgenticSearch`` scoped to one SSE request.

    Each concurrent request gets its own instance so that
    ``log_callback`` routing is isolated.  The heavy resources
    (embedding model, sentence-transformers) are cached at module
    level and shared automatically.
    """
    api_key, base_url, model_name = _read_llm_config()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_API_KEY is not configured.",
        )

    llm = OpenAIChat(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        log_callback=log_callback,
    )

    enable_cluster_reuse = os.getenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "true").lower() == "true"
    cluster_sim_threshold = float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.85"))
    cluster_sim_top_k = int(os.getenv("CLUSTER_SIM_TOP_K", "3"))

    return AgenticSearch(
        llm=llm,
        work_path=DEFAULT_SIRCHMUNK_WORK_PATH,
        verbose=False,
        log_callback=log_callback,
        reuse_knowledge=enable_cluster_reuse,
        cluster_sim_threshold=cluster_sim_threshold,
        cluster_sim_top_k=cluster_sim_top_k,
    )


# ===================================================================
#  Path coercion (HTTP API contract)
# ===================================================================

def _normalize_api_paths(
    paths: Optional[Union[str, List[str]]],
) -> Optional[Union[str, List[str]]]:
    """Return ``None`` when the client did not supply usable paths.

    ``None``, missing key, ``""``, ``[]``, or lists of only blank strings
    all mean: defer to ``AgenticSearch._resolve_paths`` (explicit request
    param absent → ``SIRCHMUNK_SEARCH_PATHS`` → cwd).

    A single non-empty path may be returned as a bare ``str`` (one path)
    or remains a one-element list from JSON; both are accepted by
    ``AgenticSearch.search``.
    """
    if paths is None:
        return None
    if isinstance(paths, str):
        stripped = paths.strip()
        return None if stripped == "" else stripped
    if not isinstance(paths, list):
        return None
    cleaned: List[str] = []
    for p in paths:
        if not isinstance(p, str):
            continue
        s = p.strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


# ===================================================================
#  SSE helpers
# ===================================================================

def _sse_event(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _build_search_kwargs(request: SearchRequest) -> dict:
    kwargs = {
        "query": request.query,
        "paths": _normalize_api_paths(request.paths),
        "mode": request.mode,
        "enable_dir_scan": request.enable_dir_scan,
        "return_context": request.return_context,
    }
    if request.max_depth is not None:
        kwargs["max_depth"] = request.max_depth
    if request.top_k_files is not None:
        kwargs["top_k_files"] = request.top_k_files
    if request.max_loops is not None:
        kwargs["max_loops"] = request.max_loops
    if request.max_token_budget is not None:
        kwargs["max_token_budget"] = request.max_token_budget
    if request.include_patterns:
        kwargs["include"] = request.include_patterns
    if request.exclude_patterns:
        kwargs["exclude"] = request.exclude_patterns
    return kwargs


def _format_result(result, request: SearchRequest) -> dict:
    """Convert search result to a JSON-serialisable dict."""
    if request.return_context and hasattr(result, "to_dict"):
        return {"type": "context", **result.to_dict()}
    if isinstance(result, list):
        return {"type": "files", "files": result, "count": len(result)}
    return {
        "type": "summary",
        "summary": str(result) if result else "No results found.",
    }


# ===================================================================
#  POST /api/v1/search  (synchronous JSON response)
# ===================================================================

@router.post("/search")
async def execute_search(request: SearchRequest) -> SearchResponse:
    """Execute an AgenticSearch query and return a JSON response."""
    try:
        async with _search_semaphore:
            searcher = _get_search_instance()
            kwargs = _build_search_kwargs(request)
            logger.debug(
                "Executing search: query='%s', mode=%s, paths(raw)=%s paths(resolved)=%s",
                request.query,
                request.mode,
                request.paths,
                kwargs.get("paths"),
            )
            result = await searcher.search(**kwargs)

        return SearchResponse(success=True, data=_format_result(result, request))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ===================================================================
#  POST /api/v1/search/stream  (SSE streaming)
# ===================================================================

@router.post("/search/stream")
async def execute_search_stream(request: SearchRequest, http_request: Request):
    """Execute a search with real-time log streaming via Server-Sent Events.

    **Event types**

    ``event: log``
        Streaming log line emitted during the search.
        ``data: {"level": "info", "message": "...", "timestamp": ...}``

    ``event: result``
        Final search result (same schema as ``POST /search``).
        ``data: {"success": true, "data": {...}}``

    ``event: error``
        Fatal error that terminated the search.
        ``data: {"error": "..."}``

    **Concurrency**

    The server limits concurrent searches to ``SIRCHMUNK_MAX_CONCURRENT_SEARCHES``
    (default 3).  If the limit is reached, the connection receives an immediate
    ``event: error`` with an appropriate message.

    **Client example** (Python)::

        import httpx

        with httpx.stream("POST", "http://host:8584/api/v1/search/stream",
                          json={"query": "find auth bugs", "mode": "FAST"}) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    print(json.loads(line[6:]))
    """
    request_id = uuid.uuid4().hex[:12]
    log_queue: asyncio.Queue = asyncio.Queue()

    # Async log_callback that matches LogCallback signature:
    #   (level: str, message: str, end: str, flush: bool) -> None
    async def _log_callback(level: str, message: str, end: str, flush: bool):
        await log_queue.put({
            "level": level,
            "message": message,
            "timestamp": time.time(),
            "flush": flush,
        })

    _SENTINEL = object()

    async def _run_search():
        """Run search in a background task, push result/error via queue."""
        try:
            async with _search_semaphore:
                searcher = _create_search_instance(log_callback=_log_callback)
                skwargs = _build_search_kwargs(request)
                logger.debug(
                    "[stream:%s] Starting search: query='%s' mode=%s paths(resolved)=%s",
                    request_id,
                    request.query,
                    request.mode,
                    skwargs.get("paths"),
                )
                result = await searcher.search(**skwargs)

            formatted = _format_result(result, request)
            await log_queue.put(("result", {"success": True, "data": formatted}))
        except Exception as exc:
            logger.error("[stream:%s] Search failed: %s", request_id, exc, exc_info=True)
            await log_queue.put(("error", {"error": str(exc)}))
        finally:
            await log_queue.put(_SENTINEL)

    async def _sse_generator() -> AsyncGenerator[str, None]:
        """Yield SSE frames from the log queue until the search completes."""
        task = asyncio.create_task(_run_search())
        try:
            while True:
                # Check for client disconnect
                if await http_request.is_disconnected():
                    task.cancel()
                    break

                try:
                    item = await asyncio.wait_for(log_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # Send keep-alive comment to prevent proxy timeouts
                    yield ": heartbeat\n\n"
                    continue

                if item is _SENTINEL:
                    break

                if isinstance(item, tuple):
                    event_type, payload = item
                    yield _sse_event(event_type, payload)
                else:
                    yield _sse_event("log", item)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-Id": request_id,
        },
    )


# ===================================================================
#  GET /api/v1/search/status
# ===================================================================

@router.get("/search/status")
async def get_search_status():
    """Get search service status."""
    try:
        api_key, _, model_name = _read_llm_config()
        has_api_key = bool(api_key)

        return {
            "success": True,
            "data": {
                "status": "ready" if has_api_key else "not_configured",
                "llm_configured": has_api_key,
                "llm_model": model_name if has_api_key else None,
                "work_path": DEFAULT_SIRCHMUNK_WORK_PATH,
                "max_concurrent_searches": _MAX_CONCURRENT_SEARCHES,
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
