"""Configured clients for Web/API entry points; explicit SDK clients stay intact."""
from contextlib import asynccontextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
import asyncio
import logging
import os
import inspect

from sirchmunk.storage.lifecycle import get_runtime

from .openai_chat import OpenAIChat
from .pool_config import PoolConfig, ConfigError, get_store
from .pool_router import RoutedChat, RoutingGuard, get_scheduler

logger = logging.getLogger(__name__)


def _legacy():
    return dict(base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                api_key=os.getenv("LLM_API_KEY", ""), model=os.getenv("LLM_MODEL_NAME", "gpt-5.2"))


@dataclass
class RequestState:
    config: PoolConfig
    path: str
    legacy: dict
    clients: list = field(default_factory=list)
    searchers: list = field(default_factory=list)
    background: set = field(default_factory=set)
    guard: RoutingGuard = field(default_factory=RoutingGuard)


_request: ContextVar[RequestState | None] = ContextVar("sirchmunk_llm_request", default=None)
_cleanup_tasks: set[asyncio.Task] = set()
_background_tasks: set[asyncio.Task] = set()
_thread_futures: set[asyncio.Future] = set()


def create_background_task(coroutine, *, runtime=None,
                           failure_code="background_work_failed", _track_request=True):
    """Reserve before dispatch; cancellation before first execution also releases.

    Search evolution and other detached coroutines must use this entry point.
    A task owns its lease until its done callback, never until an awaiter exits.
    """
    runtime = runtime or get_runtime()
    release = None
    try:
        release = runtime.reserve_background() if runtime else None
        task = asyncio.create_task(coroutine)
    except BaseException:
        if inspect.iscoroutine(coroutine):
            coroutine.close()
        if runtime and release:
            runtime.fail(failure_code)
        if release:
            release()
        raise
    _background_tasks.add(task)
    state = _request.get() if _track_request else None
    if state is not None:
        # Retain completed tasks until cleanup too: the searcher's legacy set
        # discards them immediately and would otherwise hide early failures.
        state.background.add(task)

    def finished(done):
        try:
            if done.cancelled():
                if runtime:
                    runtime.fail(failure_code)
            elif done.exception() is not None:
                if runtime:
                    runtime.fail(failure_code)
                logger.warning("Background task failed: %s", failure_code)
        finally:
            _background_tasks.discard(done)
            if release:
                release()

    task.add_done_callback(finished)
    return task


async def run_background_thread(function, *args, runtime=None, **kwargs):
    """Keep the lease until the real worker exits, even if its await is cancelled."""
    runtime = runtime or get_runtime()
    release = runtime.reserve_background() if runtime else None
    context = copy_context()

    def worker():
        try:
            return context.run(function, *args, **kwargs)
        except BaseException:
            if runtime:
                runtime.fail("background_thread_failed")
            raise
        finally:
            if release:
                release()

    try:
        future = asyncio.get_running_loop().run_in_executor(None, worker)
    except BaseException:
        if runtime:
            runtime.fail("background_thread_dispatch_failed")
        if release:
            release()
        raise
    # Retain an explicit strong reference after an awaiter is cancelled. Fetch
    # detached errors without changing the result seen by surviving awaiters.
    _thread_futures.add(future)
    def finished(done):
        try:
            if not done.cancelled():
                done.exception()
        finally:
            _thread_futures.discard(done)
    future.add_done_callback(finished)
    return await asyncio.shield(future)


def track_search_instance(searcher):
    """Own API-created storage without sharing mutable callbacks or search state."""
    state = _request.get()
    if state is not None:
        state.searchers.append(searcher)
    return searcher


async def _cleanup(state, background, runtime=None):
    # Existing evolution remains fire-and-forget. Keep its storage and LLM
    # transports alive until completion, then release them even on failure.
    errors = []
    if background:
        results = await asyncio.gather(*(asyncio.shield(task) for task in background),
                                       return_exceptions=True)
        errors.extend(result for result in results if isinstance(result, BaseException))
    for searcher in state.searchers:
        storage = getattr(searcher, "knowledge_storage", None)
        if storage is not None:
            try:
                await run_background_thread(storage.close, runtime=runtime)
            except Exception as exc:
                errors.append(exc)
                logger.warning("Request search storage cleanup failed")
    for client in state.clients:
        try:
            try:
                await client._async_client.close()
            finally:
                if runtime:
                    await run_background_thread(client._client.close, runtime=runtime)
                else:
                    client._client.close()
        except Exception as exc:
            errors.append(exc)
            logger.warning("Legacy LLM transport cleanup failed")
    state.background.clear()
    if runtime and errors:
        runtime.fail("request_cleanup_failed")
        raise RuntimeError("Managed request background/cleanup failed") from errors[0]


@asynccontextmanager
async def llm_request_scope(config=None):
    store = get_store()
    state = RequestState(config or store.read(), str(store.path), _legacy())
    runtime = get_runtime()
    token = _request.set(state)
    try:
        yield state
    finally:
        _request.reset(token)
        background = set(state.background)
        background.update(task for searcher in state.searchers
                          for task in getattr(searcher, "_background_tasks", ()))
        cleanup = create_background_task(
            _cleanup(state, background, runtime), runtime=runtime,
            failure_code="request_cleanup_failed", _track_request=False,
        )
        _cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(_cleanup_tasks.discard)
        if not background or (runtime and all(task.done() for task in background)):
            # Client cancellation must not abandon cleanup halfway through.
            await asyncio.shield(cleanup)


def create_configured_llm(log_callback=None):
    state = _request.get()
    store = get_store()
    config = state.config if state else store.read()
    if config.active_target.kind == "legacy":
        legacy = state.legacy if state else _legacy()
        if not legacy["api_key"]:
            raise ConfigError("LLM_API_KEY is not configured; configure the legacy model or select an LLM pool", 503)
        client = OpenAIChat(**legacy, log_callback=log_callback)
        if state:
            state.clients.append(client)
        return client
    return RoutedChat(config, get_scheduler(state.path if state else str(store.path)), log_callback,
                      guard=state.guard if state else None)


def configuration_identity():
    store = get_store()
    config = _request.get().config if _request.get() else store.read()
    return str(store.path), config.revision, config.active_target.kind, config.active_target.id


def configuration_status(config=None):
    config = config or get_store().read()
    target = config.active_target
    if target.kind == "legacy":
        values = _legacy()
        configured = all(values.values())
        label = values["model"]
    elif target.kind == "model":
        model = next(m for m in config.models if m.id == target.id)
        configured, label = model.enabled, model.model
    else:
        pool = next(p for p in config.pools if p.id == target.id)
        configured, label = pool.enabled and bool(pool.members), f"池：{pool.name}"
    return {"configured": bool(configured), "model": label if configured else None,
            "status": "ready" if configured else "not_configured",
            "active_target": target.model_dump(), "revision": config.revision}
