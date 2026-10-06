import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from sirchmunk.llm.pool_config import ConfigStore, get_store, Target
from sirchmunk.llm.pool_router import RoutedChat, Scheduler, PoolError
from sirchmunk.llm.factory import create_configured_llm, llm_request_scope, configuration_status
from sirchmunk.llm.openai_chat import OpenAIChat


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    store = get_store()
    for name in ("a", "b"):
        store.change(store.read().revision, "model", {
            "name": name, "model": name, "base_url": "http://mock-provider/v1", "api_key": "test-private",
        })
    c = store.read()
    c = store.change(c.revision, "pool", {"name": "test-pool", "strategy": "failover",
        "members": [{"model_id": m.id, "priority": i} for i, m in enumerate(c.models)]})
    return store.change(c.revision, "selection", {"kind": "pool", "id": c.pools[0].id})


def mock_http(monkeypatch, handler):
    from sirchmunk.llm import openai_chat
    real = openai.AsyncOpenAI
    monkeypatch.setattr(openai_chat, "AsyncOpenAI", lambda **kwargs: real(
        **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))


def response(model):
    return httpx.Response(200, json={
        "id": "test", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
    })


@pytest.mark.asyncio
async def test_real_sdk_http_failover_and_usage(pool, monkeypatch):
    calls = []
    async def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        assert request.headers["Authorization"] == "Bearer test-private"
        if body["model"] == "a":
            return httpx.Response(503, json={"error": {"message": "test-private"}})
        return response(body["model"])
    mock_http(monkeypatch, handler)
    async with llm_request_scope():
        client = create_configured_llm()
        assert isinstance(client, RoutedChat)
        result = await client.achat([{"role": "user", "content": "test"}], stream=False)
    assert calls == ["a", "b"]
    assert result.model == "b" and result.usage["total_tokens"] == 8 and result.usage["_model"] == "b"


class BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        data = {"id": "t", "object": "chat.completion.chunk", "created": 0, "model": "a",
                "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
        yield ("data: " + json.dumps(data) + "\n\n").encode()
        raise httpx.ReadError("private-error-body")


@pytest.mark.asyncio
async def test_real_sdk_stream_interruption_blocks_replay_across_clients(pool, monkeypatch):
    calls, emitted = [], []
    async def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=BrokenStream())
    mock_http(monkeypatch, handler)
    def callback(level, text, end, flush):
        emitted.append(text)
    async with llm_request_scope():
        with pytest.raises(PoolError, match="stream_interrupted_no_replay"):
            await create_configured_llm(callback).achat([])
        # Even if an outer retrieval layer retries via a fresh factory, no replay.
        with pytest.raises(PoolError, match="stream_interrupted_no_replay"):
            await create_configured_llm(callback).achat([])
    assert calls == ["a"] and emitted == ["partial"]


@pytest.mark.asyncio
async def test_selection_snapshot_and_no_legacy_key_requirement(pool):
    async with llm_request_scope():
        before = create_configured_llm()
        store = get_store()
        store.change(pool.revision, "selection", {"kind": "model", "id": pool.models[1].id})
        after = create_configured_llm()
        assert before.config.active_target == after.config.active_target == pool.active_target
    async with llm_request_scope():
        assert create_configured_llm().config.active_target.kind == "model"
        assert configuration_status()["configured"]


@pytest.mark.asyncio
async def test_legacy_key_hidden_and_preserved(pool, monkeypatch):
    from sirchmunk.api import settings
    monkeypatch.setenv("LLM_API_KEY", "legacy-private")
    monkeypatch.setenv("LLM_BASE_URL", "http://legacy.test/v1")
    store = get_store()
    store.change(pool.revision, "selection", {"kind": "legacy"})
    data = settings.get_current_env_variables()
    assert data["LLM_API_KEY"]["value"] == "***"
    assert "legacy-private" not in json.dumps(data)
    await settings.save_settings(settings.SaveSettingsRequest(environment={"LLM_API_KEY": "***"}))
    async with llm_request_scope():
        client = create_configured_llm()
        assert type(client) is OpenAIChat and client._client.api_key == "legacy-private"
        async_http = client._async_client
    assert async_http.is_closed()


@pytest.mark.asyncio
async def test_api_search_and_chat_factories_are_isolated(pool, monkeypatch):
    from sirchmunk.api import search, chat
    monkeypatch.setattr(search, "AgenticSearch", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(chat, "AgenticSearch", lambda **kw: SimpleNamespace(**kw))
    async with llm_request_scope():
        a, b = search._get_search_instance(), search._create_search_instance()
        c, d = chat.get_search_instance(), chat.get_search_instance()
        assert len({id(x) for x in (a, b, c, d)}) == 4
        assert all(isinstance(x.llm, RoutedChat) for x in (a, b, c, d))
        assert len({id(x.llm.scheduler) for x in (a, b, c, d)}) == 1


@pytest.mark.asyncio
async def test_rag_does_not_retry_pool_failure_as_pure_chat(pool, monkeypatch):
    from sirchmunk.api import chat
    monkeypatch.setattr(chat, "_resolve_rag_paths", lambda kb: (["mock"], "mock"))
    monkeypatch.setattr(chat.LogCallbackManager, "create_search_log_callback", AsyncMock(return_value=AsyncMock()))
    monkeypatch.setattr(chat, "_run_rag_search", AsyncMock(side_effect=PoolError("stream_interrupted_no_replay")))
    pure = AsyncMock()
    monkeypatch.setattr(chat, "_chat_only", pure)
    with pytest.raises(PoolError):
        await chat._chat_rag("query", "kb", object(), SimpleNamespace(send_personal_message=AsyncMock()))
    pure.assert_not_called()


@pytest.mark.asyncio
async def test_connection_probe_uses_target_without_changing_selection(pool, monkeypatch):
    from sirchmunk.api.llm_pools import test_configuration
    calls = []
    async def handler(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        return response(model)
    mock_http(monkeypatch, handler)
    snapshot = pool.model_copy(deep=True)
    snapshot.active_target = Target(kind="model", id=pool.models[1].id)
    result = await test_configuration(snapshot)
    assert result["success"] and result["model"] == "b"
    assert get_store().read().active_target == pool.active_target and calls == ["b"]


@pytest.mark.asyncio
async def test_http_capacity_failure_preserves_status(pool, monkeypatch):
    from sirchmunk.api import search
    from fastapi import HTTPException
    fake = SimpleNamespace(search=AsyncMock(side_effect=PoolError("queue_full", 429)))
    monkeypatch.setattr(search, "_get_search_instance", lambda: fake)
    with pytest.raises(HTTPException) as exc:
        await search.execute_search(search.SearchRequest(query="test"))
    assert exc.value.status_code == 429


def test_no_unrouted_openai_constructors_in_web_entrypoints():
    import inspect
    from sirchmunk.api import chat, search
    for module in (chat, search):
        assert "= OpenAIChat(" not in inspect.getsource(module)


@pytest.mark.asyncio
async def test_request_storage_is_closed_and_thread_stopped(pool, monkeypatch):
    from sirchmunk.api import search
    import sirchmunk.search as core_search
    # Storage lifecycle test, not a native-tool installation/integration test.
    monkeypatch.setattr(core_search, "check_dependencies", lambda: True)
    monkeypatch.setenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "false")
    for _ in range(3):
        async with llm_request_scope():
            instance = search._create_search_instance()
            storage = instance.knowledge_storage
            assert storage._daemon_thread.is_alive()
        assert storage.db is None and not storage._daemon_thread.is_alive()
        assert not storage._shutdown_registered


@pytest.mark.asyncio
async def test_background_evolution_keeps_resources_until_done(pool):
    from sirchmunk.llm.factory import track_search_instance, _cleanup_tasks
    gate = asyncio.Event()
    close = Mock()
    async with llm_request_scope():
        task = asyncio.create_task(gate.wait())
        track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close), _background_tasks={task}))
    close.assert_not_called()
    gate.set()
    await asyncio.gather(*list(_cleanup_tasks))
    close.assert_called_once()


@pytest.mark.asyncio
async def test_canceled_request_still_closes_storage(pool):
    from sirchmunk.llm.factory import track_search_instance, _cleanup_tasks
    entered = asyncio.Event()
    close = Mock()
    async def request():
        async with llm_request_scope():
            track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close)))
            entered.set()
            await asyncio.Event().wait()
    task = asyncio.create_task(request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.gather(*list(_cleanup_tasks))
    close.assert_called_once()
