import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from sirchmunk.llm.openai_chat import OpenAIChat, OpenAIChatResponse, LLMTokenBudgetExceeded
from sirchmunk.llm.pool_config import ModelSpec, PoolConfig, PoolSpec, Member, Target
from sirchmunk.llm.pool_router import Scheduler, RoutedChat, PoolError


def config(strategy="weighted", cap=4, **pool_options):
    return PoolConfig(revision=1, models=[
        ModelSpec(id=i, name=i, model=i, base_url="http://localhost:9001/v1", max_concurrency=cap)
        for i in ("a", "b")
    ], pools=[PoolSpec(id="p", name="P", strategy=strategy, max_concurrency=cap,
                       members=[Member(model_id="a", weight=3), Member(model_id="b", priority=1)],
                       **pool_options)], active_target=Target(kind="pool", id="p"))


class Transport:
    def __init__(self, behavior=None):
        self.behavior = behavior
        self.called = []
        self.closed = []
        self.clients = []

    def factory(self, **kwargs):
        assert kwargs["max_retries"] == 0
        model = kwargs["model"]
        client = SimpleNamespace()
        self.clients.append(client)
        client._build_request_kwargs = lambda stream, thinking, **kwargs: kwargs
        client._async_client = SimpleNamespace(close=AsyncMock())
        client._client = SimpleNamespace(close=Mock(side_effect=lambda: self.closed.append(model)))
        async def call(messages, stream, request_kwargs):
            self.called.append(model)
            if self.behavior:
                await self.behavior(model, client)
            return OpenAIChatResponse(content=model, model=model, usage={"total_tokens": 10})
        client._do_achat = call
        return client


def upstream_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "http://localhost/v1"))
    return openai.APIStatusError("private-provider-error", response=response, body={"secret": "private"})


@pytest.mark.asyncio
async def test_smooth_weighted_3_to_1_and_actual_model():
    s, t = Scheduler(), Transport()
    client = RoutedChat(config(), s, client_factory=t.factory)
    for _ in range(40):
        result = await client.achat([], stream=False)
        assert result.model == result.content
    assert Counter(t.called) == {"a": 30, "b": 10}
    assert len(t.closed) == 40 and s.pools["p"] == 0


@pytest.mark.asyncio
async def test_full_primary_uses_backup_and_prefers_recovered_primary():
    s, c = Scheduler(), config("failover", cap=1)
    c.pools[0].max_concurrency = 2
    c.models[1].max_concurrency = 2
    started, release = asyncio.Event(), asyncio.Event()
    async def behavior(model, client):
        if model == "a":
            started.set()
            await release.wait()
    t = Transport(behavior)
    client = RoutedChat(c, s, client_factory=t.factory)
    primary = asyncio.create_task(client.achat([], stream=False))
    await started.wait()
    assert (await client.achat([], stream=False)).model == "b"
    assert s.models["a"].in_flight == 1
    release.set()
    await primary
    assert (await client.achat([], stream=False)).model == "a"
    assert s.pools["p"] == 0


@pytest.mark.parametrize("status", [401, 403, 404, 408, 429, 500, 503])
@pytest.mark.asyncio
async def test_primary_failure_uses_backup_without_nested_retries(status):
    async def behavior(model, client):
        if model == "a":
            raise upstream_error(status)
    t, s = Transport(behavior), Scheduler()
    result = await RoutedChat(config("failover"), s, client_factory=t.factory).achat([])
    assert result.model == "b" and t.called == ["a", "b"]
    assert s.models["a"].cooldown_until > s.clock()
    assert s.models["a"].last_error == f"upstream_http_{status}"


@pytest.mark.asyncio
async def test_cooldown_and_auto_recovery():
    now = [100.0]
    async def behavior(model, client):
        if model == "a" and now[0] == 100:
            raise upstream_error(503)
    s, t = Scheduler(clock=lambda: now[0]), Transport(behavior)
    client = RoutedChat(config("failover", failure_cooldown_seconds=5), s, client_factory=t.factory)
    await client.achat([])
    now[0] = 102
    assert (await client.achat([])).model == "b"
    now[0] = 106
    assert (await client.achat([])).model == "a"
    assert s.models["a"].last_error is None


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.asyncio
async def test_bad_request_does_not_fan_out_or_leak(status):
    async def behavior(model, client):
        raise upstream_error(status)
    t, s = Transport(behavior), Scheduler()
    with pytest.raises(PoolError) as error:
        await RoutedChat(config(), s, client_factory=t.factory).achat([])
    assert "private" not in str(error.value)
    assert len(t.called) == 1 and s.pools["p"] == 0


@pytest.mark.asyncio
async def test_stream_output_disables_replay():
    emitted = []
    async def callback(level, text, end, flush):
        emitted.append(text)
    async def behavior(model, client):
        await client._logger_async.info("partial answer", flush=True)
        raise upstream_error(503)
    t, s = Transport(behavior), Scheduler()
    with pytest.raises(PoolError, match="stream_interrupted_no_replay"):
        await RoutedChat(config(), s, log_callback=callback, client_factory=t.factory).achat([])
    assert emitted == ["partial answer"] and t.called == ["a"]
    assert s.pools["p"] == 0


@pytest.mark.asyncio
async def test_pool_capacity_queue_full_timeout_and_cancel():
    release, started = asyncio.Event(), asyncio.Event()
    async def behavior(model, client):
        started.set()
        await release.wait()
    s, t = Scheduler(), Transport(behavior)
    c = config(cap=1, max_queue_size=1, queue_timeout_seconds=.1)
    client = RoutedChat(c, s, client_factory=t.factory)
    first = asyncio.create_task(client.achat([]))
    await started.wait()
    waiting = asyncio.create_task(client.achat([]))
    await asyncio.sleep(.01)
    assert s.waiting["p"] == 1
    with pytest.raises(PoolError, match="queue_full"):
        await client.achat([])
    with pytest.raises(PoolError, match="capacity_wait_timeout"):
        await waiting
    waiting = asyncio.create_task(client.achat([]))
    await asyncio.sleep(.01)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert s.waiting["p"] == 0
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert s.pools["p"] == 0 and s.models["a"].in_flight == 0
    assert t.closed == ["a"]


@pytest.mark.asyncio
async def test_model_capacity_shared_across_pools():
    release, started = asyncio.Event(), asyncio.Event()
    async def behavior(model, client):
        started.set()
        await release.wait()
    s, t = Scheduler(), Transport(behavior)
    c = config(cap=1, max_queue_size=0)
    c.pools[0].members = [Member(model_id="a")]
    c.pools.append(c.pools[0].model_copy(update={"id": "q"}))
    a = RoutedChat(c, s, client_factory=t.factory)
    other = c.model_copy(update={"active_target": Target(kind="pool", id="q")})
    b = RoutedChat(other, s, client_factory=t.factory)
    task = asyncio.create_task(a.achat([]))
    await started.wait()
    with pytest.raises(PoolError, match="queue_full"):
        await b.achat([])
    assert s.pools["q"] == 0
    release.set()
    await task
    assert (await b.achat([])).model == "a"


@pytest.mark.asyncio
async def test_budget_usage_success_and_unknown_failed_attempt():
    c, s, t = config(), Scheduler(), Transport()
    client = RoutedChat(c, s, client_factory=t.factory)
    token = client.bind_token_budget(1000)
    try:
        await client.achat([], max_tokens=200)
        snapshot = client.token_budget_snapshot()
        assert snapshot["consumed"] == 10 and snapshot["reserved"] == 0
    finally:
        client.reset_token_budget(token)
    async def behavior(model, transport):
        raise upstream_error(503)
    t = Transport(behavior)
    client = RoutedChat(c, s, client_factory=t.factory)
    token = client.bind_token_budget(300)
    try:
        with pytest.raises(LLMTokenBudgetExceeded):
            await client.achat([])
        snapshot = client.token_budget_snapshot()
        assert snapshot["consumed"] == 300 and snapshot["reserved"] == 0
        assert len(t.called) == 1
    finally:
        client.reset_token_budget(token)


@pytest.mark.asyncio
async def test_all_failed_once_only():
    async def behavior(model, client):
        raise upstream_error(503)
    t, s = Transport(behavior), Scheduler()
    with pytest.raises(PoolError, match="upstream_http_503"):
        await RoutedChat(config(), s, client_factory=t.factory).achat([])
    assert Counter(t.called) == {"a": 1, "b": 1} and s.pools["p"] == 0


@pytest.mark.asyncio
async def test_local_log_callbacks_do_not_cross():
    out_a, out_b = [], []
    async def behavior(model, client):
        await asyncio.sleep(.01)
        await client._logger_async.info(model)
    t, s, c = Transport(behavior), Scheduler(), config()
    def callback(output):
        return lambda level, text, end, flush: output.append(text)
    a = RoutedChat(c, s, callback(out_a), t.factory)
    b = RoutedChat(c, s, callback(out_b), t.factory)
    results = await asyncio.gather(a.achat([]), b.achat([]))
    assert out_a == [results[0].model] and out_b == [results[1].model]


@pytest.mark.asyncio
async def test_deadline_timeout_fails_over_and_releases():
    c, s = config("failover", request_timeout_seconds=1), Scheduler()
    async def behavior(model, client):
        if model == "a":
            raise asyncio.TimeoutError()
    t = Transport(behavior)
    assert (await RoutedChat(c, s, client_factory=t.factory).achat([])).model == "b"
    assert s.pools["p"] == 0


def test_sync_contract():
    t, s = Transport(), Scheduler()
    assert RoutedChat(config(), s, client_factory=t.factory).chat([]).model == "a"


@pytest.mark.asyncio
async def test_concurrent_load_never_exceeds_caps():
    s, c = Scheduler(), config(cap=2, queue_timeout_seconds=2)
    peaks = Counter()
    async def behavior(model, client):
        peaks["pool"] = max(peaks["pool"], s.pools["p"])
        peaks[model] = max(peaks[model], s.models[model].in_flight)
        await asyncio.sleep(.01)
    t = Transport(behavior)
    client = RoutedChat(c, s, client_factory=t.factory)
    await asyncio.gather(*(client.achat([]) for _ in range(24)))
    assert peaks["pool"] == 2 and peaks["a"] <= 2 and peaks["b"] <= 2
    assert s.pools["p"] == 0 and s.waiting["p"] == 0


@pytest.mark.asyncio
async def test_real_deadline_bounds_hanging_transport():
    c, s = config("failover", request_timeout_seconds=1), Scheduler()
    async def behavior(model, client):
        await asyncio.Event().wait()
    t = Transport(behavior)
    with pytest.raises(PoolError, match="route_timeout"):
        await asyncio.wait_for(RoutedChat(c, s, client_factory=t.factory).achat([]), 2)
    assert t.called == ["a"] and s.pools["p"] == 0


@pytest.mark.asyncio
async def test_lowered_limit_applies_to_old_snapshot_admission():
    s, c = Scheduler(), config(cap=2, max_queue_size=0)
    release, started = asyncio.Event(), asyncio.Event()
    async def behavior(model, client):
        started.set()
        await release.wait()
    t = Transport(behavior)
    old = RoutedChat(c, s, client_factory=t.factory)
    first = asyncio.create_task(old.achat([]))
    await started.wait()
    latest = c.model_copy(deep=True)
    latest.revision += 1
    latest.pools[0].max_concurrency = 1
    s.configure(latest)
    with pytest.raises(PoolError, match="queue_full"):
        await old.achat([])
    release.set()
    await first


@pytest.mark.asyncio
async def test_disabled_member_and_direct_selection():
    c, s, t = config(), Scheduler(), Transport()
    c.models[0].enabled = False
    routed = RoutedChat(c, s, client_factory=t.factory)
    assert (await routed.achat([])).model == "b"
    c.active_target = Target(kind="model", id="b")
    assert (await RoutedChat(c, s, client_factory=t.factory).achat([])).model == "b"
    assert t.called == ["b", "b"]
