"""Real sockets and real OpenAI SDK; no paid provider or live service access."""
import asyncio
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from sirchmunk.llm.pool_config import Member, ModelSpec, PoolConfig, PoolSpec, Target
from sirchmunk.llm.pool_router import PoolError, RoutedChat, Scheduler


@pytest.fixture
def provider():
    spec = importlib.util.spec_from_file_location("acceptance_provider", Path(__file__).parent / "support/llm_pool_provider.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield module, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()


def configuration(base, *names, strategy="weighted", cap=4, **policy):
    models = [ModelSpec(id=name, name=name, model="fixture-" + name, base_url=base,
                        max_concurrency=cap, timeout_seconds=1 if "timeout" in name else 5) for name in names]
    pool = PoolSpec(id="p", name="P", strategy=strategy, max_concurrency=cap,
                    members=[Member(model_id=m.id, weight=3 if i == 0 else 1, priority=i) for i, m in enumerate(models)],
                    failure_cooldown_seconds=0, **policy)
    return PoolConfig(revision=1, models=models, pools=[pool], active_target=Target(kind="pool", id="p"))


async def until(predicate, timeout=10):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.01)


def released(scheduler):
    assert all(state.in_flight == 0 for state in scheduler.models.values())
    assert all(count == 0 for count in scheduler.pools.values())
    assert all(count == 0 for count in scheduler.waiting.values())


@pytest.mark.asyncio
async def test_socket_weighted_and_concurrent_stress(provider):
    fixture, base = provider
    scheduler = Scheduler()
    config = configuration(base, "a", "b", cap=4, queue_timeout_seconds=120)
    client = RoutedChat(config, scheduler, log_callback=lambda *args: None)
    # Opt in to a longer acceptance run without slowing every ordinary CI run.
    count = int(os.getenv("SIRCHMUNK_SOCKET_STRESS_CALLS", "40"))
    assert 4 <= count <= 2000 and count % 4 == 0
    serial_count = Counter()
    for _ in range(count):
        result = await client.achat([], stream=False, max_tokens=16)
        serial_count[result.model] += 1
        assert result.usage["total_tokens"] == 10
    assert serial_count == {"fixture-a": count * 3 // 4, "fixture-b": count // 4}
    for _ in range(10):
        results = await asyncio.gather(*(client.achat([], stream=False, max_tokens=16) for _ in range(24)))
        assert len(results) == 24
    assert sum(fixture.calls.values()) == count + 240
    assert all(value <= 4 for value in fixture.peaks.values())
    assert scheduler.models["a"].started + scheduler.models["b"].started == count + 240
    released(scheduler)
    print(json.dumps({"socket_serial_calls": count, "weighted": dict(serial_count),
                      "socket_concurrent_calls": 240, "provider_peaks": dict(fixture.peaks), "capacity_released": True}))


@pytest.mark.parametrize("primary", ["fail", "rate", "timeout"])
@pytest.mark.asyncio
async def test_socket_failure_and_timeout_failover(provider, primary):
    fixture, base = provider
    scheduler = Scheduler()
    result = await RoutedChat(configuration(base, primary, "backup", strategy="failover"), scheduler, log_callback=lambda *args: None).achat([], stream=False)
    assert result.model == "fixture-backup"
    assert fixture.calls == {"fixture-" + primary: 1, "fixture-backup": 1}
    released(scheduler)
    await until(lambda: not any(fixture.active.values()))


@pytest.mark.asyncio
async def test_socket_stream_disconnect_cannot_replay(provider):
    fixture, base = provider
    scheduler, emitted = Scheduler(), []
    client = RoutedChat(configuration(base, "interrupt", "backup", strategy="failover"), scheduler,
                        log_callback=lambda level, text, end, flush: emitted.append(text))
    with pytest.raises(PoolError, match="stream_interrupted_no_replay"):
        await client.achat([], stream=True)
    with pytest.raises(PoolError, match="stream_interrupted_no_replay"):
        await client.achat([], stream=True)
    assert emitted and fixture.calls == {"fixture-interrupt": 1}
    released(scheduler)


@pytest.mark.asyncio
async def test_socket_shared_capacity_cancel_and_lowered_snapshot(provider):
    fixture, base = provider
    scheduler = Scheduler()
    config = configuration(base, "slow", cap=2, max_queue_size=1, queue_timeout_seconds=5)
    config.models[0].max_concurrency = 1
    config.pools.append(config.pools[0].model_copy(update={"id": "q"}))
    first_client = RoutedChat(config, scheduler, log_callback=lambda *args: None)
    other_config = config.model_copy(deep=True)
    other_config.active_target = Target(kind="pool", id="q")
    other_client = RoutedChat(other_config, scheduler, log_callback=lambda *args: None)
    first = asyncio.create_task(first_client.achat([], stream=False))
    await until(lambda: fixture.active["fixture-slow"] == 1)
    waiting = asyncio.create_task(other_client.achat([], stream=False))
    await until(lambda: scheduler.waiting["q"] == 1)
    with pytest.raises(PoolError, match="queue_full"):
        await other_client.achat([], stream=False)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert scheduler.waiting["q"] == 0 and fixture.calls["fixture-slow"] == 1
    latest = config.model_copy(deep=True)
    latest.revision += 1
    latest.pools[0].max_concurrency = 1
    scheduler.configure(latest)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    released(scheduler)
    # Remote inference may outlive local cancellation. Wait for provider
    # quiescence before checking its physical concurrent count in the next run.
    await until(lambda: not any(fixture.active.values()))
    assert (await first_client.achat([], stream=False)).model == "fixture-slow"
    assert fixture.peaks["fixture-slow"] == 1
    released(scheduler)


def test_config_survives_process_restart_and_conflict(tmp_path):
    # Actual restart: the writing process exits and releases its root lease
    # BEFORE a second process opens the persisted configuration.
    writer = r'''
import json, sys
from sirchmunk.llm.pool_config import ConfigStore, ConfigError
store = ConfigStore(sys.argv[1])
config = store.change(0, "model", {"name": "fixture", "model": "fixture-a", "base_url": "http://127.0.0.1:1/v1", "api_key_env": "TEST_ONLY_KEY"})
config = store.change(config.revision, "pool", {"name": "fixture", "members": [{"model_id": config.models[0].id}]})
config = store.change(config.revision, "selection", {"kind": "pool", "id": config.pools[0].id})
try:
    store.change(config.revision - 1, "selection", {"kind": "legacy"})
except ConfigError as exc:
    assert exc.status == 409
else:
    raise AssertionError("Stale revision accepted")
assert store.read().public() == config.public()
print(json.dumps(config.public()))
'''
    before = subprocess.run([sys.executable, "-c", writer, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert before.returncode == 0, before.stderr
    child = subprocess.run([sys.executable, "-c", "import json,sys; from sirchmunk.llm.pool_config import ConfigStore; print(json.dumps(ConfigStore(sys.argv[1]).read().public()))", str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == json.loads(before.stdout)
