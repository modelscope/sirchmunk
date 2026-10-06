"""Bound no-console stop protocol; all roots and credentials are synthetic."""
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import socket
import subprocess
import sys
import threading
import time
import uuid

import psutil
import pytest

from sirchmunk.api import supervisor as sup
from sirchmunk.api.management import ManagementHolder, ManagementState, close_managed_runtime
from sirchmunk.storage.lifecycle import bootstrap_managed_root, write_json_atomic


def intent(instance="supervisor-test", process=None):
    return dict(schema="platform-stop-intent.v1", instance_id=instance,
                service_instance_id=process or str(uuid.uuid4()), maintenance_owner="owner-0001",
                nonce=uuid.uuid4().hex + uuid.uuid4().hex, expires_at=time.time() + 120)


def protect_test_directory(path):
    """Private ACL on this test's newly created leaf only, never user folders."""
    if os.name != "nt":
        path.chmod(0o700)
        return
    import win32api
    import win32con
    import win32security as security
    token = security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        owner = security.GetTokenInformation(token, security.TokenUser)[0]
    finally:
        token.Close()
    acl = security.ACL()
    for sid in (owner, security.ConvertStringSidToSid("S-1-5-18"), security.ConvertStringSidToSid("S-1-5-32-544")):
        acl.AddAccessAllowedAceEx(security.ACL_REVISION, 3, 0x1F01FF, sid)
    security.SetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
        security.DACL_SECURITY_INFORMATION | security.PROTECTED_DACL_SECURITY_INFORMATION, None, None, acl, None)


@pytest.fixture
def config(tmp_path):
    protect_test_directory(tmp_path)
    cfg = dict(schema="sirchmunk-supervisor.v1", instance_id="supervisor-test",
               work_path=str(tmp_path / "work"), stop_intent_file=str(tmp_path / "intent.json"),
               mailbox_directory=str(tmp_path / "work" / ".control" / "stop-ipc"),
               host="127.0.0.1", port=19111, management_port=19112, shutdown_timeout_seconds=5)
    path = tmp_path / "supervisor.json"
    write_json_atomic(path, cfg)
    return path, cfg


@pytest.mark.parametrize("change", [
    {"extra": True}, {"schema": "other"}, {"instance_id": "wrong"}, {"service_instance_id": "not-uuid"},
    {"maintenance_owner": "a"}, {"nonce": "A" * 64}, {"nonce": True}, {"expires_at": True},
    {"expires_at": float("nan")}, {"expires_at": float("inf")}, {"expires_at": 0},
    {"expires_at": 9000}, {"expires_at": 9000 + 123},
])
def test_intent_strict_fields_types_and_ttl(change):
    value = {**intent(), "expires_at": 9120, **change}
    with pytest.raises(sup.SupervisorError):
        sup.validate_intent(value, "supervisor-test", now=9000)


def test_expired_intent_only_allows_observation():
    value = {**intent(), "expires_at": 8999}
    assert sup.validate_intent(value, "supervisor-test", now=9000, allow_expired=True) == value
    with pytest.raises(sup.SupervisorError):
        sup.validate_intent(value, "supervisor-test", now=9000)


@pytest.mark.parametrize("change", [
    {"command": "arbitrary"}, {"port": True}, {"host": "example.com"}, {"work_path": "relative"},
    {"shutdown_timeout_seconds": 0}, {"management_port": 19111}, {"schema": "wrong"},
])
def test_configuration_is_a_fixed_path_not_an_execution_interface(config, change):
    path, cfg = config
    write_json_atomic(path, {**cfg, **change})
    with pytest.raises(sup.SupervisorError):
        sup.load_config(path)


def test_configuration_control_path_binding(config):
    path, cfg = config
    assert sup.load_config(path) == cfg
    for change in ({"mailbox_directory": str(path.parent)},
                   {"stop_intent_file": cfg["work_path"] + "/intent.json"},
                   {"stop_intent_file": str(path)}):
        write_json_atomic(path, {**cfg, **change})
        with pytest.raises(sup.SupervisorError, match="binding"):
            sup.load_config(path)


def test_hardlinks_and_duplicate_json_rejected(config):
    path, _ = config
    alias = path.parent / "alias.json"
    os.link(path, alias)
    try:
        with pytest.raises(sup.SupervisorError, match="hardlinked"):
            sup.load_config(path)
    finally:
        alias.unlink()
    path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(sup.SupervisorError, match="duplicate"):
        sup.load_config(path)


def test_untrusted_control_file_writer_is_rejected(config):
    path, _ = config
    if os.name == "nt":
        import win32security as security
        acl = security.GetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
                                            security.DACL_SECURITY_INFORMATION).GetSecurityDescriptorDacl()
        acl.AddAccessAllowedAceEx(security.ACL_REVISION, 0, 0x1F01FF,
                                  security.ConvertStringSidToSid("S-1-1-0"))
        security.SetNamedSecurityInfo(str(path), security.SE_FILE_OBJECT,
                                       security.DACL_SECURITY_INFORMATION, None, None, acl, None)
    else:
        path.chmod(0o666)
    with pytest.raises(sup.SupervisorError, match="writer|unprotected"):
        sup.load_config(path)


@pytest.mark.parametrize("change,code", [
    ({"PLATFORM_RELEASE_VERSION": "untrue-version"}, "version_mismatch"),
    ({"SIRCHMUNK_DEPLOYMENT_ID": "other"}, "identity_conflict"),
    ({"SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "other"}, "configuration_conflict"),
])
def test_launch_identity_cannot_be_overridden(config, monkeypatch, change, code):
    _, cfg = config
    for key in tuple(os.environ):
        if key.startswith(("SIRCHMUNK_", "PLATFORM_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PLATFORM_DEPLOYMENT_ID", "synthetic-deployment")
    for key, value in change.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(sup.SupervisorError, match=code):
        sup.configure_launch(cfg)
    assert not Path(cfg["work_path"]).exists()


def test_check_only_and_helper_import_do_not_load_business_or_claim_root(config, tmp_path):
    path, cfg = config
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ("SIRCHMUNK_", "PLATFORM_", "LLM_", "EMBEDDING_", "HF_"))}
    env.update(SIRCHMUNK_WORK_PATH=cfg["work_path"], EMBEDDING_CACHE_DIR=str(tmp_path / "work/.cache/models"),
               HF_HOME=str(tmp_path / "work/.cache/models"), PLATFORM_DEPLOYMENT_ID="test-deployment")
    source = Path(__file__).resolve().parents[1]
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from sirchmunk.api.supervisor import main; "
            "code=main(['--config', sys.argv[2], '--check-only']); "
            "assert 'sirchmunk.api.main' not in sys.modules; "
            "assert 'sirchmunk.search' not in sys.modules; "
            "assert 'fastapi' not in sys.modules; "
            "from sirchmunk.storage.lifecycle import _roots; assert not _roots; sys.exit(code)")
    result = subprocess.run([sys.executable, "-I", "-c", code, str(source / "src"), str(path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["storage_opened"] is False
    assert not Path(cfg["work_path"]).exists()
    entry = source / "scripts" / "supervised_serve.py"
    result = subprocess.run([sys.executable, "-I", str(entry), "--config", str(path), "--check-only"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not Path(cfg["work_path"]).exists()
    write_json_atomic(Path(cfg["stop_intent_file"]), intent())
    helper_code = ("import sys; sys.path.insert(0, sys.argv[1]); "
                   "from sirchmunk.api.supervisor import main; "
                   "result=main(['--config',sys.argv[2],'--request-stop','--intent-file',sys.argv[3]]); "
                   "assert result == 1; "
                   "assert 'sirchmunk.api.main' not in sys.modules and 'fastapi' not in sys.modules; "
                   "from sirchmunk.storage.lifecycle import _roots; assert not _roots")
    result = subprocess.run([sys.executable, "-I", "-c", helper_code, str(source / "src"), str(path),
                              cfg["stop_intent_file"]], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert not Path(cfg["work_path"]).exists()


@pytest.fixture
def managed(config, monkeypatch):
    _, cfg = config
    root = Path(cfg["work_path"])
    launch = dict(SIRCHMUNK_WORK_PATH=str(root), SIRCHMUNK_MANAGEMENT_ENABLED="true",
                  SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED="true",
                  SIRCHMUNK_MANAGEMENT_INSTANCE_ID=cfg["instance_id"])
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(root / ".cache/models"))
    runtime = bootstrap_managed_root(launch)
    state = ManagementState(root / ".control", cfg["instance_id"], "isolated-build")
    state.enable_lifecycle(runtime)
    state.lifecycle.initialize()
    state.set_maintenance(True, "owner-0001")
    mailbox = Path(cfg["mailbox_directory"])
    mailbox.mkdir(mode=0o700)
    write_json_atomic(root / ".control/process.json", dict(
        schema="sirchmunk-process.v1", instance_id=cfg["instance_id"],
        service_instance_id=state.service_instance_id, pid=os.getpid(), create_time=psutil.Process().create_time(),
        startup_unclean=False, work_root_sha256=hashlib.sha256(str(root).encode()).hexdigest()))
    try:
        yield state, cfg
    finally:
        if not state._closed:
            if not runtime.failure and not state.lifecycle.shutdown_unresolved():
                runtime.close()
            state.close()


def test_stop_flush_receipt_and_clean_marker_are_committed_only_after_close(managed):
    state, cfg = managed
    value = intent(process=state.service_instance_id)
    assert state.lifecycle.request_stop(value) is True
    _, receipt, clean = sup._observe(cfg, value)
    assert receipt["state"] == "accepted" and not clean
    assert state.stopping and state.lifecycle.runtime.frozen
    assert state.lifecycle.request_stop(value) is False
    holder = ManagementHolder()
    holder.state = state
    asyncio.run(close_managed_runtime(holder))
    state.close()
    _, receipt, clean = sup._observe(cfg, value)
    assert receipt["state"] == "succeeded" and clean


def test_business_acceptance_rechecks_ttl_under_lock(managed, monkeypatch):
    state, _ = managed
    value = intent(process=state.service_instance_id)
    original = sup.validate_intent
    calls = []
    def advance(value, instance_id, **kwargs):
        calls.append(kwargs.get("allow_expired", False))
        kwargs.pop("now", None)
        return original(value, instance_id, now=value["expires_at"] + 1, **kwargs)
    monkeypatch.setattr(sup, "validate_intent", advance)
    with pytest.raises(sup.SupervisorError, match="expired"):
        state.lifecycle.request_stop(value)
    assert calls == [True, False]
    assert not state.lifecycle.pending() and not state.stopping


def test_failed_flush_keeps_stop_accepted_and_unclean(managed, monkeypatch):
    state, cfg = managed
    value = intent(process=state.service_instance_id)
    def fail():
        raise OSError("synthetic failure")
    with monkeypatch.context() as patch:
        patch.setattr(state.lifecycle.runtime, "flush", fail)
        with pytest.raises(OSError):
            state.lifecycle.request_stop(value)
        assert not state.stopping
        assert state.lifecycle.request_stop(value) is False
    state.close()
    _, receipt, clean = sup._observe(cfg, value)
    assert receipt["state"] == "accepted" and not clean


@pytest.mark.parametrize("jump", [-8, 8])
def test_clock_step_latches_stop_rejection_without_effect_or_auto_rebase(managed, monkeypatch, jump):
    from sirchmunk.api.management import ManagementError
    state, _ = managed
    value = intent(process=state.service_instance_id)
    original = time.time
    with monkeypatch.context() as patch:
        patch.setattr(time, "time", lambda: original() + jump)
        with pytest.raises(ManagementError, match="clock_discontinuity"):
            state.lifecycle.request_stop(value)
    assert not state.lifecycle.pending() and not state.stopping
    # Restoring the wall clock does not authorize a previously refused intent.
    with pytest.raises(ManagementError, match="clock_discontinuity"):
        state.lifecycle.request_stop(value)
    assert not state.lifecycle.pending() and state.maintenance_enabled


def test_wall_clock_rollback_cannot_resurrect_expired_stop(managed, monkeypatch):
    from sirchmunk.api.management import ManagementError
    state, _ = managed
    value = intent(process=state.service_instance_id)
    value["expires_at"] = time.time() - 1
    original = time.time
    monkeypatch.setattr(time, "time", lambda: original() - 8)
    # Naive wall-clock TTL comparison would accept this, but the generation's
    # monotonic binding must still reject it before persisting an accepted row.
    assert sup.validate_intent(value, state.instance_id) == value
    with pytest.raises(ManagementError, match="clock_discontinuity"):
        state.lifecycle.request_stop(value)
    assert not state.lifecycle.pending() and not state.stopping


@pytest.mark.parametrize("jump", [-1, -8])
def test_clock_jump_after_check_cannot_extend_expired_deadline(managed, monkeypatch, jump):
    state, _ = managed
    value = intent(process=state.service_instance_id)
    value["expires_at"] = time.time() - 0.5
    original_time, original_validate = time.time, sup.validate_intent
    def jump_before_ttl(value, instance_id, **kwargs):
        if not kwargs.get("allow_expired"):
            assert kwargs.get("now") is not None
            monkeypatch.setattr(time, "time", lambda: original_time() + jump)
        return original_validate(value, instance_id, **kwargs)
    monkeypatch.setattr(sup, "validate_intent", jump_before_ttl)
    with pytest.raises(sup.SupervisorError, match="expired"):
        state.lifecycle.request_stop(value)
    assert not state.lifecycle.pending() and not state.stopping


def test_small_clock_rollback_is_not_an_expiration_grace_period(managed, monkeypatch):
    state, _ = managed
    value = intent(process=state.service_instance_id)
    value["expires_at"] = time.time() - 0.5
    original = time.time
    monkeypatch.setattr(time, "time", lambda: original() - 1)
    with pytest.raises(sup.SupervisorError, match="expired"):
        state.lifecycle.request_stop(value)
    assert not state.lifecycle.pending() and not state.stopping


def test_small_forward_then_backward_adjustment_never_rewinds_validated_time(managed, monkeypatch):
    from types import SimpleNamespace
    from sirchmunk.api import lifecycle as module
    state, _ = managed
    wall, monotonic = state.lifecycle.stop_clock_origin
    sample = dict(wall=wall + 2, monotonic=monotonic + 1)
    monkeypatch.setattr(module, "time", SimpleNamespace(
        time=lambda: sample["wall"], monotonic=lambda: sample["monotonic"]))
    assert state.lifecycle.check_stop_clock() == wall + 2
    sample.update(wall=wall + 1.1, monotonic=monotonic + 1.1)
    value = intent(process=state.service_instance_id)
    value["expires_at"] = wall + 1.5
    with pytest.raises(sup.SupervisorError, match="expired"):
        state.lifecycle.request_stop(value)
    assert not state.lifecycle.stop_clock_unsafe  # no >2s discontinuity
    assert not state.lifecycle.pending() and not state.stopping


def test_same_intent_timeout_and_expired_retry_never_resubmits(managed, monkeypatch):
    state, cfg = managed
    value = intent(process=state.service_instance_id)
    destination = Path(cfg["mailbox_directory"]) / "request.json"
    write_json_atomic(destination, value)
    write_json_atomic(Path(cfg["stop_intent_file"]), value)
    before = destination.stat().st_mtime_ns
    def forbidden(*args):
        raise AssertionError("must not resubmit")
    monkeypatch.setattr(sup, "write_json_atomic", forbidden)
    # Force immediate timeout without sleeping; then re-observe the expired ID.
    ticks = iter([0, 2, 0, 2])
    monkeypatch.setattr(sup.time, "monotonic", lambda: next(ticks))
    for expired in (False, True):
        if expired:
            monkeypatch.setattr(sup.time, "time", lambda: value["expires_at"] + 1)
        with pytest.raises(sup.SupervisorError, match="not_confirmed"):
            sup.request_stop(cfg, cfg["stop_intent_file"], 1)
    assert destination.stat().st_mtime_ns == before
    assert not state.lifecycle.pending()


@pytest.mark.parametrize("change", [{"pid": True}, {"create_time": float("nan")},
                                   {"startup_unclean": 0}, {"extra": 1}])
def test_process_identity_document_is_strict(managed, change):
    state, cfg = managed
    path = Path(cfg["work_path"]) / ".control/process.json"
    info = json.loads(path.read_text())
    path.write_text(json.dumps({**info, **change}), encoding="utf-8")
    with pytest.raises(sup.SupervisorError):
        sup._observe(cfg, intent(process=state.service_instance_id))


@pytest.mark.parametrize("change", [{"stopped": 1}, {"effect_committed": False}, {"extra": 1},
                                   {"schema_id": "other"}, {"state": "succeeded"}])
def test_stop_receipt_is_strict(managed, change):
    state, cfg = managed
    value = intent(process=state.service_instance_id)
    state.lifecycle.request_stop(value)
    receipt = json.loads(state.lifecycle._read("stop", value["nonce"])[3])
    state.db.execute("UPDATE lifecycle_operations SET result_json=? WHERE kind='stop'", (json.dumps({**receipt, **change}),))
    with pytest.raises(sup.SupervisorError, match="receipt"):
        sup._observe(cfg, value)
    state.db.execute("UPDATE lifecycle_operations SET result_json=? WHERE kind='stop'", (json.dumps(receipt),))


def test_shutdown_waits_for_real_work_before_close(managed):
    state, _ = managed
    release = state.lifecycle.runtime.reserve_background()
    holder = ManagementHolder()
    holder.state = state
    async def run():
        closer = asyncio.create_task(close_managed_runtime(holder, 2))
        await asyncio.sleep(0.1)
        assert not closer.done() and not state.lifecycle.runtime.closed
        release()
        await closer
    asyncio.run(run())
    assert state.lifecycle.runtime.closed


def test_shutdown_waits_for_known_flush_worker_without_closing_its_ledger(managed, monkeypatch):
    state, _ = managed
    entered, finish = threading.Event(), threading.Event()
    original = state.lifecycle.runtime.flush
    def slow():
        entered.set()
        assert finish.wait(5)
        return original()
    monkeypatch.setattr(state.lifecycle.runtime, "flush", slow)
    body = dict(operation_id="flush-before-stop", instance_id=state.instance_id,
                service_instance_id=state.service_instance_id, maintenance_operation_id="owner-0001")
    holder = ManagementHolder()
    holder.state = state
    async def run():
        flusher = asyncio.create_task(asyncio.to_thread(state.lifecycle.flush, body))
        assert await asyncio.to_thread(entered.wait, 3)
        closer = asyncio.create_task(close_managed_runtime(holder, 3))
        try:
            await asyncio.sleep(0.1)
            assert not closer.done()
            assert state.lifecycle.workers == 1 and state.lifecycle.pending()
        finally:
            finish.set()
            await flusher
            await closer
    asyncio.run(run())
    assert state.lifecycle.runtime.closed and not state.lifecycle.runtime.failure


def test_shutdown_timeout_keeps_storage_open_and_unclean(managed):
    state, _ = managed
    release = state.lifecycle.runtime.reserve_background()
    holder = ManagementHolder()
    holder.state = state
    try:
        with pytest.raises(RuntimeError, match="timeout"):
            asyncio.run(close_managed_runtime(holder, 0.01))
        assert not state.lifecycle.runtime.closed and state.lifecycle.runtime.failure
        assert state.db.execute("SELECT unclean FROM process_state").fetchone() == (1,)
    finally:
        release()


def test_stop_rejection_is_not_a_service_exit(managed):
    from types import SimpleNamespace
    state, cfg = managed
    value = intent(process=str(uuid.uuid4()))
    write_json_atomic(Path(cfg["mailbox_directory"]) / "request.json", value)
    server = SimpleNamespace(should_exit=False)
    async def run():
        watcher = asyncio.create_task(sup.watch_stop(cfg, state, server))
        try:
            for _ in range(20):
                if (Path(cfg["mailbox_directory"]) / "rejection.json").exists():
                    break
                await asyncio.sleep(0.02)
            denied = sup.read_json(Path(cfg["mailbox_directory"]) / "rejection.json")
            assert denied["request_sha256"] == sup.request_digest(value)
            assert not server.should_exit and not state.stopping and not state.lifecycle.pending()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
    asyncio.run(run())


def test_shutdown_unknown_cannot_clear_unclean_marker(managed):
    state, cfg = managed
    state.counter_unknown = True
    holder = ManagementHolder()
    holder.state = state
    with pytest.raises(RuntimeError, match="unknown"):
        asyncio.run(close_managed_runtime(holder, 1))
    state.close()
    with closing(sqlite3.connect(Path(cfg["work_path"]) / ".control/management.sqlite3")) as db:
        assert db.execute("SELECT unclean FROM process_state").fetchone() == (1,)


def _free_ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


@pytest.fixture
def real_server(config, tmp_path):
    """Actual fixed-entry processes, never a system service or a user's root."""
    import httpx
    path, cfg = config
    cfg["port"], cfg["management_port"] = _free_ports()
    write_json_atomic(path, cfg)
    root = Path(cfg["work_path"])
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(
        ("SIRCHMUNK_", "PLATFORM_", "LLM_", "EMBEDDING_", "HF_", "OPENAI_"))}
    env.update(SIRCHMUNK_WORK_PATH=str(root), EMBEDDING_CACHE_DIR=str(root / ".cache/models"),
               HF_HOME=str(root / ".cache/models"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               SIRCHMUNK_MANAGEMENT_TOKEN="synthetic-management-token-with-32-plus-characters",
               SIRCHMUNK_API_TOKEN="synthetic-business-token", PLATFORM_DEPLOYMENT_ID="isolated-generation-1",
               LLM_BASE_URL="http://127.0.0.1:1/v1", LLM_API_KEY="synthetic-never-used", LLM_MODEL_NAME="no-model",
               PYTHONDONTWRITEBYTECODE="1", LOGURU_LEVEL="WARNING")
    entry = Path(__file__).resolve().parents[1] / "scripts/supervised_serve.py"
    command = [sys.executable, "-I", str(entry), "--config", str(path)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = []
    logs = []
    client = httpx.Client(base_url=f"http://127.0.0.1:{cfg['management_port']}/management/v1/",
                          headers={"Authorization": "Bearer " + env["SIRCHMUNK_MANAGEMENT_TOKEN"]},
                          trust_env=False, timeout=5)
    business = httpx.Client(base_url=f"http://127.0.0.1:{cfg['port']}", trust_env=False, timeout=5,
                            headers={"Authorization": "Bearer " + env["SIRCHMUNK_API_TOKEN"]})
    def start():
        log = tmp_path / f"process-{len(processes)}.log"
        stream = log.open("wb")
        logs.append(stream)
        child = subprocess.Popen(command, cwd=tmp_path, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
        processes.append(child)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                pytest.fail("isolated service failed to start: " + log.read_text(encoding="utf-8", errors="replace"))
            try:
                response = client.get("status")
                ready = response.status_code == 200 and response.json()["readiness"] == "ok"
                if ready and (root / ".control/process.json").exists() and business.get("/health").status_code == 200:
                    info = json.loads((root / ".control/process.json").read_text())
                    if info["service_instance_id"] == response.json()["service_instance_id"]:
                        return child, response.json()
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        pytest.fail("isolated service readiness timed out: " + log.read_text(encoding="utf-8", errors="replace"))
    def stop(value, timeout=10):
        sent_at = time.time()
        write_json_atomic(Path(cfg["stop_intent_file"]), value)
        result = subprocess.run(command + ["--request-stop", "--stop-timeout-seconds", str(timeout),
                              "--intent-file", cfg["stop_intent_file"]], cwd=tmp_path, env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              creationflags=flags, timeout=timeout + 15)
        result.test_timing = dict(sent=sent_at, observed=time.time(), expires=value.get("expires_at"))
        # Synthetic timing evidence only; keep transient VM clock failures
        # visible instead of extending/refreshing a submitted intent's TTL.
        if result.returncode:
            result.stderr += f"\ntest timing: sent={sent_at}, observed={time.time()}, expires={value.get('expires_at')}"
        return result
    try:
        yield cfg, client, business, start, stop
    finally:
        client.close()
        business.close()
        # Only a child launched by this fixture may be cleaned after a failing
        # assertion. Forced test cleanup is never counted as stop success.
        for child in processes:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
        for stream in logs:
            stream.close()


def test_real_no_console_stop_restart_owner_flush_and_backup(real_server):
    cfg, client, business, start, stop = real_server
    first, before = start()
    assert before["accepting_jobs"] and before["data_schema"] == "sirchmunk-data-v1"
    assert business.get("/api/v1/chat/sessions").status_code == 200
    assert client.post("maintenance", json={"enabled": True, "operation_id": "owner-0001"}).status_code == 200
    assert business.get("/api/v1/chat/sessions").status_code == 503
    body = dict(operation_id="flush-0001", instance_id=cfg["instance_id"],
                service_instance_id=before["service_instance_id"], maintenance_operation_id="owner-0001")
    response = client.post("flush/operations", json=body)
    assert response.status_code == 200 and response.json()["flushed"] is True
    response = client.post("backup", json={"operation_id": "owner-0001"})
    assert response.status_code == 200 and response.json()["consistent"] is True
    assert client.get("status").json()["active_jobs"] == 0
    value = intent(process=before["service_instance_id"])
    result = stop(value)
    assert result.returncode == 0, result.stderr
    assert first.wait(timeout=5) == 0
    _, receipt, clean = sup._observe(cfg, value)
    assert clean and receipt["stopped"] is True
    assert stop(value).returncode == 0  # read-only replay after process exit
    second, after = start()
    assert after["service_instance_id"] != before["service_instance_id"]
    assert after["maintenance"] == {"enabled": True, "operation_id": "owner-0001"}
    assert not after["accepting_jobs"]
    assert client.get("flush/operations/flush-0001").json()["service_instance_id"] == before["service_instance_id"]
    stale = stop(value)
    assert stale.returncode != 0 and "identity_mismatch" in stale.stderr
    assert second.poll() is None
    assert client.post("maintenance", json={"enabled": False, "operation_id": "owner-0001"}).status_code == 200
    assert business.get("/api/v1/chat/sessions").status_code == 200
    assert client.post("maintenance", json={"enabled": True, "operation_id": "owner-0001"}).status_code == 200
    current = intent(process=after["service_instance_id"])
    result = stop(current)
    assert result.returncode == 0, result.stderr
    assert second.wait(timeout=5) == 0


def test_real_stop_invalid_intents_leave_original_process_alive(real_server):
    cfg, client, _, start, stop = real_server
    child, status = start()
    assert client.post("maintenance", json={"enabled": True, "operation_id": "owner-0001"}).status_code == 200
    value = intent(process=status["service_instance_id"])
    for change in ({"service_instance_id": str(uuid.uuid4())}, {"maintenance_owner": "wrong-0001"},
                   {"expires_at": time.time() - 1}, {"extra": "forbidden"}, {"nonce": "F" * 64}):
        result = stop({**value, **change}, timeout=1)
        assert result.returncode != 0, (change, result.test_timing)
        assert child.poll() is None
        assert client.get("status").json()["service_instance_id"] == value["service_instance_id"]
        assert not (Path(cfg["mailbox_directory"]) / "request.json").exists()
    result = stop(value)
    assert result.returncode == 0, result.stderr
    assert child.wait(timeout=5) == 0
