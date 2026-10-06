"""Real isolated stores and failure injection for lifecycle receipts."""
import asyncio
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import threading

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sirchmunk.api.management import (
    AdmissionMiddleware, ManagementError, ManagementHolder, ManagementState, _management_app,
)
from sirchmunk.api.lifecycle import canonical, digest_file
from sirchmunk.storage.lifecycle import bootstrap_managed_root


def launch(root):
    return dict(SIRCHMUNK_WORK_PATH=str(root), SIRCHMUNK_MANAGEMENT_ENABLED="true",
                SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED="true",
                SIRCHMUNK_MANAGEMENT_INSTANCE_ID="lifecycle-test")


@pytest.fixture
def managed(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(root / ".cache" / "models"))
    runtime = bootstrap_managed_root(launch(root))
    state = ManagementState(root / ".control", "lifecycle-test", "isolated-build")
    state.enable_lifecycle(runtime)
    state.lifecycle.initialize()
    try:
        yield state
    finally:
        if not state._closed:
            if not runtime.failure and not state.lifecycle.pending():
                runtime.close()
            state.close()


def request(state, operation="flush-0001"):
    return dict(operation_id=operation, instance_id=state.instance_id,
                service_instance_id=state.service_instance_id, maintenance_operation_id="owner-0001")


def own(state):
    state.set_maintenance(True, "owner-0001")


def test_real_schema_readiness_and_strict_flush_receipt(managed, monkeypatch):
    life = managed.lifecycle
    assert len(managed.status()) == 14
    assert managed.readiness()["storage_validated"] is True
    assert managed.status()["data_schema"] == "sirchmunk-data-v1"
    assert "flush-operation-receipts.v1" in managed.capabilities()["capabilities"]
    with pytest.raises(ManagementError, match="maintenance_owner_required"):
        life.flush(request(managed))
    own(managed)
    assert managed.status()["readiness"] == "ok"
    assert not managed.status()["accepting_jobs"]
    calls = []
    original = life.runtime.flush
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(life.runtime, "flush", counted)
    body = request(managed)
    result = life.flush(body)
    assert len(result) == 9
    assert result["request_sha256"] == hashlib.sha256(canonical(body).encode()).hexdigest()
    assert result["schema_id"] == "sirchmunk-flush-operation-result.v1"
    assert result["state"] == "succeeded" and result["flushed"] is True and result["effect_committed"] is True
    assert life.flush(body) == life.get("flush", body["operation_id"]) == result
    assert calls == [1]
    with pytest.raises(ManagementError, match="reused"):
        life.flush({**body, "service_instance_id": "different"})
    with pytest.raises(ManagementError, match="invalid_flush_request"):
        life.flush({**body, "extra": True})
    with pytest.raises(ManagementError, match="identity_mismatch"):
        life.flush({**request(managed, "flush-0002"), "service_instance_id": "old-process"})


def test_actual_background_and_http_reservations_prevent_flush(managed):
    assert managed.admit()
    own(managed)
    with pytest.raises(ManagementError, match="not_drained"):
        managed.lifecycle.flush(request(managed))
    managed.release()
    release = managed.lifecycle.runtime.reserve_background()
    assert managed.status()["active_jobs"] == 1
    with pytest.raises(ManagementError, match="not_drained"):
        managed.lifecycle.flush(request(managed))
    release()
    assert managed.lifecycle.flush(request(managed))["flushed"] is True


def test_flush_failure_remains_accepted_and_locks_all_mutators(managed, monkeypatch):
    own(managed)
    life = managed.lifecycle
    calls = []
    def fail():
        calls.append(1)
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(life.runtime, "flush", fail)
    with pytest.raises(OSError):
        life.flush(request(managed))
    receipt = life.get("flush", "flush-0001")
    assert receipt["state"] == "accepted" and receipt["flushed"] is None
    assert receipt["effect_committed"] is None
    assert life.flush(request(managed)) == receipt
    assert calls == [1]
    assert managed.status()["readiness"] == "unknown"
    assert not managed.admit()
    with pytest.raises(ManagementError, match="locked"):
        managed.set_maintenance(False, "owner-0001")
    with pytest.raises(ManagementError, match="locked"):
        life.flush(request(managed, "flush-0002"))
    with pytest.raises(ManagementError, match="locked"):
        life.backup({"operation_id": "owner-0001"})


def test_get_responds_while_real_flush_worker_is_pending(managed, monkeypatch):
    own(managed)
    entered, finish = threading.Event(), threading.Event()
    life = managed.lifecycle
    original = life.runtime.flush
    def slow():
        entered.set()
        assert finish.wait(5)
        return original()
    monkeypatch.setattr(life.runtime, "flush", slow)
    holder = ManagementHolder()
    holder.state = managed
    headers = {"Authorization": "Bearer synthetic-test"}
    results = []
    with TestClient(_management_app(holder, "synthetic-test")) as client:
        worker = threading.Thread(target=lambda: results.append(client.post(
            "/management/v1/flush/operations", json=request(managed), headers=headers)))
        worker.start()
        try:
            assert entered.wait(3)
            response = client.get("/management/v1/flush/operations/flush-0001", headers=headers)
            assert response.json()["state"] == "accepted"
            response = client.post("/management/v1/flush/operations", json=request(managed), headers=headers)
            assert response.status_code == 202
            assert client.post("/management/v1/maintenance", headers=headers,
                               json={"enabled": False, "operation_id": "owner-0001"}).status_code == 409
            assert life.workers == 1
        finally:
            finish.set()
            worker.join(5)
        assert not worker.is_alive()
        assert results[0].status_code == 200
        assert results[0].json()["state"] == "succeeded"
        assert client.post("/management/v1/flush", json={"unexpected": 1}, headers=headers).status_code == 422
        assert client.post("/management/v1/flush", json={}, headers=headers).json()["flushed"] is True


def test_backup_includes_data_owner_and_accepted_ledger_not_recursive_snapshots(managed, tmp_path):
    root = managed.lifecycle.runtime.root
    nested = root / "uploads" / "deep" / "input.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("synthetic document", encoding="utf-8")
    (root / ".env").write_text("TEST_VALUE=synthetic-only\n", encoding="utf-8")
    models = root / ".cache" / "models"
    models.mkdir(exist_ok=True)
    (models / "regenerable").write_text("not in snapshot")
    own(managed)
    result = managed.lifecycle.backup({"operation_id": "owner-0001"})
    assert result["completed"] is True and result["consistent"] is True
    assert managed.lifecycle.backup({"operation_id": "owner-0001"}) == result
    snapshot = root / ".control" / "backups" / result["backup_id"]
    manifest = json.loads((snapshot / "manifest.json").read_text())
    paths = {entry["path"] for entry in manifest["files"]}
    assert {".control/management.sqlite3", ".control/data-schema.json", ".env",
            "uploads/deep/input.txt", ".cache/history/chat_history.db",
            ".cache/knowledge/knowledge_clusters.parquet"} <= paths
    assert not any("backups/" in path or ".cache/models/" in path or path.endswith(".lock") for path in paths)
    for entry in manifest["files"]:
        target = snapshot / entry["path"]
        assert target.stat().st_size == entry["size"]
        assert digest_file(target) == entry["sha256"]
    with closing(sqlite3.connect(str(snapshot / ".control" / "management.sqlite3"))) as db:
        assert db.execute("SELECT enabled,operation_id FROM gate").fetchone() == (1, "owner-0001")
        assert db.execute("SELECT state FROM lifecycle_operations WHERE kind='backup'").fetchone() == ("accepted",)
        assert db.execute("SELECT unclean FROM process_state").fetchone() == (1,)
    # Recovered backups must keep their unknown ledger; no delete-to-unlock.
    restored = tmp_path / "restored-work"
    shutil.copytree(snapshot, restored)
    recovered = run_child(restored, "inspect")
    assert recovered["receipt"]["completed"] is False
    assert recovered["status"]["accepting_jobs"] is False
    assert recovered["status"]["readiness"] == "unknown"


def test_backup_io_failure_never_issues_success(managed, monkeypatch):
    own(managed)
    import sirchmunk.api.lifecycle as module
    def failed_copy(*args):
        raise OSError("synthetic copy failure")
    monkeypatch.setattr(module.shutil, "copyfile", failed_copy)
    with pytest.raises(OSError):
        managed.lifecycle.backup({"operation_id": "owner-0001"})
    receipt = managed.lifecycle.get("backup", "owner-0001")
    assert receipt["completed"] is False and receipt["consistent"] is False
    assert managed.lifecycle.runtime.frozen


@pytest.mark.parametrize("busy", [False, True])
def test_maintenance_closes_idle_ws_and_allows_only_current_message(managed, busy):
    holder = ManagementHolder()
    holder.state = managed
    app = FastAPI()
    app.add_middleware(AdmissionMiddleware, holder=holder)
    finish = threading.Event()
    received = []
    @app.websocket("/api/v1/chat")
    async def chat(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                received.append(await ws.receive_text())
                await ws.send_text("started")
                while not finish.is_set():
                    await asyncio.sleep(0.01)
                await ws.send_text("finished")
        except WebSocketDisconnect:
            pass
    with TestClient(app) as client, client.websocket_connect("/api/v1/chat") as ws:
        if busy:
            ws.send_text("first")
            assert ws.receive_text() == "started"
        own(managed)
        if busy:
            ws.send_text("must-not-run")
            assert managed.status()["active_jobs"] == 1
            finish.set()
            assert ws.receive_text() == "finished"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 1013
    assert received == (["first"] if busy else [])
    assert managed.status()["active_jobs"] == 0


CHILD = r'''
import json, os, sys
from pathlib import Path
from sirchmunk.storage.lifecycle import bootstrap_managed_root
from sirchmunk.api.management import ManagementState
root = Path(os.environ['SIRCHMUNK_WORK_PATH'])
runtime = bootstrap_managed_root(dict(SIRCHMUNK_WORK_PATH=str(root), SIRCHMUNK_MANAGEMENT_ENABLED='true',
    SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED='true', SIRCHMUNK_MANAGEMENT_INSTANCE_ID='lifecycle-test'))
state = ManagementState(root / '.control', 'lifecycle-test', 'child-build')
state.enable_lifecycle(runtime)
state.lifecycle.initialize()
if sys.argv[1] == 'crash':
    state.set_maintenance(True, 'owner-0001')
    runtime.flush = lambda: os._exit(27)
    state.lifecycle.flush(dict(operation_id='crash-0001', instance_id=state.instance_id,
        service_instance_id=state.service_instance_id, maintenance_operation_id='owner-0001'))
else:
    kind, op = ('backup', 'owner-0001') if sys.argv[1] == 'inspect' else ('flush', 'crash-0001')
    receipt = state.lifecycle.get(kind, op)
    if kind == 'flush':
        body = {key: receipt[key] for key in ('operation_id','instance_id','service_instance_id','maintenance_operation_id')}
        runtime.flush = lambda: (_ for _ in ()).throw(RuntimeError('must not replay'))
        assert state.lifecycle.flush(body) == receipt
    print('RESULT=' + json.dumps(dict(status=state.status(), receipt=receipt)))
    state.close()
'''


def run_child(root, mode):
    env = os.environ.copy()
    env.update(SIRCHMUNK_WORK_PATH=str(root), EMBEDDING_CACHE_DIR=str(root / ".cache" / "models"),
               HF_HOME=str(root / ".cache" / "models"))
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child = subprocess.run([sys.executable, "-c", CHILD, mode], env=env, cwd=root.parent,
                           capture_output=True, text=True, timeout=25)
    if mode == "crash":
        assert child.returncode == 27, child.stderr
        return
    assert child.returncode == 0, child.stderr
    return json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))


def test_real_crash_after_accepted_does_not_replay_on_restart(tmp_path):
    root = tmp_path / "crash-root"
    run_child(root, "crash")
    recovered = run_child(root, "query")
    assert recovered["receipt"]["state"] == "accepted"
    assert recovered["receipt"]["service_instance_id"] != recovered["status"]["service_instance_id"]
    assert recovered["status"]["maintenance"] == {"enabled": True, "operation_id": "owner-0001"}
    assert recovered["status"]["active_jobs"] is None
    assert recovered["status"]["accepting_jobs"] is False


def test_cancelled_process_await_stays_counted_until_actual_work_exits(managed, monkeypatch):
    from sirchmunk.retrieve.text_retriever import GrepRetriever
    async def exercise():
        entered, finish = asyncio.Event(), asyncio.Event()
        async def actual(*args, **kwargs):
            entered.set()
            await finish.wait()
            return {"returncode": 0}
        monkeypatch.setattr(GrepRetriever, "_run_search_process_impl", actual)
        caller = asyncio.create_task(GrepRetriever._run_search_process(
            "unused", [], semaphore=asyncio.Semaphore(1), queue_timeout=1,
            execution_timeout=1, json_output=False))
        await entered.wait()
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        own(managed)
        assert managed.status()["active_jobs"] == 1
        with pytest.raises(ManagementError, match="not_drained"):
            managed.lifecycle.flush(request(managed))
        finish.set()
        for _ in range(20):
            if not managed.lifecycle.runtime.active_background():
                break
            await asyncio.sleep(0.01)
        assert managed.status()["active_jobs"] == 0
    asyncio.run(exercise())


def test_pending_flush_blocks_configuration_effect(managed, monkeypatch):
    class Consumer:
        limit = 100
        def get_limit(self):
            return self.limit
        def set_limit(self, value):
            self.limit = value
    managed.enable_configuration(Consumer())
    own(managed)
    def fail():
        raise OSError("synthetic flush failure")
    monkeypatch.setattr(managed.lifecycle.runtime, "flush", fail)
    with pytest.raises(OSError):
        managed.lifecycle.flush(request(managed))
    body = dict(protocol_version="1.0", operation_id="config-0001", instance_id=managed.instance_id,
                service_instance_id=managed.service_instance_id, config_schema_id="sirchmunk.runtime.v1",
                expected_revision=0, changes={"maxWebSocketConnections": 3})
    body["request_digest"] = hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with pytest.raises(ManagementError, match="lifecycle_operation_locked"):
        managed.apply_configuration(body)
    assert managed.config_document()["revision"] == 0


@pytest.mark.parametrize("kind", ["history", "knowledge"])
def test_managed_request_storage_errors_fail_closed(managed, monkeypatch, kind):
    def fail(*args, **kwargs):
        raise OSError("synthetic write failure")
    if kind == "history":
        from sirchmunk.api.components.history_storage import HistoryStorage
        storage = HistoryStorage()
        monkeypatch.setattr(storage.db, "insert_data", fail)
        with pytest.raises(OSError, match="synthetic"):
            storage.save_session({"session_id": "bad", "title": "test"})
    else:
        from sirchmunk.storage.knowledge_storage import KnowledgeStorage
        storage = KnowledgeStorage()
        monkeypatch.setattr(storage.db, "drop_table", fail)
        with pytest.raises(OSError, match="synthetic"):
            asyncio.run(storage.clear())
    assert managed.status()["readiness"] == "unknown"
    assert not managed.admit()


def test_private_snapshots_cannot_be_addressed_by_business_path_api(managed, monkeypatch):
    from sirchmunk.api.security import is_path_allowed, is_path_allowed_strict
    root = managed.lifecycle.runtime.root
    monkeypatch.setenv("SIRCHMUNK_ALLOWED_PATHS", str(root))
    for check in (is_path_allowed, is_path_allowed_strict):
        assert not check(str(root / ".control" / "backups" / "backup-synthetic" / ".env"))
        assert not check(str(root / ".control"))
        assert not check(str(root))
        assert not check(str(root.parent))
        assert check(str(root / "uploads"))


def test_http_disk_error_is_redacted_and_receipt_stays_queryable(managed, monkeypatch):
    own(managed)
    def fail():
        raise OSError("synthetic-sensitive-path-must-not-leak")
    monkeypatch.setattr(managed.lifecycle.runtime, "flush", fail)
    holder = ManagementHolder()
    holder.state = managed
    headers = {"Authorization": "Bearer synthetic-test"}
    with TestClient(_management_app(holder, "synthetic-test")) as client:
        response = client.post("/management/v1/flush/operations", json=request(managed), headers=headers)
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "lifecycle_effect_unknown"}}
        assert client.get("/management/v1/flush/operations/flush-0001", headers=headers).json()["state"] == "accepted"
