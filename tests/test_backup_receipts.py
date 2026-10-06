"""Strict backup actions over real private stores; no provider/model calls."""
from contextlib import closing
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from sirchmunk.api.lifecycle import canonical, digest_file
from sirchmunk.api.management import ManagementError, ManagementHolder, ManagementState, _management_app
from sirchmunk.storage.lifecycle import bootstrap_managed_root
from test_supervisor import protect_test_directory, intent


ROOT_ROUTE = "/management/v1/backup-operations"
RESULT_FIELDS = {"backup_id", "completed", "consistent", "data_schema", "effect_committed"}
REQUEST_FIELDS = {"operation_id", "instance_id", "service_instance_id", "maintenance_operation_id"}
RECEIPT_FIELDS = REQUEST_FIELDS | RESULT_FIELDS | {"schema_id", "request_sha256", "state"}


@pytest.fixture
def managed(tmp_path, monkeypatch):
    protect_test_directory(tmp_path)
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(root / ".cache/models"))
    monkeypatch.setenv("HF_HOME", str(root / ".cache/models"))
    runtime = bootstrap_managed_root(dict(SIRCHMUNK_WORK_PATH=str(root),
        SIRCHMUNK_MANAGEMENT_ENABLED="true", SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED="true",
        SIRCHMUNK_MANAGEMENT_INSTANCE_ID="backup-test"))
    state = ManagementState(root / ".control", "backup-test", "synthetic-build")
    state.enable_lifecycle(runtime)
    state.lifecycle.initialize()
    try:
        yield state
    finally:
        if not state._closed:
            if not runtime.failure and not state.lifecycle.pending():
                runtime.close()
            state.close()


@pytest.fixture
def client(managed):
    holder = ManagementHolder()
    holder.state = managed
    with TestClient(_management_app(holder, "synthetic-independent-management-token"),
                    headers={"Authorization": "Bearer synthetic-independent-management-token"}) as client:
        yield client


def request(state, op="backup-action-0001"):
    return dict(operation_id=op, instance_id=state.instance_id,
                service_instance_id=state.service_instance_id, maintenance_operation_id="owner-0001")


def own(state):
    state.set_maintenance(True, "owner-0001")


def check_receipt(receipt, body, state):
    assert set(receipt) == RECEIPT_FIELDS and len(receipt) == 12
    assert receipt["schema_id"] == "sirchmunk-backup-operation-result.v1"
    assert receipt["request_sha256"] == hashlib.sha256(canonical(body).encode()).hexdigest()
    assert {key: receipt[key] for key in REQUEST_FIELDS} == body
    assert receipt["state"] == state
    if state == "accepted":
        assert all(receipt[key] is None for key in RESULT_FIELDS)
    else:
        assert receipt["data_schema"] == "sirchmunk-data-v1"
        assert receipt["backup_id"].startswith("backup-") and len(receipt["backup_id"]) == 39
        assert all(receipt[key] is True for key in ("completed", "consistent", "effect_committed"))


def test_distinct_actions_same_owner_make_real_different_private_snapshots(managed, client, monkeypatch):
    root, life = managed.lifecycle.runtime.root, managed.lifecycle
    own(managed)
    source = root / "uploads/deep/document.txt"
    source.parent.mkdir(parents=True)
    source.write_text("first synthetic document", encoding="utf-8")
    (root / ".env").write_text("FAKE_TOKEN=synthetic-only\n", encoding="utf-8")
    models = root / ".cache/models"
    models.mkdir(exist_ok=True)
    (models / "ignored.txt").write_text("never copied")
    original, begun = life._snapshot, []
    def snapshot(backup_id):
        # Another connection sees FULL accepted before any snapshot IO.
        with closing(sqlite3.connect(root / ".control/management.sqlite3")) as db:
            rows = db.execute("SELECT request_json,result_json FROM lifecycle_operations "
                              "WHERE kind='backup-operation' AND state='accepted'").fetchall()
        assert len(rows) == 1
        body, receipt = map(json.loads, rows[0])
        check_receipt(receipt, body, "accepted")
        assert managed.db.execute("PRAGMA synchronous").fetchone() == (2,)
        begun.append(body["operation_id"])
        return original(backup_id)
    monkeypatch.setattr(life, "_snapshot", snapshot)
    replies = []
    for n in (1, 2):
        body = request(managed, f"backup-action-000{n}")
        response = client.post(ROOT_ROUTE, json=body)
        assert response.status_code == 200
        result = response.json()
        check_receipt(result, body, "succeeded")
        replies.append(result)
        source.write_text("second synthetic document", encoding="utf-8")
    assert begun == [r["operation_id"] for r in replies]
    assert replies[0]["backup_id"] != replies[1]["backup_id"]
    from sirchmunk.api.supervisor import require_protected
    for n, result in enumerate(replies):
        snapshot_root = root / ".control/backups" / result["backup_id"]
        assert (snapshot_root / "uploads/deep/document.txt").read_text() == (
            "first synthetic document" if n == 0 else "second synthetic document")
        manifest = json.loads((snapshot_root / "manifest.json").read_text())
        for entry in manifest["files"]:
            path = snapshot_root / entry["path"]
            assert path.stat().st_size == entry["size"] and digest_file(path) == entry["sha256"]
            require_protected(path)
        assert not any("backups/" in r["path"] or ".cache/models/" in r["path"] for r in manifest["files"])
        with closing(sqlite3.connect(snapshot_root / ".control/management.sqlite3")) as db:
            assert db.execute("SELECT enabled,operation_id FROM gate").fetchone() == (1, "owner-0001")
            saved = db.execute("SELECT result_json FROM lifecycle_operations WHERE kind='backup-operation' "
                               "AND operation_id=?", (result["operation_id"],)).fetchone()
            check_receipt(json.loads(saved[0]), request(managed, result["operation_id"]), "accepted")
    before = sorted(p.name for p in (root / ".control/backups").iterdir())
    def forbidden(*args):
        raise AssertionError("duplicate must not create or flush a snapshot")
    monkeypatch.setattr(life, "_snapshot", forbidden)
    assert client.post(ROOT_ROUTE, json=request(managed)).json() == replies[0]
    assert client.get(ROOT_ROUTE + "/backup-action-0001").json() == replies[0]
    assert sorted(p.name for p in (root / ".control/backups").iterdir()) == before
    assert len(begun) == 2


def test_legacy_and_flush_same_operation_id_never_mix_receipt_shapes(managed, client):
    own(managed)
    body = request(managed, "owner-0001")
    new = client.post(ROOT_ROUTE, json=body)
    old = client.post("/management/v1/backup", json={"operation_id": "owner-0001"})
    flush = client.post("/management/v1/flush/operations", json=body)
    assert new.status_code == old.status_code == flush.status_code == 200
    check_receipt(new.json(), body, "succeeded")
    assert set(old.json()) == {"backup_id", "completed", "consistent", "data_schema"}
    assert len(flush.json()) == 9 and flush.json()["flushed"] is True
    assert old.json()["backup_id"] != new.json()["backup_id"]
    assert client.get("/management/v1/backup/operations/owner-0001").json() == old.json()
    assert client.get(ROOT_ROUTE + "/owner-0001").json() == new.json()
    assert client.post("/management/v1/backup", json=body).status_code == 422
    assert client.post(ROOT_ROUTE, json={"operation_id": "owner-0001"}).status_code == 422
    assert {row[0] for row in managed.db.execute("SELECT kind FROM lifecycle_operations WHERE operation_id='owner-0001'")} == {
        "backup", "backup-operation", "flush"}


@pytest.mark.parametrize("change", [{"extra": 1}, {"operation_id": 1}, {"operation_id": "short"},
    {"maintenance_operation_id": False}, {"maintenance_operation_id": "bad/owner"}, {"instance_id": ""},
    {"instance_id": "x" * 81}, {"service_instance_id": None}, {"service_instance_id": "x" * 101}])
def test_strict_request_fields_types_and_limits(managed, client, change):
    own(managed)
    response = client.post(ROOT_ROUTE, json={**request(managed), **change})
    assert response.status_code == 422
    assert response.json() == {"error": {"code": "invalid_backup_operation_request"}}
    assert not managed.lifecycle.pending()
    assert not (managed.lifecycle.runtime.root / ".control/backups").exists()


@pytest.mark.parametrize("raw,status,code", [
    (b'{"operation_id":"a","operation_id":"b"}', 400, "invalid_json"),
    (b'{"x":NaN}', 400, "invalid_json"), (b'[]', 422, "invalid_request"),
    (b'{', 400, "invalid_json"), (b' ' * 16385, 413, "request_too_large")])
def test_malformed_http_bodies_are_rejected_before_backup(managed, client, raw, status, code):
    response = client.post(ROOT_ROUTE, content=raw)
    assert response.status_code == status and response.json() == {"error": {"code": code}}
    assert not managed.lifecycle.pending()


def test_authorization_missing_receipt_capability_and_disabled_lifecycle(managed, client, tmp_path):
    assert "backup-operation-receipts.v1" in client.get("/management/v1/capabilities").json()["capabilities"]
    for route in (ROOT_ROUTE, ROOT_ROUTE + "/missing-id-0001", "/management/v1/backup/operations/missing-id-0001"):
        assert client.get(route, headers={"Authorization": "Bearer wrong-business-token"}).status_code == 401
    assert client.post(ROOT_ROUTE, json=request(managed), headers={"Authorization": ""}).status_code == 401
    assert client.get(ROOT_ROUTE + "/missing-id-0001").status_code == 404
    assert client.get(ROOT_ROUTE + "/bad").status_code == 422
    disabled = ManagementState(tmp_path / "disabled/.control", "disabled-test", "synthetic-build")
    holder = ManagementHolder()
    holder.state = disabled
    try:
        with TestClient(_management_app(holder, "synthetic"), headers={"Authorization": "Bearer synthetic"}) as other:
            assert "backup-operation-receipts.v1" not in other.get("/management/v1/capabilities").json()["capabilities"]
            assert other.post(ROOT_ROUTE, json=request(managed)).status_code == 501
            assert other.get(ROOT_ROUTE + "/missing-id-0001").status_code == 501
    finally:
        disabled.close()


def test_owner_identity_and_real_counters_are_checked(managed, client):
    assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 409
    assert managed.admit()
    own(managed)
    assert client.post(ROOT_ROUTE, json=request(managed)).json()["error"]["code"] == "business_work_not_drained"
    managed.release()
    release = managed.lifecycle.runtime.reserve_background()
    try:
        assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 409
    finally:
        release()
    for change in ({"instance_id": "different"}, {"service_instance_id": "old-uuid"},
                   {"maintenance_operation_id": "wrong-owner"}):
        assert client.post(ROOT_ROUTE, json={**request(managed), **change}).status_code == 409
    assert not managed.lifecycle.pending()
    assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 200
    changed = {**request(managed), "service_instance_id": "different"}
    response = client.post(ROOT_ROUTE, json=changed)
    assert response.status_code == 409 and response.json()["error"]["code"] == "lifecycle_operation_reused"


def test_concurrent_post_returns_accepted_and_get_remains_readable(managed, client, monkeypatch):
    own(managed)
    life, replies = managed.lifecycle, []
    original = life._snapshot
    entered, finish = threading.Event(), threading.Event()
    def slow(backup_id):
        entered.set()
        assert finish.wait(10)
        return original(backup_id)
    monkeypatch.setattr(life, "_snapshot", slow)
    body = request(managed)
    worker = threading.Thread(target=lambda: replies.append(client.post(ROOT_ROUTE, json=body)))
    worker.start()
    try:
        assert entered.wait(3)
        got = client.get(ROOT_ROUTE + "/" + body["operation_id"])
        assert got.status_code == 200
        check_receipt(got.json(), body, "accepted")
        duplicate = client.post(ROOT_ROUTE, json=body)
        assert duplicate.status_code == 202 and duplicate.json() == got.json()
        assert client.post(ROOT_ROUTE, json=request(managed, "backup-action-0002")).status_code == 409
        assert client.post("/management/v1/backup", json={"operation_id": "owner-0001"}).status_code == 409
        assert client.post("/management/v1/maintenance", json={"enabled": False, "operation_id": "owner-0001"}).status_code == 409
        assert life.workers == 1 and life.runtime.frozen
    finally:
        finish.set()
        worker.join(15)
    assert not worker.is_alive() and replies[0].status_code == 200
    check_receipt(client.get(ROOT_ROUTE + "/" + body["operation_id"]).json(), body, "succeeded")


def test_cancelled_http_await_keeps_worker_and_freeze_until_real_completion(managed, monkeypatch):
    own(managed)
    life = managed.lifecycle
    original_snapshot, original_backup = life._snapshot, life.backup_operation
    entered, finish, completed = threading.Event(), threading.Event(), threading.Event()
    def slow(backup_id):
        entered.set()
        assert finish.wait(10)
        return original_snapshot(backup_id)
    def tracked(body):
        try:
            return original_backup(body)
        finally:
            # The rejected competing POST also calls this wrapper; it must
            # not be mistaken for completion of the original snapshot worker.
            if body["operation_id"] == "backup-action-0001":
                completed.set()
    monkeypatch.setattr(life, "_snapshot", slow)
    monkeypatch.setattr(life, "backup_operation", tracked)
    holder = ManagementHolder()
    holder.state = managed
    async def exercise():
        transport = httpx.ASGITransport(app=_management_app(holder, "synthetic"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                     headers={"Authorization": "Bearer synthetic"}) as client:
            caller = asyncio.create_task(client.post(ROOT_ROUTE, json=request(managed)))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                assert life.workers == 1 and life.runtime.frozen and not completed.is_set()
                check_receipt((await client.get(ROOT_ROUTE + "/backup-action-0001")).json(),
                              request(managed), "accepted")
                assert (await client.post(ROOT_ROUTE, json=request(managed, "new-action-0002"))).status_code == 409
                assert (await client.post("/management/v1/maintenance",
                    json={"enabled": False, "operation_id": "owner-0001"})).status_code == 409
            finally:
                finish.set()
                await asyncio.gather(caller, return_exceptions=True)
                assert await asyncio.to_thread(completed.wait, 10)
            assert life.workers == 0 and not life.runtime.frozen
            check_receipt((await client.get(ROOT_ROUTE + "/backup-action-0001")).json(),
                          request(managed), "succeeded")
            assert managed.maintenance_enabled and not managed.admit()
    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["backup", "flush"])
def test_legacy_accepted_operation_blocks_new_backup(managed, client, monkeypatch, kind):
    own(managed)
    life = managed.lifecycle
    def fail():
        raise OSError("synthetic flush failure")
    with monkeypatch.context() as patch:
        patch.setattr(life.runtime, "flush", fail)
        route, body = ("/management/v1/backup", {"operation_id": "owner-0001"}) if kind == "backup" else (
            "/management/v1/flush/operations", request(managed, "flush-failed-0001"))
        assert client.post(route, json=body).status_code == 503
    def forbidden(*args):
        raise AssertionError("other-kind unknown must block new snapshot")
    monkeypatch.setattr(life, "_snapshot", forbidden)
    blocked = client.post(ROOT_ROUTE, json=request(managed))
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "lifecycle_operation_locked"
    assert managed.db.execute("SELECT kind,state FROM lifecycle_operations").fetchall() == [(kind, "accepted")]
    assert life.unknown and life.runtime.frozen


def test_configuration_unknown_blocks_new_backup(managed, client, monkeypatch):
    from test_management_config import Consumer, mutation
    consumer = Consumer()
    managed.enable_configuration(consumer)
    own(managed)
    consumer.fail = True
    receipt = client.post("/management/v1/config/operations",
                          json=mutation(managed, "config-failed-0001", 0, 7)).json()
    assert receipt["status"] == "unknown"
    consumer.fail = False
    def forbidden(*args):
        raise AssertionError("configuration unknown must block new snapshot")
    monkeypatch.setattr(managed.lifecycle, "_snapshot", forbidden)
    response = client.post(ROOT_ROUTE, json=request(managed))
    assert response.status_code == 409 and response.json()["error"]["code"] == "configuration_unknown_locked"
    assert not managed.lifecycle.pending()
    assert not (managed.lifecycle.runtime.root / ".control/backups").exists()


@pytest.mark.parametrize("stage", ["flush", "copy", "directory", "publish", "after-publish", "receipt"])
def test_real_io_or_receipt_failure_keeps_unknown_and_all_mutators_locked(managed, client, monkeypatch, stage):
    import sirchmunk.api.lifecycle as module
    from sirchmunk.api.security import is_path_allowed, is_path_allowed_strict
    own(managed)
    life, root = managed.lifecycle, managed.lifecycle.runtime.root
    class Consumer:
        limit = 100
        def get_limit(self): return self.limit
        def set_limit(self, value): self.limit = value
    managed.enable_configuration(Consumer())
    def fail(*args, **kwargs):
        raise OSError("synthetic-private-secret-path-do-not-leak")
    original_publish = module.durable_publish_directory
    def after_publish(*args):
        original_publish(*args)
        fail()
    db = managed.db
    class FailedReceiptDB:
        def execute(self, sql, *args):
            if sql.startswith("UPDATE lifecycle_operations"):
                raise sqlite3.OperationalError("synthetic-sensitive-receipt-error")
            return db.execute(sql, *args)
        def __getattr__(self, name):
            return getattr(db, name)
    with monkeypatch.context() as patch:
        if stage == "flush": patch.setattr(life.runtime, "flush", fail)
        elif stage == "copy": patch.setattr(module.shutil, "copyfile", fail)
        elif stage == "directory": patch.setattr(module, "durable_directory", fail)
        elif stage == "publish": patch.setattr(module, "durable_publish_directory", fail)
        elif stage == "after-publish": patch.setattr(module, "durable_publish_directory", after_publish)
        else: patch.setattr(managed, "db", FailedReceiptDB())
        response = client.post(ROOT_ROUTE, json=request(managed))
    assert response.status_code == 503
    assert response.json() == {"error": {"code": "lifecycle_effect_unknown"}}
    check_receipt(client.get(ROOT_ROUTE + "/backup-action-0001").json(), request(managed), "accepted")
    assert life.unknown and life.runtime.frozen and life.workers == 0
    assert managed.status()["readiness"] == "unknown" and not managed.admit()
    if stage in {"after-publish", "receipt"}:
        assert any(not p.name.endswith(".partial") for p in (root / ".control/backups").iterdir())
    before = {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in (root / ".control/backups").rglob("*")}
    assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 202
    assert client.post(ROOT_ROUTE, json=request(managed, "another-action")).status_code == 409
    assert client.post("/management/v1/backup", json={"operation_id": "owner-0001"}).status_code == 409
    assert client.post("/management/v1/flush/operations", json=request(managed, "flush-new-0001")).status_code == 409
    assert client.post("/management/v1/maintenance", json={"enabled": False, "operation_id": "owner-0001"}).status_code == 409
    cfg = dict(protocol_version="1.0", operation_id="config-new-0001", instance_id=managed.instance_id,
               service_instance_id=managed.service_instance_id, config_schema_id="sirchmunk.runtime.v1",
               expected_revision=0, changes={"maxWebSocketConnections": 2})
    cfg["request_digest"] = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    assert client.post("/management/v1/config/operations", json=cfg).status_code == 409
    with pytest.raises(ManagementError, match="locked"):
        life.request_stop(intent(instance=managed.instance_id, process=managed.service_instance_id))
    assert {p.relative_to(root).as_posix(): p.stat().st_mtime_ns for p in (root / ".control/backups").rglob("*")} == before
    monkeypatch.setenv("SIRCHMUNK_ALLOWED_PATHS", str(root))
    for check in (is_path_allowed, is_path_allowed_strict):
        assert not check(str(root / ".control/backups")) and not check(str(root))


def test_accepted_sqlite_write_failure_never_starts_snapshot(managed, client, monkeypatch):
    own(managed)
    db, calls = managed.db, []
    class FailedDB:
        def execute(self, sql, *args):
            if sql.startswith("INSERT INTO lifecycle_operations"):
                raise sqlite3.OperationalError("synthetic-sensitive-write-error")
            return db.execute(sql, *args)
    with monkeypatch.context() as patch:
        patch.setattr(managed, "db", FailedDB())
        patch.setattr(managed.lifecycle, "_snapshot", lambda _: calls.append(1))
        response = client.post(ROOT_ROUTE, json=request(managed))
    assert response.status_code == 503 and not calls
    assert managed.lifecycle.workers == 0 and managed.lifecycle.unknown
    assert not managed.lifecycle.pending() and managed.lifecycle.runtime.frozen
    assert client.get(ROOT_ROUTE + "/backup-action-0001").status_code == 404
    assert client.post(ROOT_ROUTE, json=request(managed, "backup-action-0002")).status_code == 409


def test_snapshot_directory_and_ancestor_fsync_precede_success(managed, client, monkeypatch):
    import sirchmunk.api.lifecycle as module
    import sirchmunk.storage.lifecycle as storage
    own(managed)
    seen = []
    original_file, original_dir = module.durable_file, storage.durable_directory
    def file_sync(path):
        seen.append(("file", Path(path)))
        return original_file(path)
    def directory_sync(path):
        seen.append(("directory", Path(path)))
        return original_dir(path)
    monkeypatch.setattr(module, "durable_file", file_sync)
    monkeypatch.setattr(module, "durable_directory", directory_sync)
    monkeypatch.setattr(storage, "durable_directory", directory_sync)
    original_finish = managed.lifecycle._finish
    def finish(kind, body, result):
        parent = managed.lifecycle.runtime.root / ".control/backups"
        dirs = [p for kind, p in seen if kind == "directory"]
        assert parent in dirs and all(p in dirs for p in parent.parents)
        assert any(p.name.endswith(".partial") for p in dirs)
        assert any(p.name == "management.sqlite3" for kind, p in seen if kind == "file")
        return original_finish(kind, body, result)
    monkeypatch.setattr(managed.lifecycle, "_finish", finish)
    assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 200


def test_linked_backup_parent_never_writes_outside_root(managed, client, tmp_path):
    own(managed)
    parent = managed.lifecycle.runtime.root / ".control/backups"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        parent.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlinks")
    try:
        assert client.post(ROOT_ROUTE, json=request(managed)).status_code == 503
        assert not list(outside.iterdir())
        check_receipt(client.get(ROOT_ROUTE + "/backup-action-0001").json(), request(managed), "accepted")
    finally:
        parent.unlink()
