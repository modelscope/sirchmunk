"""Real HTTP ACK loss and separate-process ledger recovery, private roots only."""
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading

import httpx
import pytest

from test_supervisor import config, real_server, intent, protect_test_directory
from test_backup_receipts import check_receipt, RESULT_FIELDS


def test_real_http_lost_ack_only_queries_original_backup_and_restart_keeps_receipt(real_server):
    cfg, client, business, start, stop = real_server
    child, status = start()
    assert "backup-operation-receipts.v1" in client.get("capabilities").json()["capabilities"]
    assert client.post("maintenance", json={"enabled": True, "operation_id": "owner-0001"}).status_code == 200
    body = dict(operation_id="backup-real-0001", instance_id=cfg["instance_id"],
                service_instance_id=status["service_instance_id"], maintenance_operation_id="owner-0001")
    completed, calls, proxy_errors = threading.Event(), [], []
    class LostAck(BaseHTTPRequestHandler):
        def do_POST(self):
            # A test-only proxy completes the real upstream request, consumes
            # its response, then drops the caller's socket without sending ACK.
            try:
                payload = self.rfile.read(int(self.headers["Content-Length"]))
                calls.append(self.path)
                response = client.post("backup-operations", content=payload)
                assert response.status_code == 200, response.text
            except BaseException as exc:
                proxy_errors.append(exc)
            finally:
                completed.set()
                self.close_connection = True
        def log_message(self, *args):
            pass
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), LostAck)
    thread = threading.Thread(target=proxy.serve_forever)
    thread.start()
    try:
        with httpx.Client(trust_env=False, timeout=10) as caller:
            with pytest.raises(httpx.TransportError):
                caller.post(f"http://127.0.0.1:{proxy.server_port}/backup", json=body)
        assert completed.wait(5) and not proxy_errors
        assert calls == ["/backup"]
        response = client.get("backup-operations/" + body["operation_id"])
        assert response.status_code == 200
        first = response.json()
        check_receipt(first, body, "succeeded")
        assert client.get("status").json()["maintenance"]["operation_id"] == "owner-0001"
        assert business.get("/api/v1/chat/sessions").status_code == 503
    finally:
        proxy.shutdown()
        thread.join(5)
        proxy.server_close()
    root = Path(cfg["work_path"])
    snapshots = lambda: sorted(p.name for p in (root / ".control/backups").iterdir())
    assert snapshots() == [first["backup_id"]]
    assert client.post("backup-operations", json=body).json() == first
    assert snapshots() == [first["backup_id"]]
    second_body = {**body, "operation_id": "backup-real-0002"}
    second = client.post("backup-operations", json=second_body)
    assert second.status_code == 200
    assert second.json()["backup_id"] != first["backup_id"] and len(snapshots()) == 2
    stopped = stop(intent(process=status["service_instance_id"]))
    assert stopped.returncode == 0, stopped.stderr
    assert child.wait(timeout=5) == 0
    restarted, current = start()
    assert current["service_instance_id"] != status["service_instance_id"]
    assert current["maintenance"] == {"enabled": True, "operation_id": "owner-0001"}
    assert client.get("backup-operations/" + body["operation_id"]).json() == first
    assert client.post("backup-operations", json=body).json() == first
    assert len(snapshots()) == 2
    assert client.post("backup-operations", json={**body, "operation_id": "backup-stale-0003"}).status_code == 409
    assert len(snapshots()) == 2
    fresh = {**body, "operation_id": "backup-fresh-0003", "service_instance_id": current["service_instance_id"]}
    assert client.post("backup-operations", json=fresh).status_code == 200
    assert len(snapshots()) == 3
    stopped = stop(intent(process=current["service_instance_id"]))
    assert stopped.returncode == 0, stopped.stderr
    assert restarted.wait(timeout=5) == 0


CHILD = r'''
import json, os, sys
from pathlib import Path
from sirchmunk.storage.lifecycle import bootstrap_managed_root
from sirchmunk.api.management import ManagementState, ManagementError
from sirchmunk.api import lifecycle as module
root = Path(os.environ["SIRCHMUNK_WORK_PATH"])
runtime = bootstrap_managed_root(dict(SIRCHMUNK_WORK_PATH=str(root), SIRCHMUNK_MANAGEMENT_ENABLED="true",
    SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED="true", SIRCHMUNK_MANAGEMENT_INSTANCE_ID="backup-process-test"))
state = ManagementState(root / ".control", "backup-process-test", "synthetic-build")
state.enable_lifecycle(runtime)
state.lifecycle.initialize()
life = state.lifecycle
if sys.argv[1] in ("before-io", "after-publish", "success"):
    state.set_maintenance(True, "owner-0001")
    body = dict(operation_id="backup-crash-0001", instance_id=state.instance_id,
        service_instance_id=state.service_instance_id, maintenance_operation_id="owner-0001")
    if sys.argv[1] == "before-io":
        life._snapshot = lambda backup_id: os._exit(27)
    elif sys.argv[1] == "after-publish":
        publish = module.durable_publish_directory
        def fail_after_publish(*args):
            publish(*args)
            os._exit(28)
        module.durable_publish_directory = fail_after_publish
    receipt = life.backup_operation(body)
    runtime.close()
    state.close()
    print("RESULT=" + json.dumps(dict(receipt=receipt)))
else:
    receipt = life.get("backup-operation", "backup-crash-0001")
    body = {key: receipt[key] for key in ("operation_id","instance_id","service_instance_id","maintenance_operation_id")}
    def forbidden(*args):
        raise AssertionError("query/replay must not run IO")
    life._snapshot = forbidden
    assert life.backup_operation(body) == receipt
    if receipt["state"] == "accepted":
        assert life.unknown and runtime.frozen and state.counter_unknown
        for action in (
            lambda: life.backup_operation({**body, "operation_id":"new-action-0002","service_instance_id":state.service_instance_id}),
            lambda: life.backup({"operation_id":"owner-0001"}),
            lambda: life.flush({**body, "operation_id":"new-flush-0002","service_instance_id":state.service_instance_id}),
            lambda: state.set_maintenance(False,"owner-0001"),
        ):
            try: action()
            except ManagementError as exc: assert exc.code == "lifecycle_operation_locked"
            else: raise AssertionError("unknown operation must keep every write gate locked")
    else:
        try: life.backup_operation({**body, "operation_id":"new-action-0002"})
        except ManagementError as exc: assert exc.code == "lifecycle_identity_mismatch"
        else: raise AssertionError("old UUID cannot authorize new action")
    print("RESULT=" + json.dumps(dict(status=state.status(),receipt=receipt)))
    if not life.pending(): runtime.close()
    state.close()
'''


def run_child(root, mode):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith(
        ("SIRCHMUNK_", "LLM_", "PLATFORM_", "HF_", "EMBEDDING_"))}
    env.update(SIRCHMUNK_WORK_PATH=str(root), EMBEDDING_CACHE_DIR=str(root / ".cache/models"),
               HF_HOME=str(root / ".cache/models"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"), PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, "-B", "-c", CHILD, mode], env=env, cwd=root.parent,
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if mode in {"before-io", "after-publish"}:
        assert child.returncode == (27 if mode == "before-io" else 28), child.stderr
        return None
    assert child.returncode == 0, child.stderr
    return json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))


@pytest.mark.parametrize("stage", ["before-io", "after-publish"])
def test_real_process_death_leaves_accepted_and_restarts_only_read_original(stage, tmp_path):
    protect_test_directory(tmp_path)
    root = tmp_path / "work"
    run_child(root, stage)
    parent = root / ".control/backups"
    before = sorted(p.name for p in parent.iterdir()) if parent.exists() else []
    assert len(before) == (0 if stage == "before-io" else 1)
    with closing(sqlite3.connect(root / ".control/management.sqlite3")) as db:
        payload, receipt = db.execute("SELECT request_json,result_json FROM lifecycle_operations "
                                     "WHERE kind='backup-operation'").fetchone()
        check_receipt(json.loads(receipt), json.loads(payload), "accepted")
    observed = run_child(root, "inspect")
    assert observed["receipt"] == json.loads(receipt)
    assert all(observed["receipt"][key] is None for key in RESULT_FIELDS)
    assert observed["status"]["service_instance_id"] != observed["receipt"]["service_instance_id"]
    assert observed["status"]["active_jobs"] is None and observed["status"]["readiness"] == "unknown"
    assert observed["status"]["maintenance"] == {"enabled": True, "operation_id": "owner-0001"}
    assert not observed["status"]["accepting_jobs"]
    assert (sorted(p.name for p in parent.iterdir()) if parent.exists() else []) == before
    with closing(sqlite3.connect(root / ".control/management.sqlite3")) as db:
        assert db.execute("SELECT unclean FROM process_state").fetchone() == (1,)


def test_success_receipt_survives_clean_restart_and_restored_snapshot_stays_unknown(tmp_path):
    protect_test_directory(tmp_path)
    root = tmp_path / "work"
    first = run_child(root, "success")["receipt"]
    restored = tmp_path / "restored"
    shutil.copytree(root / ".control/backups" / first["backup_id"], restored)
    second = run_child(root, "inspect")
    assert second["receipt"] == first
    assert second["status"]["service_instance_id"] != first["service_instance_id"]
    assert second["status"]["readiness"] == "ok" and not second["status"]["accepting_jobs"]
    recovered = run_child(restored, "inspect")
    assert recovered["receipt"]["state"] == "accepted"
    assert recovered["status"]["readiness"] == "unknown"
    assert recovered["status"]["maintenance"] == {"enabled": True, "operation_id": "owner-0001"}
