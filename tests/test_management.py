"""Isolated tests for the optional Web API management boundary."""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def management(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    from sirchmunk.api.management import ManagementState

    state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    try:
        yield state
    finally:
        state.close()


def test_maintenance_owner_and_restart_persistence(management, tmp_path):
    from sirchmunk.api.management import ManagementError, ManagementState
    from sirchmunk.version import __version__

    assert management.status()["platform_type"] == "sirchmunk"
    assert management.status()["running_version"] == __version__
    assert management.status()["accepting_jobs"] is True
    first_id = management.service_instance_id
    assert management.admit()
    assert management.status()["active_jobs"] == 1
    assert management.set_maintenance(True, "owner-0001") == {
        "enabled": True, "operation_id": "owner-0001"
    }
    assert not management.admit()
    assert management.status()["active_jobs"] == 1
    management.release()
    assert management.status()["active_jobs"] == 0
    assert management.set_maintenance(True, "owner-0001")["enabled"]
    with pytest.raises(ManagementError) as conflict:
        management.set_maintenance(False, "other-0001")
    assert conflict.value.status == 409
    with pytest.raises(ManagementError) as second_worker:
        ManagementState(tmp_path / "state", "sirch-test", "test-build")
    assert second_worker.value.code == "managed_state_already_in_use"
    management.close()
    restarted = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    try:
        assert restarted.service_instance_id != first_id
        assert restarted.status()["maintenance"] == {
            "enabled": True, "operation_id": "owner-0001"
        }
        assert not restarted.admit()
        assert restarted.set_maintenance(False, "owner-0001") == {
            "enabled": False, "operation_id": None
        }
        assert restarted.admit()
        restarted.release()
    finally:
        restarted.close()


def test_unclean_active_work_fails_closed(management, tmp_path):
    from sirchmunk.api.management import ManagementError, ManagementState

    assert management.admit()
    management.set_maintenance(True, "owner-0001")
    management.close()  # Simulate process death with an unfinished reservation.
    restarted = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    try:
        status = restarted.status()
        assert status["active_jobs"] is None
        assert status["accepting_jobs"] is False
        assert status["reason"] == "unclean_previous_process"
        assert not restarted.admit()
        with pytest.raises(ManagementError) as cannot_leave:
            restarted.set_maintenance(False, "owner-0001")
        assert cannot_leave.value.code == "unfinished_work_unknown"
    finally:
        restarted.close()


def test_admission_and_maintenance_share_linearization(management):
    start = threading.Barrier(3)
    admitted = []

    def reserve():
        start.wait()
        admitted.append(management.admit())

    def close_gate():
        start.wait()
        management.set_maintenance(True, "owner-0001")

    workers = [threading.Thread(target=reserve), threading.Thread(target=close_gate)]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive()
    assert management.status()["maintenance"]["enabled"] is True
    assert management.status()["active_jobs"] == int(admitted[0])
    assert not management.admit()
    if admitted[0]:
        management.release()


def test_http_and_websocket_admission(management):
    from sirchmunk.api.management import AdmissionMiddleware, ManagementHolder

    holder = ManagementHolder()
    holder.state = management
    app = FastAPI()
    app.add_middleware(AdmissionMiddleware, holder=holder)

    @app.get("/api/v1/work")
    async def work():
        return {"active": management.status()["active_jobs"]}

    @app.websocket("/api/v1/chat")
    async def chat(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"active": management.status()["active_jobs"]})
        await ws.close()

    with TestClient(app) as client:
        assert client.get("/api/v1/work").json() == {"active": 1}
        assert management.status()["active_jobs"] == 0
        with client.websocket_connect("/api/v1/chat") as ws:
            assert ws.receive_json() == {"active": 1}
        assert management.status()["active_jobs"] == 0
        management.set_maintenance(True, "owner-0001")
        response = client.get("/api/v1/work")
        assert response.status_code == 503
        assert response.json() == {"error": {"code": "maintenance_active"}}
        with client.websocket_connect("/api/v1/chat") as ws:
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
            assert closed.value.code == 1013
        assert management.status()["active_jobs"] == 0


def test_auth_and_strict_maintenance_body(management):
    from sirchmunk.api.management import ManagementHolder, _management_app

    holder = ManagementHolder()
    holder.state = management
    with TestClient(_management_app(holder, "secret")) as client:
        assert client.get("/management/v1/status").status_code == 401
        headers = {"Authorization": "Bearer secret"}
        status = client.get("/management/v1/status", headers=headers).json()
        assert status["instance_id"] == "sirch-test"
        assert status["service_instance_id"] == management.service_instance_id
        assert "flush.v1" not in client.get("/management/v1/capabilities", headers=headers).json()["capabilities"]
        duplicate = client.post(
            "/management/v1/maintenance", headers=headers,
            content='{"enabled":true,"enabled":false,"operation_id":"owner-0001"}',
        )
        assert duplicate.status_code == 400
        assert management.status()["maintenance"]["enabled"] is False
        assert client.post(
            "/management/v1/maintenance", headers=headers,
            json={"enabled": True, "operation_id": "owner-0001"},
        ).json() == {"enabled": True, "operation_id": "owner-0001"}


def test_literal_loopback_listener(management, tmp_path, monkeypatch):
    from sirchmunk.api.management import ManagementHolder, start_management, stop_management

    management.close()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_ENABLED", "true")
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_INSTANCE_ID", "sirch-test")
    monkeypatch.setenv("SIRCHMUNK_DEPLOYMENT_ID", "test-build")
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_TOKEN", "x" * 40)
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_PORT", str(port))
    holder = ManagementHolder()
    work = tmp_path / "work"
    async def exercise():
        await start_management(holder, work, {
            "SIRCHMUNK_WORK_PATH": str(work),
            "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
            "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "sirch-test",
            "SIRCHMUNK_DEPLOYMENT_ID": "test-build",
            "SIRCHMUNK_MANAGEMENT_TOKEN": "x" * 40,
            "SIRCHMUNK_MANAGEMENT_PORT": str(port),
        })
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/management/v1/status",
                    headers={"Authorization": "Bearer " + "x" * 40},
                )
                assert response.status_code == 200
                assert response.json()["platform_type"] == "sirchmunk"
        finally:
            await stop_management(holder)
    asyncio.run(exercise())


def test_launch_snapshot_does_not_trust_business_env(management, tmp_path, monkeypatch):
    from sirchmunk.api.management import ManagementHolder, start_management

    management.close()
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_ENABLED", "true")
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_TOKEN", "business-controlled-token-value-should-not-be-trusted")
    holder = ManagementHolder()
    asyncio.run(start_management(holder, tmp_path / "work", {
        "SIRCHMUNK_MANAGEMENT_ENABLED": "false",
    }))
    assert holder.state is None
    with pytest.raises(RuntimeError, match="supervisor"):
        asyncio.run(start_management(holder, tmp_path / "work", {
            "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
            "SIRCHMUNK_MANAGEMENT_TOKEN": "business-controlled-token-value-should-not-be-trusted",
        }))


def test_business_settings_cannot_change_management_identity_or_managed_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    from sirchmunk.api.management import ManagementHolder, ManagementState
    from sirchmunk.api.settings import router

    holder = ManagementHolder()
    holder.state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    app = FastAPI()
    app.state.management_holder = holder
    app.include_router(router)
    try:
        with TestClient(app) as client:
            for key in (
                "SIRCHMUNK_MANAGEMENT_TOKEN", "SIRCHMUNK_MANAGEMENT_ENABLED",
                "SIRCHMUNK_DEPLOYMENT_ID", "SIRCHMUNK_WORK_PATH",
                "SIRCHMUNK_MAX_WS_CONNECTIONS",
            ):
                response = client.post("/api/v1/settings", json={
                    "ui": {"theme": "dark", "language": "zh"},
                    "environment": {key: "unsafe"},
                })
                assert response.status_code == 403, key
            assert not (tmp_path / "work" / ".env").exists()
    finally:
        holder.state.close()


def test_business_settings_auth_and_env_line_injection(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    monkeypatch.setenv("SIRCHMUNK_API_TOKEN", "synthetic-business-token")
    from sirchmunk.api.main import app

    with TestClient(app) as client:
        assert client.get("/api/v1/settings/ui").status_code == 200
        assert client.post(
            "/api/v1/settings/ui", json={"theme": "dark", "language": "zh"}
        ).status_code == 401
        auth = {"Authorization": "Bearer synthetic-business-token"}
        assert client.post(
            "/api/v1/settings/ui", headers=auth,
            json={"theme": "light\nSIRCHMUNK_MANAGEMENT_ENABLED=true", "language": "en"},
        ).status_code == 422
        assert client.post(
            "/api/v1/settings", headers=auth,
            json={"environment": {"UI_THEME": "light\nSIRCHMUNK_MANAGEMENT_ENABLED=true"}},
        ).status_code == 422
        assert client.post(
            "/api/v1/settings", headers=auth,
            json={"environment": {"UI_THEME\nSIRCHMUNK_MANAGEMENT_ENABLED": "true"}},
        ).status_code == 422
        assert client.post(
            "/api/v1/settings", headers=auth,
            json={"environment": {"SIRCHMUNK_MANAGEMENT_TOKEN": "synthetic-stolen-token"}},
        ).status_code == 403
        assert not (tmp_path / "work" / ".env").exists()
        assert client.post(
            "/api/v1/settings/ui", headers=auth,
            json={"theme": "dark", "language": "zh"},
        ).status_code == 200


def test_cli_dotenv_cannot_interpolate_supervisor_management_token(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path))
    token = "synthetic-supervisor-management-token-1234567890"
    monkeypatch.setenv("SIRCHMUNK_MANAGEMENT_TOKEN", token)
    monkeypatch.delenv("SIRCHMUNK_MANAGEMENT_ENABLED", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL_NAME=${SIRCHMUNK_MANAGEMENT_TOKEN}\n"
        "SIRCHMUNK_MANAGEMENT_ENABLED=true\n", encoding="utf-8",
    )
    from sirchmunk.cli.cli import _load_env_file

    try:
        assert _load_env_file(env_file)
        assert os.getenv("LLM_MODEL_NAME") != token
        assert os.getenv("SIRCHMUNK_MANAGEMENT_TOKEN") == token
        assert os.getenv("SIRCHMUNK_MANAGEMENT_ENABLED") is None
    finally:
        os.environ.pop("LLM_MODEL_NAME", None)


def test_business_dotenv_cannot_enable_management_or_read_supervisor_token(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    token = "synthetic-supervisor-management-token-1234567890"
    (work / ".env").write_text(
        "SIRCHMUNK_MANAGEMENT_ENABLED=true\n"
        "SIRCHMUNK_MANAGEMENT_PORT=43210\n"
        "LLM_MODEL_NAME=${SIRCHMUNK_MANAGEMENT_TOKEN}\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    for key in tuple(env):
        if key.upper().startswith("SIRCHMUNK_MANAGEMENT_") or key.upper() == "SIRCHMUNK_DEPLOYMENT_ID":
            env.pop(key)
    env["SIRCHMUNK_WORK_PATH"] = str(work)
    env["SIRCHMUNK_MANAGEMENT_TOKEN"] = token
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = os.pathsep.join((str(repo_src), env.get("PYTHONPATH", "")))
    code = (
        "import json, os; from sirchmunk.api import main; "
        "print('SNAPSHOT=' + json.dumps({"
        "'enabled': main._management_launch_env['SIRCHMUNK_MANAGEMENT_ENABLED'], "
        "'token_removed': os.getenv('SIRCHMUNK_MANAGEMENT_TOKEN') is None, "
        "'model_name': os.getenv('LLM_MODEL_NAME')}))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code), cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30, check=True,
    )
    snapshot = json.loads(next(line.removeprefix("SNAPSHOT=") for line in result.stdout.splitlines()
                               if line.startswith("SNAPSHOT=")))
    assert snapshot["enabled"] is None
    assert snapshot["token_removed"] is True
    assert snapshot["model_name"] != token
