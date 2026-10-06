"""Configuration contract tests use only synthetic, non-secret consumers."""

import hashlib
import json

import pytest
from fastapi.testclient import TestClient


class Consumer:
    def __init__(self, limit=100):
        self.limit = limit
        self.calls = 0
        self.fail = False

    def get_limit(self):
        return self.limit

    def set_limit(self, value):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic consumer failure")
        self.limit = value


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    from sirchmunk.api.management import ManagementHolder, ManagementState, _management_app

    state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    consumer = Consumer()
    state.enable_configuration(consumer)
    holder = ManagementHolder()
    holder.state = state
    try:
        with TestClient(_management_app(holder, "secret")) as client:
            yield state, consumer, client, tmp_path
    finally:
        state.close()


def mutation(state, operation_id, expected_revision, value):
    body = {
        "protocol_version": "1.0",
        "operation_id": operation_id,
        "instance_id": state.instance_id,
        "service_instance_id": state.service_instance_id,
        "config_schema_id": "sirchmunk.runtime.v1",
        "expected_revision": expected_revision,
        "changes": {"maxWebSocketConnections": value},
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    body["request_digest"] = hashlib.sha256(canonical).hexdigest()
    return body


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    from sirchmunk.api.management import ManagementHolder, ManagementState, _management_app

    state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    holder = ManagementHolder()
    holder.state = state
    try:
        with TestClient(_management_app(holder, "secret")) as client:
            headers = {"Authorization": "Bearer secret"}
            assert "configuration.v1" not in client.get(
                "/management/v1/capabilities", headers=headers
            ).json()["capabilities"]
            assert client.get("/management/v1/config", headers=headers).status_code == 404
    finally:
        state.close()


def test_cas_hot_readback_and_idempotent_receipt(configured):
    state, consumer, client, _ = configured
    headers = {"Authorization": "Bearer secret"}
    assert "configuration.v1" in client.get(
        "/management/v1/capabilities", headers=headers
    ).json()["capabilities"]
    document = client.get("/management/v1/config", headers=headers).json()
    assert document["stored"] == {"maxWebSocketConnections": None}
    assert document["effective"] == {"maxWebSocketConnections": 100}
    assert document["revision"] == 0
    body = mutation(state, "config-op-0001", 0, 4)
    validated = client.post(
        "/management/v1/config/validate", json=body, headers=headers
    ).json()
    assert validated["valid"] is True
    assert validated["revision"] == 0
    assert state.config_document()["revision"] == 0
    receipt = client.post(
        "/management/v1/config/operations", json=body, headers=headers
    ).json()
    assert receipt["status"] == "succeeded"
    assert receipt["effect_committed"] is True
    assert (receipt["before_revision"], receipt["after_revision"]) == (0, 1)
    assert consumer.get_limit() == 4
    assert state.config_document()["effective"] == {"maxWebSocketConnections": 4}
    assert state.config_document()["apply_state"] == "applied"
    calls = consumer.calls
    assert client.post(
        "/management/v1/config/operations", json=body, headers=headers
    ).json() == receipt
    assert consumer.calls == calls
    assert client.get(
        "/management/v1/config/operations/config-op-0001", headers=headers
    ).json() == receipt
    reused = mutation(state, "config-op-0001", 1, 9)
    assert client.post(
        "/management/v1/config/operations", json=reused, headers=headers
    ).status_code == 409

    # The deployment default is a real hot value, not a string sentinel.
    reset = mutation(state, "config-op-0002", 1, None)
    assert client.post(
        "/management/v1/config/operations", json=reset, headers=headers
    ).json()["status"] == "succeeded"
    assert state.config_document()["stored"] == {"maxWebSocketConnections": None}
    assert state.config_document()["effective"] == {"maxWebSocketConnections": 100}


def test_revision_conflict_and_strict_values(configured):
    state, consumer, client, _ = configured
    headers = {"Authorization": "Bearer secret"}
    good = mutation(state, "config-op-0001", 0, 5)
    assert client.post(
        "/management/v1/config/operations", json=good, headers=headers
    ).status_code == 200
    stale = mutation(state, "config-op-0002", 0, 6)
    response = client.post(
        "/management/v1/config/operations", json=stale, headers=headers
    )
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["status"] == "rejected"
    assert receipt["before_revision"] == 1
    assert receipt["after_revision"] is None
    assert receipt["effect_committed"] is False
    assert receipt["error_code"] == "configuration_revision_conflict"
    assert consumer.get_limit() == 5
    assert client.post(
        "/management/v1/config/validate", json=stale, headers=headers
    ).status_code == 409
    invalid = mutation(state, "config-op-0003", 1, True)
    assert client.post(
        "/management/v1/config/operations", json=invalid, headers=headers
    ).status_code == 422
    tampered = mutation(state, "config-op-0004", 1, 7)
    tampered["changes"]["maxWebSocketConnections"] = 8
    assert client.post(
        "/management/v1/config/operations", json=tampered, headers=headers
    ).status_code == 422
    assert state.config_document()["revision"] == 1


def test_unknown_receipt_not_replayed_and_restart_readback(configured):
    state, consumer, client, tmp_path = configured
    headers = {"Authorization": "Bearer secret"}
    consumer.fail = True
    body = mutation(state, "config-op-0001", 0, 7)
    receipt = client.post(
        "/management/v1/config/operations", json=body, headers=headers
    ).json()
    assert receipt["status"] == "unknown"
    assert receipt["effect_committed"] is None
    assert state.config_document()["stored"] == {"maxWebSocketConnections": 7}
    assert state.config_document()["effective"] == {"maxWebSocketConnections": 100}
    assert state.config_document()["apply_state"] == "unknown"
    consumer.fail = False
    calls = consumer.calls
    assert client.post(
        "/management/v1/config/operations", json=body, headers=headers
    ).json() == receipt
    assert consumer.calls == calls
    blocked = mutation(state, "config-op-0002", 1, 8)
    assert client.post(
        "/management/v1/config/validate", json=blocked, headers=headers
    ).status_code == 409
    assert client.post(
        "/management/v1/config/operations", json=blocked, headers=headers
    ).status_code == 409
    state.close()
    from sirchmunk.api.management import ManagementState

    next_state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    next_consumer = Consumer()
    try:
        next_state.enable_configuration(next_consumer)
        assert next_state.service_instance_id != receipt["service_instance_id"]
        assert next_state.config_document()["revision"] == 1
        assert next_state.config_document()["effective"] == {"maxWebSocketConnections": 7}
        assert next_state.get_configuration_operation("config-op-0001") == receipt
        from sirchmunk.api.management import ManagementError
        with pytest.raises(ManagementError) as locked:
            next_state.apply_configuration(mutation(next_state, "config-op-0002", 1, 8))
        assert locked.value.code == "configuration_unknown_locked"
    finally:
        next_state.close()


def test_real_connection_consumer_without_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    from sirchmunk.api.chat import ChatConnectionManager
    from sirchmunk.api.management import ManagementState

    class Socket:
        def __init__(self):
            self.accepted = False
            self.closed = None

        async def accept(self):
            self.accepted = True

        async def close(self, code, reason):
            self.closed = (code, reason)

    manager = ChatConnectionManager()
    state = ManagementState(tmp_path / "state", "sirch-test", "test-build")
    try:
        state.enable_configuration(manager)
        receipt = state.apply_configuration(mutation(state, "config-op-0001", 0, 1))
        assert receipt["status"] == "succeeded"
        assert manager.get_limit() == 1
    finally:
        state.close()
    first, second, third = Socket(), Socket(), Socket()

    async def exercise():
        assert await manager.connect(first)
        assert not await manager.connect(second)
        assert second.closed == (1013, "Server at capacity")
        manager.disconnect(first)
        assert await manager.connect(third)
        manager.disconnect(third)

    import asyncio
    asyncio.run(exercise())
