import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path))
    monkeypatch.setenv("SIRCHMUNK_API_TOKEN", "admin-test-token")
    from sirchmunk.api.llm_pools import router
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer admin-test-token"
        yield c


def test_auth_read_and_write(client):
    for method, path in [("GET", "/config"), ("POST", "/models"), ("DELETE", "/models/a")]:
        response = client.request(method, "/api/v1/llm" + path, headers={"Authorization": ""})
        assert response.status_code == 401


def test_api_crud_and_safe_validation(client):
    assert client.get("/api/v1/llm/config").headers["cache-control"] == "no-store"
    payload = {"revision": 0, "model": {"name": "test", "model": "test", "base_url": "http://localhost:9001/v1", "api_key": "super-private"}}
    response = client.post("/api/v1/llm/models", json=payload)
    assert response.status_code == 200
    assert "super-private" not in response.text
    data = response.json()["data"]
    item_id = data["models"][0]["id"]
    response = client.post("/api/v1/llm/models/" + item_id, json={"revision": 1, "model": {"timeout_seconds": "super-private"}})
    assert response.status_code == 422 and "super-private" not in response.text
    assert client.post("/api/v1/llm/models", json=payload).status_code == 409
    assert client.post("/api/v1/llm/selection", json={"revision": 1, "target": {"kind": "model", "id": item_id}}).status_code == 200
    assert client.delete("/api/v1/llm/models/" + item_id, params={"revision": 2}).status_code == 422
    assert client.request("DELETE", "/api/v1/llm/models/" + item_id, json={"revision": 2}).status_code == 409
    assert client.post("/api/v1/llm/selection", json={"revision": 2, "target": {"kind": "legacy"}}).status_code == 200
    assert client.request("DELETE", "/api/v1/llm/models/" + item_id, json={"revision": 3}).status_code == 200


@pytest.mark.parametrize("body", ['null', '[]', 'bad-json', '{"revision":0,"model":"super-private"}'])
def test_malformed_body_is_safe(client, body):
    response = client.post("/api/v1/llm/models", content=body)
    assert response.status_code == 422
    assert "super-private" not in response.text


def test_body_limit(client):
    assert client.post("/api/v1/llm/models", content=" " * 140000).status_code == 413
