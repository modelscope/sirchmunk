import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from sirchmunk.llm.pool_config import ConfigError, ConfigStore


def add_model(store, name="A", **extra):
    return store.change(store.read().revision, "model", {
        "name": name, "base_url": "http://127.0.0.1:9001/v1", "model": "test-model",
        "api_key": "private-test-secret", **extra,
    })


def test_legacy_default_is_read_only(tmp_path):
    store = ConfigStore(tmp_path)
    assert store.read().active_target.kind == "legacy"
    assert not store.path.exists()


def test_secret_write_only_keep_replace_clear(tmp_path):
    store = ConfigStore(tmp_path)
    config = add_model(store)
    model_id = config.models[0].id
    assert "private-test-secret" not in json.dumps(config.public())
    assert "api_key" not in config.public()["models"][0]
    assert config.public()["models"][0]["has_api_key"]
    for update in ({"name": "B"}, {"api_key": ""}, {"api_key": "***"}, {"api_key": None}):
        config = store.change(config.revision, "model", update, model_id)
        assert store.read().models[0].key() == "private-test-secret"
    config = store.change(config.revision, "model", {"api_key": "replacement"}, model_id)
    assert store.read().models[0].key() == "replacement"
    store.change(config.revision, "model", {"clear_api_key": True}, model_id)
    assert store.read().models[0].key() == ""
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600


def test_env_reference_is_not_copied(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_POOL_KEY", "env-only-secret")
    store = ConfigStore(tmp_path)
    config = add_model(store, api_key="", api_key_env="TEST_POOL_KEY")
    assert config.models[0].key() == "env-only-secret"
    assert "env-only-secret" not in store.path.read_text()
    assert "env-only-secret" not in json.dumps(config.public())


def test_references_selection_and_deletion(tmp_path):
    store = ConfigStore(tmp_path)
    config = add_model(store)
    model_id = config.models[0].id
    config = store.change(config.revision, "pool", {"name": "P", "members": [{"model_id": model_id}]})
    pool_id = config.pools[0].id
    with pytest.raises(ConfigError, match="Remove model"):
        store.change(config.revision, "delete_model", item_id=model_id)
    config = store.change(config.revision, "selection", {"kind": "pool", "id": pool_id})
    for operation, payload, item_id in [("delete_pool", None, pool_id), ("model", {"enabled": False}, model_id),
                                      ("pool", {"members": []}, pool_id)]:
        with pytest.raises(ConfigError):
            store.change(config.revision, operation, payload, item_id)
    config = store.change(config.revision, "selection", {"kind": "legacy"})
    config = store.change(config.revision, "delete_pool", item_id=pool_id)
    config = store.change(config.revision, "delete_model", item_id=model_id)
    assert not config.models and not config.pools


@pytest.mark.parametrize("update", [
    {"base_url": "ftp://example.com"}, {"base_url": "https://user:secret@example.com"},
    {"base_url": "https://example.com?key=secret"}, {"max_concurrency": 0},
    {"max_concurrency": True}, {"max_concurrency": 1.5}, {"timeout_seconds": float("nan")},
    {"timeout_seconds": 0}, {"provider": "invalid"}, {"unexpected": "secret"},
])
def test_invalid_fields(tmp_path, update):
    with pytest.raises(ConfigError):
        add_model(ConfigStore(tmp_path), **update)


def test_pool_members_and_empty_selection(tmp_path):
    store = ConfigStore(tmp_path)
    config = add_model(store)
    model_id = config.models[0].id
    for members in ([{"model_id": "missing"}], [{"model_id": model_id}] * 2):
        with pytest.raises(ConfigError):
            store.change(config.revision, "pool", {"name": "bad", "members": members})
    config = store.change(config.revision, "pool", {"name": "draft"})
    with pytest.raises(ConfigError):
        store.change(config.revision, "selection", {"kind": "pool", "id": config.pools[0].id})


def test_atomic_failure_retains_previous_config(tmp_path, monkeypatch):
    store = ConfigStore(tmp_path)
    config = add_model(store)
    before = store.path.read_bytes()
    def fail(*args):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(ConfigError, match="Cannot save"):
        store.change(config.revision, "model", {"name": "changed"}, config.models[0].id)
    assert store.path.read_bytes() == before
    assert not list(tmp_path.glob(".llm-pools-*"))


def test_concurrent_revision_guard(tmp_path):
    store = ConfigStore(tmp_path)
    def write(_):
        try:
            store.change(0, "pool", {"name": "P"})
            return True
        except ConfigError as exc:
            assert exc.status == 409
            return False
    with ThreadPoolExecutor(8) as executor:
        assert sum(executor.map(write, range(8))) == 1
    assert store.read().revision == 1


def test_corrupt_config_fails_closed(tmp_path):
    store = ConfigStore(tmp_path)
    store.path.write_text('{"private-secret":', encoding="utf-8")
    with pytest.raises(ConfigError, match="Cannot read") as error:
        store.read()
    assert "private-secret" not in str(error.value)
