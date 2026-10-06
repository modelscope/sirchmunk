"""Cross-process ownership checks; all roots and caches are isolated."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sirchmunk.storage.lifecycle import (
    DATA_SCHEMA, LifecycleError, bootstrap_managed_root, claim_work_root,
    durable_replace, get_runtime,
)


def launch(root):
    return {
        "SIRCHMUNK_WORK_PATH": str(root),
        "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "isolated-root",
    }


def child(root, code, *arguments):
    env = os.environ.copy()
    for key in tuple(env):
        if key.upper().startswith("SIRCHMUNK_MANAGEMENT_"):
            env.pop(key)
    env.update(SIRCHMUNK_WORK_PATH=str(root), EMBEDDING_CACHE_DIR=str(root / ".cache" / "models"))
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.run(
        [sys.executable, "-c", code, *arguments], env=env, cwd=root.parent,
        capture_output=True, text=True, timeout=20,
    )


def test_exclusive_root_blocks_second_process(tmp_path):
    root = tmp_path / "work"
    lease = claim_work_root(root)
    result = child(root, "from sirchmunk.storage.lifecycle import claim_work_root; claim_work_root()")
    assert result.returncode != 0
    assert "work_root_already_in_use" in result.stderr
    assert not lease.file.closed


def test_managed_root_rejects_cli_mcp_and_sdk_even_between_restarts(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    runtime = bootstrap_managed_root(launch(root))
    assert get_runtime() is runtime
    marker = json.loads((root / ".control" / "data-schema.json").read_text())
    assert marker["schema_id"] == DATA_SCHEMA
    for code in (
        "from sirchmunk.storage.lifecycle import claim_work_root; claim_work_root()",
        "from sirchmunk.cli.cli import run_cmd; import sys; sys.argv=['sirchmunk','init']; run_cmd()",
        "from sirchmunk_mcp.config import Config; Config.from_env()",
        "from sirchmunk.llm.pool_config import ConfigStore; import os; ConfigStore(os.environ['SIRCHMUNK_WORK_PATH'])",
    ):
        result = child(root, code)
        assert result.returncode != 0
        assert "managed_work_root_requires_supervisor" in result.stderr
    assert not (root / ".env").exists()


def test_root_lock_released_by_process_exit_and_schema_identity_checked(tmp_path):
    root = tmp_path / "work"
    code = "from sirchmunk.storage.lifecycle import claim_work_root; claim_work_root(); print('owned')"
    assert child(root, code).returncode == 0
    assert child(root, code).returncode == 0
    runtime = bootstrap_managed_root(launch(root))
    assert runtime.active_background() == 0
    token = runtime.reserve_background()
    with pytest.raises(LifecycleError, match="not_drained"):
        runtime.flush()
    token()
    token()
    assert runtime.active_background() == 0


def test_schema_mismatch_does_not_overwrite_marker(tmp_path):
    root = tmp_path / "work"
    (root / ".control").mkdir(parents=True)
    marker = root / ".control" / "data-schema.json"
    data = '{"schema_id":"future-incompatible-schema"}'
    marker.write_text(data)
    with pytest.raises(LifecycleError, match="mismatch"):
        bootstrap_managed_root(launch(root))
    assert marker.read_text() == data


def test_durable_replace_failure_preserves_old_file(tmp_path, monkeypatch):
    old, candidate = tmp_path / "current.json", tmp_path / "candidate.json"
    old.write_text("old")
    candidate.write_text("new")
    def failed_sync(_fd):
        raise OSError("synthetic fsync failure")
    monkeypatch.setattr(os, "fsync", failed_sync)
    with pytest.raises(OSError, match="synthetic"):
        durable_replace(candidate, old)
    assert old.read_text() == "old"


def test_flush_rejects_linked_persistent_file(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    runtime = bootstrap_managed_root(launch(root))
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve")
    try:
        os.link(outside, root / "outside-hardlink.txt")
    except OSError:
        pytest.skip("filesystem does not support hard links")
    with pytest.raises(LifecycleError, match="hardlinked"):
        runtime.flush()
    assert outside.read_text() == "preserve"


def test_managed_schema_cannot_change_via_business_settings(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    bootstrap_managed_root(launch(root))
    # Import only after the explicit isolated root is claimed.
    for dependency in ("fastapi", "httpx", "numpy", "loguru"):
        pytest.importorskip(dependency, reason="Web API test dependencies required")
    from sirchmunk.api.settings import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        for key in ("EMBEDDING_DIM", "EMBEDDING_MODEL_ID", "EMBEDDING_CACHE_DIR", "SIRCHMUNK_WORK_PATH"):
            response = client.post("/api/v1/settings", json={
                "ui": {"theme": "dark", "language": "zh"},
                "environment": {key: "unsafe"},
            })
            assert response.status_code == 403
    assert not (root / ".env").exists()


def test_flush_syncs_all_new_directory_ancestors_and_propagates_failure(tmp_path, monkeypatch):
    import sirchmunk.storage.lifecycle as module
    root = tmp_path / "new" / "work"
    runtime = bootstrap_managed_root(launch(root))
    target = root / ".cache" / "deep" / "store" / "data"
    target.parent.mkdir(parents=True)
    target.write_text("isolated")
    visited = []
    monkeypatch.setattr(module, "durable_directory", visited.append)
    runtime.flush()
    assert all(parent in visited for parent in target.parents)
    assert visited.index(target.parent) < visited.index(root / ".cache") < visited.index(root)
    def fail_middle(path):
        if path == root / ".cache":
            raise OSError("synthetic intermediate directory fsync failure")
    monkeypatch.setattr(module, "durable_directory", fail_middle)
    with pytest.raises(OSError, match="intermediate"):
        runtime.flush()
