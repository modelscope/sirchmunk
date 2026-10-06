"""Launch a temporary API/provider pair and exercise HTTP, SSE and WebSocket.

Requires the project web/test dependencies. No production port/root is used.
Native document-tool detection is stubbed in the child: this is conversational
routing acceptance, not a PDF extraction or knowledge retrieval benchmark.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import httpx


CHILD = r'''
import sys, threading
import sirchmunk.search as core
core.check_dependencies = lambda: True
import uvicorn
server = uvicorn.Server(uvicorn.Config("sirchmunk.api.main:app", host="127.0.0.1", port=int(sys.argv[1]), log_level="error"))
def stop():
    sys.stdin.readline()
    server.should_exit = True
threading.Thread(target=stop, daemon=True).start()
server.run()
'''


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop(child):
    if child.poll() is None:
        child.stdin.write("stop\n")
        child.stdin.flush()
        child.wait(timeout=30)
    assert child.returncode == 0, "Isolated API did not exit cleanly"


def run():
    fixture = load("llm_pool_provider")
    server = fixture.ThreadingHTTPServer(("127.0.0.1", 0), fixture.Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    provider = f"http://127.0.0.1:{server.server_port}"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix="sirchmunk-api-acceptance-") as folder:
            root = Path(folder)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("SIRCHMUNK_", "PLATFORM_", "LLM_"))}
            env.update(SIRCHMUNK_WORK_PATH=str(root), LLM_API_KEY="fixture-only", LLM_MODEL_NAME="fixture-legacy",
                       LLM_BASE_URL=provider + "/v1", EMBEDDING_CACHE_DIR=str(root / "models"),
                       HF_HOME=str(root / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", GREP_FALLBACK_TO_RG="true",
                       SIRCHMUNK_ENABLE_CLUSTER_REUSE="false")

            def start():
                process = subprocess.Popen([sys.executable, "-c", CHILD, str(port)], env=env,
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
                deadline = time.monotonic() + 30
                with httpx.Client(trust_env=False, timeout=1) as client:
                    while time.monotonic() < deadline and process.poll() is None:
                        try:
                            if client.get(base + "/health").status_code == 200:
                                return process
                        except httpx.TransportError:
                            pass
                        time.sleep(.05)
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError("Isolated API startup failed")

            child = start()
            try:
                asyncio.run(load("accept_llm_pools").run(SimpleNamespace(
                    base=base, provider=provider, allow_isolated_writes=True)))
                with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
                    original = client.get("/api/v1/llm/config").json()["data"]
                    changed = client.post("/api/v1/llm/models", json={"revision": original["revision"],
                        "model": {"name": "fixture-persistent", "model": "fixture-persistent", "base_url": provider + "/v1"}})
                    changed.raise_for_status()
                    persisted = changed.json()["data"]
                stop(child)
                child = start()
                with httpx.Client(base_url=base, trust_env=False, timeout=5) as client:
                    assert client.get("/api/v1/llm/config").json()["data"] == persisted
                    stale = client.post("/api/v1/llm/selection", json={"revision": original["revision"], "target": {"kind": "legacy"}})
                    assert stale.status_code == 409
                print(json.dumps({"api_clean_restart_persistence": True, "stale_revision_rejected": True}), flush=True)
            finally:
                stop(child)
                child = None
    finally:
        if child is not None and child.poll() is None:
            stop(child)
        server.shutdown()
        thread.join(5)
        server.server_close()


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
