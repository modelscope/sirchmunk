"""Opt-in bounded real-provider acceptance; never logs credentials or prompts.

Pass an OpenAI-compatible HTTPS base URL and a preferred model ID. The key is
read from non-echoing terminal input (or private stdin). Private/self-signed TLS
requires explicit opt-in and is trusted only in this test process. No product
TLS policy or existing service configuration is changed.
"""
import argparse
import asyncio
import getpass
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import sys
import tempfile
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx
import openai


def test_tls(base, allow_private):
    url = urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ValueError("An HTTPS API base without embedded credentials is required")
    system = ssl.create_default_context()
    try:
        with socket.create_connection((url.hostname, url.port or 443), timeout=10) as sock:
            with system.wrap_socket(sock, server_hostname=url.hostname):
                return system, {"mode": "system_trust"}
    except ssl.SSLCertVerificationError:
        if not allow_private:
            raise
    # Explicit test-only trust-on-first-use. Subsequent authenticated requests
    # verify against this exact certificate, not an unrestricted verify=False.
    probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    probe.check_hostname = False
    probe.verify_mode = ssl.CERT_NONE
    with socket.create_connection((url.hostname, url.port or 443), timeout=10) as sock:
        with probe.wrap_socket(sock, server_hostname=url.hostname) as tls:
            der = tls.getpeercert(binary_form=True)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(der))
    context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    # Private gateways often use a DNS certificate on their LAN IP. Only this
    # explicitly opted-in client disables hostname matching, retaining chain
    # verification against the ephemeral leaf trust anchor.
    context.check_hostname = False
    return context, {"mode": "test_only_leaf_trust", "sha256": hashlib.sha256(der).hexdigest()}


async def run(args, key, root):
    context, tls_report = test_tls(args.base, args.allow_private_certificate)
    async with httpx.AsyncClient(verify=context, trust_env=False, timeout=30) as http:
        response = await http.get(args.base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
        response.raise_for_status()
        ids = sorted(item["id"] for item in response.json()["data"])
    if args.preferred_model not in ids:
        print(json.dumps({"preferred_available": False, "models": ids}), flush=True)
        raise ValueError("Preferred model unavailable; no automatic paid-model substitution")

    from sirchmunk.llm.openai_chat import OpenAIChat
    from sirchmunk.llm.pool_config import ConfigStore
    from sirchmunk.llm.pool_router import RoutedChat, Scheduler

    os.environ["SIRCHMUNK_ACCEPTANCE_KEY"] = key
    store = ConfigStore(root)
    for name in ("primary", "backup"):
        store.change(store.read().revision, "model", {
            "name": name, "model": args.preferred_model, "base_url": args.base,
            "api_key_env": "SIRCHMUNK_ACCEPTANCE_KEY", "max_concurrency": 1,
            "timeout_seconds": 45,
        })
    config = store.read()
    members = [{"model_id": m.id, "weight": 3 if i == 0 else 1, "priority": i}
               for i, m in enumerate(config.models)]
    config = store.change(config.revision, "pool", {"name": "real-weighted", "strategy": "weighted",
        "members": members, "max_concurrency": 2, "request_timeout_seconds": 90})
    config = store.change(config.revision, "selection", {"kind": "pool", "id": config.pools[0].id})
    assert key not in store.path.read_text(encoding="utf-8")
    assert key not in json.dumps(config.public())
    records, emitted, scheduler = [], [], Scheduler()
    sync_sdk, async_sdk = openai.OpenAI, openai.AsyncOpenAI

    def sync_client(**kwargs):
        return sync_sdk(**kwargs, http_client=httpx.Client(verify=context, trust_env=False))

    def async_client(**kwargs):
        return async_sdk(**kwargs, http_client=httpx.AsyncClient(verify=context, trust_env=False))

    async def call(client, stream=False, label="pool"):
        start = time.monotonic()
        result = await client.achat([{"role": "user", "content": "Reply with exactly OK."}],
                                  stream=stream, max_tokens=128)
        assert result.content.strip(), "Provider returned no visible answer"
        assert result.model, "Provider did not report model identity"
        records.append({"kind": label, "stream": stream, "model": result.model,
                        "seconds": round(time.monotonic() - start, 3),
                        "usage": {k: v for k, v in result.usage.items() if k in
                                  ("prompt_tokens", "completion_tokens", "total_tokens")}})
        return result

    # Only test transport construction is customized; real SDK parsing, product
    # request construction and pool routing remain the code under acceptance.
    with patch("sirchmunk.llm.openai_chat.OpenAI", sync_client), patch("sirchmunk.llm.openai_chat.AsyncOpenAI", async_client):
        direct = OpenAIChat(api_key=key, base_url=args.base, model=args.preferred_model,
                            max_retries=0, log_callback=lambda level, text, end, flush: emitted.append(text))
        try:
            await call(direct, label="direct")
            await call(direct, stream=True, label="direct")
            assert emitted and any(emitted)
        finally:
            await direct._async_client.close()
            direct._client.close()
        for _ in range(4):
            await call(RoutedChat(config, scheduler, log_callback=lambda *args: None))
        started = [scheduler.models[m.id].started for m in config.models]
        assert started == [3, 1], started
        config = store.change(config.revision, "pool", {"strategy": "failover", "failure_cooldown_seconds": 0}, config.pools[0].id)
        attempts = []

        def controlled_primary_failure(**kwargs):
            attempts.append(kwargs["model"])
            if len(attempts) == 1:
                raise openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1/synthetic-failure"))
            return OpenAIChat(**kwargs)

        await call(RoutedChat(config, scheduler, client_factory=controlled_primary_failure, log_callback=lambda *args: None), label="synthetic_failure_real_backup")
        assert len(attempts) == 2
    status = scheduler.status(config)
    assert all(m["in_flight"] == 0 for m in status["models"])
    assert all(p["in_flight"] == p["waiting"] == 0 for p in status["pools"])
    print(json.dumps({"success": True, "available_model_count": len(ids),
        "preferred_model": args.preferred_model, "tls": tls_report, "real_calls": len(records),
        "calls": records, "weighted_connection_counts": started, "capacity_released": True,
        "boundaries": "One real gateway/model; primary failure is synthetic, backup is real. Not an independent-provider uptime or retrieval-quality test."}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--preferred-model", required=True)
    parser.add_argument("--allow-private-certificate", action="store_true")
    args = parser.parse_args()
    print("Ready for API key via private input", flush=True)
    key = getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
    if not key:
        raise SystemExit("Empty API key")
    with tempfile.TemporaryDirectory(prefix="sirchmunk-real-acceptance-") as folder:
        root = Path(folder)
        os.environ.update(SIRCHMUNK_WORK_PATH=str(root), EMBEDDING_CACHE_DIR=str(root / "embedding"),
                          HF_HOME=str(root / "hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        try:
            asyncio.run(run(args, key, root))
        except Exception as exc:
            # Exception messages/URLs/provider bodies may expose credentials.
            print(json.dumps({"success": False, "error_type": type(exc).__name__}), flush=True)
            raise SystemExit(1) from None
        finally:
            os.environ.pop("SIRCHMUNK_ACCEPTANCE_KEY", None)
            key = ""
            from sirchmunk.storage.lifecycle import claim_work_root
            # This standalone harness has no background storage workers. Close
            # its sole lease before Windows tries to remove the temporary root.
            claim_work_root(root).close()
