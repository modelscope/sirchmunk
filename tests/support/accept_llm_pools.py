"""Opt-in network acceptance against an isolated, fixture-only work directory.

No source imports: use this to validate an installed wheel through real sockets.
Requires llm_pool_provider.py, fake legacy model fixture-legacy and explicit opt-in.
"""
import argparse
import asyncio
from collections import Counter
import json
from urllib.parse import urlsplit
import uuid

import httpx
import websockets


def loopback(url):
    p = urlsplit(url)
    return p.scheme == "http" and p.hostname in ("127.0.0.1", "localhost") and not p.username and not p.password


async def run(args):
    assert args.allow_isolated_writes and loopback(args.base) and loopback(args.provider)
    report = {}
    async with httpx.AsyncClient(base_url=args.base, timeout=30, trust_env=False) as client:
        async def config():
            r = await client.get("/api/v1/llm/config")
            r.raise_for_status()
            return r.json()["data"]

        original = await config()
        assert original["legacy"]["model"] == "fixture-legacy", "Refusing a non-fixture service"
        assert all(loopback(m["base_url"]) and m["model"].startswith("fixture-") for m in original["models"])
        tag = "fixture-" + uuid.uuid4().hex[:8]
        created_models, created_pools = [], []

        async def change(path, field, value, method="POST"):
            current = await config()
            payload = {"revision": current["revision"]}
            if field:
                payload[field] = value
            r = await client.request(method, "/api/v1/llm/" + path, json=payload)
            r.raise_for_status()
            return r.json()["data"]

        async def model(suffix):
            name = tag + suffix
            c = await change("models", "model", {"name": name, "model": name,
                "base_url": args.provider + "/v1", "max_concurrency": 1})
            item = next(m for m in c["models"] if m["name"] == name)
            created_models.append(item["id"])
            return item

        async def pool(suffix, members, **kwargs):
            c = await change("pools", "pool", {"name": tag + suffix, "members": members,
                "failure_cooldown_seconds": 0, **kwargs})
            item = next(p for p in c["pools"] if p["name"] == tag + suffix)
            created_pools.append(item["id"])
            return item

        async def probe(p):
            c = await config()
            r = await client.post("/api/v1/llm/test", json={"revision": c["revision"],
                "target": {"kind": "pool", "id": p["id"]}})
            r.raise_for_status()
            return r.json()

        async def stats():
            r = await client.get(args.provider + "/stats")
            r.raise_for_status()
            return r.json()

        async def until_active(name, expected=1):
            async with asyncio.timeout(5):
                while (await stats())["active"].get(name, 0) != expected:
                    await asyncio.sleep(.01)

        try:
            a, b = await model("-a"), await model("-b")
            balanced = await pool("-weighted", [{"model_id": a["id"], "weight": 3},
                {"model_id": b["id"], "weight": 1}], strategy="weighted", max_concurrency=2)
            count = Counter()
            for _ in range(40):
                result = await probe(balanced)
                assert result["success"], result
                count[result["model"]] += 1
            assert count == {a["model"]: 30, b["model"]: 10}, count
            report["weighted_40"] = {"primary": 30, "secondary": 10}

            failed = await model("-fail")
            standby = await pool("-failover", [{"model_id": failed["id"], "priority": 0},
                {"model_id": b["id"], "priority": 1}], strategy="failover")
            result = await probe(standby)
            assert result["success"] and result["model"] == b["model"], result
            assert (await stats())["calls"][failed["model"]] == 1
            report["http_503_failover"] = True

            slow_a, slow_b = await model("-slow-a"), await model("-slow-b")
            bounded = await pool("-bounded", [{"model_id": slow_a["id"], "priority": 0},
                {"model_id": slow_b["id"], "priority": 1}], strategy="failover",
                max_concurrency=2, max_queue_size=0)
            first = asyncio.create_task(probe(bounded))
            await until_active(slow_a["model"])
            second = asyncio.create_task(probe(bounded))
            await until_active(slow_b["model"])
            full = await probe(bounded)
            assert not full["success"] and full["message"] == "queue_full", full
            results = await asyncio.gather(first, second)
            assert all(r["success"] for r in results)
            assert [r["model"] for r in results] == [slow_a["model"], slow_b["model"]]
            peaks = (await stats())["peaks"]
            assert peaks[slow_a["model"]] == peaks[slow_b["model"]] == 1
            report["primary_full_backup_and_capacity"] = {"model_peaks": [1, 1], "pool_max": 2, "third": "queue_full"}

            waiting = await pool("-waiting", [{"model_id": slow_a["id"]}],
                strategy="failover", max_concurrency=1, max_queue_size=1, queue_timeout_seconds=.1)
            first = asyncio.create_task(probe(waiting))
            await until_active(slow_a["model"])
            expired = await probe(waiting)
            assert not expired["success"] and expired["message"] == "capacity_wait_timeout", expired
            assert (await first)["success"]
            report["queue_timeout"] = True

            await change("selection", "target", {"kind": "pool", "id": standby["id"]})
            before = (await stats())["calls"].get(b["model"], 0)
            r = await client.post("/api/v1/search", json={"query": "hello", "mode": "FAST", "response_format": "minimal"})
            r.raise_for_status()
            assert r.json()["success"] and "OK" in json.dumps(r.json())
            after_http = (await stats())["calls"][b["model"]]
            assert after_http - before == 1
            r = await client.post("/api/v1/search/stream", json={"query": "hello", "mode": "FAST", "response_format": "minimal"})
            r.raise_for_status()
            assert "event: result" in r.text and "event: error" not in r.text and "OK" in r.text
            after_sse = (await stats())["calls"][b["model"]]
            assert after_sse - after_http == 1
            async with websockets.connect(args.base.replace("http:", "ws:") + "/api/v1/chat", proxy=None) as ws:
                await ws.send(json.dumps({"message": "Reply with OK", "enable_rag": False, "enable_web_search": False}))
                async with asyncio.timeout(15):
                    while True:
                        event = json.loads(await ws.recv())
                        assert event["type"] != "error", event
                        if event["type"] == "result":
                            assert b["model"] in event["content"], event
                            break
            # Existing WebSocket history filter sees the just-saved user message;
            # one relevance-filter call plus one response call must both route.
            assert (await stats())["calls"][b["model"]] - after_sse == 2
            report["http_sse_ws_selected_pool"] = {"http_calls": 1, "sse_calls": 1, "ws_calls": 2}
            report["retrieval_quality"] = "not tested; conversational routing and protocol only"
            status = (await client.get("/api/v1/llm/status")).json()["data"]
            assert all(m["in_flight"] == 0 for m in status["models"])
            assert all(p["in_flight"] == 0 and p["waiting"] == 0 for p in status["pools"])
            report["all_capacity_released"] = True
        finally:
            await change("selection", "target", original["active_target"])
            for item_id in reversed(created_pools):
                await change("pools/" + item_id, None, None, "DELETE")
            for item_id in reversed(created_models):
                await change("models/" + item_id, None, None, "DELETE")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--allow-isolated-writes", action="store_true")
    asyncio.run(run(parser.parse_args()))
