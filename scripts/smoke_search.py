"""Bounded SSE acceptance client. Writes results only to an explicit local file.

Works on Windows and Ubuntu; no browser automation or private config changes.
SIRCHMUNK_API_TOKEN, if required, is read from the caller's environment only.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx


async def main(args):
    headers = {}
    if os.getenv("SIRCHMUNK_API_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["SIRCHMUNK_API_TOKEN"]
    request = dict(query=args.query, paths=args.paths, mode=args.mode,
                   max_token_budget=args.token_budget, max_loops=args.max_loops,
                   response_format="rich", llm_fallback=False)
    start = time.monotonic()
    result = None
    async with asyncio.timeout(args.deadline):
        async with httpx.AsyncClient(timeout=30, headers=headers, trust_env=False) as client:
            async with client.stream("POST", args.url.rstrip("/") + "/api/v1/search/stream", json=request) as response:
                response.raise_for_status()
                event = None
                async for line in response.aiter_lines():
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:") and event in {"result", "error"}:
                        result = json.loads(line[5:].strip())
                        if event == "error":
                            raise RuntimeError("Search endpoint returned an error; inspect server's local log")
    if result is None:
        raise RuntimeError("Stream ended without a result")
    record = dict(request=request, elapsed_seconds=round(time.monotonic() - start, 2), result=result)
    if args.output:
        output = Path(args.output).resolve()
        with output.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
    data = result.get("data", {})
    print(json.dumps({"elapsed_seconds": record["elapsed_seconds"], "success": result.get("success"),
                      "answer": str(data.get("answer", data.get("summary", data.get("content", data))))[:1800]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--mode", choices=["FAST", "DEEP"], default="DEEP")
    parser.add_argument("--token-budget", type=int, default=24000)
    parser.add_argument("--max-loops", type=int, default=4)
    parser.add_argument("--deadline", type=float, default=180)
    parser.add_argument("--output")
    asyncio.run(main(parser.parse_args()))
