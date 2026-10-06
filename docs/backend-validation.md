# Backend validation

Acceptance date: 2026-10-07. Base: upstream main
`b314e11fcea87cf8146b2844cf2f991208c021d9`.
Runtime/test checkpoint: `ebf2cd5f9a10866fe57f26159309f76fdbbffc09`.
Documentation-only changes follow that checkpoint. This report deliberately
excludes private provider addresses, credentials, machine paths and runtime data.

## Results

| Check | Result | Boundary |
| --- | --- | --- |
| Windows full Python suite | 457 passed, 7 deprecation warnings, 173.20s | Python 3.12.10, isolated roots/offline model caches |
| Ubuntu full Python suite | 456 passed, 1 skipped, 6 deprecation warnings, 102.84s | Python 3.12.3, Ubuntu 24.04 under WSL; only Windows junction behavior skipped |
| Focused pool/config/API/integration/socket tests | 68 passed, 1 existing warning, 50.20s | Stress count set to 1000; actual TCP and real OpenAI SDK parsing |
| Weighted socket stress | 1000 sequential calls: 750/250; 240 additional concurrent calls | Local protocol provider, not a paid-model throughput benchmark |
| Socket faults and cleanup | 503/429/attempt timeout fallback, stream truncation/no replay, shared-model admission, queued/active cancellation passed | Every scenario ends with zero local in-flight/waiting |
| Full API process acceptance (Windows and Ubuntu) | HTTP/SSE/WS routes, 3:1 30/10, saturated-primary backup, queue-full and wait timeout passed | Temporary loopback service/provider; native document-tool detection stubbed |
| Configuration restart | API clean exit/restart retains configuration; stale CAS rejected; independent writer-exit/reader-process check passed | Private temporary roots, no live configuration edits |
| Real preferred-model calls | Final run 7/7 successful; response model `gpt-6-luna`, 1.656–3.078s/call | One authorized gateway, not multiple independent providers |

No tests were deselected to obtain the full-suite results. Existing FastAPI
`on_event` and Windows Starlette/AnyIO deprecation warnings remain visible.
The full suites include real HTTP backup ACK-loss, accepted/publication process
death, preserved unknown locks, supervised clean stop/restart and old-UUID
rejection. They do not constitute deployment-platform acceptance.

## Real-provider details

Discovery reported 15 available model IDs; the exact requested `gpt-6-luna` was
available. The final bounded run issued two direct calls (non-stream/stream),
four calls through a 3:1 pool of two connection IDs, and one real backup call
after a synthetic primary connection failure. Actual connection starts were
3/1 in the weighted phase. All outputs were nonempty; response identity and
usage were returned. Final reported aggregate usage was 2266 tokens across
seven calls. Counts describe connection routing, not independent model servers.

An initial seven-call run also completed successfully but failed its Windows
temporary-directory cleanup because the harness retained its work-root lease.
That test-only cleanup was corrected, then the entire real run was repeated
with a successful process exit. Total real calls during this acceptance: 14.
The restart test initially attempted a second writer while the first still held
the root lease; the expected protective refusal was not a product regression.
The harness now exits the writer before starting the reader. Expected provider
disconnects also handle Windows `ConnectionAbortedError` without noisy traces.
No routing/runtime defect was found during this acceptance.

The key was entered through non-echoing private input, remained in test-process
memory/environment, and was not written to the pool JSON. The gateway required
a private TLS certificate. Only the explicitly opted-in test client used a
temporary leaf trust anchor; subsequent authenticated connections verified
against that anchor, with hostname matching disabled for this test only.
Product TLS verification was not relaxed. This is trust-on-first-use, not an
independently authenticated certificate-provisioning process.

## Reproduction

Install project/web/test dependencies in an isolated environment: `pytest`,
`pytest-asyncio`, `httpx`, `websockets` (15+ for the optional acceptance script),
OpenAI SDK, DuckDB/pandas and the project's other declared runtime dependencies.
Windows ACL tests additionally require `pywin32`; Ubuntu does not install it.
The lightweight test environment used substitutes for expensive model loading;
no real embedding download or paper-grade corpus benchmark is implied.

Before importing product modules, set `SIRCHMUNK_WORK_PATH`,
`EMBEDDING_CACHE_DIR` and `HF_HOME` to fresh private temporary paths; set
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1` and
`PYTHONPATH` to the checkout's `src`. On Windows also set
`PYTHONIOENCODING=utf-8`. Never point acceptance at a live knowledge base.

```sh
python -m pytest tests -q --tb=short -rs
```

For extended fixture stress, set `SIRCHMUNK_SOCKET_STRESS_CALLS=1000` and run:

```sh
python -m pytest tests/test_llm_pool_sockets.py tests/test_llm_pool_router.py tests/test_llm_pool_api.py tests/test_llm_pool_config.py tests/test_llm_pool_integration.py -q --tb=short
python tests/support/run_api_pool_acceptance.py
```

The default socket test uses 40 sequential calls plus 240 concurrent calls.
The extended count is explicit (multiple of four, 4–2000). Fault tests use
ephemeral loopback listeners and fixture model names; no external paid calls.
The API runner starts/stops its own single-worker child with private roots and
synthetic credentials. The lower-level `accept_llm_pools.py` additionally refuses
non-loopback or non-fixture configurations and requires `--allow-isolated-writes`.

Only with an authorized real endpoint/key, run the opt-in real-provider harness:

```sh
python tests/support/accept_real_llm.py --base https://provider.example/v1 --preferred-model your-supported-model-id
```

The script reads the key privately, never from an argv parameter. If explicitly
testing a private certificate, add `--allow-private-certificate`; do not use that
as a production certificate-provisioning policy. No real provider is invoked by
the ordinary pytest suite.

## Acceptance limits

- Admission is local to one service process. Multi-worker/container-wide quotas
  and remote inference cancellation require additional infrastructure.
- Local cancellation cannot prove the provider's GPU job stopped. The fixture
  waits for provider quiescence before checking a subsequent physical peak.
- Synthetic primary failures with a real backup test routing, not uptime of two
  independent providers. This is bounded acceptance, not a long-duration SLA.
- No new chip-manual retrieval-quality evaluation, real embedding/vector-model
  quality benchmark, frontend build/browser acceptance or graph-renderer test
  is claimed. UI source and assets are outside this contribution.
- Physical power-loss, network filesystems, signed release envelopes, systemd/
  SCM installation, Docker, multi-host platform deployment, production upgrade
  and rollback are not signed off by these isolated tests.
- Existing live services, private configurations and knowledge bases were not
  replaced or edited during acceptance. No data-schema migration was performed.
