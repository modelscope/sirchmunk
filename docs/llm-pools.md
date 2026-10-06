# LLM connections and pools

The Web/API service can retain multiple OpenAI-compatible connections and route
calls through a selected connection or pool. No frontend changes are required
to use these APIs. Creating an item does **not** select it. The default target
remains `legacy`, using `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL_NAME`.
Explicit SDK clients, standalone CLI and MCP processes are not automatically
redirected to Web pools and do not share the Web process's capacity.

## Admission and failure semantics

- `weighted`: smooth weighted round robin over currently enabled, available
  members. Weights describe **call counts**, not tokens or processing time.
  A healthy sequential 3:1 pool selects its members 30/10 over 40 calls.
  Capacity, cooldown or failures can change the observed ratio.
- `failover`: try the eligible member with the lowest `priority` first; ties
  retain member order. A saturated primary uses an available backup. Recovering
  primaries become eligible again when their cooldown ends.
- Connection `max_concurrency` is shared across pools and direct selection
  using that connection ID. Pool `max_concurrency` additionally bounds that
  pool's attempts. Aliases with different connection IDs are separate quotas,
  even when they point to the same remote model/server.
- Admission is **single-process**, across event loops/threads. Use one Web
  worker. Multi-worker/container-wide quotas require an external coordinator.
- `max_queue_size` bounds waiting calls. Queue timeout and total request timeout
  are finite. Lowered connection/pool concurrency limits constrain new
  admissions from older request snapshots, without aborting admitted calls.
- Each candidate is attempted at most once per call; nested SDK retries are
  disabled. Connection errors, timeouts, HTTP 401/403/404/408/429 and 5xx can
  trigger fallback. HTTP 400/422 and other non-retryable failures do not fan out.
  There is no implicit fallback to the legacy model.
- After visible streaming output, interruption returns
  `stream_interrupted_no_replay`: no alternate-model replay, including through
  another factory within the same request. Before output, retryable failures
  may use a backup. Capacity is released in cleanup on success/failure/cancel.
- Client cancellation releases local network-call admission; the remote
  provider may continue inference. This is **not** a remote GPU-job quota.
- Token budgets charge unknown failed-attempt usage conservatively at the
  reservation ceiling. Success includes actual response model and usage.

## Configuration and security

Private `<work>/llm-pools.json` has schema version 1 and a monotonically
increasing `revision`. Writes are serialized and atomically replaced after file
fsync; lifecycle-managed writes use the durable root transaction boundary.
This is not a distributed configuration store or encryption-at-rest facility.
Protect the work volume with deployment-account ACLs/modes and do not publish it.

Set `SIRCHMUNK_API_TOKEN` before exposing the business API. Every route below
uses that existing Bearer authentication boundary; leaving the token unset
retains the project's unauthenticated local-development behavior. Management
tokens are separate and cannot authorize business APIs.

Credential fields are write-only. Public configuration excludes `api_key` and
returns `has_api_key`; validation/probe failures do not return provider bodies
or supplied secrets. Prefer `api_key_env` naming a process environment variable
to keep provider keys outside the JSON file. An omitted, empty or `"***"`
`api_key` preserves an existing key. `clear_api_key: true` clears both the stored
key and environment reference. A newly supplied literal key replaces an
environment reference unless an explicit `api_key_env` is also provided.
Ordinary TLS verification remains enabled in product clients.

Each Web request captures one configuration/selection snapshot. Updates affect
new requests; callbacks, searchers and transport cleanup are request-local.
Existing background evolution retains its resources until completion, then
closes them. Active/referenced items cannot be deleted; switch selection and
remove pool references first. Empty/disabled selections are rejected.

## HTTP API

All paths have prefix `/api/v1/llm`. GET success responses are
`{"success": true, "data": ...}` with `Cache-Control: no-store`. Every mutation
requires the latest integer `revision`; stale writes return HTTP 409. Refresh
configuration after each successful mutation. DELETE accepts a JSON body,
not a query-string revision. Request bodies are limited to 128 KiB.

| Method/path | JSON body / result |
| --- | --- |
| `GET /config` | Public models, pools, active target, revision and legacy status |
| `POST /models` | `{"revision": R, "model": {...}}`; server assigns ID |
| `POST /models/{id}` | Same wrapper; partial update, immutable ID |
| `DELETE /models/{id}` | `{"revision": R}`; rejects active/referenced item |
| `POST /pools` | `{"revision": R, "pool": {...}}`; server assigns ID |
| `POST /pools/{id}` | Same wrapper; partial update |
| `DELETE /pools/{id}` | `{"revision": R}`; rejects active pool |
| `POST /selection` | `{"revision": R, "target": {"kind": "legacy"}}`, or kind `model`/`pool` plus `id` |
| `GET /status` | Single-process model/pool in-flight, waiting, success/failure/cancel counts, cooldown and selection |
| `POST /test` | `{"revision": R, "target": {...}}`; optional target, temporary probe, does not change selection |

Probe results use `success`, `status`, `message` and `model`; an upstream failure
can be an HTTP-200 response with `success: false`. The explicit probe sends a
small prompt, limits output to 128 tokens and waits at most 20 seconds. It is a
real billable model call, not a read-only health check.

Example model object (replace host/model/env name with your deployment values):

```json
{
  "name": "Primary connection",
  "base_url": "https://provider.example/v1",
  "model": "your-supported-model-id",
  "api_key_env": "PRIMARY_LLM_KEY",
  "max_concurrency": 4,
  "timeout_seconds": 120
}
```

`provider` optionally selects an existing request-parameter profile; an unknown
profile is rejected. It does not add a new provider protocol.

Example pool object after obtaining connection IDs:

```json
{
  "name": "Balanced pool",
  "strategy": "weighted",
  "max_concurrency": 8,
  "max_queue_size": 64,
  "queue_timeout_seconds": 15,
  "request_timeout_seconds": 180,
  "failure_cooldown_seconds": 15,
  "members": [
    {"model_id": "primary-connection-id", "weight": 3},
    {"model_id": "backup-connection-id", "weight": 1}
  ]
}
```

For a primary/backup pool, use `strategy: "failover"` and priorities 0/1 on the
two members. Concurrency limits are strict integers: connections 1–256, pools
1–1024; weights 1–10000, priorities 0–10000. Up to 128 connections and 64 pools
are accepted. Queue size 0–4096, queue timeout 0–120s, request timeout 1–1800s,
connection timeout 1–600s and cooldown 0–300s are validated centrally.

Relevant routing codes include `queue_full`, `capacity_wait_timeout`,
`route_timeout`, `no_eligible_model`, `upstream_http_503`, `connection_error`,
`timeout`, `model_override_not_allowed` and `stream_interrupted_no_replay`.
HTTP, SSE and WebSocket consumers must preserve their respective error/status
semantics rather than treating every HTTP-200 transport as model success.

See [validation and reproducible acceptance](backend-validation.md).
