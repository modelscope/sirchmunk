# Backend improvements and compatibility

This contribution changes Python backend/configuration/tests only. It does not
change `web/`, the graph renderer, frontend navigation, styles or bundles.
The source version remains upstream's declared version, not a new release.

## Review map

| Area | Implementation | Validation |
| --- | --- | --- |
| Configurable embeddings | `utils/embedding_util.py`, `storage/knowledge_storage.py` | `test_sync_embeddings.py` |
| Retrieval/LLM safeguards | `search.py`, `agentic/react_agent.py`, `llm/openai_chat.py`, retriever/scanner/extractor | `test_sync_retrieval.py`, `test_sync_stream.py` |
| Multiple LLMs and pools | `llm/pool_config.py`, `pool_router.py`, `factory.py`, `api/llm_pools.py`, chat/search/settings | `test_llm_pool_*.py` and `tests/support/` acceptance |
| Durable storage and lifecycle | `storage/lifecycle.py`, DuckDB/knowledge/history storage, evolver | durable storage, work-root and background/evolver lifecycle tests |
| Optional management/supervision | `api/management.py`, `lifecycle.py`, `supervisor.py`, portable scripts | management/config/lifecycle/backup/supervisor/release-boundary tests |

## Retrieval and embedding behavior

`config/env.example` documents centralized rollback/configuration switches.
`EMBEDDING_MODEL_ID`, `EMBEDDING_DIM` and `EMBEDDING_QUERY_INSTRUCTION` allow an
alternate model and optional query prefix. Dimension/model incompatibility is
rejected before replacing persisted vectors; this contribution does not migrate
or rebuild an existing knowledge base. Provision compatible models explicitly.

LLM attempts have a finite `LLM_TIMEOUT`; SDK nested retries are disabled while
the wrapper retains its application retry policy. Default token/loop budgets
apply only when API callers omit explicit values. ReAct history retains paired
recent exchanges and a bounded older digest; digests are not evidence.
Duplicate-query normalization respects scope. Identifier/CJK routing and safe
spec-cache JSON serialization use corpus-independent behavior with environment
switches; arbitrary unknown Python objects are not silently serialized.
Token-budget exhaustion can return a partial/degraded result without issuing a
second answer-generation call. These are behavior changes, not claims of
improved paper/benchmark headline metrics or newly evaluated retrieval quality.

Pool behavior is opt-in by selection; see [LLM pool API](llm-pools.md).
Legacy selection is the default. Explicit SDK clients remain explicit.

## Optional management contract

Management is disabled unless injected by a trusted launcher **before imports**.
Business `.env`/settings cannot enable it or change its identity/credentials.
Use a separate loopback-only listener and independent Bearer token; never ship
credentials in source, static files, image layers or logs.

- `SIRCHMUNK_MANAGEMENT_ENABLED` enables observation/maintenance.
- `SIRCHMUNK_MANAGEMENT_CONFIG_ENABLED` additionally enables CAS configuration.
- `SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED` additionally enables persistent flush,
  consistent backup, durable receipts and supervised shutdown.
- Instance/deployment IDs, explicit work root, management port/token and a
  suitable immutable release root are mandatory as documented in
  [the full management protocol](platform-control.md) (Chinese).

The work volume and immutable code/interpreter slot must not overlap. A process
owns an OS work-root lease before opening persistent storage. Cooperative
CLI/MCP/SDK/storage users obey that lease; a second writer is refused. Provisioned
lifecycle roots cannot be opened by downgrading to an unaware legacy process.
Shared authoritative storage is specific to lifecycle-managed Web processes.

Maintenance persists an owner and prevents new work while admitted HTTP/WS,
background tasks and real worker threads drain. Cancelling an awaiter does not
release a still-running physical worker. Real durable barriers flush storage,
fsync and publish snapshots atomically on the same local filesystem. Knowledge
Parquet shape and `sirchmunk-data-v1` remain unchanged; no migrations are added.

Operations persist `accepted` before effects, then `succeeded` only after the
effect and receipt commit. Lost ACKs require GET of the **original** operation
ID; do not invent another ID or infer failure from a missing ACK. Accepted or
unknown outcomes retain maintenance/write locks across restart, do not replay,
and require read-only reconciliation. Never delete the ledger/WAL/marker to
unlock an unknown result. [Backup receipt details](backup-operation-receipts.md)
specify identities, digest, strict result fields and error codes (Chinese).

Legacy backup/flush routes remain separate from new operation-receipt routes.
Disabled lifecycle returns 501 without claiming unsupported capabilities.
Readiness verifies local storage/activity, not real model-provider availability.
Private backups may contain `.env` or model credentials: retain restrictive
volume ACLs/modes; snapshots are not downloadable through business APIs.

## Cross-platform deployment and rollback

Use one worker with `reload=false`, reliable local storage and an externally
mounted private work volume. Production targets include Ubuntu; the Windows
ACL adapter is isolated with a conditional `pywin32` dependency. Ubuntu uses
native uid/mode checks, not PowerShell, drive letters or Windows process rules.

`scripts/supervised_serve.py` runs in the foreground and supports a protected
external config, `--check-only`, and a bounded `--request-stop` helper. The helper
binds to the original instance UUID/owner/PID/create time, persists stop intent,
waits for a successful durable receipt and clean real exit, and never force-kills
or silently retries an unknown stop. Wall/monotonic-clock discontinuity latches
a protective refusal; tests must not relax TTLs to bypass it.

`scripts/build_release_metadata.py` reads fixed raw Git blobs without importing
the product, emitting source hashes/version/data schema/actual build ABI. It is
not a signed deployment envelope or a claim of cross-platform wheel acceptance.
Production signing, systemd/SCM wiring, multi-host control integration, upgrade
and rollback approval remain deployment-system responsibilities.

Pool-only rollback selects `legacy` without modifying knowledge data. Managed
root rollback requires code that understands the same locks/schema/receipts;
rollback code only, never automatically restore older business data. Preserve
private configuration and ledgers for reconciliation.

See [test results and acceptance limits](backend-validation.md).
