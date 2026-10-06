"""Opt-in, single-process management listener for the Sirchmunk Web API.

The listener is deliberately separate from the public API socket.  It does not
manage the independently launched MCP or CLI processes.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sirchmunk.version import __version__


_INSTANCE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
_OPERATION = re.compile(r"^[a-zA-Z0-9-]{8,100}$")
_MAX_BODY = 16 * 1024
_CONFIG_SCHEMA = "sirchmunk.runtime.v1"
_CONFIG_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class ManagementError(Exception):
    def __init__(self, code: str, status: int):
        self.code = code
        self.status = status
        super().__init__(code)


def _error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code}}, status_code=status)


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


async def strict_body(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > _MAX_BODY:
        raise ManagementError("request_too_large", 413)
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_no_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non_finite")),
        )
    except (UnicodeError, ValueError) as exc:
        raise ManagementError("invalid_json", 400) from exc
    if not isinstance(value, dict):
        raise ManagementError("invalid_request", 422)
    return value


class _ProcessLock:
    """Prevent a second managed Web worker using the same state root."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a+b")
        try:
            self.file.seek(0)
            if self.file.read(1) != b"\0":
                self.file.seek(0)
                self.file.write(b"\0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.file.close()
            raise ManagementError("managed_state_already_in_use", 409) from None

    def close(self):
        if self.file.closed:
            return
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()


class ManagementState:
    def __init__(self, state_dir: Path, instance_id: str, deployment_id: str):
        self.state_dir = state_dir
        self.instance_id = instance_id
        self.deployment_id = deployment_id
        self.service_instance_id = str(uuid.uuid4())
        self.lock = threading.RLock()
        self.active_jobs = 0
        self._closed = False
        self.config_enabled = False
        self.ws_consumer = None
        self.ws_default = None
        self.lifecycle = None
        self.stopping = False
        self._process_lock = _ProcessLock(state_dir / "web-instance.lock")
        try:
            self.db = sqlite3.connect(
                str(state_dir / "management.sqlite3"),
                isolation_level=None,
                check_same_thread=False,
            )
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS gate (id INTEGER PRIMARY KEY CHECK(id=1), "
                "enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), "
                "operation_id TEXT)"
            )
            self.db.execute(
                "INSERT OR IGNORE INTO gate(id, enabled, operation_id) VALUES(1, 0, NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS process_state (id INTEGER PRIMARY KEY CHECK(id=1), "
                "unclean INTEGER NOT NULL CHECK(unclean IN (0,1)))"
            )
            self.db.execute("INSERT OR IGNORE INTO process_state(id, unclean) VALUES(1, 0)")
            previous = self.db.execute(
                "SELECT unclean FROM process_state WHERE id=1"
            ).fetchone()
            if previous is None or type(previous[0]) is not int or previous[0] not in (0, 1):
                raise ManagementError("invalid_process_state", 503)
            self.counter_unknown = bool(previous[0])
            # Only a clean shutdown can clear this marker. A hard process death
            # must not turn unfinished work into a reported zero on restart.
            self.db.execute("UPDATE process_state SET unclean=1 WHERE id=1")
            row = self.db.execute(
                "SELECT enabled, operation_id FROM gate WHERE id=1"
            ).fetchone()
            if row is None or bool(row[0]) != (row[1] is not None) or (
                row[1] is not None and not _OPERATION.fullmatch(row[1])
            ):
                raise ManagementError("invalid_persistent_gate", 503)
            self.maintenance_enabled = bool(row[0])
            self.maintenance_owner = row[1]
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self._process_lock.close()
            raise

    def close(self):
        with self.lock:
            if self._closed:
                return
            if self.lifecycle and self.lifecycle.workers:
                raise RuntimeError("management_operation_still_running")
            clean_storage = not self.lifecycle or (
                self.lifecycle.runtime.closed and not self.lifecycle.runtime.failure
                and not self.lifecycle.shutdown_unresolved()
            )
            if self.active_jobs == 0 and not self.counter_unknown and clean_storage:
                if self.lifecycle:
                    self.lifecycle.finalize_shutdown()
                else:
                    self.db.execute("UPDATE process_state SET unclean=0 WHERE id=1")
            self.db.close()
            self._process_lock.close()
            self._closed = True

    def admit(self) -> bool:
        with self.lock:
            if self.business_blocked():
                return False
            self.active_jobs += 1
            return True

    def release(self):
        with self.lock:
            if self.active_jobs < 1:
                raise RuntimeError("management admission counter underflow")
            self.active_jobs -= 1

    def business_blocked(self):
        with self.lock:
            return bool(self.maintenance_enabled or self.counter_unknown or self.stopping
                        or (self.lifecycle and (self.lifecycle.reason() or self.lifecycle.runtime.frozen)))

    def enable_lifecycle(self, runtime):
        from .lifecycle import LifecycleControl
        with self.lock:
            if self.lifecycle:
                raise ManagementError("lifecycle_already_enabled", 409)
            self.lifecycle = LifecycleControl(self, runtime)

    def require_lifecycle(self):
        if not self.lifecycle:
            raise ManagementError("business_flush_not_proven", 501)
        return self.lifecycle

    def readiness(self):
        with self.lock:
            status = self.status()
            return {"instance_id": self.instance_id, "service_instance_id": self.service_instance_id,
                    "readiness": status["readiness"], "reason": status["reason"],
                    "data_schema": status["data_schema"],
                    "background_jobs": self.lifecycle.runtime.active_background() if self.lifecycle else None,
                    "storage_validated": bool(self.lifecycle and self.lifecycle.validated)}

    def set_maintenance(self, enabled: bool, operation_id: str) -> dict:
        if not _OPERATION.fullmatch(operation_id):
            raise ManagementError("invalid_operation_id", 422)
        with self.lock:
            if self.lifecycle:
                self.lifecycle.guard_mutation()
            if enabled:
                if self.maintenance_enabled:
                    if self.maintenance_owner != operation_id:
                        raise ManagementError("maintenance_owned_by_other", 409)
                else:
                    self.db.execute(
                        "UPDATE gate SET enabled=1, operation_id=? WHERE id=1",
                        (operation_id,),
                    )
                    self.maintenance_enabled = True
                    self.maintenance_owner = operation_id
            else:
                if self.maintenance_enabled:
                    if self.maintenance_owner != operation_id:
                        raise ManagementError("maintenance_owned_by_other", 409)
                    if self.counter_unknown:
                        raise ManagementError("unfinished_work_unknown", 409)
                    if self.lifecycle and (self.lifecycle.reason() or self.stopping):
                        raise ManagementError("lifecycle_not_ready", 409)
                    self.db.execute(
                        "UPDATE gate SET enabled=0, operation_id=NULL WHERE id=1"
                    )
                    self.maintenance_enabled = False
                    self.maintenance_owner = None
                else:
                    # No stale owner can be used to open a different generation.
                    raise ManagementError("maintenance_not_enabled", 409)
            return {
                "enabled": self.maintenance_enabled,
                "operation_id": self.maintenance_owner,
            }

    def status(self) -> dict:
        with self.lock:
            reason = "unclean_previous_process" if self.counter_unknown else (
                self.lifecycle.reason() if self.lifecycle else ""
            )
            if not reason and self.stopping:
                reason = "stopping"
            background = self.lifecycle.runtime.active_background() if self.lifecycle else 0
            return {
                "protocol_version": "1.0",
                "instance_id": self.instance_id,
                "platform_type": "sirchmunk",
                "liveness": "ok",
                "readiness": "unknown" if reason else "ok",
                "running_version": __version__,
                "deployment_id": self.deployment_id,
                "service_instance_id": self.service_instance_id,
                "data_schema": "sirchmunk-data-v1" if self.lifecycle and self.lifecycle.validated else None,
                "maintenance": {
                    "enabled": self.maintenance_enabled,
                    "operation_id": self.maintenance_owner,
                },
                "active_jobs": None if self.counter_unknown else self.active_jobs + background,
                "queued_jobs": None if self.counter_unknown else 0,
                "accepting_jobs": not self.business_blocked(),
                "reason": reason,
            }

    def capabilities(self) -> dict:
        capabilities = [
            "health.v1", "identity.v1", "maintenance.v1",
            "persistent-maintenance.v1", "drain.v1",
        ]
        if self.config_enabled:
            capabilities.append("configuration.v1")
        if self.lifecycle and self.lifecycle.validated:
            capabilities.extend(["flush.v1", "flush-operation-receipts.v1", "backup.v1",
                                 "backup-operation-receipts.v1"])
        return {
            "protocol_version": "1.0",
            "instance_id": self.instance_id,
            "capabilities": capabilities,
        }

    def enable_configuration(self, ws_consumer):
        """Load stored capacity into the real WebSocket admission consumer."""
        with self.lock:
            if self.config_enabled:
                raise ManagementError("configuration_already_enabled", 409)
            default = ws_consumer.get_limit()
            if type(default) is not int or not 1 <= default <= 1000:
                raise ManagementError("invalid_deployment_websocket_limit", 503)
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS runtime_config ("
                "id INTEGER PRIMARY KEY CHECK(id=1), "
                "revision INTEGER NOT NULL CHECK(revision>=0), "
                "stored INTEGER CHECK(stored BETWEEN 1 AND 1000))"
            )
            self.db.execute(
                "INSERT OR IGNORE INTO runtime_config(id,revision,stored) VALUES(1,0,NULL)"
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS config_operations ("
                "operation_id TEXT PRIMARY KEY, request_digest TEXT NOT NULL, "
                "service_instance_id TEXT NOT NULL, status TEXT NOT NULL, "
                "before_revision INTEGER NOT NULL, after_revision INTEGER, "
                "effect_committed INTEGER, apply_state TEXT NOT NULL, error_code TEXT)"
            )
            row = self.db.execute(
                "SELECT revision,stored FROM runtime_config WHERE id=1"
            ).fetchone()
            if row is None or type(row[0]) is not int or row[0] < 0 or (
                row[1] is not None and (type(row[1]) is not int or not 1 <= row[1] <= 1000)
            ):
                raise ManagementError("invalid_runtime_config", 503)
            ws_consumer.set_limit(row[1] if row[1] is not None else default)
            self.ws_consumer = ws_consumer
            self.ws_default = default
            self.config_enabled = True

    def _require_configuration(self):
        if not self.config_enabled:
            raise ManagementError("configuration_disabled", 404)

    def config_document(self) -> dict:
        self._require_configuration()
        with self.lock:
            revision, stored = self.db.execute(
                "SELECT revision,stored FROM runtime_config WHERE id=1"
            ).fetchone()
            effective = self.ws_consumer.get_limit()
            desired = self.ws_default if stored is None else stored
            return {
                "protocol_version": "1.0",
                "instance_id": self.instance_id,
                "platform_type": "sirchmunk",
                "service_instance_id": self.service_instance_id,
                "config_schema_id": _CONFIG_SCHEMA,
                "revision": revision,
                "stored": {"maxWebSocketConnections": stored},
                "effective": {"maxWebSocketConnections": effective},
                "apply_state": "applied" if effective == desired else "unknown",
            }

    def _configuration_request(self, body: dict) -> dict:
        self._require_configuration()
        required = {
            "protocol_version", "operation_id", "instance_id",
            "service_instance_id", "config_schema_id", "expected_revision",
            "changes", "request_digest",
        }
        if set(body) != required:
            raise ManagementError("invalid_configuration_request", 422)
        operation_id = body["operation_id"]
        revision = body["expected_revision"]
        changes = body["changes"]
        digest = body["request_digest"]
        if (
            body["protocol_version"] != "1.0"
            or body["instance_id"] != self.instance_id
            or body["service_instance_id"] != self.service_instance_id
            or body["config_schema_id"] != _CONFIG_SCHEMA
            or not isinstance(operation_id, str)
            or not _CONFIG_ID.fullmatch(operation_id)
            or type(revision) is not int or revision < 0
            or type(changes) is not dict
            or set(changes) != {"maxWebSocketConnections"}
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
        ):
            raise ManagementError("invalid_configuration_request", 422)
        value = changes["maxWebSocketConnections"]
        if value is not None and (type(value) is not int or not 1 <= value <= 1000):
            raise ManagementError("invalid_configuration_value", 422)
        canonical = json.dumps(
            {key: item for key, item in body.items() if key != "request_digest"},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if not hmac.compare_digest(hashlib.sha256(canonical).hexdigest(), digest):
            raise ManagementError("configuration_digest_mismatch", 422)
        return body

    def validate_configuration(self, body: dict) -> dict:
        request = self._configuration_request(body)
        with self.lock:
            if self.db.execute(
                "SELECT 1 FROM config_operations WHERE status='unknown' LIMIT 1"
            ).fetchone() is not None:
                raise ManagementError("configuration_unknown_locked", 409)
            revision = self.db.execute(
                "SELECT revision FROM runtime_config WHERE id=1"
            ).fetchone()[0]
            if revision != request["expected_revision"]:
                raise ManagementError("configuration_revision_conflict", 409)
            return {
                "protocol_version": "1.0",
                "instance_id": self.instance_id,
                "service_instance_id": self.service_instance_id,
                "operation_id": request["operation_id"],
                "request_digest": request["request_digest"],
                "valid": True,
                "revision": revision,
                "apply_mode": "hot",
            }

    def _operation_receipt(self, row) -> dict:
        return {
            "protocol_version": "1.0",
            "instance_id": self.instance_id,
            "service_instance_id": row[2],
            "operation_id": row[0],
            "request_digest": row[1],
            "config_schema_id": _CONFIG_SCHEMA,
            "status": row[3],
            "before_revision": row[4],
            "after_revision": row[5],
            "effect_committed": None if row[6] is None else bool(row[6]),
            "apply_state": row[7],
            "error_code": row[8],
        }

    def _read_operation(self, operation_id: str):
        return self.db.execute(
            "SELECT operation_id,request_digest,service_instance_id,status,"
            "before_revision,after_revision,effect_committed,apply_state,error_code "
            "FROM config_operations WHERE operation_id=?",
            (operation_id,),
        ).fetchone()

    def get_configuration_operation(self, operation_id: str) -> dict:
        self._require_configuration()
        if not _CONFIG_ID.fullmatch(operation_id):
            raise ManagementError("invalid_operation_id", 422)
        with self.lock:
            row = self._read_operation(operation_id)
            if row is None:
                raise ManagementError("configuration_operation_not_found", 404)
            return self._operation_receipt(row)

    def apply_configuration(self, body: dict) -> dict:
        request = self._configuration_request(body)
        operation_id = request["operation_id"]
        digest = request["request_digest"]
        with self.lock:
            prior = self._read_operation(operation_id)
            if prior is not None:
                if prior[1] != digest:
                    raise ManagementError("configuration_operation_reused", 409)
                return self._operation_receipt(prior)
            if self.lifecycle:
                self.lifecycle.guard_mutation()
            if self.db.execute(
                "SELECT 1 FROM config_operations WHERE status='unknown' LIMIT 1"
            ).fetchone() is not None:
                raise ManagementError("configuration_unknown_locked", 409)
            revision = self.db.execute(
                "SELECT revision FROM runtime_config WHERE id=1"
            ).fetchone()[0]
            if revision != request["expected_revision"]:
                if revision < request["expected_revision"]:
                    raise ManagementError("configuration_revision_conflict", 409)
                self.db.execute("BEGIN IMMEDIATE")
                try:
                    self.db.execute(
                        "INSERT INTO config_operations VALUES(?,?,?,?,?,?,?,?,?)",
                        (operation_id, digest, self.service_instance_id, "rejected",
                         revision, None, 0, "applied", "configuration_revision_conflict"),
                    )
                    self.db.execute("COMMIT")
                except BaseException:
                    self.db.execute("ROLLBACK")
                    raise
                return self._operation_receipt(self._read_operation(operation_id))

            value = request["changes"]["maxWebSocketConnections"]
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute(
                    "INSERT INTO config_operations VALUES(?,?,?,?,?,?,?,?,?)",
                    (operation_id, digest, self.service_instance_id, "unknown",
                     revision, revision + 1, None, "unknown", "configuration_effect_unknown"),
                )
                self.db.execute(
                    "UPDATE runtime_config SET revision=?,stored=? WHERE id=1",
                    (revision + 1, value),
                )
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            desired = self.ws_default if value is None else value
            try:
                self.ws_consumer.set_limit(desired)
                if self.ws_consumer.get_limit() != desired:
                    return self._operation_receipt(self._read_operation(operation_id))
                self.db.execute(
                    "UPDATE config_operations SET status='succeeded',"
                    "effect_committed=1,apply_state='applied',error_code=NULL "
                    "WHERE operation_id=? AND status='unknown'",
                    (operation_id,),
                )
            except Exception:
                # The stored intent remains unknown; never replay this POST.
                return self._operation_receipt(self._read_operation(operation_id))
            return self._operation_receipt(self._read_operation(operation_id))


class ManagementHolder:
    def __init__(self):
        self.state: ManagementState | None = None
        self.server = None
        self.thread = None


class AdmissionMiddleware:
    def __init__(self, app, holder: ManagementHolder):
        self.app = app
        self.holder = holder

    async def __call__(self, scope, receive, send):
        state = self.holder.state
        if (
            state is None
            or scope["type"] not in ("http", "websocket")
            or not scope.get("path", "").startswith("/api/v1/")
            or (scope["type"] == "http" and scope.get("method") == "OPTIONS")
        ):
            return await self.app(scope, receive, send)
        if not state.admit():
            if scope["type"] == "websocket":
                await send({"type": "websocket.accept"})
                await send({"type": "websocket.close", "code": 1013, "reason": "maintenance"})
            else:
                raw = b'{"error":{"code":"maintenance_active"}}'
                await send({
                    "type": "http.response.start", "status": 503,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(raw)).encode("ascii"))],
                })
                await send({"type": "http.response.body", "body": raw})
            return
        try:
            if scope["type"] == "websocket" and state.lifecycle:
                await self._managed_websocket(state, scope, receive, send)
            else:
                await self.app(scope, receive, send)
        finally:
            state.release()

    async def _managed_websocket(self, state, scope, receive, send):
        accepted = False
        closed = False

        async def guarded_send(message):
            nonlocal accepted, closed
            if message["type"] == "websocket.accept":
                accepted = True
            if closed:
                return
            if message["type"] == "websocket.close":
                closed = True
            await send(message)

        async def drain_close():
            nonlocal closed
            if not closed:
                closed = True
                await send({"type": "websocket.close", "code": 1013, "reason": "maintenance"})
            return {"type": "websocket.disconnect", "code": 1013}

        async def guarded_receive():
            # The current message may finish. The next message is linearized
            # against the same lock as maintenance, including already queued
            # frames. Idle sockets wake within 100ms without cancelling work.
            if not accepted:
                return await receive()
            pending = None
            try:
                while True:
                    if state.business_blocked():
                        return await drain_close()
                    if pending is None:
                        pending = asyncio.create_task(receive())
                    done, _ = await asyncio.wait({pending}, timeout=0.1)
                    if done:
                        message = pending.result()
                        if state.business_blocked():
                            return await drain_close()
                        return message
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

        await self.app(scope, guarded_receive, guarded_send)


def _management_app(holder: ManagementHolder, token: str) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        auth = request.headers.get("authorization", "")
        if len(auth) > 256 or not auth.startswith("Bearer ") or not hmac.compare_digest(auth[7:], token):
            return _error("unauthorized", 401)
        return await call_next(request)

    @app.exception_handler(ManagementError)
    async def handle_management_error(_request: Request, exc: ManagementError):
        return _error(exc.code, exc.status)

    @app.get("/management/v1/capabilities")
    async def capabilities():
        return holder.state.capabilities()

    @app.get("/management/v1/status")
    async def status():
        return holder.state.status()

    @app.get("/management/v1/readiness")
    async def readiness():
        return holder.state.readiness()

    @app.post("/management/v1/maintenance")
    async def maintenance(request: Request):
        body = await strict_body(request)
        if set(body) != {"enabled", "operation_id"} or type(body["enabled"]) is not bool or not isinstance(body["operation_id"], str):
            raise ManagementError("invalid_request", 422)
        return holder.state.set_maintenance(body["enabled"], body["operation_id"])

    @app.get("/management/v1/config")
    async def get_configuration():
        return holder.state.config_document()

    @app.post("/management/v1/config/validate")
    async def validate_configuration(request: Request):
        return holder.state.validate_configuration(await strict_body(request))

    @app.post("/management/v1/config/operations")
    async def apply_configuration(request: Request):
        return holder.state.apply_configuration(await strict_body(request))

    @app.get("/management/v1/config/operations/{operation_id}")
    async def get_configuration_operation(operation_id: str):
        return holder.state.get_configuration_operation(operation_id)

    async def durable_operation(method, request):
        body = await strict_body(request)
        # GET/status stays responsive while physical IO runs. Cancellation of
        # the HTTP await does NOT release the operation's persistent freeze.
        try:
            result = await asyncio.to_thread(method, body)
        except ManagementError:
            raise
        except Exception:
            # A failed physical effect may have partially committed. No paths,
            # exception messages, keys or stack traces belong in this response.
            return _error("lifecycle_effect_unknown", 503)
        accepted = result.get("state") == "accepted" or result.get("completed") is False
        return JSONResponse(result, status_code=202 if accepted else 200)

    @app.post("/management/v1/flush")
    async def flush(request: Request):
        return await durable_operation(holder.state.require_lifecycle().flush_compat, request)

    @app.post("/management/v1/flush/operations")
    async def flush_operation(request: Request):
        return await durable_operation(holder.state.require_lifecycle().flush, request)

    @app.get("/management/v1/flush/operations/{operation_id}")
    async def get_flush_operation(operation_id: str):
        return holder.state.require_lifecycle().get("flush", operation_id)

    @app.post("/management/v1/backup")
    async def backup(request: Request):
        return await durable_operation(holder.state.require_lifecycle().backup, request)

    @app.get("/management/v1/backup/operations/{operation_id}")
    async def get_backup_operation(operation_id: str):
        return holder.state.require_lifecycle().get("backup", operation_id)

    # Independent namespace and kind: the legacy GET above must always keep
    # returning the legacy four-field receipt, including for overlapping IDs.
    @app.post("/management/v1/backup-operations")
    async def backup_operation(request: Request):
        return await durable_operation(holder.state.require_lifecycle().backup_operation, request)

    @app.get("/management/v1/backup-operations/{operation_id}")
    async def get_durable_backup_operation(operation_id: str):
        return holder.state.require_lifecycle().get("backup-operation", operation_id)

    return app


async def start_management(holder: ManagementHolder, work_path: Path, launch_env):
    # The caller must pass the process-launch snapshot captured before loading
    # the business-editable .env. Never fall back to the mutable process env.
    source = launch_env
    if (source.get("SIRCHMUNK_MANAGEMENT_ENABLED") or "false").lower() != "true":
        return
    launch_work = source.get("SIRCHMUNK_WORK_PATH")
    if not launch_work or Path(launch_work).expanduser().resolve() != work_path:
        raise RuntimeError("managed work path must be supplied by the supervisor")
    instance_id = source.get("SIRCHMUNK_MANAGEMENT_INSTANCE_ID") or ""
    deployment_id = source.get("SIRCHMUNK_DEPLOYMENT_ID") or ""
    token = source.get("SIRCHMUNK_MANAGEMENT_TOKEN") or ""
    port_text = source.get("SIRCHMUNK_MANAGEMENT_PORT") or ""
    if not _INSTANCE.fullmatch(instance_id) or not deployment_id or len(deployment_id) > 100:
        raise RuntimeError("invalid management instance or deployment identity")
    if len(token) < 32 or any(ch.isspace() for ch in token) or hmac.compare_digest(token, os.getenv("SIRCHMUNK_API_TOKEN", "")):
        raise RuntimeError("invalid or reused management token")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise RuntimeError("invalid management port") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("invalid management port")
    state_dir = work_path / ".control"
    state = ManagementState(state_dir, instance_id, deployment_id)
    holder.state = state
    try:
        if (source.get("SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED") or "false").lower() == "true":
            from sirchmunk.storage.lifecycle import get_runtime
            runtime = get_runtime(work_path)
            if runtime is None:
                raise RuntimeError("managed root must be bootstrapped before storage imports")
            state.enable_lifecycle(runtime)
        if (source.get("SIRCHMUNK_MANAGEMENT_CONFIG_ENABLED") or "false").lower() == "true":
            from .chat import manager as websocket_manager

            state.enable_configuration(websocket_manager)
        import uvicorn

        server = uvicorn.Server(uvicorn.Config(
            _management_app(holder, token), host="127.0.0.1", port=port,
            log_level="warning", access_log=False, lifespan="off", proxy_headers=False,
        ))
        thread = threading.Thread(target=server.run, name="sirchmunk-management", daemon=True)
        holder.server = server
        holder.thread = thread
        thread.start()
        for _ in range(200):
            if server.started:
                return
            if not thread.is_alive():
                break
            await asyncio.sleep(0.05)
        raise RuntimeError("management loopback listener failed to start")
    except BaseException:
        await stop_management(holder)
        raise


async def stop_management(holder: ManagementHolder):
    if holder.server is not None:
        holder.server.should_exit = True
    if holder.thread is not None:
        await asyncio.to_thread(holder.thread.join, 5)
        if holder.thread.is_alive():
            raise RuntimeError("management listener failed to stop")
    if holder.state is not None:
        holder.state.close()
    holder.state = None
    holder.thread = None
    holder.server = None


async def close_managed_runtime(holder: ManagementHolder, timeout: float = 60):
    """Drain real work before closing; timeout never means force or clean exit."""
    state = holder.state
    if state is None or state.lifecycle is None:
        return
    lifecycle = state.lifecycle
    runtime = lifecycle.runtime
    with state.lock:
        state.stopping = True
    deadline = asyncio.get_running_loop().time() + timeout
    try:
        while True:
            with state.lock, runtime.activity_lock:
                if state.counter_unknown or runtime.failure or lifecycle.unknown:
                    raise RuntimeError("shutdown_state_unknown")
                if state.active_jobs == 0 and runtime.activities == 0 and lifecycle.workers == 0:
                    if lifecycle.shutdown_unresolved():
                        raise RuntimeError("shutdown_state_unknown")
                    runtime.frozen = True
                    break
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError("shutdown_drain_timeout")
            await asyncio.sleep(0.05)
        await asyncio.to_thread(runtime.close)
    except BaseException:
        runtime.fail("shutdown_not_clean")
        raise
