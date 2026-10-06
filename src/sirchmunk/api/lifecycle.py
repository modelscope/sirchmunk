"""Durable management operations over the exclusively owned business root.

Lock order: state.lock -> runtime.activity_lock for admission/freeze; release
both before taking runtime.io_lock. Storage failures never acquire state.lock.
An accepted operation is deliberately not replayable, including after restart.
"""
from __future__ import annotations

import hashlib
from contextlib import closing
import json
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from sirchmunk.storage.lifecycle import (
    DATA_SCHEMA, durable_directory, durable_file, durable_publish_directory, write_json_atomic,
)


def reject(code, status=409):
    from .management import ManagementError
    raise ManagementError(code, status)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LifecycleControl:
    def __init__(self, state, runtime):
        if runtime.instance_id != state.instance_id or runtime.root / ".control" != state.state_dir:
            reject("lifecycle_identity_mismatch", 503)
        self.state, self.runtime = state, runtime
        self.validated = False
        self.workers = 0
        self.unknown = False
        self.active_stop = None
        self.stop_flushed = False
        # A wall-clock step must not resurrect an expired stop intent. The
        # UUID binds the process generation; this pair binds its clock epoch.
        self.stop_clock_origin = (time.time(), time.monotonic())
        self.stop_clock_highwater = self.stop_clock_origin[0]
        self.stop_clock_unsafe = False
        runtime.state = state
        state.db.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_operations ("
            "operation_id TEXT NOT NULL, kind TEXT NOT NULL, request_sha256 TEXT NOT NULL, "
            "request_json TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('accepted','succeeded')), "
            "result_json TEXT NOT NULL, PRIMARY KEY(kind,operation_id))"
        )
        self.unknown = self.pending()
        runtime.frozen = self.unknown
        durable_directory(state.state_dir)

    def pending(self):
        return self.state.db.execute(
            "SELECT 1 FROM lifecycle_operations WHERE state='accepted' LIMIT 1"
        ).fetchone() is not None

    def reason(self):
        if self.runtime.failure:
            return self.runtime.failure
        if self.unknown:
            return "lifecycle_effect_unknown"
        if not self.validated:
            return "storage_initializing"
        if self.runtime.closed or self.runtime.closing:
            return "storage_closed"
        return ""

    def initialize(self):
        # Constructors validate the actual DB/Parquet schemas, not just marker
        # metadata. They share the very same consumers subsequently used by API.
        from .components.history_storage import HistoryStorage
        from sirchmunk.storage.knowledge_storage import KnowledgeStorage
        from sirchmunk.llm.pool_config import ConfigStore
        from sirchmunk.scheduler.evolver import EvolveManifest
        try:
            HistoryStorage(work_path=str(self.runtime.root))
            KnowledgeStorage(work_path=str(self.runtime.root))
            ConfigStore(self.runtime.root).read()
            manifest = self.runtime.root / ".cache" / "evolve" / "manifest.json"
            if manifest.exists():
                EvolveManifest.from_json(manifest.read_text(encoding="utf-8"), strict=True)
            # Never turn restart into an implicit retry of an accepted effect.
            if not self.unknown and not self.state.counter_unknown:
                self.runtime.flush()
                with self.state.lock:
                    self._checkpoint()
            with self.state.lock:
                self.validated = True
        except BaseException:
            self.runtime.fail("storage_initialization_failed")
            raise

    def _checkpoint(self):
        busy, pages, copied = self.state.db.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        if busy or pages != copied:
            raise RuntimeError("management_checkpoint_incomplete")
        durable_directory(self.state.state_dir)

    def guard_mutation(self):
        if self.runtime.frozen or self.unknown or self.pending():
            reject("lifecycle_operation_locked")

    def _preflight(self, owner):
        state = self.state
        self.guard_mutation()
        if state.counter_unknown or self.reason() or state.stopping:
            reject("lifecycle_not_ready")
        if not state.maintenance_enabled or state.maintenance_owner != owner:
            reject("maintenance_owner_required")
        if state.active_jobs or self.runtime.active_background():
            reject("business_work_not_drained")
        if state.config_enabled and state.db.execute(
            "SELECT 1 FROM config_operations WHERE status='unknown' LIMIT 1"
        ).fetchone():
            reject("configuration_unknown_locked")

    def _read(self, kind, operation_id):
        return self.state.db.execute(
            "SELECT request_sha256,request_json,state,result_json FROM lifecycle_operations "
            "WHERE kind=? AND operation_id=?", (kind, operation_id),
        ).fetchone()

    def get(self, kind, operation_id):
        from .management import _OPERATION
        if not _OPERATION.fullmatch(operation_id):
            reject("invalid_operation_id", 422)
        with self.state.lock:
            row = self._read(kind, operation_id)
            if row is None:
                reject("lifecycle_operation_not_found", 404)
            return json.loads(row[3])

    def _begin(self, kind, request, result):
        """Under state/activity locks, persist intent BEFORE any business IO."""
        state = self.state
        payload = canonical(request)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        prior = self._read(kind, request["operation_id"])
        if prior is not None:
            if prior[0] != digest:
                reject("lifecycle_operation_reused")
            return json.loads(prior[3])
        owner = request.get("maintenance_operation_id", request["operation_id"])
        self._preflight(owner)
        if kind in {"flush", "backup-operation"} and (
            request["instance_id"] != state.instance_id
            or request["service_instance_id"] != state.service_instance_id
        ):
            reject("lifecycle_identity_mismatch")
        self.runtime.frozen = True
        try:
            state.db.execute(
                "INSERT INTO lifecycle_operations VALUES(?,?,?,?,?,?)",
                (request["operation_id"], kind, digest, payload, "accepted", canonical(result)),
            )
        except BaseException:
            self.unknown = True
            self.runtime.fail("lifecycle_intent_persistence_failed")
            raise
        self.workers += 1
        return None

    def _finish(self, kind, request, result):
        with self.state.lock:
            changed = self.state.db.execute(
                "UPDATE lifecycle_operations SET state='succeeded',result_json=? "
                "WHERE kind=? AND operation_id=? AND state='accepted'",
                (canonical(result), kind, request["operation_id"]),
            )
            if changed.rowcount != 1:
                raise RuntimeError("lifecycle_receipt_transition_failed")
            self.runtime.frozen = False

    def _failed(self):
        with self.state.lock:
            self.unknown = True
            # Keep accepted and freeze. Do not infer whether an effect committed.
            self.runtime.fail("lifecycle_effect_unknown")

    def flush(self, request):
        from .management import _OPERATION
        required = {"operation_id", "instance_id", "service_instance_id", "maintenance_operation_id"}
        if set(request) != required or any(type(v) is not str for v in request.values()):
            reject("invalid_flush_request", 422)
        if (not _OPERATION.fullmatch(request["operation_id"])
                or not _OPERATION.fullmatch(request["maintenance_operation_id"])
                or not 1 <= len(request["instance_id"]) <= 80
                or not 1 <= len(request["service_instance_id"]) <= 100):
            reject("invalid_flush_request", 422)
        result = dict(schema_id="sirchmunk-flush-operation-result.v1", **request,
                      request_sha256=hashlib.sha256(canonical(request).encode("utf-8")).hexdigest(),
                      state="accepted", flushed=None, effect_committed=None)
        with self.state.lock, self.runtime.activity_lock:
            prior = self._begin("flush", request, result)
            if prior is not None:
                return prior
        try:
            self.runtime.flush()
            with self.state.lock:
                self._checkpoint()
            result.update(state="succeeded", flushed=True, effect_committed=True)
            self._finish("flush", request, result)
            return result
        except BaseException:
            self._failed()
            raise
        finally:
            with self.state.lock:
                self.workers -= 1

    def flush_compat(self, body):
        if body:
            reject("invalid_flush_request", 422)
        with self.state.lock:
            owner = self.state.maintenance_owner
            if owner is None:
                reject("maintenance_owner_required")
            request = dict(operation_id=str(uuid.uuid4()), instance_id=self.state.instance_id,
                           service_instance_id=self.state.service_instance_id,
                           maintenance_operation_id=owner)
        return self.flush(request)

    def backup(self, request):
        from .management import _OPERATION
        if (set(request) != {"operation_id"} or type(request["operation_id"]) is not str
                or not _OPERATION.fullmatch(request["operation_id"])):
            reject("invalid_backup_request", 422)
        result = dict(backup_id="backup-" + uuid.uuid4().hex, completed=False,
                      consistent=False, data_schema=DATA_SCHEMA)
        with self.state.lock, self.runtime.activity_lock:
            prior = self._begin("backup", request, result)
            if prior is not None:
                return prior
        try:
            self._snapshot(result["backup_id"])
            result.update(completed=True, consistent=True)
            self._finish("backup", request, result)
            return result
        except BaseException:
            self._failed()
            raise
        finally:
            with self.state.lock:
                self.workers -= 1

    def backup_operation(self, request):
        """A separately named action under the original maintenance owner.

        Do not change the legacy backup route, ledger kind or receipt shape.
        Five null result fields deliberately conceal an unproven snapshot.
        Only the accepted request is durable before any snapshot work starts.
        """
        from .management import _OPERATION
        required = {"operation_id", "instance_id", "service_instance_id", "maintenance_operation_id"}
        if (type(request) is not dict or set(request) != required
                or any(type(value) is not str for value in request.values())):
            reject("invalid_backup_operation_request", 422)
        if (not _OPERATION.fullmatch(request["operation_id"])
                or not _OPERATION.fullmatch(request["maintenance_operation_id"])
                or not 1 <= len(request["instance_id"]) <= 80
                or not 1 <= len(request["service_instance_id"]) <= 100):
            reject("invalid_backup_operation_request", 422)
        result = dict(schema_id="sirchmunk-backup-operation-result.v1", **request,
                      request_sha256=hashlib.sha256(canonical(request).encode("utf-8")).hexdigest(),
                      state="accepted", backup_id=None, completed=None, consistent=None,
                      data_schema=None, effect_committed=None)
        with self.state.lock, self.runtime.activity_lock:
            prior = self._begin("backup-operation", request, result)
            if prior is not None:
                return prior
        try:
            backup_id = "backup-" + uuid.uuid4().hex
            self._snapshot(backup_id)
            result.update(state="succeeded", backup_id=backup_id, completed=True,
                          consistent=True, data_schema=DATA_SCHEMA, effect_committed=True)
            self._finish("backup-operation", request, result)
            return result
        except BaseException:
            self._failed()
            raise
        finally:
            with self.state.lock:
                self.workers -= 1

    def _snapshot(self, backup_id):
        runtime = self.runtime
        # Private root ACLs apply to all snapshots. Never serve these via API.
        parent = runtime.root / ".control" / "backups"
        if parent.is_symlink() or (hasattr(parent, "is_junction") and parent.is_junction()):
            raise RuntimeError("linked_backup_directory")
        parent.mkdir(mode=0o700, exist_ok=True)
        temporary = parent / (backup_id + ".partial")
        final = parent / backup_id
        temporary.mkdir(mode=0o700)
        with runtime.io_lock:
            runtime.flush()
            files = list(runtime.persistent_files())
            files.append(runtime.root / ".control" / "data-schema.json")
            manifest = []
            for source in files:
                relative = source.relative_to(runtime.root)
                target = temporary / relative
                target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                shutil.copyfile(source, target)
                durable_file(target)
                manifest.append(dict(path=relative.as_posix(), size=target.stat().st_size,
                                     sha256=digest_file(target)))
            target = temporary / ".control" / "management.sqlite3"
            # Online backup includes maintenance owner + accepted operation.
            # The success receipt is written only AFTER this snapshot exists.
            with self.state.lock, closing(sqlite3.connect(str(target))) as snapshot:
                self.state.db.backup(snapshot)
                if snapshot.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise RuntimeError("backup_management_integrity_failed")
            durable_file(target)
            manifest.append(dict(path=".control/management.sqlite3", size=target.stat().st_size,
                                 sha256=digest_file(target)))
            write_json_atomic(temporary / "manifest.json", dict(
                schema_id="sirchmunk-backup.v1", data_schema=DATA_SCHEMA,
                backup_id=backup_id, instance_id=self.state.instance_id,
                service_instance_id=self.state.service_instance_id, files=manifest,
            ))
            for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()),
                                    key=lambda p: len(p.parts), reverse=True):
                durable_directory(directory)
            durable_directory(temporary)
            # Same volume, unique name; never replace an existing snapshot.
            durable_publish_directory(temporary, final)
        # A failure leaves the private partial directory for manual diagnosis;
        # no successful receipt is issued and it is never recursively included.

    def request_stop(self, intent):
        """Trusted local supervisor only; never exposed as an HTTP endpoint.

        The supervisor validates the signed-in OS user's fixed config/mailbox
        and the six-field intent before calling this method. Identity, owner,
        admission and actual counters are rechecked atomically here.
        """
        from .supervisor import validate_intent
        validate_intent(intent, self.state.instance_id, allow_expired=True)
        payload = canonical(intent)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        nonce = intent["nonce"]
        result = dict(schema_id="sirchmunk-stop-result.v1", nonce=nonce,
                      instance_id=intent["instance_id"], service_instance_id=intent["service_instance_id"],
                      maintenance_owner=intent["maintenance_owner"], request_sha256=digest,
                      state="accepted", stopped=None, effect_committed=None)
        with self.state.lock, self.runtime.activity_lock:
            prior = self._read("stop", nonce)
            if prior is not None:
                if prior[0] != digest:
                    reject("stop_nonce_reused")
                return False  # Read-only duplicate: never replay accepted work.
            if (intent["instance_id"] != self.state.instance_id
                    or intent["service_instance_id"] != self.state.service_instance_id):
                reject("stop_identity_mismatch")
            self._preflight(intent["maintenance_owner"])
            # Use the checked monotonic-derived timestamp itself, not another
            # wall-clock read that could roll back after the continuity check.
            # Keep this check after potentially blocking preflight reads.
            validated_now = self.check_stop_clock()
            validate_intent(intent, self.state.instance_id, now=validated_now)
            self.runtime.frozen = True
            try:
                self.state.db.execute("INSERT INTO lifecycle_operations VALUES(?,?,?,?,?,?)",
                    (nonce, "stop", digest, payload, "accepted", canonical(result)))
            except BaseException:
                self._failed()
                raise
            self.active_stop = intent.copy()
            self.workers += 1
        try:
            self.runtime.flush()
            with self.state.lock:
                self._checkpoint()
                self.stop_flushed = True
                self.state.stopping = True
            return True
        except BaseException:
            self._failed()
            raise
        finally:
            with self.state.lock:
                self.workers -= 1

    def check_stop_clock(self):
        """Reject new stop effects after a >2s clock step in this generation.

        Called under the state lock by both the watcher and final acceptance.
        Do not rebase automatically: a previously expired intent could become
        valid under the new wall clock. Existing accepted receipts stay readable.
        """
        from .supervisor import STOP_CLOCK_SKEW_SECONDS
        with self.state.lock:
            wall, monotonic = self.stop_clock_origin
            expected = wall + (time.monotonic() - monotonic)
            actual = time.time()
            if abs(actual - expected) > STOP_CLOCK_SKEW_SECONDS:
                self.stop_clock_unsafe = True
            if self.stop_clock_unsafe:
                reject("stop_clock_discontinuity")
            # Even a small (<2s) backwards adjustment may not extend an intent
            # deadline. Clock tolerance is NOT an expiration grace period.
            self.stop_clock_highwater = max(expected, actual, self.stop_clock_highwater)
            return self.stop_clock_highwater

    def shutdown_unresolved(self):
        rows = self.state.db.execute(
            "SELECT kind,operation_id FROM lifecycle_operations WHERE state='accepted'"
        ).fetchall()
        expected = ("stop", self.active_stop["nonce"]) if self.active_stop and self.stop_flushed else None
        return any(tuple(row) != expected for row in rows)

    def finalize_shutdown(self):
        """Commit clean-process marker and bound stop result in ONE FULL txn."""
        self.state.db.execute("BEGIN IMMEDIATE")
        try:
            if self.active_stop:
                nonce = self.active_stop["nonce"]
                result = json.loads(self._read("stop", nonce)[3])
                result.update(state="succeeded", stopped=True, effect_committed=True)
                changed = self.state.db.execute(
                    "UPDATE lifecycle_operations SET state='succeeded',result_json=? "
                    "WHERE kind='stop' AND operation_id=? AND state='accepted'",
                    (canonical(result), nonce),
                )
                if changed.rowcount != 1:
                    raise RuntimeError("stop_receipt_transition_failed")
            self.state.db.execute("UPDATE process_state SET unclean=0 WHERE id=1")
            self.state.db.execute("COMMIT")
        except BaseException:
            self.state.db.execute("ROLLBACK")
            self.runtime.fail("shutdown_receipt_unknown")
            raise
