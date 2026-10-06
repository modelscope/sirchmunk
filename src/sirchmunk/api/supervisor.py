"""Fixed local supervisor entry and non-replaying, no-console stop helper.

Importing this module never imports business routers or claims a work root.
Only the serving branch imports the application after trusted config validation.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import sys
import threading
import time
import uuid

from sirchmunk.storage.lifecycle import DATA_SCHEMA, write_json_atomic
from sirchmunk.version import __version__

STOP_INTENT_TTL_SECONDS = 120
STOP_CLOCK_SKEW_SECONDS = 2


class SupervisorError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def request_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SupervisorError("duplicate_json_key")
        result[key] = value
    return result


def safe_path(value):
    if type(value) is not str or not value or "\0" in value:
        raise SupervisorError("invalid_absolute_path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise SupervisorError("invalid_absolute_path")
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise SupervisorError("linked_supervisor_path")
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise SupervisorError("hardlinked_supervisor_file")
    return path.resolve()


def require_protected(path):
    """No untrusted OS principal may replace a control file or its parent.

    Production ancestors/owner separation are also enforced by Control's host
    config validator. The product rejects links in the full path independently.
    """
    path = safe_path(str(path))
    for target in (path, path.parent):
        details = target.stat()
        if os.name != "nt":
            if details.st_uid not in {0, os.getuid()} or details.st_mode & 0o022:
                raise SupervisorError("unprotected_supervisor_path")
        else:
            try:
                import win32api
                import win32con
                import win32security as security
            except ImportError:
                raise SupervisorError("windows_acl_validation_dependency_missing") from None
            token = security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
            try:
                sid = security.ConvertSidToStringSid(security.GetTokenInformation(token, security.TokenUser)[0])
            finally:
                token.Close()
            trusted = {sid, "S-1-5-18", "S-1-5-32-544"}
            sd = security.GetNamedSecurityInfo(str(target), security.SE_FILE_OBJECT,
                security.OWNER_SECURITY_INFORMATION | security.DACL_SECURITY_INFORMATION)
            if security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) not in trusted:
                raise SupervisorError("untrusted_supervisor_owner")
            # Python 3.13+ mkdir(mode=0o700) grants OWNER RIGHTS, not the
            # caller's literal SID. It is safe only after verifying that owner.
            trusted.add("S-1-3-4")
            acl = sd.GetSecurityDescriptorDacl()
            if acl is None:
                raise SupervisorError("unprotected_supervisor_dacl")
            write_mask = 0x40000000 | 0x10000000 | 0x10000 | 0x40000 | 0x80000 | 0x2 | 0x4 | 0x10 | 0x40 | 0x100
            for index in range(acl.GetAceCount()):
                ace = acl.GetAce(index)
                if ace[0][1] & 0x8:  # INHERIT_ONLY_ACE does not apply to this object.
                    continue
                if ace[0][0] in (0, 5, 9, 11) and ace[1] & write_mask:
                    if security.ConvertSidToStringSid(ace[-1]) not in trusted:
                        raise SupervisorError("untrusted_supervisor_writer")
    return path


def read_json(path, *, protected=True):
    path = require_protected(path) if protected else safe_path(str(path))
    if path.stat().st_size > 16384:
        raise SupervisorError("control_document_too_large")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(SupervisorError("non_finite_json")))
    if type(data) is not dict:
        raise SupervisorError("invalid_control_document")
    return data


def load_config(path):
    config_path = require_protected(safe_path(str(path)))
    cfg = read_json(config_path)
    required = {"schema", "instance_id", "work_path", "stop_intent_file", "mailbox_directory",
                "host", "port", "management_port", "shutdown_timeout_seconds"}
    if set(cfg) != required or cfg["schema"] != "sirchmunk-supervisor.v1":
        raise SupervisorError("invalid_supervisor_configuration")
    if type(cfg["instance_id"]) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", cfg["instance_id"]):
        raise SupervisorError("invalid_instance_id")
    root = safe_path(cfg["work_path"])
    intent = safe_path(cfg["stop_intent_file"])
    mailbox = safe_path(cfg["mailbox_directory"])
    from sirchmunk.storage.lifecycle import validate_managed_paths
    code_root = validate_managed_paths(root, dict(os.environ))
    if (mailbox != root / ".control" / "stop-ipc" or intent.is_relative_to(root)
            or config_path.is_relative_to(root) or intent == config_path
            or intent.is_relative_to(code_root) or config_path.is_relative_to(code_root)):
        raise SupervisorError("invalid_control_path_binding")
    if type(cfg["host"]) is not str or cfg["host"] not in {"127.0.0.1", "0.0.0.0", "::1", "::"}:
        raise SupervisorError("invalid_business_bind_host")
    for key in ("port", "management_port"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 65535:
            raise SupervisorError("invalid_port")
    if cfg["port"] == cfg["management_port"]:
        raise SupervisorError("listener_ports_must_differ")
    if type(cfg["shutdown_timeout_seconds"]) is not int or not 1 <= cfg["shutdown_timeout_seconds"] <= 600:
        raise SupervisorError("invalid_shutdown_timeout")
    cfg.update(work_path=str(root), stop_intent_file=str(intent), mailbox_directory=str(mailbox))
    return cfg


def validate_intent(value, instance_id, *, now=None, allow_expired=False):
    required = {"schema", "instance_id", "service_instance_id", "maintenance_owner", "nonce", "expires_at"}
    if type(value) is not dict or set(value) != required or value["schema"] != "platform-stop-intent.v1":
        raise SupervisorError("invalid_stop_intent")
    for key in required - {"expires_at"}:
        if type(value[key]) is not str:
            raise SupervisorError("invalid_stop_intent_type")
    if value["instance_id"] != instance_id:
        raise SupervisorError("stop_instance_mismatch")
    try:
        if str(uuid.UUID(value["service_instance_id"])) != value["service_instance_id"]:
            raise ValueError()
    except ValueError:
        raise SupervisorError("invalid_stop_process_id") from None
    if (not re.fullmatch(r"[A-Za-z0-9-]{8,100}", value["maintenance_owner"])
            or not re.fullmatch(r"[0-9a-f]{64}", value["nonce"])):
        raise SupervisorError("invalid_stop_owner_or_nonce")
    expires = value["expires_at"]
    now = time.time() if now is None else now
    if (type(expires) not in (int, float) or not math.isfinite(expires) or expires <= 0
            or (not allow_expired and not now < expires <= now + STOP_INTENT_TTL_SECONDS + STOP_CLOCK_SKEW_SECONDS)):
        raise SupervisorError("expired_or_invalid_stop_intent")
    return value


@contextmanager
def helper_lock(mailbox):
    """Serialize local helpers, not business writers; never claim the work root."""
    require_protected(mailbox)
    path = safe_path(str(mailbox / "helper.lock"))
    stream = path.open("a+b")
    acquired = False
    try:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _observe(cfg, intent):
    """Read only: exact original UUID/nonce, never substitute current identity."""
    root = Path(cfg["work_path"])
    info = read_json(root / ".control" / "process.json")
    if (set(info) != {"schema", "instance_id", "service_instance_id", "work_root_sha256",
                      "pid", "create_time", "startup_unclean"}
            or info.get("schema") != "sirchmunk-process.v1" or info.get("instance_id") != cfg["instance_id"]
            or info.get("service_instance_id") != intent["service_instance_id"]
            or info.get("work_root_sha256") != hashlib.sha256(str(root).encode()).hexdigest()
            or type(info.get("pid")) is not int or info["pid"] < 1
            or type(info.get("create_time")) not in (int, float)
            or not math.isfinite(info["create_time"]) or info["create_time"] <= 0
            or type(info.get("startup_unclean")) is not bool):
        raise SupervisorError("stop_process_identity_mismatch")
    path = require_protected(root / ".control" / "management.sqlite3")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
        gate = db.execute("SELECT enabled,operation_id FROM gate WHERE id=1").fetchone()
        if gate != (1, intent["maintenance_owner"]):
            raise SupervisorError("stop_maintenance_owner_mismatch")
        row = db.execute("SELECT request_sha256,result_json,state,request_json FROM lifecycle_operations "
                         "WHERE kind='stop' AND operation_id=?", (intent["nonce"],)).fetchone()
        marker = db.execute("SELECT unclean FROM process_state WHERE id=1").fetchone()
        if marker not in ((0,), (1,)):
            raise SupervisorError("invalid_stop_process_marker")
        clean = marker == (0,)
    result = None
    if row:
        if row[0] != request_digest(intent) or row[3] != canonical(intent):
            raise SupervisorError("stop_nonce_reused")
        result = json.loads(row[1], object_pairs_hook=_unique)
        expected = dict(nonce=intent["nonce"], instance_id=intent["instance_id"],
                        service_instance_id=intent["service_instance_id"],
                        maintenance_owner=intent["maintenance_owner"], request_sha256=request_digest(intent))
        if (type(result) is not dict
                or set(result) != {*expected, "schema_id", "state", "stopped", "effect_committed"}
                or result["schema_id"] != "sirchmunk-stop-result.v1"
                or any(result.get(key) != value for key, value in expected.items())
                or row[2] not in ("accepted", "succeeded") or result["state"] != row[2]
                or (row[2] == "accepted" and (result["stopped"] is not None or result["effect_committed"] is not None))
                or (row[2] == "succeeded" and (result["stopped"] is not True or result["effect_committed"] is not True))):
            raise SupervisorError("stop_receipt_identity_mismatch")
    return info, result, clean


def _same_process(info):
    import psutil
    try:
        process = psutil.Process(info["pid"])
        return process.create_time() == info["create_time"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        raise SupervisorError("stop_process_observation_unknown") from None


def request_stop(cfg, intent_file, timeout):
    if safe_path(str(intent_file)) != Path(cfg["stop_intent_file"]):
        raise SupervisorError("stop_intent_path_mismatch")
    # Expiration forbids a new submission, not observation of an old effect.
    intent = validate_intent(read_json(intent_file), cfg["instance_id"], allow_expired=True)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 1 <= timeout <= 600:
        raise SupervisorError("invalid_stop_timeout")
    mailbox = Path(cfg["mailbox_directory"])
    with helper_lock(mailbox):
        info, receipt, clean = _observe(cfg, intent)
        if receipt is None:
            if info.get("startup_unclean") is not False or not _same_process(info):
                raise SupervisorError("stop_live_process_not_confirmed")
            destination = safe_path(str(mailbox / "request.json"))
            submitted = False
            if destination.exists():
                previous = read_json(destination)
                if previous.get("nonce") == intent["nonce"]:
                    if request_digest(previous) != request_digest(intent):
                        raise SupervisorError("stop_nonce_reused")
                    submitted = True
                elif previous.get("service_instance_id") == intent["service_instance_id"]:
                    raise SupervisorError("stop_request_already_submitted")
            # One write only. Later loops observe; no retry or intent mutation.
            if not submitted:
                validate_intent(intent, cfg["instance_id"])
                write_json_atomic(destination, intent)
        deadline = time.monotonic() + timeout
        while True:
            info, receipt, clean = _observe(cfg, intent)
            if (receipt and receipt.get("state") == "succeeded" and receipt.get("stopped") is True
                    and receipt.get("effect_committed") is True and clean and not _same_process(info)):
                return
            rejection = mailbox / "rejection.json"
            if rejection.exists():
                denied = read_json(rejection)
                if denied.get("request_sha256") == request_digest(intent):
                    code = "stop_clock_discontinuity" if denied.get("error") == "stop_clock_discontinuity" else "stop_rejected_or_unknown"
                    raise SupervisorError(code)
            if time.monotonic() >= deadline:
                raise SupervisorError("stop_not_confirmed")
            time.sleep(0.1)


async def watch_stop(cfg, state, server):
    mailbox = Path(cfg["mailbox_directory"])
    seen = None
    while not state._closed and not server.should_exit:
        path = mailbox / "request.json"
        intent = None
        clock_safe = True
        try:
            # Observe clock jumps even between requests. Never let a rollback
            # make an expired intent valid again in this process generation.
            state.require_lifecycle().check_stop_clock()
        except Exception:
            clock_safe = False
        try:
            if path.exists():
                intent = read_json(path)
                fingerprint = request_digest(intent)
                if fingerprint != seen:
                    seen = fingerprint
                    if not clock_safe:
                        raise SupervisorError("stop_clock_discontinuity")
                    validate_intent(intent, cfg["instance_id"])
                    accepted = await asyncio.to_thread(state.require_lifecycle().request_stop, intent)
                    if accepted:
                        server.should_exit = True
                        return
        except Exception:
            if intent is not None:
                write_json_atomic(mailbox / "rejection.json", {
                    "schema": "sirchmunk-stop-rejection.v1", "request_sha256": request_digest(intent),
                    "error": ("stop_clock_discontinuity" if not clock_safe or state.lifecycle.stop_clock_unsafe
                              else "stop_rejected_or_unknown"),
                })
        await asyncio.sleep(0.1)


def configure_launch(cfg):
    from sirchmunk.storage.lifecycle import validate_managed_paths
    validate_managed_paths(Path(cfg["work_path"]), dict(os.environ))
    expected = {
        "SIRCHMUNK_WORK_PATH": cfg["work_path"],
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": cfg["instance_id"],
        "SIRCHMUNK_MANAGEMENT_ENABLED": "true", "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_PORT": str(cfg["management_port"]),
        "SIRCHMUNK_MANAGEMENT_SHUTDOWN_TIMEOUT_SECONDS": str(cfg["shutdown_timeout_seconds"]),
    }
    for key, value in expected.items():
        if key in os.environ and os.environ[key] != value:
            raise SupervisorError("launch_configuration_conflict")
    deployment = os.getenv("PLATFORM_DEPLOYMENT_ID") or os.getenv("SIRCHMUNK_DEPLOYMENT_ID")
    if not deployment or (os.getenv("SIRCHMUNK_DEPLOYMENT_ID") not in (None, deployment)):
        raise SupervisorError("launch_deployment_identity_conflict")
    if os.getenv("PLATFORM_RELEASE_VERSION", __version__) != __version__:
        raise SupervisorError("launch_release_version_mismatch")
    expected["SIRCHMUNK_DEPLOYMENT_ID"] = deployment
    os.environ.update(expected)


async def serve(cfg):
    import psutil
    import uvicorn
    from . import main as application

    class Server(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            if threading.current_thread() is not threading.main_thread():
                raise SupervisorError("supervisor_requires_main_thread")
            supported = [signal.SIGINT, signal.SIGTERM]
            if hasattr(signal, "SIGBREAK"):
                supported.append(signal.SIGBREAK)
            previous = {sig: signal.signal(sig, self.handle_exit) for sig in supported}
            try:
                yield
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)

        def handle_exit(self, sig, frame):
            # Manual OS signals remain graceful, but are not a substitute for
            # the platform's bound intent/nonce protocol and stop helper proof.
            holder = application._management_holder
            if holder.state:
                with holder.state.lock:
                    holder.state.stopping = True
            self.should_exit = True

    server = Server(uvicorn.Config(application.app, host=cfg["host"], port=cfg["port"],
        workers=1, reload=False, access_log=False, log_level="warning",
        timeout_graceful_shutdown=cfg["shutdown_timeout_seconds"]))
    runner = asyncio.create_task(server.serve())
    watcher = None
    try:
        while not server.started and not runner.done():
            await asyncio.sleep(0.02)
        if server.started:
            state = application._management_holder.state
            root = Path(cfg["work_path"])
            mailbox = Path(cfg["mailbox_directory"])
            mailbox.mkdir(mode=0o700, parents=True, exist_ok=True)
            require_protected(mailbox)
            write_json_atomic(root / ".control" / "process.json", dict(
                schema="sirchmunk-process.v1", instance_id=state.instance_id,
                service_instance_id=state.service_instance_id, pid=os.getpid(),
                create_time=psutil.Process().create_time(), startup_unclean=state.counter_unknown,
                work_root_sha256=hashlib.sha256(str(root).encode()).hexdigest(),
            ))
            watcher = asyncio.create_task(watch_stop(cfg, state, server))
        await runner
    finally:
        if watcher:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        if not runner.done():
            server.should_exit = True
            await runner
    runtime = application._managed_runtime
    if not runtime or not runtime.closed or runtime.failure:
        raise SupervisorError("supervised_process_not_clean")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed Sirchmunk supervisor and local stop helper")
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--request-stop", action="store_true")
    parser.add_argument("--intent-file")
    parser.add_argument("--stop-timeout-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.request_stop:
            if not args.intent_file:
                raise SupervisorError("stop_intent_required")
            request_stop(cfg, args.intent_file, args.stop_timeout_seconds)
            print('{"stopped":true}')
        else:
            configure_launch(cfg)
            if args.check_only:
                print(canonical(dict(platform_type="sirchmunk", running_version=__version__,
                                     data_schema=DATA_SCHEMA, checks_only=True, storage_opened=False)))
            else:
                asyncio.run(serve(cfg))
        return 0
    except Exception as exc:
        # Unexpected messages may contain private paths. Only expose our codes.
        code = str(exc) if isinstance(exc, SupervisorError) else "supervisor_failed"
        print(canonical({"error": {"code": code}}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
