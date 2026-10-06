"""Single-writer work-root ownership and durable managed storage lifecycle.

Only the trusted Web bootstrap may open a provisioned managed root. All updated
CLI/MCP/SDK storage users acquire the same OS lock before opening persistent data.
This is a cooperative product boundary, not protection from a same-user attacker.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
from pathlib import Path
import threading
import uuid

DATA_SCHEMA = "sirchmunk-data-v1"
_roots: dict[str, "WorkRootLock"] = {}
_runtimes: dict[str, "RootRuntime"] = {}
_registry_lock = threading.RLock()


class LifecycleError(RuntimeError):
    pass


def root_path(value=None) -> Path:
    return Path(value or os.getenv("SIRCHMUNK_WORK_PATH", "~/.sirchmunk")).expanduser().resolve()


def durable_file(path) -> None:
    with open(path, "r+b") as source:
        source.flush()
        os.fsync(source.fileno())


def durable_directory(path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def durable_replace(source, target) -> None:
    if Path(source).is_symlink() or Path(target).is_symlink():
        raise LifecycleError("linked_durable_replace_target")
    durable_file(source)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        # REPLACE_EXISTING | WRITE_THROUGH, same-volume paths only.
        if not move(str(Path(source).resolve()), str(Path(target).resolve()), 0x1 | 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.replace(source, target)
    durable_directory(Path(target).parent)


def durable_publish_directory(source: Path, target: Path) -> None:
    """Publish a unique same-volume snapshot, without replacing prior data."""
    if target.exists() or source.is_symlink() or target.is_symlink():
        raise LifecycleError("invalid_snapshot_publish_target")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(str(source.resolve()), str(target.resolve()), 0x8):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        source.rename(target)
    for directory in target.parents:
        durable_directory(directory)


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        durable_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class WorkRootLock:
    def __init__(self, root: Path):
        self.root, self.pid = root, os.getpid()
        control = root / ".control"
        if control.is_symlink() or (hasattr(control, "is_junction") and control.is_junction()):
            raise LifecycleError("linked_control_directory")
        control.mkdir(parents=True, exist_ok=True)
        path = control / "work-root.lock"
        if path.is_symlink():
            raise LifecycleError("linked_work_root_lock")
        self.file = path.open("a+b")
        try:
            self.file.seek(0)
            if not self.file.read(1):
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
            raise LifecycleError("work_root_already_in_use") from None

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


def claim_work_root(root=None, *, _managed=False) -> WorkRootLock:
    root = root_path(root)
    key = str(root)
    with _registry_lock:
        existing = _roots.get(key)
        if existing is not None and existing.pid != os.getpid():
            raise LifecycleError("forked_work_root_not_supported")
        runtime = _runtimes.get(key)
        if runtime is not None and runtime.closed:
            raise LifecycleError("managed_work_root_closed")
        if (root / ".control" / "data-schema.json").exists() and runtime is None and not _managed:
            raise LifecycleError("managed_work_root_requires_supervisor")
        if existing is not None:
            return existing
        lease = WorkRootLock(root)
        _roots[key] = lease
        return lease


def get_runtime(root=None):
    with _registry_lock:
        runtime = _runtimes.get(str(root_path(root)))
        if runtime is not None and runtime.pid != os.getpid():
            raise LifecycleError("forked_managed_runtime_not_supported")
        return runtime


class ManagedStorageMeta(type):
    """Share authoritative storage only in an explicitly managed Web process."""
    def __call__(cls, *args, **kwargs):
        root = root_path(kwargs.get("work_path", args[0] if args else None))
        claim_work_root(root)
        runtime = get_runtime(root)
        if runtime is None:
            return super().__call__(*args, **kwargs)
        with runtime.io_lock:
            if runtime.closing or runtime.closed:
                raise LifecycleError("managed_storage_closing")
            if cls not in runtime.storages:
                instance = super().__call__(*args, **kwargs)
                runtime.register_storage(instance)
                runtime.storages[cls] = instance
            return runtime.storages[cls]


class RootRuntime:
    def __init__(self, root: Path, instance_id: str):
        self.root, self.instance_id = root, instance_id
        self.pid = os.getpid()
        self.io_lock = threading.RLock()
        self.activity_lock = threading.RLock()
        self.storages = {}
        self._registered = []
        self.activities = 0
        self.failure = None
        self.closing = False
        self.closed = False
        self.frozen = False
        self.state = None

    def register_storage(self, storage):
        with self.io_lock:
            if storage not in self._registered:
                self._registered.append(storage)

    def fail(self, code="managed_storage_failure"):
        with self.activity_lock:
            self.failure = code

    def reserve_background(self):
        with self.activity_lock:
            if self.closed or self.closing or self.frozen:
                raise LifecycleError("managed_runtime_not_accepting_background")
            self.activities += 1
        released = False
        def release():
            nonlocal released
            with self.activity_lock:
                if not released:
                    self.activities -= 1
                    released = True
        return release

    def active_background(self):
        with self.activity_lock:
            return self.activities

    def persistent_files(self):
        # Models/static/tools/logs are regenerable deployment inputs or telemetry.
        # They are excluded from the authoritative business snapshot contract.
        excluded = {".control", ".cache/models", ".cache/web_static", "tools", "logs", ".cache/logs"}
        def walk(directory):
            for path in sorted(directory.iterdir()):
                relative = path.relative_to(self.root).as_posix()
                if relative in excluded or relative.startswith(".cache/web_static-"):
                    continue
                if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                    raise LifecycleError("linked_persistent_path")
                if path.is_dir():
                    yield from walk(path)
                elif path.is_file():
                    if path.stat().st_nlink > 1:
                        raise LifecycleError("hardlinked_persistent_file")
                    yield path
                else:
                    raise LifecycleError("unsupported_persistent_path")
        yield from walk(self.root)

    def flush(self):
        with self.io_lock:
            if self.failure:
                raise LifecycleError(self.failure)
            if self.active_background():
                raise LifecycleError("background_work_not_drained")
            for storage in self._registered:
                storage.durable_flush()
            files = list(self.persistent_files())
            for path in files:
                durable_file(path)
            # Every intermediate directory entry must be durable too. Include
            # the root's ancestors: it may itself have just been mkdir(parents).
            directories = {self.root, *self.root.parents}
            for path in files:
                directories.update(path.parents)
            for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
                durable_directory(directory)
            return {"files": len(files), "storages": len(self._registered)}

    def close(self):
        with self.io_lock:
            if self.closed:
                return
            self.flush()
            self.closing = True
            for storage in reversed(self._registered):
                storage.close()
            self.closed = True


def _checked_managed_data_path(value) -> Path:
    """Inspect the *unresolved* chain, including dangling links and junctions.

    Do not resolve first: that would hide a reparse point in a parent directory.
    This is a preflight check, not a defence against a concurrent same-user
    attacker swapping directory entries; deployments still need private ACLs.
    """
    import stat

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise LifecycleError("managed_work_root_must_be_absolute")
    try:
        for ancestor in reversed((path, *path.parents)):
            try:
                info = ancestor.lstat()
            except FileNotFoundError:
                continue
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                raise LifecycleError("linked_managed_work_root")
        resolved = path.resolve()
        if resolved.exists() and not resolved.is_dir():
            raise LifecycleError("managed_work_root_not_directory")
        return resolved
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, LifecycleError):
            raise
        raise LifecycleError("managed_work_root_uninspectable") from None


def validate_managed_paths(root, launch_env: dict, code_file=None) -> Path:
    """Return the trusted code/slot root without creating files or taking locks.

    Source checkouts are identified from the loaded ``src/sirchmunk`` layout
    plus its repo-level pyproject marker, never from cwd. Installed deployments
    must explicitly anchor the complete interpreter environment (sys.prefix),
    not just site-packages or the package directory. SIRCHMUNK_RELEASE_ROOT is
    one immutable release slot, *not* the shared release-store directory.

    ``code_file`` is an inspection/test seam; production callers omit it so the
    boundary derives from the actually imported lifecycle module.
    """
    import sys

    data_root = _checked_managed_data_path(root)
    try:
        loaded = Path(code_file if code_file is not None else __file__)
        if not loaded.is_absolute():
            raise LifecycleError("managed_code_file_must_be_absolute")
        loaded = loaded.resolve(strict=True)
        if not loaded.is_file():
            raise LifecycleError("managed_code_file_not_regular")

        source_root = None
        for package in loaded.parents:
            if package.name == "sirchmunk" and package.parent.name == "src":
                candidate = package.parent.parent
                if (candidate / "pyproject.toml").is_file():
                    source_root = candidate
                    break

        anchor_value = launch_env.get("SIRCHMUNK_RELEASE_ROOT")
        if source_root is not None:
            detected_root = source_root
        else:
            if not anchor_value:
                raise LifecycleError("managed_installed_release_root_required")
            detected_root = Path(sys.prefix).resolve(strict=True)
            if not loaded.is_relative_to(detected_root):
                raise LifecycleError("managed_installed_code_outside_prefix")

        code_root = detected_root
        if anchor_value is not None:
            anchor = Path(anchor_value)
            if not anchor.is_absolute():
                raise LifecycleError("managed_release_root_must_be_absolute")
            anchor = anchor.resolve(strict=True)
            if not anchor.is_dir() or not loaded.is_relative_to(anchor):
                raise LifecycleError("managed_release_root_does_not_contain_code")
            if not detected_root.is_relative_to(anchor):
                raise LifecycleError("managed_release_root_too_narrow")
            code_root = anchor

        if (data_root == Path(data_root.anchor) or data_root.is_relative_to(code_root)
                or code_root.is_relative_to(data_root)):
            raise LifecycleError("managed_data_must_be_separate_from_code")
        return code_root
    except (OSError, RuntimeError) as exc:
        if isinstance(exc, LifecycleError):
            raise
        raise LifecycleError("managed_code_root_uninspectable") from None


def bootstrap_managed_root(launch_env: dict) -> RootRuntime | None:
    if (launch_env.get("SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED") or "false").lower() != "true":
        return None
    if (launch_env.get("SIRCHMUNK_MANAGEMENT_ENABLED") or "false").lower() != "true":
        raise LifecycleError("lifecycle_requires_management")
    if not launch_env.get("SIRCHMUNK_WORK_PATH"):
        raise LifecycleError("managed_work_root_must_be_explicit")
    instance = launch_env.get("SIRCHMUNK_MANAGEMENT_INSTANCE_ID") or ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", instance):
        raise LifecycleError("managed_instance_required")
    # Validate the original path before root_path resolves away linked parents,
    # and before claim_work_root can create .control or any persistent marker.
    validate_managed_paths(launch_env["SIRCHMUNK_WORK_PATH"], launch_env)
    root = root_path(launch_env["SIRCHMUNK_WORK_PATH"])
    claim_work_root(root, _managed=True)
    for path in (root / ".control").iterdir():
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise LifecycleError("linked_management_state")
        if path.is_file() and path.stat().st_nlink > 1:
            raise LifecycleError("hardlinked_management_state")
    marker = root / ".control" / "data-schema.json"
    expected = {
        "schema_id": DATA_SCHEMA, "instance_id": instance,
        "embedding_dimension": int(os.getenv("EMBEDDING_DIM", "384")),
        "embedding_model": os.getenv("EMBEDDING_MODEL_ID", ""),
    }
    if not 1 <= expected["embedding_dimension"] <= 65536:
        raise LifecycleError("invalid_embedding_dimension")
    if marker.exists():
        try:
            saved = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            raise LifecycleError("invalid_managed_schema_marker") from None
        if (type(saved) is not dict or set(saved) != set(expected)
                or any(type(saved[key]) is not type(value) or saved[key] != value for key, value in expected.items())):
            raise LifecycleError("managed_schema_or_identity_mismatch")
    else:
        write_json_atomic(marker, expected)
    runtime = RootRuntime(root, instance)
    # Refuse links before any storage constructor follows them to another root.
    list(runtime.persistent_files())
    with _registry_lock:
        if str(root) in _runtimes:
            raise LifecycleError("managed_root_already_bootstrapped")
        _runtimes[str(root)] = runtime
    return runtime


def schema_digest(root: Path) -> str:
    return hashlib.sha256((root / ".control" / "data-schema.json").read_bytes()).hexdigest()


@atexit.register
def _release_root_locks():
    # Registered before storage shutdown hooks; atexit executes LIFO.
    for lease in tuple(_roots.values()):
        if lease.pid == os.getpid():
            lease.close()
