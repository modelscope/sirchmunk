#!/usr/bin/env python3
"""Generate unsigned sirchmunk-release-build.v1 metadata from a fixed Git commit.

Usage (Python >= 3.11, Git on PATH):
    python scripts/build_release_metadata.py --repo REPO --commit FULL_SHA
    python scripts/build_release_metadata.py --repo REPO --commit FULL_SHA --output NEW_FILE

No checkout, imports of product code, packaging, provider IO, signature or Git
mutation. Files are raw Git blobs (no smudge filters, CRLF conversion or replace
objects). Runtime/credential paths in a commit are rejected before blob reads;
untracked/dirty worktree contents are never consulted. This is build provenance,
not a signature or proof that dependencies/native wheels have been installed.

source_sha256 is SHA256 of the UTF-8 canonical JSON files array (sorted paths,
sorted object keys, separators=(',', ':'), ensure_ascii=False). The ABI is the
actual builder, so platform overrides/cross-build claims are deliberately absent.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import struct
import subprocess
import sys
import sysconfig


class MetadataError(ValueError):
    pass


def _git(repo: Path, *args: str, payload: bytes | None = None) -> bytes:
    # Ignore inherited Git redirection/config injection and replacement refs.
    # Missing promisor objects must fail offline, never trigger a lazy fetch.
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1",
               GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_ALLOW_PROTOCOL="")
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(repo), *args], input=payload,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise MetadataError("git object query unavailable or timed out") from None
    if result.returncode:
        # Git errors may contain local repository paths. Do not publish them.
        raise MetadataError("git object query failed; require a complete local commit")
    return result.stdout


def _check_source_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (not path or parsed.is_absolute() or str(parsed) != path
            or any(part in (".", "..") for part in parsed.parts)
            or any(char in path for char in "\\:\x00\r\n\t")):
        raise MetadataError("non-portable source path in commit")
    parts = tuple(part.lower() for part in parsed.parts)
    private_dirs = {".git", ".cache", ".control", ".venv", "__pycache__",
                    "node_modules", ".pytest_cache", ".next"}
    runtime_roots = {"data", "work", "sirchmunk-work", "knowledge_base", "knowledge-base",
                     "backups", "logs", "venv"}
    name = parts[-1]
    env_file = (name in {".env", ".envrc"} or name.startswith(".env.")) and not name.endswith((".example", ".sample"))
    private_file = name in {"credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
    if (private_dirs.intersection(parts) or parts[0] in runtime_roots or env_file or private_file
            or parsed.suffix.lower() in {".db", ".duckdb", ".parquet", ".sqlite", ".sqlite3",
                                         ".wal", ".pyc", ".pem", ".key"}):
        raise MetadataError("runtime or credential path in commit; refusing blob reads")


def _tree_entries(repo: Path, commit: str):
    entries = []
    paths = set()
    for row in _git(repo, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not row:
            continue
        try:
            header, raw_path = row.split(b"\t", 1)
            mode, kind, oid = header.decode("ascii").split(" ")
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeError):
            raise MetadataError("invalid Git tree entry") from None
        _check_source_path(path)
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise MetadataError("release tree contains a symlink, submodule or unsupported mode")
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid) or path in paths:
            raise MetadataError("invalid or duplicate Git object entry")
        paths.add(path)
        entries.append((path, mode, oid))
    if not entries:
        raise MetadataError("empty source commit")
    return sorted(entries)


def _raw_blobs(repo: Path, entries):
    # Bounded batches avoid one Git process per file and avoid buffering the
    # complete source tree. Binary payloads are framed by byte count, not lines.
    for start in range(0, len(entries), 32):
        batch = entries[start:start + 32]
        output = _git(repo, "cat-file", "--batch",
                      payload=("\n".join(oid for _path, _mode, oid in batch) + "\n").encode("ascii"))
        offset = 0
        for path, mode, oid in batch:
            end = output.find(b"\n", offset)
            if end < 0:
                raise MetadataError("truncated Git blob header")
            try:
                actual_oid, kind, length = output[offset:end].decode("ascii").split(" ")
                size = int(length)
            except (ValueError, UnicodeError):
                raise MetadataError("unreadable Git blob") from None
            if actual_oid != oid or kind != "blob" or size < 0:
                raise MetadataError("Git blob identity mismatch")
            begin, offset = end + 1, end + 2 + size
            data = output[begin:begin + size]
            if len(data) != size or output[begin + size:offset] != b"\n":
                raise MetadataError("truncated Git blob")
            git_hash = hashlib.sha1 if len(oid) == 40 else hashlib.sha256
            if git_hash(b"blob " + str(size).encode("ascii") + b"\0" + data).hexdigest() != oid:
                raise MetadataError("Git blob hash mismatch")
            yield path, mode, data
        if offset != len(output):
            raise MetadataError("unexpected Git batch output")


def _literal_string(source: bytes, name: str) -> str:
    """Parse, never execute, the committed version/schema source."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeError):
        raise MetadataError("invalid committed Python metadata source") from None
    values = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                raise MetadataError("metadata requires a literal string assignment")
            values.append(node.value.value)
    if len(values) != 1:
        raise MetadataError("metadata requires exactly one literal assignment")
    return values[0]


def _abi_token(value):
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.+\-]+", value):
        raise MetadataError("unrecognized or non-portable build ABI value")
    return value


def _build_abi():
    bits = struct.calcsize("P") * 8
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "x86_64": "x86_64", "aarch64": "arm64", "arm64": "arm64",
            "i386": "x86", "i686": "x86", "x86": "x86"}.get(machine, machine)
    if bits == 32 and arch == "x86_64":
        arch = "x86"
    return dict(os=_abi_token(platform.system().lower()), arch=_abi_token(arch),
                python_version=".".join(map(str, sys.version_info[:3])),
                implementation=_abi_token(sys.implementation.name),
                cache_tag=_abi_token(sys.implementation.cache_tag),
                soabi=_abi_token(sysconfig.get_config_var("SOABI")),
                extension_suffix=_abi_token(sysconfig.get_config_var("EXT_SUFFIX")),
                platform_tag=_abi_token(sysconfig.get_platform()), pointer_bits=bits)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _validate_python_range(value):
    """Validate/evaluate numeric Requires-Python clauses without packaging deps.

    Reject unsupported prerelease/arbitrary-equality syntax, not approximate it.
    The product's ordinary PEP 440 numeric ranges and ==/!= wildcards work here.
    """
    if not isinstance(value, str) or not value.strip():
        raise MetadataError("missing Python compatibility expression")
    current = tuple(sys.version_info[:3])
    for clause in value.split(","):
        match = re.fullmatch(r"\s*(~=|==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+){0,2})(\.\*)?\s*", clause)
        if not match:
            raise MetadataError("unsupported Python compatibility expression")
        operator, release, wildcard = match.groups()
        version = tuple(map(int, release.split(".")))
        target = version + (0,) * (3 - len(version))
        if wildcard:
            if operator not in {"==", "!="}:
                raise MetadataError("unsupported Python compatibility wildcard")
            equal = current[:len(version)] == version
            accepted = equal if operator == "==" else not equal
        elif operator == "~=":
            if len(version) < 2:
                raise MetadataError("invalid compatible Python release")
            prefix = version[:-1]
            upper = (*prefix[:-1], prefix[-1] + 1) + (0,) * (3 - len(prefix))
            accepted = target <= current < upper
        else:
            accepted = {"==": current == target, "!=": current != target,
                        "<=": current <= target, ">=": current >= target,
                        "<": current < target, ">": current > target}[operator]
        if not accepted:
            raise MetadataError("builder Python does not satisfy committed requires-python")


def build_metadata(repo, commit: str) -> dict:
    if sys.version_info < (3, 11):
        raise MetadataError("metadata builder requires Python 3.11+ for stdlib TOML parsing")
    import tomllib

    if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit):
        raise MetadataError("source commit must be a full 40/64-hex Git commit SHA")
    commit = commit.lower()
    repo = Path(repo).resolve(strict=True)
    if not repo.is_dir():
        raise MetadataError("repository must be a directory")
    if _git(repo, "cat-file", "-t", commit).strip() != b"commit":
        raise MetadataError("source object must be a commit, not a tag or tree")
    entries = _tree_entries(repo, commit)  # Validate every path before reading any blob.
    required = {"src/sirchmunk/version.py", "src/sirchmunk/storage/lifecycle.py", "pyproject.toml"}
    if not required.issubset(path for path, _mode, _oid in entries):
        raise MetadataError("commit lacks required product metadata sources")
    sources, files = {}, []
    for path, mode, data in _raw_blobs(repo, entries):
        files.append(dict(path=path, mode=mode, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if path in required:
            sources[path] = data
    version = _literal_string(sources["src/sirchmunk/version.py"], "__version__")
    if not re.fullmatch(r"[0-9][A-Za-z0-9.!+_\-]*", version):
        raise MetadataError("invalid source version")
    schema = _literal_string(sources["src/sirchmunk/storage/lifecycle.py"], "DATA_SCHEMA")
    if schema != "sirchmunk-data-v1":
        raise MetadataError("unsupported data schema; migrations require an explicit release contract")
    try:
        project = tomllib.loads(sources["pyproject.toml"].decode("utf-8"))["project"]
        requires_python = project["requires-python"]
    except (ValueError, KeyError, TypeError):
        raise MetadataError("invalid committed pyproject metadata") from None
    if project.get("name") != "sirchmunk" or not isinstance(requires_python, str):
        raise MetadataError("incorrect product identity or missing Python compatibility")
    _validate_python_range(requires_python)
    abi = _build_abi()
    return dict(schema="sirchmunk-release-build.v1", product="sirchmunk", version=version,
                source_commit=commit, source_sha256=hashlib.sha256(canonical(files)).hexdigest(),
                data_schema=schema, migrations="none",
                compatibility=dict(os=[abi["os"]], arch=[abi["arch"]], python=requires_python),
                build_abi=abi, files=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="local Git repository (a bare repository is supported)")
    parser.add_argument("--commit", required=True, help="full immutable commit SHA, never a branch/ref")
    parser.add_argument("--output", help="create a new JSON file; default is stdout; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        payload = canonical(build_metadata(args.repo, args.commit)) + b"\n"
        if args.output:
            with Path(args.output).open("xb") as output:
                output.write(payload)
        else:
            sys.stdout.buffer.write(payload)
        return 0
    except MetadataError as exc:
        # These messages are fixed, path/data-free codes; never print Git stderr.
        print(f"release metadata failed: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("release metadata failed: filesystem IO; output is not a validated artifact", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
