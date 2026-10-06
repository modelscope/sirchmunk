"""Read-only release/data preflight and fixed-object build metadata tests.

All code layouts, Git repositories and writable roots are synthetic temp data.
No provider, real deployment root, installed package tree or production port IO.
"""
import os
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    monkeypatch.delenv("SIRCHMUNK_RELEASE_ROOT", raising=False)


@pytest.fixture
def lifecycle():
    from sirchmunk.storage import lifecycle
    return lifecycle


def source_layout(base):
    repo = base / "repo"
    code = repo / "src/sirchmunk/storage/lifecycle.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic loaded code\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text('[project]\nname="sirchmunk"\n', encoding="utf-8")
    return repo, code


def installed_layout(base, monkeypatch):
    prefix = base / "venv"
    relative = "Lib/site-packages" if os.name == "nt" else "lib/python3.12/site-packages"
    code = prefix / relative / "sirchmunk/storage/lifecycle.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic installed code\n", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    return prefix, code


def launch(root, **extra):
    return dict(SIRCHMUNK_WORK_PATH=str(root), SIRCHMUNK_MANAGEMENT_ENABLED="true",
                SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED="true",
                SIRCHMUNK_MANAGEMENT_INSTANCE_ID="release-boundary-test", **extra)


@pytest.mark.parametrize("relative", ["data", "src/data", "docs/state", ".cache/models", "."])
def test_source_rejects_data_anywhere_in_repo_without_writing(tmp_path, lifecycle, relative):
    repo, code = source_layout(tmp_path)
    before = set(tmp_path.rglob("*"))
    with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
        lifecycle.validate_managed_paths(repo / relative, {}, code_file=code)
    assert set(tmp_path.rglob("*")) == before


def test_source_rejects_data_ancestor_containing_code(tmp_path, lifecycle):
    _repo, code = source_layout(tmp_path)
    with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
        lifecycle.validate_managed_paths(tmp_path, {}, code_file=code)


def test_source_detection_ignores_cwd_and_ambient_anchor(tmp_path, lifecycle, monkeypatch):
    repo, code = source_layout(tmp_path / "loaded")
    unrelated, _ = source_layout(tmp_path / "cwd")
    monkeypatch.chdir(unrelated)
    monkeypatch.setenv("SIRCHMUNK_RELEASE_ROOT", str(unrelated))
    work = tmp_path / "absent/work"
    assert lifecycle.validate_managed_paths(work, {}, code_file=code) == repo
    assert not work.parent.exists()


def test_explicit_slot_must_cover_whole_repo_and_excludes_slot_data(tmp_path, lifecycle):
    slot = tmp_path / "slots/v1"
    _repo, code = source_layout(slot)
    env = {"SIRCHMUNK_RELEASE_ROOT": str(slot)}
    assert lifecycle.validate_managed_paths(tmp_path / "work", env, code) == slot
    with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
        lifecycle.validate_managed_paths(slot / "data", env, code)
    # A different slot is not implicitly part of this slot's boundary.
    assert lifecycle.validate_managed_paths(tmp_path / "slots/v2/data", env, code) == slot


@pytest.mark.parametrize("kind", ["relative", "empty", "wrong", "missing", "src", "package", "file"])
def test_invalid_source_anchors_rejected_without_writes(tmp_path, lifecycle, kind):
    repo, code = source_layout(tmp_path)
    wrong = tmp_path / "unrelated"
    wrong.mkdir()
    anchors = {"relative": "repo", "empty": "", "wrong": str(wrong),
               "missing": str(tmp_path / "nonexistent"), "src": str(repo / "src"),
               "package": str(code.parent.parent), "file": str(code)}
    before = set(tmp_path.rglob("*"))
    with pytest.raises(lifecycle.LifecycleError):
        lifecycle.validate_managed_paths(tmp_path / "work", {"SIRCHMUNK_RELEASE_ROOT": anchors[kind]}, code)
    assert set(tmp_path.rglob("*")) == before


def test_source_requires_marker_at_actual_loaded_repo_not_cwd(tmp_path, lifecycle, monkeypatch):
    repo, code = source_layout(tmp_path / "loaded")
    (repo / "pyproject.toml").unlink()
    cwd_repo, _ = source_layout(tmp_path / "cwd")
    monkeypatch.chdir(cwd_repo)
    with pytest.raises(lifecycle.LifecycleError, match="installed_release_root_required"):
        lifecycle.validate_managed_paths(tmp_path / "work", {}, code)


def test_installed_layout_requires_explicit_complete_environment_anchor(tmp_path, lifecycle, monkeypatch):
    prefix, code = installed_layout(tmp_path, monkeypatch)
    with pytest.raises(lifecycle.LifecycleError, match="installed_release_root_required"):
        lifecycle.validate_managed_paths(tmp_path / "work", {}, code)
    env = {"SIRCHMUNK_RELEASE_ROOT": str(prefix)}
    assert lifecycle.validate_managed_paths(tmp_path / "work", env, code) == prefix
    for root in (prefix / "data", code.parents[2] / "data"):
        with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
            lifecycle.validate_managed_paths(root, env, code)
        assert not root.exists()
    for narrow in (code.parent.parent, code.parents[2]):
        with pytest.raises(lifecycle.LifecycleError, match="too_narrow"):
            lifecycle.validate_managed_paths(tmp_path / "work", {"SIRCHMUNK_RELEASE_ROOT": str(narrow)}, code)


def test_installed_slot_may_be_wider_than_prefix(tmp_path, lifecycle, monkeypatch):
    slot = tmp_path / "release"
    _prefix, code = installed_layout(slot, monkeypatch)
    env = {"SIRCHMUNK_RELEASE_ROOT": str(slot)}
    assert lifecycle.validate_managed_paths(tmp_path / "work", env, code) == slot
    with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
        lifecycle.validate_managed_paths(slot / "data", env, code)


def test_installed_code_outside_current_prefix_is_rejected(tmp_path, lifecycle, monkeypatch):
    _prefix, code = installed_layout(tmp_path / "external", monkeypatch)
    own_prefix = tmp_path / "interpreter"
    own_prefix.mkdir()
    monkeypatch.setattr(sys, "prefix", str(own_prefix))
    with pytest.raises(lifecycle.LifecycleError, match="installed_code_outside_prefix"):
        lifecycle.validate_managed_paths(tmp_path / "work", {"SIRCHMUNK_RELEASE_ROOT": str(tmp_path)}, code)


@pytest.mark.parametrize("missing_target", [False, True])
def test_root_ancestor_symlink_rejected_before_resolve(tmp_path, lifecycle, missing_target):
    _repo, code = source_layout(tmp_path)
    target = tmp_path / "target"
    if not missing_target:
        target.mkdir()
    link = tmp_path / "alias"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not permit unprivileged symlinks")
    for root in (link, link / "work", link / ".." / "work"):
        with pytest.raises(lifecycle.LifecycleError, match="linked_managed_work_root"):
            lifecycle.validate_managed_paths(root, {}, code)
    assert not (target / "work").exists()


def test_root_ancestor_reparse_attribute_rejected_on_all_platforms(tmp_path, lifecycle, monkeypatch):
    _repo, code = source_layout(tmp_path)
    junction = tmp_path / "junction"
    junction.mkdir()
    original = Path.lstat
    def fake_reparse(path, *args, **kwargs):
        if path == junction:
            return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", fake_reparse)
    with pytest.raises(lifecycle.LifecycleError, match="linked_managed_work_root"):
        lifecycle.validate_managed_paths(junction / "work", {}, code)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behaviour")
def test_real_windows_junction_is_rejected(tmp_path, lifecycle):
    _repo, code = source_layout(tmp_path)
    target, junction = tmp_path / "target", tmp_path / "junction"
    target.mkdir()
    command = "New-Item -ItemType Junction -Path '{}' -Target '{}' | Out-Null".format(
        str(junction).replace("'", "''"), str(target).replace("'", "''"))
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                            capture_output=True, timeout=15)
    assert result.returncode == 0
    try:
        with pytest.raises(lifecycle.LifecycleError, match="linked_managed_work_root"):
            lifecycle.validate_managed_paths(junction / "work", {}, code)
    finally:
        junction.rmdir()  # Remove only the link, not the target directory.
    assert target.is_dir() and not (target / "work").exists()


def test_relative_or_file_work_root_rejected(tmp_path, lifecycle):
    _repo, code = source_layout(tmp_path)
    with pytest.raises(lifecycle.LifecycleError, match="must_be_absolute"):
        lifecycle.validate_managed_paths("relative-work", {}, code)
    path = tmp_path / "file"
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(lifecycle.LifecycleError, match="not_directory"):
        lifecycle.validate_managed_paths(path, {}, code)
    assert path.read_text(encoding="utf-8") == "preserve"


def test_bootstrap_preflights_fake_source_before_claiming_or_writing(tmp_path, lifecycle, monkeypatch):
    repo, code = source_layout(tmp_path)
    monkeypatch.setattr(lifecycle, "__file__", str(code))
    invalid = repo / "data"
    with pytest.raises(lifecycle.LifecycleError, match="separate_from_code"):
        lifecycle.bootstrap_managed_root(launch(invalid))
    assert not invalid.exists() and str(invalid) not in lifecycle._roots
    root = tmp_path / "valid-work"
    try:
        runtime = lifecycle.bootstrap_managed_root(launch(root))
        assert runtime.root == root
    finally:
        lifecycle._runtimes.pop(str(root), None)
        lease = lifecycle._roots.pop(str(root), None)
        if lease:
            lease.close()


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_release_metadata.py"
    spec = importlib.util.spec_from_file_location("isolated_release_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo, *args, payload=None):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
    result = subprocess.run(["git", "-C", str(repo), *args], input=payload,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=15)
    assert result.returncode == 0, "synthetic Git fixture command failed"
    return result.stdout


@pytest.fixture
def committed(tmp_path):
    repo = tmp_path / "git-source"
    repo.mkdir()
    git(repo, "init", "--quiet")
    git(repo, "config", "user.name", "Isolated Builder")
    git(repo, "config", "user.email", "builder@example.invalid")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "commit.gpgsign", "false")
    payloads = {
        "src/sirchmunk/version.py": b'__version__ = "0.2.0"\r\nraise RuntimeError("never execute source")\r\n',
        "src/sirchmunk/storage/lifecycle.py": b'DATA_SCHEMA = "sirchmunk-data-v1"\n',
        "pyproject.toml": b'[project]\nname = "sirchmunk"\nrequires-python = ">=3.10"\n',
        "assets/bytes.bin": b"\x00\xffbinary\r\n\x00\n",
        "assets/text.txt": b"Git LF bytes\nsecond line\n",
    }
    for relative, payload in payloads.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "synthetic fixed source")
    sha = git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    return repo, sha, payloads


def commit_fixture(repo):
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "-m", "synthetic next source")
    return git(repo, "rev-parse", "HEAD").decode("ascii").strip()


def test_metadata_fixed_sha_uses_raw_bytes_not_worktree_or_head(builder, committed, tmp_path):
    repo, sha, payloads = committed
    expected = builder.build_metadata(repo, sha)
    assert expected["schema"] == "sirchmunk-release-build.v1"
    assert expected["product"] == "sirchmunk" and expected["version"] == "0.2.0"
    assert expected["source_commit"] == sha
    assert expected["data_schema"] == "sirchmunk-data-v1" and expected["migrations"] == "none"
    for item in expected["files"]:
        assert item["sha256"] == hashlib.sha256(payloads[item["path"]]).hexdigest()
        assert item["size"] == len(payloads[item["path"]])
        assert not Path(item["path"]).is_absolute()
    assert expected["source_sha256"] == hashlib.sha256(builder.canonical(expected["files"])).hexdigest()
    assert str(tmp_path) not in builder.canonical(expected).decode("utf-8")
    abi = expected["build_abi"]
    assert abi["python_version"] == ".".join(map(str, sys.version_info[:3]))
    assert abi["cache_tag"] == sys.implementation.cache_tag
    assert expected["compatibility"] == dict(os=[abi["os"]], arch=[abi["arch"]], python=">=3.10")

    # Even advancing HEAD to a different release cannot change a fixed build.
    (repo / "src/sirchmunk/version.py").write_bytes(b'__version__ = "99.0.0"\n')
    (repo / "new-file.txt").write_text("not in old release", encoding="utf-8")
    commit_fixture(repo)
    git(repo, "config", "core.autocrlf", "true")
    (repo / "assets/text.txt").write_bytes(b"Git LF bytes\r\nsecond line\r\n")
    (repo / "src/sirchmunk/version.py").write_text("broken dirty source", encoding="utf-8")
    assert builder.build_metadata(repo, sha) == expected


def test_metadata_never_reads_untracked_runtime_files(builder, committed, monkeypatch):
    repo, sha, _payloads = committed
    for relative in (".env", ".cache/models/model", ".cache/knowledge/knowledge.parquet", ".control/state.db"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private test value must not be read")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("metadata must not read worktree files")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    result = builder.build_metadata(repo, sha)
    assert all(not item["path"].startswith((".env", ".cache", ".control")) for item in result["files"])
    assert b"private test value" not in builder.canonical(result)


@pytest.mark.parametrize("bad_commit", ["HEAD", "main", "abcd1234", "-h", "HEAD~1", "g" * 40])
def test_metadata_requires_full_commit_sha_before_git_access(builder, tmp_path, monkeypatch, bad_commit):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid commit must not invoke Git")
    monkeypatch.setattr(builder, "_git", forbidden)
    with pytest.raises(builder.MetadataError, match="full 40/64-hex"):
        builder.build_metadata(tmp_path, bad_commit)


def test_metadata_rejects_full_sha_for_non_commit_object(builder, committed):
    repo, _sha, _payloads = committed
    tree = git(repo, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    with pytest.raises(builder.MetadataError, match="must be a commit"):
        builder.build_metadata(repo, tree)


def test_metadata_ignores_git_replace_and_inherited_git_redirect(builder, committed, monkeypatch, tmp_path):
    repo, sha, _payloads = committed
    version_oid = git(repo, "rev-parse", f"{sha}:src/sirchmunk/version.py").decode("ascii").strip()
    substitute = git(repo, "hash-object", "-w", "--stdin", payload=b'__version__ = "666.0.0"\n').decode("ascii").strip()
    git(repo, "replace", version_oid, substitute)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "wrong-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wrong-worktree"))
    monkeypatch.chdir(tmp_path)
    assert builder.build_metadata(repo, sha)["version"] == "0.2.0"


@pytest.mark.parametrize("relative", [".env", "config/.env.production", ".cache/knowledge/data.parquet",
                                      "data/chat_sessions.db", "credentials.json"])
def test_metadata_rejects_committed_runtime_paths_before_blob_reads(builder, committed, monkeypatch, relative):
    repo, _sha, _payloads = committed
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"must not be read even from Git")
    sha = commit_fixture(repo)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("private path detection must precede every blob read")
    monkeypatch.setattr(builder, "_raw_blobs", forbidden)
    with pytest.raises(builder.MetadataError, match="runtime or credential"):
        builder.build_metadata(repo, sha)


@pytest.mark.parametrize("mode", ["120000", "160000"])
def test_metadata_rejects_symlink_and_submodule_tree_modes(builder, committed, mode):
    repo, sha, _payloads = committed
    oid = sha if mode == "160000" else git(repo, "hash-object", "-w", "--stdin", payload=b"../outside").decode("ascii").strip()
    git(repo, "update-index", "--add", "--cacheinfo", f"{mode},{oid},external")
    git(repo, "commit", "--quiet", "-m", "unsupported tree mode")
    bad_sha = git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    with pytest.raises(builder.MetadataError, match="symlink, submodule"):
        builder.build_metadata(repo, bad_sha)


@pytest.mark.parametrize("relative,payload,reason", [
    ("src/sirchmunk/version.py", b'__version__ = str("1.0.0")\n', "literal"),
    ("src/sirchmunk/version.py", b'__version__ = "1"\n__version__ = "2"\n', "exactly one"),
    ("src/sirchmunk/storage/lifecycle.py", b'DATA_SCHEMA = "sirchmunk-data-v9"\n', "unsupported data schema"),
    ("pyproject.toml", b'[project]\nname="other-product"\nrequires-python=">=3.10"\n', "product identity"),
])
def test_metadata_rejects_ambiguous_or_incompatible_identity(builder, committed, relative, payload, reason):
    repo, _sha, _payloads = committed
    (repo / relative).write_bytes(payload)
    sha = commit_fixture(repo)
    with pytest.raises(builder.MetadataError, match=reason):
        builder.build_metadata(repo, sha)


def test_metadata_cli_outputs_json_and_refuses_overwrite(builder, committed, tmp_path):
    repo, sha, _payloads = committed
    command = [sys.executable, builder.__file__, "--repo", str(repo), "--commit", sha]
    result = subprocess.run(command, capture_output=True, timeout=30)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout)["source_commit"] == sha
    destination = tmp_path / "release-build.json"
    result = subprocess.run([*command, "--output", str(destination)], capture_output=True, timeout=30)
    assert result.returncode == 0
    saved = destination.read_bytes()
    assert json.loads(saved)["version"] == "0.2.0"
    result = subprocess.run([*command, "--output", str(destination)], capture_output=True, timeout=30)
    assert result.returncode != 0 and destination.read_bytes() == saved
    assert str(tmp_path).encode() not in result.stderr


@pytest.mark.parametrize("value,allowed", [
    (">=3.10, <4", True), ("~=3.10", True), ("==3.*", True), ("!=2.*", True),
    (">=99", False), ("!=3.*", False), (">=3.*", False), (">", False),
])
def test_python_compatibility_checks_actual_builder(builder, value, allowed):
    if allowed:
        builder._validate_python_range(value)
    else:
        with pytest.raises(builder.MetadataError):
            builder._validate_python_range(value)


def test_metadata_abi_cannot_leak_private_absolute_paths(builder, monkeypatch):
    monkeypatch.setattr(builder.sysconfig, "get_config_var", lambda _key: "/private/build/interpreter")
    with pytest.raises(builder.MetadataError, match="build ABI"):
        builder._build_abi()
