"""Managed evolver ownership/durability, with fake LLM/embed and isolated roots."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    # No Sirchmunk import may precede the isolated work/cache environment.
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    for key in ("EMBEDDING_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME", "MODELSCOPE_CACHE",
                "TORCH_HOME", "HF_MODULES_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR",
                "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.setenv(key, str(root / ".cache/models" / key.lower()))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("This suite must not load models or contact providers")

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=forbidden))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=forbidden))
    from sirchmunk.llm.openai_chat import OpenAIChat
    monkeypatch.setattr(OpenAIChat, "__init__", forbidden)
    yield root


@pytest.fixture
def module(isolated):
    from sirchmunk.scheduler import evolver
    return evolver


@pytest.fixture
def managed(isolated):
    from sirchmunk.storage import lifecycle
    runtime = lifecycle.bootstrap_managed_root({
        "SIRCHMUNK_WORK_PATH": str(isolated), "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "evolver-test",
    })
    yield runtime
    assert runtime.active_background() == 0
    # Failure injection must not retry a failed durability barrier in teardown.
    for storage in runtime._registered:
        db = getattr(storage, "db", None)
        if db is not None and db.connection is not None:
            db.connection.close()
            db.connection = None
    lifecycle._runtimes.pop(str(isolated.resolve()), None)
    lease = lifecycle._roots.pop(str(isolated.resolve()), None)
    if lease:
        lease.close()


def cluster(identifier):
    from sirchmunk.schema.knowledge import KnowledgeCluster
    now = datetime.now(timezone.utc)
    return KnowledgeCluster(id=identifier, name=identifier, description="test",
                            content="test content", queries=["test query"],
                            create_time=now, last_modified=now, related_clusters=[])


def storage_for(*clusters):
    records = {item.id: item for item in clusters}
    return SimpleNamespace(
        get=AsyncMock(side_effect=records.get),
        get_with_embedding=AsyncMock(side_effect=lambda key: (records.get(key), [1., 0., 0., 0.])),
        update=AsyncMock(return_value=True), insert=AsyncMock(return_value=True),
        store_embedding=AsyncMock(return_value=True), merge=AsyncMock(return_value=next(iter(records.values()), None)),
        combine_cluster_fields=lambda values: "\n".join(values),
        search_similar_clusters=AsyncMock(return_value=[]), table_name="knowledge_clusters",
        db=SimpleNamespace(fetch_one=Mock(return_value=(0,)), execute=Mock(), fetch_all=Mock(return_value=[])),
    )


def make_evolver(module, root, storage=None):
    llm = SimpleNamespace(achat=AsyncMock(return_value=SimpleNamespace(
        content='{"query":"refined query","content":"refined content"}')))
    embedding = SimpleNamespace(model_id="fake-local", embed=AsyncMock(return_value=[[1., 0., 0., 0.]]))
    result = module.KnowledgeEvolver(llm, embedding, storage or storage_for(), root)
    result._log = SimpleNamespace(info=AsyncMock(), warning=AsyncMock(), error=AsyncMock())
    return result


def read_manifest(evolver):
    return json.loads(evolver._manifest_path.read_text(encoding="utf-8"))


class CheckedLock:
    """Portable ownership assertion for the real runtime RLock contract."""
    def __init__(self):
        self.lock = threading.RLock()
        self.owner = None
        self.depth = 0

    def __enter__(self):
        self.lock.acquire()
        self.owner = threading.get_ident()
        self.depth += 1
        return self

    def __exit__(self, *_exc):
        self.depth -= 1
        if not self.depth:
            self.owner = None
        self.lock.release()

    def assert_owned(self):
        assert self.owner == threading.get_ident() and self.depth > 0


def test_strict_roundtrip_preserves_v1_and_repeated_ids(module):
    manifest = module.EvolveManifest(current_step=2, cluster_ids_buffer=["a", "a"],
                                     last_evolve_at="2026-09-30T00:00:00+00:00")
    assert module.EvolveManifest.from_json(manifest.to_json(), strict=True) == manifest
    assert module.EvolveManifest.from_json("{}") == module.EvolveManifest()


@pytest.mark.parametrize("field,value", [
    ("version", "2.0"), ("version", 1.0), ("version", None),
    ("current_step", True), ("current_step", -1), ("current_step", 1.0),
    ("last_meta_detection_step", "1"), ("last_meta_detection_cluster_count", None),
    ("last_global_update_step", []), ("last_global_update_cluster_count", False),
    ("cluster_ids_buffer", {}), ("cluster_ids_buffer", "a"), ("cluster_ids_buffer", [1]),
    ("connect_merge_cluster_ids", [None]), ("edge_refresh_cluster_ids", [["a"]]),
    ("last_evolve_at", 1), ("last_evolve_at", {}),
])
def test_strict_rejects_wrong_field_types(module, field, value):
    data = asdict(module.EvolveManifest())
    data[field] = value
    with pytest.raises(ValueError):
        module.EvolveManifest.from_json(json.dumps(data), strict=True)


@pytest.mark.parametrize("kind", ["missing", "unknown", "duplicate", "nan", "array", "null"])
def test_strict_rejects_unknown_or_incomplete_format(module, kind):
    data = asdict(module.EvolveManifest())
    if kind == "missing":
        del data["edge_refresh_cluster_ids"]
    elif kind == "unknown":
        data["other_queue"] = []
    payload = json.dumps(data)
    if kind == "duplicate":
        payload = payload[:-1] + ', "version": "1.0"}'
    elif kind == "nan":
        payload = payload.replace('"current_step": 0', '"current_step": NaN')
    elif kind == "array":
        payload = "[]"
    elif kind == "null":
        payload = "null"
    with pytest.raises(ValueError):
        module.EvolveManifest.from_json(payload, strict=True)


def test_constructor_threads_share_authority_and_lock_without_reloading(module, managed, monkeypatch):
    lock = CheckedLock()
    managed.io_lock = lock
    loads = []
    original_load = module.KnowledgeEvolver._load_manifest

    def checked_load(self):
        lock.assert_owned()
        loads.append(True)
        return original_load(self)

    monkeypatch.setattr(module.KnowledgeEvolver, "_load_manifest", checked_load)
    with ThreadPoolExecutor(max_workers=4) as executor:
        instances = list(executor.map(lambda _: make_evolver(module, managed.root), range(8)))
    first = instances[0]
    assert len(loads) == 1
    for item in instances:
        assert item._manifest is managed._evolve_manifest is first._manifest
        assert item._evolve_lock is managed._evolve_lock is first._evolve_lock
    assert first._llm is not instances[1]._llm

    original_write = module.write_json_atomic
    def checked_write(path, value):
        lock.assert_owned()
        original_write(path, value)
    monkeypatch.setattr(module, "write_json_atomic", checked_write)
    first._manifest.cluster_ids_buffer.append("unflushed")
    another = make_evolver(module, managed.root)
    assert another._manifest.cluster_ids_buffer == ["unflushed"]
    another._save_manifest(another._manifest)
    assert read_manifest(first)["cluster_ids_buffer"] == ["unflushed"]


@pytest.mark.asyncio
async def test_concurrent_instances_keep_arrivals_durable_during_classification(module, managed):
    a, b, c = cluster("a"), cluster("b"), cluster("c")
    storage = storage_for(a, b, c)
    entered, finish = asyncio.Event(), asyncio.Event()
    original_get = storage.get
    async def blocked_get(identifier):
        if identifier == "a":
            entered.set()
            await finish.wait()
        return await original_get(identifier)
    storage.get = blocked_get
    first = make_evolver(module, managed.root, storage)
    second = make_evolver(module, managed.root, storage)
    task = asyncio.create_task(first.step(a))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await second.step(b)
        saved = read_manifest(first)
        assert saved["current_step"] == 2
        assert saved["cluster_ids_buffer"] == ["a", "b"]
        assert not saved["connect_merge_cluster_ids"]
        assert not task.done() and first._evolve_lock.locked()
    finally:
        finish.set()
        await task
    assert read_manifest(first)["cluster_ids_buffer"] == ["b"]
    assert read_manifest(first)["connect_merge_cluster_ids"] == ["a"]
    await second.step(c)
    saved = read_manifest(first)
    assert saved["current_step"] == 3 and not saved["cluster_ids_buffer"]
    assert set(saved["connect_merge_cluster_ids"]) == {"a", "b", "c"}
    assert managed.failure is None


@pytest.mark.asyncio
async def test_real_storage_connect_phase_persists_without_model_provider(module, managed):
    from sirchmunk.storage.knowledge_storage import KnowledgeStorage
    knowledge = KnowledgeStorage(work_path=managed.root)
    first = make_evolver(module, managed.root, knowledge)
    second = make_evolver(module, managed.root, knowledge)
    for index in range(module._EVO_CONNECT_MERGE_COUNT):
        item = cluster(f"stored-{index}")
        assert await knowledge.insert(item)
        await first._update_cluster_embedding(item)  # Fake embedding, real DB.
        await (first if index % 2 == 0 else second).step(item)
    assert not first._manifest.cluster_ids_buffer
    assert not first._manifest.connect_merge_cluster_ids
    assert first._manifest.last_evolve_at is not None
    managed.flush()
    assert Path(knowledge.parquet_file).is_file()
    count = knowledge.db.fetch_one("SELECT count(*) FROM read_parquet(?)", [knowledge.parquet_file])[0]
    assert count == module._EVO_CONNECT_MERGE_COUNT
    assert read_manifest(first)["current_step"] == count
    assert managed.failure is None


@pytest.mark.asyncio
async def test_managed_storage_read_failure_retains_buffer_and_blocks_further_writes(module, managed):
    item = cluster("pending")
    storage = storage_for(item)
    storage.get.side_effect = OSError("read failed")
    evolver = make_evolver(module, managed.root, storage)
    with pytest.raises(OSError, match="read failed"):
        await evolver.step(item)
    saved = evolver._manifest_path.read_bytes()
    assert read_manifest(evolver)["cluster_ids_buffer"] == ["pending"]
    assert managed.failure and not evolver._evolve_lock.locked()
    with pytest.raises(RuntimeError):
        await evolver.step(cluster("must-not-overwrite"))
    assert evolver._manifest_path.read_bytes() == saved


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["connect", "refresh", "meta", "global"])
async def test_phase_errors_fail_runtime_without_success_checkpoint(module, managed, monkeypatch, phase):
    item = cluster("pending")
    evolver = make_evolver(module, managed.root, storage_for(item))
    phases = {
        "connect": ("_should_connect_merge", "_connect_and_merge"),
        "refresh": ("_should_refresh_edges", "_refresh_edges"),
        "meta": ("_should_detect_meta_clusters", "_detect_meta_clusters"),
        "global": ("_should_global_update", "_global_update"),
    }
    for name, (trigger, operation) in phases.items():
        monkeypatch.setattr(evolver, trigger, lambda enabled=name == phase: enabled)
        monkeypatch.setattr(evolver, operation, AsyncMock(side_effect=OSError("phase write failed")))
    with pytest.raises(OSError, match="phase write failed"):
        await evolver.step(item)
    saved = read_manifest(evolver)
    assert saved["connect_merge_cluster_ids"] == ["pending"]
    assert saved["last_evolve_at"] is None
    assert saved["last_meta_detection_step"] == saved["last_global_update_step"] == 0
    assert managed.failure
    evolver._log.info.assert_any_await("[Evolve] Starting evolver step with 1 clusters in the buffer.")
    assert not any("step completed" in str(call) for call in evolver._log.info.await_args_list)
    with pytest.raises(RuntimeError):
        managed.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["meta", "global"])
@pytest.mark.parametrize("cancel_child", [False, True])
async def test_phase_gather_settles_siblings_before_propagating(module, managed, monkeypatch, phase, cancel_child):
    from sirchmunk.llm.factory import create_background_task
    evolver = make_evolver(module, managed.root)
    evolver._manifest.current_step = 20
    entered, finish = asyncio.Event(), asyncio.Event()
    completed = []

    async def child(identifier, *_args):
        if identifier in (0, "bad"):
            if cancel_child:
                raise asyncio.CancelledError()
            raise OSError("child storage failed")
        entered.set()
        await finish.wait()
        completed.append(identifier)

    if phase == "meta":
        monkeypatch.setattr(module, "ig", SimpleNamespace(Graph=Mock()))
        monkeypatch.setattr(module, "leidenalg", SimpleNamespace(
            ModularityVertexPartition=object(), find_partition=Mock(return_value=SimpleNamespace(membership=[0, 0, 1, 1]))))
        monkeypatch.setattr(evolver, "_cleanup_meta_clusters", Mock())
        monkeypatch.setattr(evolver, "_get_non_meta_cluster_count", lambda: 4)
        monkeypatch.setattr(evolver, "_get_edges_and_weights", lambda: ([("a", "b"), ("c", "d")], [1., 1.]))
        monkeypatch.setattr(evolver, "_build_meta_cluster", child)
        operation = evolver._detect_meta_clusters
    else:
        monkeypatch.setattr(evolver, "_get_all_embed_sim_edges", lambda: [])
        monkeypatch.setattr(evolver, "_get_cluster_ids_to_refine", lambda: ["bad", "slow"])
        monkeypatch.setattr(evolver, "_refine_cluster", child)
        operation = evolver._global_update
    task = create_background_task(operation(), runtime=managed)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        await asyncio.sleep(0)
        assert not task.done() and managed.active_background() == 1
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError if cancel_child else OSError):
            await task
        await asyncio.sleep(0)
    assert completed and managed.failure and managed.active_background() == 0
    assert evolver._manifest.last_meta_detection_step == evolver._manifest.last_global_update_step == 0


@pytest.mark.asyncio
async def test_cancelling_parent_gather_keeps_lease_until_child_cleanup_exits(module, managed):
    from sirchmunk.llm.factory import create_background_task
    evolver = make_evolver(module, managed.root)
    entered, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def child():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
            await finish.wait()
    task = create_background_task(evolver._gather_phase([child()]), runtime=managed)
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        await asyncio.wait_for(cancelled.wait(), 3)
        assert not task.done() and managed.active_background() == 1
    finally:
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
    assert managed.active_background() == 0 and managed.failure


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", ["refine", "meta", "embedding", "merge", "edge"])
@pytest.mark.parametrize("false_result", [False, True])
async def test_storage_failures_in_helpers_are_not_success(module, managed, helper, false_result):
    a, b = cluster("a"), cluster("b")
    storage = storage_for(a, b)
    storage.db.fetch_one.return_value = ('["query"]', None)
    evolver = make_evolver(module, managed.root, storage)
    if helper == "refine":
        target, operation = storage.update, lambda: evolver._refine_cluster("a")
    elif helper == "meta":
        target, operation = storage.insert, lambda: evolver._build_meta_cluster(0, ["a", "b"])
    elif helper == "embedding":
        target, operation = storage.store_embedding, lambda: evolver._update_cluster_embedding(a)
    elif helper == "merge":
        target, operation = storage.merge, lambda: evolver._merge_clusters(a, b)
    else:
        target, operation = storage.update, lambda: evolver._create_weak_semantic_edge(a, b, "embed_sim", 0.7)
    if false_result:
        target.return_value = None if helper == "merge" else False
    else:
        target.side_effect = OSError("storage write failed")
    with pytest.raises(RuntimeError if false_result else OSError):
        await operation()
    assert managed.failure
    assert not evolver._manifest.edge_refresh_cluster_ids


@pytest.mark.asyncio
async def test_invalid_refine_response_fails_managed_runtime(module, managed):
    evolver = make_evolver(module, managed.root, storage_for(cluster("a")))
    evolver._llm.achat.return_value.content = "not JSON"
    with pytest.raises(ValueError):
        await evolver._refine_cluster("a")
    assert managed.failure
    evolver._storage.update.assert_not_awaited()


@pytest.mark.parametrize("payload", ["{broken", "{}", '{"version":"99"}', "[]"])
def test_bad_manifest_is_never_reset_or_overwritten(module, managed, payload):
    path = managed.root / ".cache/evolve/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        make_evolver(module, managed.root)
    assert managed.failure and path.read_text(encoding="utf-8") == payload
    assert not hasattr(managed, "_evolve_manifest")
    with pytest.raises(RuntimeError):
        make_evolver(module, managed.root)
    assert path.read_text(encoding="utf-8") == payload


def test_unreadable_manifest_is_not_treated_as_missing(module, managed):
    path = managed.root / ".cache/evolve/manifest.json"
    path.mkdir(parents=True)
    with pytest.raises(OSError):
        make_evolver(module, managed.root)
    assert managed.failure and path.is_dir()


@pytest.mark.parametrize("invalid_memory", [False, True])
def test_cannot_publish_non_authoritative_or_malformed_manifest(module, managed, invalid_memory):
    evolver = make_evolver(module, managed.root)
    evolver._save_manifest(evolver._manifest)
    saved = evolver._manifest_path.read_bytes()
    manifest = evolver._manifest if invalid_memory else module.EvolveManifest()
    if invalid_memory:
        manifest.edge_refresh_cluster_ids.append(123)
    with pytest.raises((ValueError, RuntimeError)):
        evolver._save_manifest(manifest)
    assert managed.failure and evolver._manifest_path.read_bytes() == saved


@pytest.mark.parametrize("failing_fsync", [1, 2])
def test_atomic_fsync_failure_retains_old_file_and_only_cleans_own_temp(module, managed, monkeypatch, failing_fsync):
    from sirchmunk.storage import lifecycle
    evolver = make_evolver(module, managed.root)
    evolver._save_manifest(evolver._manifest)
    saved = evolver._manifest_path.read_bytes()
    old_temp = evolver._manifest_path.with_name("manifest.json.prior.tmp")
    old_temp.write_text("unrelated recovery data", encoding="utf-8")
    fsync = lifecycle.os.fsync
    calls = []
    def fail(descriptor):
        calls.append(descriptor)
        if len(calls) == failing_fsync:
            raise OSError("injected fsync failure")
        return fsync(descriptor)
    monkeypatch.setattr(lifecycle.os, "fsync", fail)
    evolver._manifest.current_step += 1
    with pytest.raises(OSError, match="fsync failure"):
        evolver._save_manifest(evolver._manifest)
    assert managed.failure and evolver._manifest_path.read_bytes() == saved
    assert list(evolver._evolve_dir.glob("*.tmp")) == [old_temp]
    assert old_temp.read_text(encoding="utf-8") == "unrelated recovery data"


def test_directory_fsync_failure_is_not_reported_as_durable_success(module, managed, monkeypatch):
    from sirchmunk.storage import lifecycle
    evolver = make_evolver(module, managed.root)
    evolver._save_manifest(evolver._manifest)
    evolver._manifest.current_step = 4
    monkeypatch.setattr(lifecycle, "durable_directory", Mock(side_effect=OSError("directory fsync failed")))
    with pytest.raises(OSError, match="directory fsync failed"):
        evolver._save_manifest(evolver._manifest)
    # Publication may already have happened. Keep the valid published data and
    # fail closed; attempting rollback cannot prove durability either.
    assert module.EvolveManifest.from_json(evolver._manifest_path.read_text(), strict=True).current_step == 4
    assert managed.failure
    with pytest.raises(RuntimeError):
        managed.flush()


def test_valid_saved_manifest_reloads_after_runtime_restart(module, managed):
    from sirchmunk.storage import lifecycle
    evolver = make_evolver(module, managed.root)
    evolver._manifest.current_step = 7
    evolver._manifest.cluster_ids_buffer.extend(["pending", "pending"])
    evolver._save_manifest(evolver._manifest)
    managed.close()
    lifecycle._runtimes.pop(str(managed.root))
    restarted = lifecycle.bootstrap_managed_root({
        "SIRCHMUNK_WORK_PATH": str(managed.root), "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "evolver-test",
    })
    restored = make_evolver(module, managed.root)
    assert restored._manifest == evolver._manifest
    assert restored._manifest is not evolver._manifest
    assert restarted.failure is None


@pytest.mark.asyncio
async def test_unmanaged_legacy_error_handling_and_independent_instances(module, isolated, monkeypatch):
    path = isolated / ".cache/evolve/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("broken legacy file", encoding="utf-8")
    item = cluster("a")
    first = make_evolver(module, isolated, storage_for(item))
    second = make_evolver(module, isolated, storage_for(item))
    assert first._manifest is not second._manifest and first._evolve_lock is not second._evolve_lock
    assert first._manifest == module.EvolveManifest()
    monkeypatch.setattr(first, "_should_connect_merge", lambda: True)
    monkeypatch.setattr(first, "_connect_and_merge", AsyncMock(side_effect=OSError("legacy phase failure")))
    await first.step(item)
    first._storage.update.side_effect = OSError("legacy write failure")
    await first._refine_cluster("a")
    first._storage.store_embedding.side_effect = OSError("legacy embedding failure")
    await first._update_cluster_embedding(item)
    result = await first._gather_phase([AsyncMock(side_effect=OSError("legacy child failure"))()])
    assert isinstance(result[0], OSError)
    assert read_manifest(first)["current_step"] == 1
