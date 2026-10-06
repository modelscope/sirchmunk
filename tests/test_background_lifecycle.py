"""Isolated lifecycle tests: real tasks/threads, fake models, no provider IO."""
import asyncio
import os
import multiprocessing as mp
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


_CACHE_ENV = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE",
              "SENTENCE_TRANSFORMERS_HOME", "MODELSCOPE_CACHE", "TORCH_HOME",
              "HF_MODULES_CACHE", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")
_OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(root))
    monkeypatch.setenv("EMBEDDING_CACHE_DIR", str(root / ".cache/models"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    for key in _CACHE_ENV:
        monkeypatch.setenv(key, str(root / ".cache/models" / key.lower()))
    for key in _OFFLINE_ENV:
        monkeypatch.setenv(key, "0")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No real model loading/download is permitted in this suite")

    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=forbidden))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=forbidden))


@pytest.fixture
def managed(tmp_path):
    from sirchmunk.storage import lifecycle
    root = tmp_path / "work"
    runtime = lifecycle.bootstrap_managed_root({
        "SIRCHMUNK_WORK_PATH": str(root), "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "background-test",
    })
    yield runtime
    assert runtime.active_background() == 0, "test left actual background work running"
    lifecycle._runtimes.pop(str(root.resolve()), None)
    lease = lifecycle._roots.pop(str(root.resolve()), None)
    if lease:
        lease.close()


async def _until(predicate, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("background work did not reach the expected state")
        await asyncio.sleep(0.002)


async def _join_cleanup():
    from sirchmunk.llm.factory import _cleanup_tasks
    tasks = list(_cleanup_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)  # Deliver ownership-release callbacks.


@pytest.mark.asyncio
async def test_evolution_lease_precedes_dispatch_and_survives_scope_exit(managed):
    from sirchmunk.llm.factory import create_background_task, llm_request_scope, track_search_instance
    gate = asyncio.Event()
    close = Mock()
    try:
        async with llm_request_scope():
            task = create_background_task(gate.wait())
            assert managed.active_background() == 1  # Task has not run yet.
            track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close),
                                                 _background_tasks={task}))
        assert managed.active_background() >= 1
        close.assert_not_called()
        with pytest.raises(Exception, match="not_drained"):
            managed.flush()
    finally:
        gate.set()
        await _join_cleanup()
    close.assert_called_once()
    assert managed.failure is None


@pytest.mark.asyncio
async def test_completed_evolution_error_is_not_lost_when_search_set_discards_it(managed):
    from sirchmunk.llm.factory import create_background_task, llm_request_scope, track_search_instance
    async def fail():
        raise OSError("storage write failed")
    with pytest.raises(RuntimeError, match="background/cleanup"):
        async with llm_request_scope():
            tasks = set()
            task = create_background_task(fail())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            track_search_instance(SimpleNamespace(knowledge_storage=None, _background_tasks=tasks))
            with pytest.raises(OSError, match="storage write"):
                await task
            assert not tasks
    await _join_cleanup()
    assert managed.failure


@pytest.mark.asyncio
async def test_task_cancelled_before_first_instruction_releases_and_fails(managed):
    from sirchmunk.llm.factory import create_background_task
    ran = []
    async def worker():
        ran.append(True)
    task = create_background_task(worker())
    assert managed.active_background() == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)
    assert not ran and managed.active_background() == 0
    assert managed.failure


@pytest.mark.asyncio
async def test_failed_task_dispatch_releases_lease(managed, monkeypatch):
    from sirchmunk.llm.factory import create_background_task
    async def work():
        pass
    def fail(*_args, **_kwargs):
        raise RuntimeError("task dispatch failed")
    coroutine = work()
    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", fail)
        with pytest.raises(RuntimeError, match="dispatch"):
            create_background_task(coroutine)
    assert coroutine.cr_frame is None
    assert managed.active_background() == 0 and managed.failure


@pytest.mark.asyncio
async def test_scope_cancellation_leaves_cleanup_and_thread_leased(managed):
    from sirchmunk.llm.factory import llm_request_scope, track_search_instance
    entered, finish = threading.Event(), threading.Event()
    def close():
        entered.set()
        assert finish.wait(5)
    async def request():
        async with llm_request_scope():
            track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close),
                                                 _background_tasks=set()))
    request_task = asyncio.create_task(request())
    try:
        await _until(entered.is_set)
        assert managed.active_background() == 2  # Cleanup task + actual worker.
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task
        assert managed.active_background() == 2
        assert managed.failure is None
    finally:
        finish.set()
        await _join_cleanup()
        await asyncio.gather(request_task, return_exceptions=True)
    assert managed.active_background() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_after_cancel", [False, True])
async def test_cancelled_await_does_not_release_running_thread(managed, fail_after_cancel):
    from sirchmunk.llm.factory import run_background_thread, _thread_futures
    entered, finish = threading.Event(), threading.Event()
    def work():
        entered.set()
        assert finish.wait(5)
        if fail_after_cancel:
            raise OSError("late worker failure")
        return 17
    task = asyncio.create_task(run_background_thread(work))
    try:
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
        assert _thread_futures
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0)
        await asyncio.gather(task, return_exceptions=True)
    assert bool(managed.failure) == fail_after_cancel


@pytest.mark.asyncio
async def test_cancelled_cleanup_keeps_running_thread_counted(managed):
    from sirchmunk.llm.factory import llm_request_scope, track_search_instance, _cleanup_tasks
    entered, finish = threading.Event(), threading.Event()
    def close():
        entered.set()
        assert finish.wait(5)
    async def request():
        async with llm_request_scope():
            track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close)))
    task = asyncio.create_task(request())
    try:
        await _until(entered.is_set)
        cleanup = next(iter(_cleanup_tasks))
        cleanup.cancel()
        await asyncio.gather(task, cleanup, return_exceptions=True)
        assert managed.active_background() == 1
        assert managed.failure
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0)
        await _join_cleanup()


@pytest.mark.asyncio
async def test_storage_cleanup_error_propagates_and_marks_runtime(managed):
    from sirchmunk.llm.factory import llm_request_scope, track_search_instance
    close = Mock(side_effect=OSError("fsync failed"))
    with pytest.raises(RuntimeError, match="background/cleanup"):
        async with llm_request_scope():
            track_search_instance(SimpleNamespace(knowledge_storage=SimpleNamespace(close=close)))
    await _join_cleanup()
    assert managed.failure


def _local_model(root):
    path = root / "preprovisioned"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


class FakeModel:
    max_seq_length = 32
    def get_sentence_embedding_dimension(self):
        return 4
    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 4))


@pytest.mark.asyncio
async def test_managed_local_model_load_is_offline_and_uses_root_caches(managed, monkeypatch):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    calls = []
    path = _local_model(managed.root)
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    model = EmbeddingUtil(str(path), device="cpu")
    assert (await model.embed(["fixture"])) == [[0.0] * 4]
    await _until(lambda: managed.active_background() == 0)
    assert len(calls) == 1
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["trust_remote_code"] is False
    assert Path(calls[0][1]["cache_folder"]).is_relative_to(managed.root)
    assert all(Path(os.environ[key]).is_relative_to(managed.root) for key in _CACHE_ENV)
    assert all(os.environ[key] == "1" for key in _OFFLINE_ENV)


@pytest.mark.asyncio
async def test_missing_managed_model_never_attempts_online_download(managed):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    model = EmbeddingUtil("not-provisioned/model", device="cpu")
    with pytest.raises(RuntimeError, match="not provisioned locally"):
        await model._ensure_model_async()
    await _until(lambda: managed.active_background() == 0)
    assert managed.failure == "embedding_model_load_failed"
    assert not model.is_ready()


@pytest.mark.asyncio
async def test_cancelled_model_wait_does_not_cancel_loading_or_release_thread(managed, monkeypatch):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    path = _local_model(managed.root)
    entered, finish = threading.Event(), threading.Event()
    def constructor(*_args, **_kwargs):
        entered.set()
        assert finish.wait(5)
        return FakeModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=constructor))
    model = EmbeddingUtil(str(path), device="cpu")
    task = asyncio.create_task(model._ensure_model_async())
    try:
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
        assert not model._model_future.cancelled()
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0)
        await asyncio.gather(task, return_exceptions=True)
    assert isinstance(await model._ensure_model_async(), FakeModel)
    assert managed.failure is None


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_error", [False, True])
async def test_embedding_encode_thread_retains_lease_after_cancellation(managed, worker_error):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    entered, finish = threading.Event(), threading.Event()
    class SlowModel(FakeModel):
        def encode(self, texts, **kwargs):
            entered.set()
            assert finish.wait(5)
            if worker_error:
                raise RuntimeError("encode failed")
            return super().encode(texts, **kwargs)
    model = EmbeddingUtil("unused/local", device="cpu")
    model.model = SlowModel()
    task = asyncio.create_task(model.embed(["fixture"]))
    try:
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0)
        await asyncio.gather(task, return_exceptions=True)
    assert bool(managed.failure) == worker_error


def test_thread_start_failure_releases_model_load_lease(managed, monkeypatch):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    model = EmbeddingUtil("unused/local", device="cpu")
    def fail(_self):
        raise RuntimeError("no worker thread")
    monkeypatch.setattr(threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="no worker"):
        model.start_loading()
    assert managed.active_background() == 0
    assert managed.failure == "embedding_load_dispatch_failed"
    assert isinstance(model._model_future.exception(), RuntimeError)


@pytest.mark.parametrize("layout", ["modelscope", "hf"])
def test_managed_preload_resolves_existing_cache_without_hub_calls(managed, layout):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    cache = managed.root / ".cache/models"
    if layout == "modelscope":
        path = cache / "models" / "fixture" / "model"
    else:
        repo = cache / "models--fixture--model"
        path = repo / "snapshots" / ("a" * 40)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/main").write_text("a" * 40, encoding="utf-8")
    path.mkdir(parents=True)
    (path / "modules.json").write_text("[]", encoding="utf-8")
    assert EmbeddingUtil.preload_model(str(cache), "fixture/model") == str(path.resolve())


def test_managed_cache_outside_root_is_rejected(managed, tmp_path):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    with pytest.raises(ValueError, match="inside the work root"):
        EmbeddingUtil("unused/local", device="cpu", cache_dir=str(tmp_path / "outside"))
    assert managed.failure
    assert not (tmp_path / "outside").exists()


def test_unmanaged_offline_first_online_fallback_remains_compatible(monkeypatch):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    calls = []
    def download(**kwargs):
        calls.append(kwargs["local_files_only"])
        if kwargs["local_files_only"]:
            raise FileNotFoundError("fixture cache miss")
        return "fake-provisioned-directory"
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    assert EmbeddingUtil._download_model("fixture/model") == "fake-provisioned-directory"
    assert calls == [True, False]
    assert all(os.environ[key] == "0" for key in _OFFLINE_ENV)


def _delayed_extraction_child(file_path, profile_dict, pipe, *args):
    """Real spawned child: returning a result is deliberately not process exit."""
    pipe.send({"content": "child result"})
    pipe.close()
    profile_dict["entered"].set()
    if not profile_dict["finish"].wait(8):
        raise RuntimeError("test did not release child")


@pytest.mark.asyncio
async def test_extraction_thread_lease_lasts_until_real_child_exit(managed, monkeypatch):
    from sirchmunk.llm.factory import run_background_thread
    from sirchmunk.utils import document_extractor as module
    context = mp.get_context("spawn")
    entered, finish = context.Event(), context.Event()
    monkeypatch.setattr(module, "_extraction_worker", _delayed_extraction_child)
    before = {child.pid for child in mp.active_children()}
    task = asyncio.create_task(run_background_thread(
        module._run_extraction_in_child, "unused-file",
        {"entered": entered, "finish": finish},
    ))
    try:
        await _until(entered.is_set, timeout=6)
        assert managed.active_background() == 1
        assert {child.pid for child in mp.active_children()} - before
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0, timeout=8)
        await asyncio.gather(task, return_exceptions=True)
    assert not ({child.pid for child in mp.active_children()} - before)
    assert managed.failure is None


@pytest.mark.asyncio
async def test_managed_native_extraction_is_routed_through_child_barrier(managed, monkeypatch):
    from sirchmunk.utils import document_extractor as module
    entered, finish = threading.Event(), threading.Event()
    def extract(file_path, profile, *args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return {"content": "isolated"}
    monkeypatch.setattr(module, "_run_extraction_in_child", extract)
    task = asyncio.create_task(module.DocumentExtractor.extract("unused.txt"))
    try:
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
    finally:
        finish.set()
        await _until(lambda: managed.active_background() == 0)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_managed_extraction_does_not_fallback_after_child_failure(managed, monkeypatch):
    from sirchmunk.utils import document_extractor as module
    def fail(*args, **kwargs):
        raise OSError("child failed")
    monkeypatch.setattr(module, "_run_extraction_in_child", fail)
    with pytest.raises(OSError, match="child failed"):
        await module.DocumentExtractor.extract("unused.txt")
    assert managed.failure
    assert managed.active_background() == 0


@pytest.mark.asyncio
async def test_bytes_and_batch_use_managed_child_dispatch(managed, monkeypatch):
    from sirchmunk.utils import document_extractor as module
    calls = []
    def extract(payload, profile, operation="file", mime_type=None):
        assert managed.active_background() == 1
        calls.append((operation, payload, mime_type))
        return [{"content": "batch"}] if operation == "batch" else {"content": "bytes"}
    monkeypatch.setattr(module, "_run_extraction_in_child", extract)
    assert (await module.DocumentExtractor.extract_bytes(b"text", "text/plain")).content == "bytes"
    assert (await module.DocumentExtractor.batch_extract(["file.txt"]))[0].content == "batch"
    assert calls == [("bytes", b"text", "text/plain"), ("batch", ["file.txt"], None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["file", "bytes", "batch"])
async def test_real_native_text_extraction_exits_child_before_barrier(managed, operation):
    from sirchmunk.utils.document_extractor import DocumentExtractor
    path = managed.root / "local-document.txt"
    path.write_text("A local extraction fixture with no model or provider.", encoding="utf-8")
    before = {child.pid for child in mp.active_children()}
    if operation == "bytes":
        result = await DocumentExtractor.extract_bytes(path.read_bytes(), "text/plain")
    elif operation == "batch":
        result = (await DocumentExtractor.batch_extract([path]))[0]
    else:
        result = await DocumentExtractor.extract(path)
    assert "local extraction fixture" in result.content
    assert managed.active_background() == 0
    assert not ({child.pid for child in mp.active_children()} - before)


@pytest.mark.asyncio
async def test_frozen_runtime_refuses_dispatch_before_thread_starts(managed):
    from sirchmunk.llm.factory import run_background_thread
    ran = Mock()
    managed.frozen = True
    with pytest.raises(Exception, match="not_accepting"):
        await run_background_thread(ran)
    ran.assert_not_called()
    assert managed.active_background() == 0


@pytest.mark.asyncio
async def test_queued_thread_remains_reserved_after_await_cancel(managed, monkeypatch):
    from sirchmunk.llm.factory import run_background_thread
    blocker_entered, unblock = threading.Event(), threading.Event()
    ran = threading.Event()
    def blocker():
        blocker_entered.set()
        assert unblock.wait(5)
    executor = ThreadPoolExecutor(max_workers=1)
    blocked = executor.submit(blocker)
    loop = asyncio.get_running_loop()
    original = loop.run_in_executor
    monkeypatch.setattr(loop, "run_in_executor", lambda _executor, fn, *args:
                        original(executor, fn, *args))
    task = asyncio.create_task(run_background_thread(ran.set))
    try:
        await _until(blocker_entered.is_set)
        await _until(lambda: managed.active_background() == 1)
        assert not ran.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert managed.active_background() == 1
    finally:
        unblock.set()
        await _until(ran.is_set)
        await _until(lambda: managed.active_background() == 0)
        blocked.result(timeout=2)
        executor.shutdown(wait=True)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_thread_dispatch_failure_propagates_without_leaking_lease(managed, monkeypatch):
    from sirchmunk.llm.factory import run_background_thread
    def fail(*_args, **_kwargs):
        raise RuntimeError("executor rejected work")
    monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail)
    with pytest.raises(RuntimeError, match="executor rejected"):
        await run_background_thread(lambda: None)
    assert managed.active_background() == 0
    assert managed.failure == "background_thread_dispatch_failed"


def test_frozen_model_loader_does_not_poison_future_or_dispatch(managed, monkeypatch):
    from sirchmunk.utils.embedding_util import EmbeddingUtil
    managed.frozen = True
    model = EmbeddingUtil("unused/local", device="cpu")
    worker = Mock()
    monkeypatch.setattr(threading.Thread, "start", worker)
    with pytest.raises(Exception, match="not_accepting"):
        model.start_loading()
    worker.assert_not_called()
    assert not model._loading_started and not model._model_future.done()
    assert managed.active_background() == 0 and managed.failure is None
