"""Real storage, concurrency and crash/restart tests; no model or real root IO."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import duckdb
import pytest


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    # Import Sirchmunk only in fixtures/tests, after isolating every work root.
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path / "work"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "models"))
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    monkeypatch.delenv("EMBEDDING_MODEL_ID", raising=False)


def _launch_env(root):
    return {
        "SIRCHMUNK_MANAGEMENT_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_LIFECYCLE_ENABLED": "true",
        "SIRCHMUNK_MANAGEMENT_INSTANCE_ID": "durable-storage-test",
        "SIRCHMUNK_WORK_PATH": str(root),
    }


@pytest.fixture
def managed(tmp_path):
    from sirchmunk.storage import lifecycle

    root = tmp_path / "work"
    runtime = lifecycle.bootstrap_managed_root(_launch_env(root))
    yield runtime
    # Failure tests intentionally leave the runtime fail-closed. Teardown only
    # closes handles; it must not silently retry/acknowledge a failed barrier.
    for storage in runtime._registered:
        db = getattr(storage, "db", storage)
        if getattr(db, "connection", None) is not None:
            db.connection.close()
            db.connection = None
    lifecycle._runtimes.pop(str(root.resolve()), None)
    lease = lifecycle._roots.pop(str(root.resolve()), None)
    if lease:
        lease.close()


def _stores(runtime):
    from sirchmunk.storage.knowledge_storage import KnowledgeStorage
    from sirchmunk.api.components.history_storage import HistoryStorage

    return KnowledgeStorage(str(runtime.root)), HistoryStorage(str(runtime.root))


def _cluster(identifier):
    from sirchmunk.schema.knowledge import KnowledgeCluster

    return KnowledgeCluster(id=identifier, name=identifier, description="description", content="content")


def _session(history, identifier="session"):
    assert history.save_session({
        "session_id": identifier, "created_at": "2026-09-30T00:00:00",
        "updated_at": "2026-09-30T00:00:00",
    })


def _child(root, code, *args):
    # Only OS runtime variables are inherited; no model credentials/config.
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP")
           if key in os.environ}
    env.update(_launch_env(root))
    env.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONDONTWRITEBYTECODE": "1", "EMBEDDING_DIM": "4",
        "HF_HOME": str(root / "models"),
    })
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], env=env,
                          capture_output=True, text=True, timeout=40)


def test_shared_consumers_and_close_do_not_destroy_live_storage(managed):
    knowledge, history = _stores(managed)
    with ThreadPoolExecutor(max_workers=6) as executor:
        instances = list(executor.map(lambda _: _stores(managed), range(12)))
    assert all(k is knowledge and h is history for k, h in instances)
    assert knowledge._io_lock is history.db._io_lock is managed.io_lock
    assert knowledge._daemon_thread is history.db._daemon_thread is None
    assert not knowledge._shutdown_registered and not history.db._shutdown_registered
    knowledge.close()
    history.close()
    history.db.close()
    _session(history)
    assert asyncio.run(knowledge.insert(_cluster("survives-close")))
    managed.flush()
    managed.close()
    assert knowledge.db is None and history.db.connection is None


def test_concurrent_real_writes_reads_and_flush(managed):
    knowledge, history = _stores(managed)
    _session(history)
    start = threading.Barrier(5)

    def writer(worker):
        start.wait(timeout=10)
        for number in range(10):
            identifier = f"{worker}-{number}"
            assert asyncio.run(knowledge.insert(_cluster(identifier)))
            assert history.save_message("session", {"role": "user", "content": identifier})
            assert history.db.fetch_one("SELECT ?::VARCHAR", [identifier]) == (identifier,)
            assert knowledge.db.fetch_df("SELECT ?::VARCHAR AS value", [identifier]).iloc[0, 0] == identifier

    def flusher():
        start.wait(timeout=10)
        for _ in range(5):
            managed.flush()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(writer, n) for n in range(4)] + [executor.submit(flusher)]
        for future in futures:
            future.result(timeout=30)
    managed.flush()
    assert history.get_session("session")["message_count"] == 40
    with duckdb.connect(history.db_path, read_only=True) as disk:
        assert disk.execute("SELECT count(*) FROM chat_messages").fetchone() == (40,)
    with duckdb.connect() as disk:
        assert disk.execute("SELECT count(*) FROM read_parquet(?)", [knowledge.parquet_file]).fetchone() == (40,)


def test_fetch_holds_lock_through_result_consumption(managed, monkeypatch):
    _, history = _stores(managed)
    original = history.db.execute

    def slow_execute(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.001)  # Widen the execute/fetch race using a real connection.
        return result

    monkeypatch.setattr(history.db, "execute", slow_execute)
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(lambda n: history.db.fetch_one("SELECT ?::INTEGER", [n]), range(64)))
    assert values == [(n,) for n in range(64)]


def test_managed_reload_cannot_erase_dirty_knowledge(managed):
    knowledge, _ = _stores(managed)
    assert asyncio.run(knowledge.insert(_cluster("old")))
    knowledge.durable_flush()
    assert asyncio.run(knowledge.insert(_cluster("new")))
    with pytest.raises(RuntimeError, match="unflushed"):
        knowledge.reload()
    knowledge._check_and_reload()
    assert asyncio.run(knowledge.get("new")).id == "new"
    knowledge.durable_flush()
    knowledge.reload()
    assert asyncio.run(knowledge.get("new")).id == "new"


@pytest.mark.parametrize("kind", ["knowledge", "history"])
@pytest.mark.parametrize("failure_point", ["durable_file", "durable_replace", "durable_directory"])
def test_flush_failure_propagates_and_stays_dirty(managed, monkeypatch, kind, failure_point):
    import sirchmunk.storage.duckdb as db_module
    import sirchmunk.storage.knowledge_storage as knowledge_module

    knowledge, history = _stores(managed)
    _session(history)
    assert asyncio.run(knowledge.insert(_cluster("old")))
    managed.flush()
    store, module = (knowledge, knowledge_module) if kind == "knowledge" else (history, db_module)
    path = Path(knowledge.parquet_file if kind == "knowledge" else history.db_path)
    before = path.read_bytes()
    assert asyncio.run(knowledge.insert(_cluster("new")))
    assert history.save_message("session", {"role": "user", "content": "new"})

    def fail(*_args):
        raise OSError("injected durability failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, failure_point, fail)
        with pytest.raises(OSError, match="injected durability"):
            store.durable_flush()
    assert managed.failure
    assert (knowledge._parquet_dirty_count if kind == "knowledge" else history.db._dirty_count) > 0
    if failure_point != "durable_directory":
        assert path.read_bytes() == before  # Failure before atomic publication.
    with pytest.raises(Exception, match="flush_failed"):
        managed.flush()


@pytest.mark.parametrize("kind", ["knowledge", "history"])
def test_corrupt_existing_storage_fails_start_without_overwriting(managed, kind):
    from sirchmunk.storage.knowledge_storage import KnowledgeStorage
    from sirchmunk.api.components.history_storage import HistoryStorage

    relative = ".cache/knowledge/knowledge_clusters.parquet" if kind == "knowledge" else ".cache/history/chat_history.db"
    path = managed.root / relative
    path.parent.mkdir(parents=True)
    damaged = b"not-a-valid-storage-file"
    path.write_bytes(damaged)
    with pytest.raises(Exception):
        (KnowledgeStorage if kind == "knowledge" else HistoryStorage)(str(managed.root))
    assert path.read_bytes() == damaged
    assert managed.failure


def test_failed_wal_recovery_never_deletes_original_wal(managed):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = managed.root / "broken.db"
    path.write_bytes(b"damaged database")
    wal = Path(str(path) + ".wal")
    wal.write_bytes(b"only surviving journal bytes")
    with pytest.raises(Exception):
        DuckDBManager(persist_path=str(path))
    assert wal.read_bytes() == b"only surviving journal bytes"
    assert path.read_bytes() == b"damaged database"


def test_cleared_and_empty_storage_have_verified_durable_snapshots(managed):
    knowledge, history = _stores(managed)
    managed.flush()
    assert Path(knowledge.parquet_file).is_file() and Path(history.db_path).is_file()
    assert asyncio.run(knowledge.insert(_cluster("delete-me")))
    knowledge.durable_flush()
    before = Path(knowledge.parquet_file).read_bytes()
    assert asyncio.run(knowledge.clear())
    assert Path(knowledge.parquet_file).read_bytes() == before
    knowledge.durable_flush()
    assert knowledge.db.fetch_one("SELECT count(*) FROM read_parquet(?)", [knowledge.parquet_file]) == (0,)


def test_restart_recovers_flushed_data_and_primary_keys(tmp_path):
    root = tmp_path / "restart"
    setup = """
import os
from sirchmunk.storage.lifecycle import bootstrap_managed_root
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.api.components.history_storage import HistoryStorage
runtime = bootstrap_managed_root(dict(os.environ))
knowledge, history = KnowledgeStorage(), HistoryStorage()
"""
    write = _child(root, setup + """
history.db.execute("INSERT INTO chat_sessions VALUES ('session','title',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,'{}',0)")
knowledge.db.execute("INSERT INTO knowledge_clusters (id,name) VALUES ('cluster','name')")
runtime.flush()
os._exit(0)  # No atexit/final close: durability must come from the barrier.
""")
    assert write.returncode == 0, write.stderr
    read = _child(root, setup + """
assert history.db.fetch_one('SELECT session_id FROM chat_sessions') == ('session',)
assert knowledge.db.fetch_one('SELECT id FROM knowledge_clusters') == ('cluster',)
constraints = history.db.fetch_all("SELECT constraint_type FROM duckdb_constraints() WHERE table_name='chat_sessions'")
assert ('PRIMARY KEY',) in constraints
runtime.close()
print('restart-verified')
""")
    assert read.returncode == 0, read.stderr
    assert "restart-verified" in read.stdout


def test_sdk_cannot_bypass_managed_root_or_open_second_copy(managed, tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = managed.root / "direct.db"
    db = DuckDBManager(persist_path=str(path))
    assert db._runtime is managed and db._io_lock is managed.io_lock
    db.execute("CREATE TABLE example (id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO example VALUES (9)")
    with pytest.raises(RuntimeError, match="already open"):
        DuckDBManager(persist_path=str(path))
    with pytest.raises(RuntimeError, match="outside"):
        DuckDBManager(persist_path=str(tmp_path / "escape.db"), runtime=managed)
    managed.flush()
    other = _child(managed.root, """
import os, sys
from sirchmunk.storage.duckdb import DuckDBManager
try:
    DuckDBManager(persist_path=sys.argv[1])
except Exception as error:
    print(str(error))
else:
    raise AssertionError('second process acquired managed root')
""", path)
    assert other.returncode == 0, other.stderr
    assert ("managed_work_root_requires_supervisor" in other.stdout
            or "work_root_already_in_use" in other.stdout)


def test_sdk_root_lock_applies_to_unmanaged_parent(tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = tmp_path / "outside-env" / "db.duckdb"
    db = DuckDBManager(persist_path=str(path))
    try:
        assert (path.parent / ".control/work-root.lock").is_file()
        other = _child(tmp_path / "different-env", """
import sys
from sirchmunk.storage.duckdb import DuckDBManager
try:
    DuckDBManager(persist_path=sys.argv[1])
except Exception as error:
    print(str(error))
else:
    raise AssertionError('second writer was not blocked')
""", path)
        assert other.returncode == 0, other.stderr
        assert "work_root_already_in_use" in other.stdout
    finally:
        db.close()


def test_strict_unmanaged_flush_and_constraint_roundtrip(tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = tmp_path / "standalone.db"
    with DuckDBManager(persist_path=str(path)) as db:
        db.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value VARCHAR NOT NULL)")
        db.execute("INSERT INTO sample VALUES (1, 'saved')")
        db.sync_to_disk(strict=True)
    with DuckDBManager(persist_path=str(path)) as db:
        assert db.fetch_one("SELECT * FROM sample") == (1, "saved")
        with pytest.raises(duckdb.ConstraintException):
            db.execute("INSERT INTO sample VALUES (1, 'duplicate')")


def test_failed_parquet_reload_preserves_live_table(managed):
    knowledge, _ = _stores(managed)
    assert asyncio.run(knowledge.insert(_cluster("intact")))
    knowledge.durable_flush()
    Path(knowledge.parquet_file).write_bytes(b"corrupt external replacement")
    with pytest.raises(Exception):
        knowledge.reload()
    assert knowledge.db.fetch_one("SELECT id FROM knowledge_clusters") == ("intact",)
    assert managed.failure


@pytest.mark.parametrize("damage", ["missing-table", "missing-column", "wrong-type", "reordered"])
def test_managed_history_schema_is_checked_before_create(managed, damage):
    from sirchmunk.api.components.history_storage import HistoryStorage

    path = managed.root / ".cache/history/chat_history.db"
    path.parent.mkdir(parents=True)
    with duckdb.connect(str(path)) as disk:
        disk.execute("""CREATE TABLE chat_sessions (
            session_id VARCHAR, title VARCHAR, created_at TIMESTAMP,
            updated_at TIMESTAMP, settings JSON, message_count INTEGER)""")
        if damage != "missing-table":
            disk.execute("""CREATE TABLE chat_messages (
                id VARCHAR, session_id VARCHAR, role VARCHAR, content VARCHAR,
                timestamp TIMESTAMP, search_logs JSON, is_streaming BOOLEAN)""")
        if damage == "missing-column":
            disk.execute("ALTER TABLE chat_sessions DROP title")
        elif damage == "wrong-type":
            disk.execute("ALTER TABLE chat_sessions ALTER message_count TYPE VARCHAR")
        elif damage == "reordered":
            disk.execute("ALTER TABLE chat_sessions RENAME TO old_sessions")
            disk.execute("CREATE TABLE chat_sessions AS SELECT title, session_id, created_at, updated_at, settings, message_count FROM old_sessions")
            disk.execute("DROP TABLE old_sessions")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="history"):
        HistoryStorage(str(managed.root))
    assert managed.failure
    assert path.read_bytes() == before


def test_ctas_legacy_history_without_primary_key_is_accepted(managed):
    from sirchmunk.api.components.history_storage import HistoryStorage

    path = managed.root / ".cache/history/chat_history.db"
    path.parent.mkdir(parents=True)
    with duckdb.connect(str(path)) as disk:
        disk.execute("""CREATE TABLE chat_sessions AS SELECT
            'old'::VARCHAR AS session_id, 'legacy'::VARCHAR AS title,
            TIMESTAMP '2026-09-30' AS created_at, TIMESTAMP '2026-09-30' AS updated_at,
            '{}'::JSON AS settings, 1::INTEGER AS message_count""")
        disk.execute("""CREATE TABLE chat_messages AS SELECT
            'message'::VARCHAR AS id, 'old'::VARCHAR AS session_id, 'user'::VARCHAR AS role,
            'preserved'::VARCHAR AS content, TIMESTAMP '2026-09-30' AS timestamp,
            '[]'::JSON AS search_logs, false::BOOLEAN AS is_streaming""")
    before = path.read_bytes()
    history = HistoryStorage(str(managed.root))
    assert path.read_bytes() == before
    assert history.get_session("old")["messages"][0]["content"] == "preserved"
    assert history.db.fetch_all("SELECT * FROM duckdb_constraints()") == []
    history.durable_flush()


@pytest.mark.parametrize("damage", ["missing-required", "wrong-type", "unknown-column", "bad-dimension"])
def test_managed_parquet_validates_actual_schema(managed, damage):
    from sirchmunk.storage.knowledge_storage import KnowledgeStorage

    # Derive a real canonical snapshot from an isolated, unmanaged sibling.
    source = KnowledgeStorage(str(managed.root.parent / "parquet-source"))
    try:
        assert asyncio.run(source.insert(_cluster("legacy")))
        if damage == "missing-required":
            source.db.execute("ALTER TABLE knowledge_clusters DROP description")
        elif damage == "wrong-type":
            source.db.execute("ALTER TABLE knowledge_clusters ALTER hotness TYPE VARCHAR")
        elif damage == "unknown-column":
            source.db.execute("ALTER TABLE knowledge_clusters ADD unknown INTEGER")
        else:
            source.db.execute("ALTER TABLE knowledge_clusters ALTER embedding_vector TYPE FLOAT[]")
            source.db.execute("UPDATE knowledge_clusters SET embedding_vector=[1,2]")
        path = managed.root / ".cache/knowledge/knowledge_clusters.parquet"
        path.parent.mkdir(parents=True)
        source.db.export_to_parquet(source.table_name, str(path))
    finally:
        source.close()
    before = path.read_bytes()
    with pytest.raises(ValueError):
        KnowledgeStorage(str(managed.root))
    assert path.read_bytes() == before
    assert managed.failure


def test_known_pre_embedding_parquet_projection_is_accepted(managed):
    from sirchmunk.storage.knowledge_storage import KnowledgeStorage

    source = KnowledgeStorage(str(managed.root.parent / "old-knowledge"))
    try:
        assert asyncio.run(source.insert(_cluster("legacy")))
        for column in ("merge_count", "embedding_vector", "embedding_model", "embedding_timestamp", "embedding_text_hash"):
            source.db.execute(f"ALTER TABLE knowledge_clusters DROP {column}")
        path = managed.root / ".cache/knowledge/knowledge_clusters.parquet"
        path.parent.mkdir(parents=True)
        source.db.export_to_parquet(source.table_name, str(path))
    finally:
        source.close()
    before = path.read_bytes()
    knowledge = KnowledgeStorage(str(managed.root))
    assert path.read_bytes() == before
    assert asyncio.run(knowledge.get("legacy")).id == "legacy"
    assert knowledge.db.fetch_one("SELECT merge_count, embedding_vector FROM knowledge_clusters") == (0, None)
    knowledge.durable_flush()


def test_wal_from_real_crashed_process_is_recovered(tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    root = tmp_path / "wal-recovery"
    root.mkdir()
    path = root / "history.db"
    result = _child(root, """
import os, sys, duckdb
db = duckdb.connect(sys.argv[1])
db.execute("CREATE TABLE journal (id INTEGER PRIMARY KEY)")
db.execute("CHECKPOINT")
db.execute("INSERT INTO journal VALUES (42)")
os._exit(0)
""", path)
    assert result.returncode == 0, result.stderr
    assert Path(str(path) + ".wal").exists()
    with DuckDBManager(persist_path=str(path)) as db:
        assert db.fetch_one("SELECT * FROM journal") == (42,)
        db.durable_flush()


def test_strict_flush_does_not_resurrect_last_dropped_table(tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = tmp_path / "empty.db"
    with DuckDBManager(persist_path=str(path)) as db:
        db.execute("CREATE TABLE obsolete (id INTEGER)")
        db.durable_flush()
        db.execute("DROP TABLE obsolete")
        db.durable_flush()
    with DuckDBManager(persist_path=str(path)) as db:
        assert db.list_tables() == []


def test_direct_file_sdk_is_included_in_barrier(managed):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = managed.root / "direct-mode.db"
    db = DuckDBManager(db_path=str(path))
    db.execute("CREATE TABLE state (value VARCHAR)")
    db.execute("INSERT INTO state VALUES ('checkpointed')")
    assert db._runtime is managed
    managed.flush()
    assert db._suspended and db.connection is None
    with pytest.raises(RuntimeError, match="already open"):
        DuckDBManager(db_path=str(path))
    assert db.fetch_one("SELECT value FROM state") == ("checkpointed",)
    assert not db._suspended
    managed.close()
    with duckdb.connect(str(path), read_only=True) as disk:
        assert disk.execute("SELECT value FROM state").fetchone() == ("checkpointed",)


def test_memory_only_sdk_does_not_claim_work_root(tmp_path):
    from sirchmunk.storage.duckdb import DuckDBManager

    with DuckDBManager(db_path=":memory:") as db:
        db.execute("SELECT 1")
        db.durable_flush()
    assert not (tmp_path / "work/.control").exists()


def test_orphan_wal_rejects_empty_startup(managed):
    from sirchmunk.storage.duckdb import DuckDBManager

    path = managed.root / "missing.db"
    wal = Path(str(path) + ".wal")
    wal.write_bytes(b"unrecovered journal")
    with pytest.raises(IOError, match="WAL"):
        DuckDBManager(persist_path=str(path))
    assert wal.read_bytes() == b"unrecovered journal"
    assert not path.exists()
    assert managed.failure


def test_direct_file_close_failure_is_not_acknowledged(managed, monkeypatch):
    from sirchmunk.storage.duckdb import DuckDBManager

    db = DuckDBManager(db_path=str(managed.root / "close-failure.db"))
    db.execute("CREATE TABLE state (id INTEGER)")
    connection = db.connection

    class CloseFailure:
        def execute(self, *args, **kwargs):
            return connection.execute(*args, **kwargs)

        def close(self):
            raise OSError("injected close failure")

    with monkeypatch.context() as patch:
        patch.setattr(db, "connection", CloseFailure())
        with pytest.raises(OSError, match="close failure"):
            managed.flush()
    assert managed.failure
    assert not db._suspended


def test_direct_file_reopen_failure_is_fail_closed(managed, monkeypatch):
    from sirchmunk.storage.duckdb import DuckDBManager

    db = DuckDBManager(db_path=str(managed.root / "reopen-failure.db"))
    db.execute("CREATE TABLE state (id INTEGER)")
    managed.flush()

    def fail():
        raise OSError("injected reopen failure")

    with monkeypatch.context() as patch:
        patch.setattr(db, "_connect", fail)
        with pytest.raises(OSError, match="reopen failure"):
            db.fetch_one("SELECT count(*) FROM state")
    assert managed.failure == "database_reopen_failed"
    assert db._suspended
