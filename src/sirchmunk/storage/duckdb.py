# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DuckDB database manager for Sirchmunk
Provides a comprehensive interface for DuckDB operations including
connection management, table operations, data manipulation, and analytics.

Supports two operational modes:
1. Direct mode (default): connects to file-based or pure in-memory DB
2. Persist mode (persist_path): in-memory DB with periodic disk writeback
   via daemon thread, eliminating file locking issues for multi-process scenarios
"""

import os
import atexit
import duckdb
import threading
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from uuid import uuid4

from .lifecycle import (
    claim_work_root, get_runtime, durable_file, durable_directory, durable_replace,
)

logger = logging.getLogger(__name__)


def serialized_io(method):
    """Keep a synchronous storage operation (including result fetch) atomic."""
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._io_lock:
            return method(self, *args, **kwargs)
    return locked


def _claim_database_root(path, runtime):
    """Apply the same single-writer boundary to direct SDK database users."""
    if not path or str(path) == ":memory:":
        return runtime
    target = Path(path).expanduser().resolve()
    if runtime is not None and not target.is_relative_to(runtime.root):
        raise RuntimeError("Managed database path is outside its work root")
    configured = os.getenv("SIRCHMUNK_WORK_PATH")
    root = None
    if configured:
        configured = Path(configured).expanduser().resolve()
        if target.is_relative_to(configured):
            root = configured
    # A provisioned ancestor takes precedence over a deeper ad-hoc directory.
    # This also prevents callers from changing the environment to evade a
    # previously provisioned root's supervisor requirement.
    for parent in target.parents:
        control = parent / ".control"
        if ((control / "data-schema.json").exists()
                or (control / "work-root.lock").exists()):
            root = parent
            break
    root = root or (runtime.root if runtime else target.parent)
    claim_work_root(root)
    inferred = get_runtime(root)
    if runtime is not None and runtime is not inferred:
        raise RuntimeError("Database runtime does not own the claimed work root")
    if inferred is not None and not target.is_relative_to(inferred.root):
        raise RuntimeError("Managed database path is outside its work root")
    return inferred


class DuckDBManager:
    """
    A comprehensive DuckDB database manager providing common operations
    for data storage, retrieval, and analytics in the Sirchmunk system.

    Supports two operational modes:
    - Direct mode: Standard file-based or in-memory connection (original behavior)
    - Persist mode (persist_path set): In-memory DB with periodic disk writeback,
      designed to eliminate file locking conflicts in multi-process deployments.
      All read/write ops run in memory; a daemon thread periodically flushes
      dirty data to disk via atomic temp-file + rename.
    """

    # SQL keywords that indicate write operations (for dirty tracking)
    _WRITE_KEYWORDS = frozenset([
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'COPY'
    ])

    def __init__(self, db_path: Optional[str] = None, read_only: bool = False,
                 persist_path: Optional[str] = None,
                 sync_interval: int = 60,
                 sync_threshold: int = 100, runtime=None):
        """
        Initialize DuckDB connection

        Args:
            db_path: Path to database file. If None, creates in-memory database.
                     Ignored when persist_path is set.
            read_only: Whether to open database in read-only mode.
                       Ignored when persist_path is set.
            persist_path: Path to disk file for persistence. When set, uses
                          in-memory DB with periodic writeback (eliminates file locks).
            sync_interval: Seconds between periodic disk syncs (default: 60).
                           Only effective when persist_path is set.
            sync_threshold: Dirty write count that triggers immediate sync (default: 100).
                            Only effective when persist_path is set.
        """
        if db_path and str(db_path) != ":memory:":
            db_path = str(Path(db_path).expanduser().resolve())
        if persist_path:
            persist_path = str(Path(persist_path).expanduser().resolve())
        self.db_path = db_path
        self.read_only = read_only
        self.persist_path = persist_path
        self.connection = None
        self._runtime = _claim_database_root(persist_path or db_path, runtime)
        self._io_lock = self._runtime.io_lock if self._runtime else threading.RLock()
        self._initialized = False
        self._transaction_depth = 0
        self._suspended = False

        # Writeback state (only active in persist mode)
        self._dirty_count = 0
        self._sync_threshold = sync_threshold
        self._sync_lock = self._io_lock
        self._stop_event = threading.Event()
        self._daemon_thread = None
        self._shutdown_registered = False

        with self._io_lock:
            opening = False
            try:
                if self._runtime and (self._runtime.closing or self._runtime.closed):
                    raise RuntimeError("Managed database runtime is closing")
                persistent = persist_path or (db_path if db_path != ":memory:" else None)
                if self._runtime and persistent:
                    # Never permit two authoritative in-memory copies of the
                    # same file, including callers bypassing HistoryStorage.
                    key = str(Path(persistent).expanduser().resolve())
                    managers = getattr(self._runtime, "_duckdb_managers", {})
                    previous = managers.get(key)
                    if previous is not None and (previous.connection is not None or previous._suspended):
                        raise RuntimeError("Managed database already open; use the shared storage")
                opening = True
                if persist_path:
                    Path(persist_path).parent.mkdir(parents=True, exist_ok=True)
                    self.connection = duckdb.connect(":memory:")
                    self._load_from_disk()
                else:
                    self._connect()
                self._initialized = True
                if self._runtime and persistent:
                    managers[key] = self
                    self._runtime._duckdb_managers = managers
                    self._runtime.register_storage(self)
            except BaseException:
                # Failed startup must never write a partial/empty replacement.
                if self.connection is not None:
                    self.connection.close()
                    self.connection = None
                if opening and self._runtime:
                    self._runtime.fail("database_initialization_failed")
                raise
        if persist_path and not self._runtime:
            self._start_sync_daemon(sync_interval)
            self._register_shutdown_hook()

    def _connect(self):
        """Establish database connection (direct mode only)"""
        try:
            if self.db_path:
                self.connection = duckdb.connect(self.db_path, read_only=self.read_only)
                logger.info(f"Connected to DuckDB at {self.db_path}")
            else:
                self.connection = duckdb.connect(":memory:")
                logger.info("Connected to in-memory DuckDB")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise

    def _resume_direct(self):
        """Reopen a direct file connection released by the managed barrier.

        The root lock and runtime io_lock remain owned while its native file
        handle is released, so no cooperating writer can enter this interval.
        """
        if self._suspended:
            if self._runtime and (self._runtime.closing or self._runtime.closed):
                raise RuntimeError("Managed database runtime is closing")
            try:
                self._connect()
                self._suspended = False
            except Exception:
                if self._runtime:
                    self._runtime.fail("database_reopen_failed")
                raise

    # ------------------------------------------------------------------ #
    #  Persist mode: disk load / sync / daemon thread / shutdown hook     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _checkpoint_wal(db_file: str):
        """
        Checkpoint a stale WAL file by briefly opening the database in
        read-write mode.  DuckDB cannot open a file in READ_ONLY mode when
        a WAL file exists because recovery requires write access.  This
        helper applies the WAL and removes it so that subsequent READ_ONLY
        ATTACH operations succeed.
        """
        wal_file = Path(f"{db_file}.wal")
        if not wal_file.exists():
            return

        logger.info(
            f"Stale WAL detected at {wal_file}, checkpointing before load"
        )
        # A WAL may contain the only committed copy of a write. Never discard
        # it to make a damaged database appear readable.
        with duckdb.connect(db_file) as tmp_conn:
            tmp_conn.execute("CHECKPOINT")
        durable_file(Path(db_file))
        durable_directory(Path(db_file).parent)

    @serialized_io
    def _load_from_disk(self):
        """
        Load all tables from disk DuckDB file into in-memory database.
        Uses ATTACH (READ_ONLY) to briefly open the disk file, copy tables,
        then DETACH. READ_ONLY prevents exclusive file locks so multiple
        processes can load from the same disk file concurrently.

        Before loading, any stale WAL file is checkpointed so that
        READ_ONLY ATTACH does not fail.
        """
        if not self.persist_path or not Path(self.persist_path).exists():
            if self.persist_path and Path(f"{self.persist_path}.wal").exists():
                raise IOError("Database is missing but its recovery WAL still exists")
            logger.info(
                f"No disk file at {self.persist_path}, starting with empty database"
            )
            return

        # Checkpoint stale WAL (if any) so READ_ONLY ATTACH succeeds
        self._checkpoint_wal(self.persist_path)

        try:
            escaped_path = str(self.persist_path).replace("'", "''")
            self.connection.execute(
                f"ATTACH '{escaped_path}' AS disk_db (READ_ONLY)"
            )

            # COPY FROM DATABASE preserves constraints, indexes and sequences;
            # CREATE TABLE AS would silently drop PRIMARY KEY on every restart.
            self.connection.execute("COPY FROM DATABASE disk_db TO memory")

            self.connection.execute("DETACH disk_db")
            logger.info(f"Loaded database from {self.persist_path}")

        except Exception as e:
            logger.error(f"Failed to load from disk {self.persist_path}: {e}")
            try:
                self.connection.execute("DETACH disk_db")
            except Exception:
                pass
            raise

    def sync_to_disk(self, strict: bool = False):
        """
        Sync in-memory data to disk file atomically.
        Uses ATTACH to a temp DuckDB file, copies all tables, then
        performs atomic os.replace() to swap the disk file.
        Thread-safe via _sync_lock.
        """
        with self._sync_lock:
            if self._suspended:
                # No writes since the last verified checkpoint. Leave the file
                # handle closed so the root's final fsync/backup can read it.
                if strict:
                    try:
                        durable_file(Path(self.db_path))
                        durable_directory(Path(self.db_path).parent)
                        self._verify_disk(self.db_path)
                    except Exception:
                        if self._runtime:
                            self._runtime.fail("database_durable_flush_failed")
                        raise
                return
            if not self._initialized or self.connection is None:
                if strict:
                    raise RuntimeError("Cannot flush an uninitialized/closed database")
                return
            if self._transaction_depth:
                raise RuntimeError("Cannot flush inside a database transaction")
            if not self.persist_path:
                if strict and self.db_path and self.db_path != ":memory:":
                    try:
                        if not self.read_only:
                            self.connection.execute("CHECKPOINT")
                        # Windows DuckDB denies a second writable file handle.
                        # Close under the shared IO lock before fsync/verification;
                        # managed consumers resume lazily after the full barrier.
                        self.connection.close()
                        self.connection = None
                        self._suspended = True
                        durable_file(Path(self.db_path))
                        durable_directory(Path(self.db_path).parent)
                        self._verify_disk(self.db_path)
                        self._dirty_count = 0
                    except Exception:
                        if self._runtime:
                            self._runtime.fail("database_durable_flush_failed")
                        raise
                    finally:
                        if not self._runtime:
                            self._resume_direct()
                return
            if self._dirty_count == 0 and not strict:
                return

            temp_path = f"{self.persist_path}.{uuid4().hex}.tmp"
            try:
                escaped_temp = temp_path.replace("'", "''")
                self.connection.execute(f"ATTACH '{escaped_temp}' AS sync_db")

                try:
                    self.connection.execute("COPY FROM DATABASE memory TO sync_db")
                    self.connection.execute("CHECKPOINT sync_db")
                    self.connection.execute("DETACH sync_db")
                except Exception:
                    # Ensure DETACH even on failure
                    try:
                        self.connection.execute("DETACH sync_db")
                    except Exception:
                        pass
                    raise

                self._checkpoint_wal(temp_path)
                self._verify_disk(temp_path)
                durable_file(Path(temp_path))
                if Path(f"{self.persist_path}.wal").exists():
                    raise RuntimeError("Refusing to replace a database with an unrecovered WAL")
                durable_replace(Path(temp_path), Path(self.persist_path))
                durable_directory(Path(self.persist_path).parent)
                self._verify_disk(self.persist_path)

                synced_ops = self._dirty_count
                self._dirty_count = 0
                logger.info(
                    f"Synced database ({synced_ops} dirty ops) "
                    f"to {self.persist_path}"
                )

            except Exception as e:
                logger.error(f"Failed to sync to disk: {e}")
                # Keep dirty even if a strict forced sync began at zero.
                self._dirty_count = max(1, self._dirty_count)
                # Clean up temp file and its WAL on failure
                for p in (temp_path, f"{temp_path}.wal"):
                    if Path(p).exists():
                        try:
                            Path(p).unlink()
                        except Exception:
                            pass
                if strict or self._runtime:
                    if self._runtime:
                        self._runtime.fail("database_durable_flush_failed")
                    raise

    @staticmethod
    def _verify_disk(path):
        """Reopen the checkpointed snapshot before acknowledging durability."""
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise IOError("Database snapshot is missing or empty")
        with duckdb.connect(str(path), read_only=True) as check:
            tables = check.execute(
                "SELECT schema_name, table_name FROM duckdb_tables() "
                "WHERE NOT internal"
            ).fetchall()
            for schema, table in tables:
                schema = schema.replace('"', '""')
                table = table.replace('"', '""')
                check.execute(f'SELECT * FROM "{schema}"."{table}" LIMIT 1').fetchall()

    def durable_flush(self):
        """Synchronous flush: any checkpoint, fsync or verification error raises."""
        self.sync_to_disk(strict=True)

    @serialized_io
    def force_sync(self):
        """Force immediate sync to disk regardless of dirty count"""
        if self.persist_path and self.connection:
            if self._dirty_count == 0:
                self._dirty_count = 1
            self.sync_to_disk(strict=bool(self._runtime))

    def _mark_dirty(self):
        """Increment dirty counter and trigger sync if threshold reached"""
        self._dirty_count += 1
        if (self.persist_path and not self._runtime and not self._transaction_depth
                and self._dirty_count >= self._sync_threshold):
            self.sync_to_disk()

    def _start_sync_daemon(self, interval: int):
        """Start a daemon thread for periodic disk sync"""
        def _sync_loop():
            while not self._stop_event.is_set():
                self._stop_event.wait(interval)
                if not self._stop_event.is_set():
                    try:
                        self.sync_to_disk()
                    except Exception as e:
                        logger.error(f"Daemon sync failed: {e}")

        thread_name = f"duckdb-sync-{Path(self.persist_path).stem}"
        self._daemon_thread = threading.Thread(
            target=_sync_loop, daemon=True, name=thread_name
        )
        self._daemon_thread.start()
        logger.info(
            f"Started sync daemon '{thread_name}' "
            f"(interval={interval}s, threshold={self._sync_threshold})"
        )

    def _register_shutdown_hook(self):
        """Register atexit handler for graceful shutdown with final sync"""
        if self._shutdown_registered:
            return
        atexit.register(self._shutdown_sync)
        self._shutdown_registered = True
        logger.debug("Registered atexit shutdown hook for disk sync")

    def _shutdown_sync(self):
        """Stop daemon thread and perform final sync (called by atexit)"""
        if self._runtime:
            return  # Shared storage is flushed/closed by the root runtime.
        self._stop_event.set()
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_thread.join(timeout=5)

        # Force a final sync regardless of dirty count
        if self.persist_path and self.connection:
            if self._dirty_count == 0:
                self._dirty_count = 1  # Force sync
            try:
                self.sync_to_disk()
            except Exception as e:
                logger.error(f"Final shutdown sync failed: {e}")

    # ------------------------------------------------------------------ #
    #  Core database operations (API unchanged)                           #
    # ------------------------------------------------------------------ #

    def close(self):
        """Close database connection (with final sync in persist mode)"""
        if self._runtime and not self._runtime.closing:
            return
        if self._runtime:
            self.durable_close()
            return
        if self.persist_path:
            self._shutdown_sync()
        with self._io_lock:
            if self.connection:
                self.connection.close()
                self.connection = None
                logger.info("DuckDB connection closed")
            self._suspended = False
            if self._shutdown_registered:
                atexit.unregister(self._shutdown_sync)
                self._shutdown_registered = False

    @serialized_io
    def durable_close(self):
        """Final close; failed durability keeps the live connection recoverable."""
        if self.connection is not None or self._suspended:
            self.durable_flush()
            if self.connection is not None:
                self.connection.close()
            self.connection = None
            self._suspended = False

    @contextmanager
    def transaction(self):
        """Context manager for database transactions"""
        with self._io_lock:
            self._resume_direct()
            self.connection.begin()
            self._transaction_depth += 1
            try:
                yield self.connection
                self.connection.commit()
                if self.persist_path:
                    self._dirty_count += 1
            except BaseException:
                self.connection.rollback()
                raise
            finally:
                self._transaction_depth -= 1

    @serialized_io
    def execute(self, query: str, parameters: Optional[List] = None):
        """
        Execute SQL query

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Query result
        """
        try:
            self._resume_direct()
            if parameters:
                result = self.connection.execute(query, parameters)
            else:
                result = self.connection.execute(query)

            # Track memory-only writes too: the knowledge store persists them
            # to Parquet and must not reload a stale snapshot over raw SQL edits.
            first_word = (
                query.strip().split(maxsplit=1)[0].upper()
                if query.strip() else ""
            )
            if first_word in self._WRITE_KEYWORDS:
                self._mark_dirty()

            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise

    @serialized_io
    def fetch_all(self, query: str, parameters: Optional[List] = None) -> List[Tuple]:
        """
        Execute query and fetch all results

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            List of result tuples
        """
        result = self.execute(query, parameters)
        return result.fetchall()

    @serialized_io
    def fetch_one(self, query: str, parameters: Optional[List] = None) -> Optional[Tuple]:
        """
        Execute query and fetch one result

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Single result tuple or None
        """
        result = self.execute(query, parameters)
        return result.fetchone()

    @serialized_io
    def fetch_df(self, query: str, parameters: Optional[List] = None) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Results as DataFrame
        """
        result = self.execute(query, parameters)
        return result.df()

    def create_table(self, table_name: str, schema: Dict[str, str], if_not_exists: bool = True):
        """
        Create table with specified schema

        Args:
            table_name: Name of the table
            schema: Dictionary mapping column names to types
            if_not_exists: Whether to use IF NOT EXISTS clause
        """
        columns = ", ".join([f"{col} {dtype}" for col, dtype in schema.items()])
        if_not_exists_clause = "IF NOT EXISTS" if if_not_exists else ""

        query = f"CREATE TABLE {if_not_exists_clause} {table_name} ({columns})"
        self.execute(query)
        logger.info(f"Table {table_name} created successfully")

    def drop_table(self, table_name: str, if_exists: bool = True):
        """
        Drop table

        Args:
            table_name: Name of the table to drop
            if_exists: Whether to use IF EXISTS clause
        """
        if_exists_clause = "IF EXISTS" if if_exists else ""
        query = f"DROP TABLE {if_exists_clause} {table_name}"
        self.execute(query)
        logger.info(f"Table {table_name} dropped successfully")

    @serialized_io
    def insert_data(self, table_name: str, data: Union[Dict, List[Dict], pd.DataFrame]):
        """
        Insert data into table

        Args:
            table_name: Target table name
            data: Data to insert (dict, list of dicts, or DataFrame)
        """
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            if not data:
                return

            columns = list(data[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            column_names = ", ".join(columns)

            query = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

            for row in data:
                values = [row.get(col) for col in columns]
                self.execute(query, values)

        elif isinstance(data, pd.DataFrame):
            # Use DuckDB's efficient DataFrame insertion
            self._resume_direct()
            self.connection.register("temp_df", data)
            try:
                self.execute(f"INSERT INTO {table_name} SELECT * FROM temp_df")
            finally:
                self.connection.unregister("temp_df")

        logger.info(f"Data inserted into {table_name}")

    def update_data(self, table_name: str, set_clause: Dict[str, Any],
                   where_clause: str, where_params: Optional[List] = None):
        """
        Update data in table

        Args:
            table_name: Target table name
            set_clause: Dictionary of column-value pairs to update
            where_clause: WHERE condition
            where_params: Parameters for WHERE clause
        """
        set_parts = [f"{col} = ?" for col in set_clause.keys()]
        set_string = ", ".join(set_parts)

        query = f"UPDATE {table_name} SET {set_string} WHERE {where_clause}"
        params = list(set_clause.values())
        if where_params:
            params.extend(where_params)

        self.execute(query, params)
        logger.info(f"Data updated in {table_name}")

    def delete_data(self, table_name: str, where_clause: str, where_params: Optional[List] = None):
        """
        Delete data from table

        Args:
            table_name: Target table name
            where_clause: WHERE condition
            where_params: Parameters for WHERE clause
        """
        query = f"DELETE FROM {table_name} WHERE {where_clause}"
        self.execute(query, where_params)
        logger.info(f"Data deleted from {table_name}")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
        """
        result = self.fetch_one(query, [table_name])
        return result[0] > 0 if result else False

    def get_table_info(self, table_name: str) -> List[Dict]:
        """
        Get table schema information

        Args:
            table_name: Name of the table

        Returns:
            List of column information dictionaries
        """
        query = f"DESCRIBE {table_name}"
        result = self.fetch_all(query)

        columns = []
        for row in result:
            columns.append({
                "column_name": row[0],
                "column_type": row[1],
                "null": row[2],
                "key": row[3] if len(row) > 3 else None,
                "default": row[4] if len(row) > 4 else None,
                "extra": row[5] if len(row) > 5 else None
            })

        return columns

    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for table

        Args:
            table_name: Name of the table

        Returns:
            Number of rows in table
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.fetch_one(query)
        return result[0] if result else 0

    def list_tables(self) -> List[str]:
        """
        Get list of all tables in database

        Returns:
            List of table names
        """
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        result = self.fetch_all(query)
        return [row[0] for row in result]

    def export_to_csv(self, table_name: str, file_path: str, delimiter: str = ","):
        """
        Export table data to CSV file

        Args:
            table_name: Source table name
            file_path: Output CSV file path
            delimiter: CSV delimiter
        """
        query = f"COPY {table_name} TO '{file_path}' (DELIMITER '{delimiter}', HEADER)"
        self.execute(query)
        logger.info(f"Table {table_name} exported to {file_path}")

    def import_from_csv(self, table_name: str, file_path: str,
                       delimiter: str = ",", header: bool = True,
                       create_table: bool = True):
        """
        Import data from CSV file

        Args:
            table_name: Target table name
            file_path: CSV file path
            delimiter: CSV delimiter
            header: Whether CSV has header row
            create_table: Whether to create table automatically
        """
        if create_table:
            # Let DuckDB auto-detect schema and create table
            query = f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path}', delim='{delimiter}', header={header})
            """
        else:
            # Insert into existing table
            query = f"""
            INSERT INTO {table_name}
            SELECT * FROM read_csv_auto('{file_path}', delim='{delimiter}', header={header})
            """

        self.execute(query)
        logger.info(f"Data imported from {file_path} to {table_name}")

    def export_to_parquet(self, table_name: str, file_path: str):
        """
        Export table data to Parquet file

        Args:
            table_name: Source table name
            file_path: Output Parquet file path
        """
        self.execute(f"COPY {table_name} TO ? (FORMAT PARQUET)", [str(file_path)])
        logger.info(f"Table {table_name} exported to Parquet: {file_path}")

    def import_from_parquet(self, table_name: str, file_path: str, create_table: bool = True):
        """
        Import data from Parquet file

        Args:
            table_name: Target table name
            file_path: Parquet file path
            create_table: Whether to create table automatically
        """
        if create_table:
            query = f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
        else:
            query = f"INSERT INTO {table_name} SELECT * FROM read_parquet('{file_path}')"

        self.execute(query)
        logger.info(f"Data imported from Parquet {file_path} to {table_name}")

    def create_index(self, table_name: str, column_names: Union[str, List[str]],
                    index_name: Optional[str] = None):
        """
        Create index on table columns

        Args:
            table_name: Target table name
            column_names: Column name(s) for index
            index_name: Optional custom index name
        """
        if isinstance(column_names, str):
            column_names = [column_names]

        columns_str = ", ".join(column_names)

        if not index_name:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"

        query = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
        self.execute(query)
        logger.info(f"Index {index_name} created on {table_name}({columns_str})")

    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive table statistics

        Args:
            table_name: Name of the table to analyze

        Returns:
            Dictionary containing table statistics
        """
        # Basic table info
        row_count = self.get_table_count(table_name)
        columns = self.get_table_info(table_name)

        # Column statistics
        column_stats = {}
        for col in columns:
            col_name = col["column_name"]
            col_type = col["column_type"]

            if "INT" in col_type.upper() or "FLOAT" in col_type.upper() or "DOUBLE" in col_type.upper():
                # Numeric column statistics
                stats_query = f"""
                SELECT
                    MIN({col_name}) as min_val,
                    MAX({col_name}) as max_val,
                    AVG({col_name}) as avg_val,
                    COUNT(DISTINCT {col_name}) as distinct_count,
                    COUNT({col_name}) as non_null_count
                FROM {table_name}
                """
                stats = self.fetch_one(stats_query)
                if stats:
                    column_stats[col_name] = {
                        "type": "numeric",
                        "min": stats[0],
                        "max": stats[1],
                        "avg": stats[2],
                        "distinct_count": stats[3],
                        "non_null_count": stats[4],
                        "null_count": row_count - stats[4]
                    }
            else:
                # Text/other column statistics
                stats_query = f"""
                SELECT
                    COUNT(DISTINCT {col_name}) as distinct_count,
                    COUNT({col_name}) as non_null_count
                FROM {table_name}
                """
                stats = self.fetch_one(stats_query)
                if stats:
                    column_stats[col_name] = {
                        "type": "categorical",
                        "distinct_count": stats[0],
                        "non_null_count": stats[1],
                        "null_count": row_count - stats[1]
                    }

        return {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
            "column_statistics": column_stats,
            "analyzed_at": datetime.now().isoformat()
        }

    def search_tables(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for tables and columns containing the search term

        Args:
            search_term: Term to search for

        Returns:
            List of matching tables and columns
        """
        search_term = search_term.lower()
        results = []

        # Search table names
        tables = self.list_tables()
        for table in tables:
            if search_term in table.lower():
                results.append({
                    "type": "table",
                    "table_name": table,
                    "match_type": "table_name",
                    "match_value": table
                })

        # Search column names
        for table in tables:
            columns = self.get_table_info(table)
            for col in columns:
                if search_term in col["column_name"].lower():
                    results.append({
                        "type": "column",
                        "table_name": table,
                        "column_name": col["column_name"],
                        "column_type": col["column_type"],
                        "match_type": "column_name",
                        "match_value": col["column_name"]
                    })

        return results

    def backup_database(self, backup_path: str):
        """
        Create database backup

        Args:
            backup_path: Path for backup file
        """
        if not self.db_path and not self.persist_path:
            raise ValueError("Cannot backup pure in-memory database")

        query = f"EXPORT DATABASE '{backup_path}'"
        self.execute(query)
        logger.info(f"Database backed up to {backup_path}")

    def restore_database(self, backup_path: str):
        """
        Restore database from backup

        Args:
            backup_path: Path to backup file
        """
        query = f"IMPORT DATABASE '{backup_path}'"
        self.execute(query)
        logger.info(f"Database restored from {backup_path}")

    def optimize_database(self):
        """Run database optimization operations"""
        try:
            # Analyze all tables for query optimization
            tables = self.list_tables()
            for table in tables:
                self.execute(f"ANALYZE {table}")

            # Run VACUUM to reclaim space
            self.execute("VACUUM")

            logger.info("Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise

    def get_database_size(self) -> Dict[str, Any]:
        """
        Get database size information

        Returns:
            Dictionary with size information
        """
        path_to_check = self.db_path or self.persist_path
        if not path_to_check:
            return {"type": "in_memory", "size": "N/A"}

        try:
            db_file = Path(path_to_check)
            if db_file.exists():
                size_bytes = db_file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                return {
                    "type": "file" if self.db_path else "persist",
                    "path": str(db_file),
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "size_human": f"{size_mb:.2f} MB" if size_mb < 1024 else f"{size_mb/1024:.2f} GB"
                }
            else:
                return {"type": "file", "path": str(db_file), "exists": False}
        except Exception as e:
            logger.error(f"Failed to get database size: {e}")
            return {"type": "error", "error": str(e)}

    def execute_script(self, script_path: str):
        """
        Execute SQL script from file

        Args:
            script_path: Path to SQL script file
        """
        script_file = Path(script_path)
        if not script_file.exists():
            raise FileNotFoundError(f"Script file not found: {script_path}")

        with open(script_file, 'r', encoding='utf-8') as f:
            script_content = f.read()

        # Split script into individual statements
        statements = [stmt.strip() for stmt in script_content.split(';') if stmt.strip()]

        for statement in statements:
            self.execute(statement)

        logger.info(f"SQL script executed: {script_path}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed"""
        if hasattr(self, 'connection') and self.connection:
            try:
                self.close()
            except Exception:
                logger.exception("Database finalizer could not flush; explicit close is required")


# Utility functions for common operations
def create_knowledge_base_tables(db_manager: DuckDBManager):
    """Create standard tables for knowledge base operations"""

    # Documents table
    documents_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "kb_name": "VARCHAR NOT NULL",
        "filename": "VARCHAR NOT NULL",
        "file_path": "VARCHAR",
        "file_size": "BIGINT",
        "file_type": "VARCHAR",
        "content": "TEXT",
        "metadata": "JSON",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("documents", documents_schema)

    # Chunks table for RAG
    chunks_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "document_id": "VARCHAR NOT NULL",
        "kb_name": "VARCHAR NOT NULL",
        "chunk_index": "INTEGER NOT NULL",
        "content": "TEXT NOT NULL",
        "embedding": "FLOAT[]",
        "metadata": "JSON",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("chunks", chunks_schema)

    # Search history table
    search_history_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "kb_name": "VARCHAR NOT NULL",
        "query": "TEXT NOT NULL",
        "results_count": "INTEGER",
        "response_time_ms": "INTEGER",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("search_history", search_history_schema)

    logger.info("Knowledge base tables created successfully")


def create_analytics_tables(db_manager: DuckDBManager):
    """Create tables for analytics and monitoring"""

    # User activities table
    activities_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "user_id": "VARCHAR",
        "activity_type": "VARCHAR NOT NULL",
        "activity_data": "JSON",
        "duration_ms": "INTEGER",
        "success": "BOOLEAN DEFAULT TRUE",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("user_activities", activities_schema)

    # System metrics table
    metrics_schema = {
        "id": "VARCHAR PRIMARY KEY",
        "metric_name": "VARCHAR NOT NULL",
        "metric_value": "DOUBLE NOT NULL",
        "metric_unit": "VARCHAR",
        "tags": "JSON",
        "recorded_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    }
    db_manager.create_table("system_metrics", metrics_schema)

    logger.info("Analytics tables created successfully")
