# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Chat History Storage using DuckDB
Provides persistent storage for chat sessions and messages.

Uses in-memory DuckDB with periodic disk writeback (persist mode)
to eliminate file locking issues in multi-process deployments.
"""

import os
import json
import threading
from uuid import uuid4
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from loguru import logger

from sirchmunk.storage.duckdb import DuckDBManager, serialized_io
from sirchmunk.storage.lifecycle import ManagedStorageMeta, get_runtime
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH


class HistoryStorage(metaclass=ManagedStorageMeta):
    """
    Manages persistent storage of chat history using DuckDB

    Architecture:
    - In-memory DuckDB for all read/write operations (zero lock contention)
    - Daemon thread in DuckDBManager syncs dirty data to disk periodically
    - Stores chat sessions and messages
    - Follows Single Responsibility Principle (SRP)
    - Decoupled from API layer (Dependency Inversion Principle)
    """

    def __init__(self, work_path: Optional[str] = None):
        """
        Initialize History Storage

        Args:
            work_path: Base work path. If None, uses SIRCHMUNK_WORK_PATH env variable
        """
        # Get work path from env if not provided, and expand ~ in path
        if work_path is None:
            work_path = os.getenv("SIRCHMUNK_WORK_PATH", DEFAULT_SIRCHMUNK_WORK_PATH)
        self._runtime = get_runtime(work_path)
        self._io_lock = self._runtime.io_lock if self._runtime else threading.RLock()
        self.db = None

        # Create history storage path (expand ~ and resolve to absolute path)
        self.history_path = Path(work_path).expanduser().resolve() / ".cache" / "history"
        if self._runtime and not self.history_path.resolve().is_relative_to(self._runtime.root):
            self._runtime.fail("history_path_outside_managed_root")
            raise ValueError("History path is outside the managed root")
        self.history_path.mkdir(parents=True, exist_ok=True)

        # Initialize DuckDB in persist mode (in-memory + periodic disk writeback)
        self.db_path = str(self.history_path / "chat_history.db")
        try:
            self.db = DuckDBManager(
                persist_path=self.db_path, sync_interval=60, sync_threshold=50,
                runtime=self._runtime,
            )
            self._io_lock = self.db._io_lock
            if self._runtime and Path(self.db_path).exists():
                self._validate_managed_schema()
            self._initialize_tables()
        except BaseException:
            if self.db is not None:
                # Do not write a partially initialized schema on startup failure.
                self.db._stop_event.set()
                with self._io_lock:
                    self.db.connection.close()
                    self.db.connection = None
            if self._runtime:
                self._runtime.fail("history_storage_initialization_failed")
            raise

        logger.info(f"History storage initialized at: {self.db_path}")

    @serialized_io
    def _validate_managed_schema(self):
        """Validate v0.2.0 data, including snapshots whose CTAS lost keys.

        This deliberately does not add constraints or migrate column types.
        A recognized legacy file remains usable; an unknown schema never gains
        a compatibility claim merely through CREATE TABLE IF NOT EXISTS.
        """
        expected = {
            "chat_sessions": {
                "session_id": "VARCHAR", "title": "VARCHAR",
                "created_at": "TIMESTAMP", "updated_at": "TIMESTAMP",
                "settings": "JSON", "message_count": "INTEGER",
            },
            "chat_messages": {
                "id": "VARCHAR", "session_id": "VARCHAR", "role": "VARCHAR",
                "content": "VARCHAR", "timestamp": "TIMESTAMP",
                "search_logs": "JSON", "is_streaming": "BOOLEAN",
            },
        }
        if set(self.db.list_tables()) != set(expected):
            raise ValueError("Unrecognized managed history tables")
        for table, columns in expected.items():
            actual = {row[0]: row[1] for row in self.db.fetch_all(f"DESCRIBE {table}")}
            # Several read APIs unpack SELECT * positionally, so order is part
            # of this format contract, not merely a presentation preference.
            if list(actual.items()) != list(columns.items()):
                raise ValueError(f"Incompatible managed history schema: {table}")

    def _initialize_tables(self):
        """Create database tables for chat history"""

        # Chat sessions table
        sessions_schema = {
            "session_id": "VARCHAR PRIMARY KEY",
            "title": "VARCHAR",
            "created_at": "TIMESTAMP NOT NULL",
            "updated_at": "TIMESTAMP NOT NULL",
            "settings": "JSON",
            "message_count": "INTEGER DEFAULT 0",
        }
        self.db.create_table("chat_sessions", sessions_schema, if_not_exists=True)

        # Chat messages table
        messages_schema = {
            "id": "VARCHAR PRIMARY KEY",
            "session_id": "VARCHAR NOT NULL",
            "role": "VARCHAR NOT NULL",
            "content": "TEXT NOT NULL",
            "timestamp": "TIMESTAMP NOT NULL",
            "search_logs": "JSON",
            "is_streaming": "BOOLEAN DEFAULT FALSE",
        }
        self.db.create_table("chat_messages", messages_schema, if_not_exists=True)

        # Create index for faster queries
        if not self.db.table_exists("chat_messages"):
            self.db.create_index("chat_messages", ["session_id"], "idx_messages_session")

    @serialized_io
    def save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save or update a chat session

        Args:
            session_data: Dictionary containing session information
                - session_id: str
                - title: str (optional)
                - created_at: str (ISO format)
                - updated_at: str (ISO format)
                - settings: dict (optional)
                - message_count: int (optional)

        Returns:
            True if successful
        """
        try:
            session_id = session_data["session_id"]

            # Check if session exists
            existing = self.db.fetch_one(
                "SELECT session_id FROM chat_sessions WHERE session_id = ?",
                [session_id]
            )

            # Prepare data
            data_to_save = {
                "session_id": session_id,
                "title": session_data.get("title", "Chat Session"),
                "created_at": session_data.get("created_at"),
                "updated_at": session_data.get("updated_at"),
                "settings": json.dumps(session_data.get("settings", {})),
                "message_count": session_data.get("message_count", 0),
            }

            if existing:
                # Update existing session
                set_clause = {k: v for k, v in data_to_save.items() if k != "session_id"}
                self.db.update_data(
                    "chat_sessions",
                    set_clause=set_clause,
                    where_clause="session_id = ?",
                    where_params=[session_id]
                )
                logger.debug(f"Updated session: {session_id}")
            else:
                # Insert new session
                self.db.insert_data("chat_sessions", data_to_save)
                logger.debug(f"Created new session: {session_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return False

    @serialized_io
    def save_message(self, session_id: str, message_data: Dict[str, Any]) -> bool:
        """
        Save a chat message

        Args:
            session_id: Session ID
            message_data: Dictionary containing message information
                - role: str ("user" or "assistant")
                - content: str
                - timestamp: str (ISO format) or int (Unix timestamp)
                - search_logs: list (optional)
                - is_streaming: bool (optional)

        Returns:
            True if successful
        """
        try:
            # Generate message ID
            message_id = f"{session_id}_{uuid4().hex}"

            # Handle timestamp conversion
            timestamp = message_data.get("timestamp")
            if isinstance(timestamp, int):
                timestamp = datetime.fromtimestamp(timestamp).isoformat()
            elif not timestamp:
                timestamp = datetime.now().isoformat()

            # Prepare data
            data_to_save = {
                "id": message_id,
                "session_id": session_id,
                "role": message_data["role"],
                "content": message_data["content"],
                "timestamp": timestamp,
                "search_logs": json.dumps(message_data.get("searchLogs", [])),
                "is_streaming": message_data.get("isStreaming", False),
            }

            # Insert message
            self.db.insert_data("chat_messages", data_to_save)

            # Update session message count
            self.db.execute(
                """
                UPDATE chat_sessions
                SET message_count = (
                    SELECT COUNT(*) FROM chat_messages WHERE session_id = ?
                ),
                updated_at = ?
                WHERE session_id = ?
                """,
                [session_id, datetime.now().isoformat(), session_id]
            )

            logger.debug(f"Saved message to session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save message: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return False

    @serialized_io
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chat session with all messages

        Args:
            session_id: Session ID

        Returns:
            Dictionary containing session data and messages, or None if not found
        """
        try:
            # Get session info
            session_row = self.db.fetch_one(
                "SELECT * FROM chat_sessions WHERE session_id = ?",
                [session_id]
            )

            if not session_row:
                return None

            # Parse session data
            session_data = {
                "session_id": session_row[0],
                "title": session_row[1],
                "created_at": session_row[2],
                "updated_at": session_row[3],
                "settings": json.loads(session_row[4]) if session_row[4] else {},
                "message_count": session_row[5],
            }

            # Get messages
            message_rows = self.db.fetch_all(
                """
                SELECT id, role, content, timestamp, search_logs, is_streaming
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                [session_id]
            )

            messages = []
            for row in message_rows:
                messages.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3],
                    "searchLogs": json.loads(row[4]) if row[4] else [],
                    "isStreaming": row[5],
                })

            session_data["messages"] = messages
            return session_data

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return None

    def get_all_sessions(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve all chat sessions (without messages)

        Args:
            limit: Maximum number of sessions to retrieve
            offset: Number of sessions to skip

        Returns:
            List of session dictionaries
        """
        try:
            rows = self.db.fetch_all(
                """
                SELECT session_id, title, created_at, updated_at, settings, message_count
                FROM chat_sessions
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                [limit, offset]
            )

            sessions = []
            for row in rows:
                sessions.append({
                    "session_id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "settings": json.loads(row[4]) if row[4] else {},
                    "message_count": row[5],
                })

            return sessions

        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return []

    @serialized_io
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session and all its messages

        Args:
            session_id: Session ID

        Returns:
            True if successful
        """
        try:
            # Delete messages first
            self.db.delete_data("chat_messages", "session_id = ?", [session_id])

            # Delete session
            self.db.delete_data("chat_sessions", "session_id = ?", [session_id])

            logger.info(f"Deleted session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return False

    def get_recent_messages(
        self, session_id: str, limit: int = 20,
    ) -> List[Dict[str, str]]:
        """Return the most recent messages in OpenAI chat format.

        Only ``role`` and ``content`` are included — suitable for direct
        injection into an LLM messages list.

        Args:
            session_id: Session to query.
            limit: Maximum number of messages to return.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts, ordered
            chronologically (oldest first).
        """
        try:
            rows = self.db.fetch_all(
                """
                SELECT role, content FROM (
                    SELECT role, content, timestamp
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ) sub
                ORDER BY timestamp ASC
                """,
                [session_id, limit],
            )
            return [{"role": r[0], "content": r[1]} for r in rows]
        except Exception as e:
            logger.warning(f"Failed to get recent messages for {session_id}: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return []

    def get_session_count(self) -> int:
        """Get total number of sessions"""
        return self.db.get_table_count("chat_sessions")

    def search_sessions(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search sessions by title or message content

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        try:
            # Search in session titles
            title_matches = self.db.fetch_all(
                """
                SELECT DISTINCT session_id, title, created_at, updated_at, settings, message_count
                FROM chat_sessions
                WHERE LOWER(title) LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                [f"%{query.lower()}%", limit]
            )

            # Search in message content
            content_matches = self.db.fetch_all(
                """
                SELECT DISTINCT s.session_id, s.title, s.created_at, s.updated_at, s.settings, s.message_count
                FROM chat_sessions s
                JOIN chat_messages m ON s.session_id = m.session_id
                WHERE LOWER(m.content) LIKE ?
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                [f"%{query.lower()}%", limit]
            )

            # Combine and deduplicate results
            session_ids = set()
            sessions = []

            for row in title_matches + content_matches:
                if row[0] not in session_ids:
                    session_ids.add(row[0])
                    sessions.append({
                        "session_id": row[0],
                        "title": row[1],
                        "created_at": row[2],
                        "updated_at": row[3],
                        "settings": json.loads(row[4]) if row[4] else {},
                        "message_count": row[5],
                    })

            return sessions[:limit]

        except Exception as e:
            logger.error(f"Failed to search sessions: {e}")
            if self._runtime:
                self._runtime.fail("history_operation_failed")
                raise
            return []

    def close(self):
        """Close database connection (triggers final sync to disk)"""
        if self._runtime and not self._runtime.closing:
            return
        if self.db:
            self.db.close()
            logger.info("History storage closed")

    @serialized_io
    def durable_flush(self):
        """Flush the real history database; errors propagate to the barrier."""
        if self.db is None:
            raise RuntimeError("History storage is not initialized")
        self.db.durable_flush()

    @serialized_io
    def durable_close(self):
        if self.db:
            self.db.durable_close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed"""
        if hasattr(self, 'db') and self.db:
            try:
                self.close()
            except Exception:
                logger.exception("History finalizer failed; explicit durable close required")
