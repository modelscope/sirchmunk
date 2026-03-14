# Copyright (c) ModelScope Contributors. All rights reserved.
"""FeedbackMemory — implicit/explicit signal storage.

Stores raw :class:`FeedbackSignal` records in ``feedback.duckdb``
(table ``signals``).  Signal dispatch to other memory layers is
handled by :class:`RetrievalMemory` (the manager), keeping this
store dependency-free.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore
from .schemas import FeedbackSignal

try:
    from sirchmunk.storage.duckdb import DuckDBManager
except ImportError:
    DuckDBManager = None  # type: ignore[assignment,misc]


class FeedbackMemory(MemoryStore):
    """Append-only time-series store for search feedback signals."""

    def __init__(self, db: "DuckDBManager"):
        self._db = db

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "FeedbackMemory"

    def initialize(self) -> None:
        if self._db.table_exists("signals"):
            return
        self._db.create_table("signals", {
            "id": "VARCHAR PRIMARY KEY",
            "signal_type": "VARCHAR NOT NULL",
            "query": "VARCHAR NOT NULL",
            "answer_found": "BOOLEAN DEFAULT FALSE",
            "cluster_confidence": "FLOAT DEFAULT 0.0",
            "react_loops": "INTEGER DEFAULT 0",
            "files_read_count": "INTEGER DEFAULT 0",
            "files_useful_count": "INTEGER DEFAULT 0",
            "total_tokens": "INTEGER DEFAULT 0",
            "latency_sec": "FLOAT DEFAULT 0.0",
            "mode": "VARCHAR DEFAULT 'FAST'",
            "keywords_json": "VARCHAR DEFAULT '[]'",
            "paths_json": "VARCHAR DEFAULT '[]'",
            "files_read_json": "VARCHAR DEFAULT '[]'",
            "user_verdict": "VARCHAR",
            "em_score": "FLOAT",
            "f1_score": "FLOAT",
            "llm_judge_verdict": "VARCHAR",
            "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        })

    def decay(self, now: Optional[datetime] = None) -> int:
        return 0

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        max_entries = max_entries or 100000
        try:
            count = self._db.get_table_count("signals")
            if count <= max_entries:
                return 0
            excess = count - max_entries
            self._db.execute(
                f"DELETE FROM signals WHERE rowid IN ("
                f"  SELECT rowid FROM signals "
                f"  ORDER BY timestamp ASC LIMIT {excess}"
                f")"
            )
            return excess
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        try:
            count = self._db.get_table_count("signals")
            row = self._db.fetch_one(
                "SELECT AVG(latency_sec), "
                "SUM(CASE WHEN answer_found THEN 1 ELSE 0 END) "
                "FROM signals"
            )
            avg_latency = row[0] if row and row[0] else 0.0
            success_count = int(row[1]) if row and row[1] else 0
        except Exception:
            count, avg_latency, success_count = 0, 0.0, 0
        return {
            "name": self.name,
            "signals_count": count,
            "avg_latency": round(avg_latency, 2),
            "success_count": success_count,
        }

    def close(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────

    def record_signal(self, signal: FeedbackSignal) -> None:
        """Persist a feedback signal."""
        sig_id = uuid.uuid4().hex[:20]
        try:
            self._db.insert_data("signals", {
                "id": sig_id,
                "signal_type": signal.signal_type,
                "query": signal.query,
                "answer_found": signal.answer_found,
                "cluster_confidence": signal.cluster_confidence,
                "react_loops": signal.react_loops,
                "files_read_count": signal.files_read_count,
                "files_useful_count": signal.files_useful_count,
                "total_tokens": signal.total_tokens,
                "latency_sec": signal.latency_sec,
                "mode": signal.mode,
                "keywords_json": json.dumps(
                    signal.keywords_used, ensure_ascii=False
                ),
                "paths_json": json.dumps(
                    signal.paths_searched, ensure_ascii=False
                ),
                "files_read_json": json.dumps(
                    signal.files_read, ensure_ascii=False
                ),
                "user_verdict": signal.user_verdict,
                "em_score": signal.em_score,
                "f1_score": signal.f1_score,
                "llm_judge_verdict": signal.llm_judge_verdict,
                "timestamp": signal.timestamp,
            })
        except Exception as exc:
            logger.debug(f"FeedbackMemory: record failed: {exc}")

    def recent_signals(
        self,
        limit: int = 50,
        mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the most recent signals (newest first)."""
        try:
            where = ""
            params: list = []
            if mode:
                where = "WHERE mode = ?"
                params.append(mode)
            rows = self._db.fetch_all(
                f"SELECT * FROM signals {where} "
                f"ORDER BY timestamp DESC LIMIT ?",
                params + [limit],
            )
            cols = [c["column_name"] for c in self._db.get_table_info("signals")]
            return [dict(zip(cols, row)) for row in rows]
        except Exception:
            return []
