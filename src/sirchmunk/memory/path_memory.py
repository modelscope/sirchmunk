# Copyright (c) ModelScope Contributors. All rights reserved.
"""PathMemory — file path hotness and utility statistics.

Tracks per-path retrieval frequency, usefulness ratio, and a composite
*hot_score* that can be used as a boost factor during BM25 reranking
or candidate merging.

Backed by the shared ``corpus.duckdb`` (table ``path_stats``).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore

try:
    from sirchmunk.storage.duckdb import DuckDBManager
except ImportError:
    DuckDBManager = None  # type: ignore[assignment,misc]


class PathMemory(MemoryStore):
    """Per-path retrieval statistics and hotness scoring."""

    def __init__(self, db: "DuckDBManager"):
        self._db = db

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "PathMemory"

    def initialize(self) -> None:
        if self._db.table_exists("path_stats"):
            return
        self._db.create_table("path_stats", {
            "path": "VARCHAR PRIMARY KEY",
            "total_retrievals": "INTEGER DEFAULT 0",
            "useful_retrievals": "INTEGER DEFAULT 0",
            "useful_ratio": "FLOAT DEFAULT 0.0",
            "avg_file_size": "FLOAT DEFAULT 0.0",
            "entity_density": "FLOAT DEFAULT 0.0",
            "last_useful": "TIMESTAMP",
            "hot_score": "FLOAT DEFAULT 0.0",
        })

    def decay(self, now: Optional[datetime] = None) -> int:
        try:
            row = self._db.fetch_one(
                "SELECT COUNT(*) FROM path_stats WHERE hot_score > 0.01"
            )
            count = row[0] if row else 0
            if count > 0:
                self._db.execute(
                    "UPDATE path_stats SET hot_score = hot_score * 0.95 "
                    "WHERE hot_score > 0.01"
                )
            return count
        except Exception:
            return 0

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        max_entries = max_entries or 20000
        try:
            count = self._db.get_table_count("path_stats")
            if count <= max_entries:
                return 0
            excess = count - max_entries
            self._db.execute(
                f"DELETE FROM path_stats WHERE rowid IN ("
                f"  SELECT rowid FROM path_stats "
                f"  ORDER BY hot_score ASC, total_retrievals ASC "
                f"  LIMIT {excess}"
                f")"
            )
            return excess
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        try:
            count = self._db.get_table_count("path_stats")
            row = self._db.fetch_one(
                "SELECT AVG(hot_score), MAX(hot_score) FROM path_stats"
            )
            avg_score = row[0] if row and row[0] else 0.0
            max_score = row[1] if row and row[1] else 0.0
        except Exception:
            count, avg_score, max_score = 0, 0.0, 0.0
        return {
            "name": self.name,
            "paths_tracked": count,
            "avg_hot_score": round(avg_score, 4),
            "max_hot_score": round(max_score, 4),
        }

    def close(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────

    def get_path_scores(self, paths: List[str]) -> Dict[str, float]:
        """Return ``{path: hot_score}`` for known paths."""
        if not paths:
            return {}
        try:
            placeholders = ", ".join(["?" for _ in paths])
            rows = self._db.fetch_all(
                f"SELECT path, hot_score FROM path_stats "
                f"WHERE path IN ({placeholders})",
                paths,
            )
            return {r[0]: r[1] for r in rows}
        except Exception:
            return {}

    def record_retrieval(
        self,
        path: str,
        useful: bool = False,
        file_size: float = 0.0,
    ) -> None:
        """Record one retrieval event for *path*."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._db.fetch_one(
                "SELECT total_retrievals, useful_retrievals FROM path_stats "
                "WHERE path = ?",
                [path],
            )
            if existing:
                total = existing[0] + 1
                useful_count = existing[1] + (1 if useful else 0)
                ratio = useful_count / max(total, 1)

                # For hot_score recency: use existing last_useful from DB
                # when the current retrieval is not useful, so recency
                # correctly reflects the age of the last *actually useful* hit.
                if useful:
                    last_useful_for_score = now
                else:
                    row_lu = self._db.fetch_one(
                        "SELECT last_useful FROM path_stats WHERE path = ?",
                        [path],
                    )
                    last_useful_for_score = (
                        str(row_lu[0]) if row_lu and row_lu[0] else None
                    )

                score = self._compute_hot_score(
                    ratio, total, last_useful_for_score,
                )
                update_cols = {
                    "total_retrievals": total,
                    "useful_retrievals": useful_count,
                    "useful_ratio": ratio,
                    "hot_score": score,
                }
                if file_size > 0:
                    update_cols["avg_file_size"] = file_size
                if useful:
                    update_cols["last_useful"] = now
                self._db.update_data(
                    "path_stats", update_cols, "path = ?", [path],
                )
            else:
                ratio = 1.0 if useful else 0.0
                score = self._compute_hot_score(
                    ratio, 1, now if useful else None,
                )
                self._db.insert_data("path_stats", {
                    "path": path,
                    "total_retrievals": 1,
                    "useful_retrievals": 1 if useful else 0,
                    "useful_ratio": ratio,
                    "avg_file_size": file_size,
                    "entity_density": 0.0,
                    "last_useful": now if useful else None,
                    "hot_score": score,
                })
        except Exception as exc:
            logger.debug(f"PathMemory: record_retrieval failed: {exc}")

    @staticmethod
    def _compute_hot_score(
        useful_ratio: float,
        total: int,
        last_useful_iso: Optional[str],
    ) -> float:
        """hot_score = useful_ratio × log₂(1 + total) × recency_factor."""
        recency = 1.0
        if last_useful_iso:
            try:
                last_dt = datetime.fromisoformat(last_useful_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                days = (datetime.now(timezone.utc) - last_dt).days
                recency = max(0.0, 1.0 - days / 90.0)
            except (ValueError, TypeError):
                pass
        return useful_ratio * math.log2(1 + total) * recency
