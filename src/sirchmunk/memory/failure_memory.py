# Copyright (c) ModelScope Contributors. All rights reserved.
"""FailureMemory — noise keywords, dead paths, failed strategies.

Learns from unsuccessful searches to avoid repeating the same mistakes:

- **Noise keywords**: Keywords that produce thousands of matches but yield
  few useful files — recommendation is ``skip`` or ``modify``.
- **Dead paths**: File paths repeatedly retrieved yet never useful.
- **Failed strategies**: Parameter combinations that consistently fail for
  a given query pattern — prevents retrying known bad strategies.

Backed by the shared ``corpus.duckdb``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore

try:
    from sirchmunk.storage.duckdb import DuckDBManager
except ImportError:
    DuckDBManager = None  # type: ignore[assignment,misc]


class FailureMemory(MemoryStore):
    """Negative-pattern memory for noise keywords, dead paths, and
    failed strategy combinations.
    """

    _NOISE_SKIP_THRESHOLD = 0.1
    _NOISE_MODIFY_THRESHOLD = 0.25
    _NOISE_MIN_SAMPLES = 3
    _DEAD_PATH_MIN_RETRIEVALS = 5

    def __init__(self, db: "DuckDBManager"):
        self._db = db

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "FailureMemory"

    def initialize(self) -> None:
        self._ensure_tables()

    def decay(self, now: Optional[datetime] = None) -> int:
        count = 0
        try:
            if self._db.table_exists("failed_strategies"):
                row = self._db.fetch_one(
                    "SELECT COUNT(*) FROM failed_strategies "
                    "WHERE failure_count > 0"
                )
                n = row[0] if row else 0
                if n > 0:
                    self._db.execute(
                        "UPDATE failed_strategies SET failure_count = "
                        "GREATEST(failure_count - 1, 0)"
                    )
                count += n
        except Exception:
            pass
        return count

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        removed = 0
        try:
            if self._db.table_exists("failed_strategies"):
                row = self._db.fetch_one(
                    "SELECT COUNT(*) FROM failed_strategies "
                    "WHERE failure_count <= 0"
                )
                n = row[0] if row else 0
                if n > 0:
                    self._db.execute(
                        "DELETE FROM failed_strategies "
                        "WHERE failure_count <= 0"
                    )
                    removed += n
        except Exception:
            pass
        return removed

    def stats(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"name": self.name}
        for table in ("noise_keywords", "dead_paths", "failed_strategies"):
            try:
                result[f"{table}_count"] = self._db.get_table_count(table)
            except Exception:
                result[f"{table}_count"] = 0
        return result

    def close(self) -> None:
        pass

    # ── Table creation ────────────────────────────────────────────────

    def _ensure_tables(self) -> None:
        if not self._db.table_exists("noise_keywords"):
            self._db.create_table("noise_keywords", {
                "keyword": "VARCHAR PRIMARY KEY",
                "avg_files_found": "FLOAT DEFAULT 0.0",
                "avg_useful_ratio": "FLOAT DEFAULT 0.0",
                "sample_count": "INTEGER DEFAULT 0",
                "recommendation": "VARCHAR DEFAULT 'monitor'",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            })

        if not self._db.table_exists("dead_paths"):
            self._db.create_table("dead_paths", {
                "path": "VARCHAR PRIMARY KEY",
                "times_retrieved": "INTEGER DEFAULT 0",
                "times_useful": "INTEGER DEFAULT 0",
                "last_checked": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            })

        if not self._db.table_exists("failed_strategies"):
            self._db.create_table("failed_strategies", {
                "pattern_id": "VARCHAR NOT NULL",
                "params_hash": "VARCHAR NOT NULL",
                "failure_count": "INTEGER DEFAULT 1",
                "last_failed": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            })
            try:
                self._db.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_failed_strat "
                    "ON failed_strategies (pattern_id, params_hash)"
                )
            except Exception:
                pass

    # ── Seeding ────────────────────────────────────────────────────────

    _DEFAULT_NOISE_WORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "shall", "should", "may", "might", "must", "can",
        "could", "of", "in", "to", "for", "with", "on", "at", "from",
        "by", "about", "as", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "and", "but", "or", "nor", "not",
        "so", "yet", "both", "either", "neither", "each", "every",
        "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just",
        "because", "also", "however", "although", "though", "while",
        # Common CJK stop words
        "的", "了", "是", "在", "有", "和", "与", "及", "或", "也",
        "都", "就", "不", "但", "而", "这", "那", "它", "他", "她",
        "吗", "呢", "吧", "啊", "哪", "什么", "怎么", "为什么",
    })

    def seed_noise_keywords(self) -> int:
        """Pre-fill noise_keywords with common stop words (idempotent).

        Uses a single query to find existing keywords, then batch-inserts
        only the missing ones.  Returns the number of newly inserted keywords.
        """
        try:
            rows = self._db.fetch_all(
                "SELECT keyword FROM noise_keywords", [],
            )
            existing = {r[0] for r in rows} if rows else set()
        except Exception:
            existing = set()

        to_insert = [w for w in self._DEFAULT_NOISE_WORDS if w not in existing]
        if not to_insert:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        for word in to_insert:
            try:
                self._db.insert_data("noise_keywords", {
                    "keyword": word,
                    "avg_files_found": 9999.0,
                    "avg_useful_ratio": 0.0,
                    "sample_count": self._NOISE_MIN_SAMPLES,
                    "recommendation": "skip",
                    "updated_at": now,
                })
                inserted += 1
            except Exception:
                pass
        if inserted:
            logger.debug(f"FailureMemory: seeded {inserted} noise keywords")
        return inserted

    # ── Noise keywords ────────────────────────────────────────────────

    def filter_noise_keywords(self, keywords: List[str]) -> List[str]:
        """Remove keywords marked as ``skip``."""
        if not keywords:
            return keywords
        try:
            placeholders = ", ".join(["?" for _ in keywords])
            rows = self._db.fetch_all(
                f"SELECT keyword FROM noise_keywords "
                f"WHERE keyword IN ({placeholders}) "
                f"AND recommendation = 'skip'",
                [k.lower() for k in keywords],
            )
            noise_set = {r[0] for r in rows}
            return [k for k in keywords if k.lower() not in noise_set]
        except Exception:
            return keywords

    def get_keyword_recommendation(self, keyword: str) -> Optional[str]:
        """Return recommendation for *keyword*: ``skip``, ``modify``, or ``monitor``."""
        try:
            row = self._db.fetch_one(
                "SELECT recommendation FROM noise_keywords WHERE keyword = ?",
                [keyword.lower()],
            )
            return row[0] if row else None
        except Exception:
            return None

    def record_keyword_result(
        self,
        keyword: str,
        files_found: int,
        useful: bool,
    ) -> None:
        """Update noise statistics for *keyword*."""
        key = keyword.lower()
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._db.fetch_one(
                "SELECT avg_files_found, avg_useful_ratio, sample_count "
                "FROM noise_keywords WHERE keyword = ?",
                [key],
            )
            if existing:
                old_files, old_ratio, old_count = existing
                new_count = old_count + 1
                alpha = 0.3
                new_files = alpha * files_found + (1 - alpha) * old_files
                new_ratio = alpha * (1.0 if useful else 0.0) + (1 - alpha) * old_ratio
                rec = self._classify_noise(new_ratio, new_count)
                self._db.update_data(
                    "noise_keywords",
                    {
                        "avg_files_found": new_files,
                        "avg_useful_ratio": new_ratio,
                        "sample_count": new_count,
                        "recommendation": rec,
                        "updated_at": now,
                    },
                    "keyword = ?", [key],
                )
            else:
                ratio = 1.0 if useful else 0.0
                rec = "monitor"
                self._db.insert_data("noise_keywords", {
                    "keyword": key,
                    "avg_files_found": float(files_found),
                    "avg_useful_ratio": ratio,
                    "sample_count": 1,
                    "recommendation": rec,
                    "updated_at": now,
                })
        except Exception as exc:
            logger.debug(f"FailureMemory: keyword record failed: {exc}")

    def _classify_noise(self, useful_ratio: float, sample_count: int) -> str:
        if sample_count < self._NOISE_MIN_SAMPLES:
            return "monitor"
        if useful_ratio <= self._NOISE_SKIP_THRESHOLD:
            return "skip"
        if useful_ratio <= self._NOISE_MODIFY_THRESHOLD:
            return "modify"
        return "monitor"

    # ── Dead paths ────────────────────────────────────────────────────

    def filter_dead_paths(self, paths: List[str]) -> List[str]:
        """Remove paths that are persistently useless."""
        if not paths:
            return paths
        try:
            placeholders = ", ".join(["?" for _ in paths])
            rows = self._db.fetch_all(
                f"SELECT path FROM dead_paths "
                f"WHERE path IN ({placeholders}) "
                f"AND times_retrieved >= {self._DEAD_PATH_MIN_RETRIEVALS} "
                f"AND times_useful = 0",
                paths,
            )
            dead = {r[0] for r in rows}
            return [p for p in paths if p not in dead]
        except Exception:
            return paths

    def record_path_result(self, path: str, useful: bool) -> None:
        """Update dead-path tracking for *path*."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._db.fetch_one(
                "SELECT times_retrieved, times_useful FROM dead_paths "
                "WHERE path = ?",
                [path],
            )
            if existing:
                self._db.update_data(
                    "dead_paths",
                    {
                        "times_retrieved": existing[0] + 1,
                        "times_useful": existing[1] + (1 if useful else 0),
                        "last_checked": now,
                    },
                    "path = ?", [path],
                )
            else:
                self._db.insert_data("dead_paths", {
                    "path": path,
                    "times_retrieved": 1,
                    "times_useful": 1 if useful else 0,
                    "last_checked": now,
                })
        except Exception as exc:
            logger.debug(f"FailureMemory: path record failed: {exc}")

    # ── Failed strategies ─────────────────────────────────────────────

    def is_strategy_failed(
        self,
        pattern_id: str,
        params_hash: str,
        threshold: int = 3,
    ) -> bool:
        """Return True if this (pattern, params) combo has failed repeatedly."""
        try:
            row = self._db.fetch_one(
                "SELECT failure_count FROM failed_strategies "
                "WHERE pattern_id = ? AND params_hash = ?",
                [pattern_id, params_hash],
            )
            return (row[0] >= threshold) if row else False
        except Exception:
            return False

    def record_strategy_failure(
        self,
        pattern_id: str,
        params_hash: str,
    ) -> None:
        """Increment failure count for a strategy combination."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._db.fetch_one(
                "SELECT failure_count FROM failed_strategies "
                "WHERE pattern_id = ? AND params_hash = ?",
                [pattern_id, params_hash],
            )
            if existing:
                self._db.execute(
                    "UPDATE failed_strategies SET failure_count = ?, "
                    "last_failed = ? "
                    "WHERE pattern_id = ? AND params_hash = ?",
                    [existing[0] + 1, now, pattern_id, params_hash],
                )
            else:
                self._db.insert_data("failed_strategies", {
                    "pattern_id": pattern_id,
                    "params_hash": params_hash,
                    "failure_count": 1,
                    "last_failed": now,
                })
        except Exception as exc:
            logger.debug(f"FailureMemory: strategy record failed: {exc}")
