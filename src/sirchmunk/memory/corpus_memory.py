# Copyright (c) ModelScope Contributors. All rights reserved.
"""CorpusMemory — entity-path index (DuckDB) + semantic bridge (JSON).

The entity-path index maps entities found in successful searches to the
file paths that contained them.  The semantic bridge stores learned
keyword expansion rules (synonyms, aliases, hypernyms).
"""
from __future__ import annotations

import itertools
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore
from .schemas import SemanticBridgeEntry, SemanticExpansion

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_CJK_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fff]{2,}(?:市|省|县|国|河|山|湖|大学|学院|公司|集团)"
    r"|《[^》]+》"
)
_EN_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "and",
    "or", "but", "not", "no", "for", "this", "that", "how", "what",
    "who", "where", "when", "which", "than", "also", "both", "from",
})

try:
    from sirchmunk.storage.duckdb import DuckDBManager
except ImportError:
    DuckDBManager = None  # type: ignore[assignment,misc]


class CorpusMemory(MemoryStore):
    """Entity-path inverted index + semantic keyword bridge.

    The entity index lives in the shared ``corpus.duckdb`` (table
    ``entity_index``), while the semantic bridge is a small JSON file
    for human readability and rapid schema evolution.
    """

    _BRIDGE_MIN_CONFIDENCE = 0.4
    _BRIDGE_MIN_HITS = 2
    _DECAY_STALE_DAYS = 7
    _BRIDGE_DECAY_DAYS = 30

    def __init__(
        self,
        db: "DuckDBManager",
        bridge_file: Path,
    ):
        self._db = db
        self._bridge_file = bridge_file
        self._bridge: Dict[str, SemanticBridgeEntry] = {}
        self._lock = threading.RLock()

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "CorpusMemory"

    def initialize(self) -> None:
        self._create_entity_table()
        self._load_bridge()

    def decay(self, now: Optional[datetime] = None) -> int:
        if not now:
            now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(days=self._DECAY_STALE_DAYS)).isoformat()
        count = 0

        # Decay stale entity index entries
        try:
            row = self._db.fetch_one(
                "SELECT COUNT(*) FROM entity_index "
                "WHERE last_hit < ? AND confidence > 0.1",
                [cutoff],
            )
            n = row[0] if row else 0
            if n > 0:
                self._db.execute(
                    "UPDATE entity_index SET confidence = confidence * 0.95 "
                    "WHERE last_hit < ? AND confidence > 0.1",
                    [cutoff],
                )
            count += n
        except Exception:
            pass

        # Decay stale semantic bridge entries
        bridge_cutoff = (
            now - timedelta(days=self._BRIDGE_DECAY_DAYS)
        ).isoformat()
        with self._lock:
            bridge_decayed = 0
            for entry in self._bridge.values():
                if entry.updated_at and entry.updated_at < bridge_cutoff:
                    for exp in entry.expansions:
                        if exp.confidence > 0.1:
                            exp.confidence *= 0.9
                            bridge_decayed += 1
            if bridge_decayed:
                self._save_bridge()
            count += bridge_decayed

        return count

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        max_entries = max_entries or 50000
        try:
            count = self._db.get_table_count("entity_index")
            if count <= max_entries:
                return 0
            excess = count - max_entries
            self._db.execute(
                f"DELETE FROM entity_index WHERE rowid IN ("
                f"  SELECT rowid FROM entity_index "
                f"  ORDER BY confidence ASC, hit_count ASC LIMIT {excess}"
                f")",
            )
            return excess
        except Exception:
            return 0

    def stats(self) -> Dict[str, Any]:
        try:
            count = self._db.get_table_count("entity_index")
        except Exception:
            count = 0
        with self._lock:
            bridge_count = len(self._bridge)
        return {
            "name": self.name,
            "entity_index_count": count,
            "semantic_bridge_count": bridge_count,
        }

    def close(self) -> None:
        self._save_bridge()

    # ── DuckDB entity index ──────────────────────────────────────────

    def _create_entity_table(self) -> None:
        if self._db.table_exists("entity_index"):
            return
        self._db.create_table("entity_index", {
            "entity": "VARCHAR NOT NULL",
            "entity_type": "VARCHAR DEFAULT 'unknown'",
            "path": "VARCHAR NOT NULL",
            "confidence": "FLOAT DEFAULT 0.5",
            "hit_count": "INTEGER DEFAULT 1",
            "last_hit": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        })
        try:
            self._db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_path "
                "ON entity_index (entity, path)"
            )
        except Exception:
            pass

    def get_entity_paths(
        self,
        entities: List[str],
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> List[str]:
        """Return file paths known to contain any of the given entities."""
        if not entities:
            return []
        try:
            placeholders = ", ".join(["?" for _ in entities])
            rows = self._db.fetch_all(
                f"SELECT DISTINCT path FROM entity_index "
                f"WHERE entity IN ({placeholders}) "
                f"AND confidence >= ? "
                f"ORDER BY confidence DESC, hit_count DESC "
                f"LIMIT ?",
                [e.lower() for e in entities] + [min_confidence, limit],
            )
            return [r[0] for r in rows]
        except Exception:
            return []

    def record_entity_path(
        self,
        entity: str,
        path: str,
        entity_type: str = "unknown",
        success: bool = True,
    ) -> None:
        """Record that *entity* was found in *path*."""
        entity_lower = entity.lower()
        now = datetime.now(timezone.utc).isoformat()
        try:
            existing = self._db.fetch_one(
                "SELECT hit_count, confidence FROM entity_index "
                "WHERE entity = ? AND path = ?",
                [entity_lower, path],
            )
            if existing:
                new_hits = existing[0] + 1
                delta = 0.05 if success else -0.02
                new_conf = max(0.0, min(1.0, existing[1] + delta))
                self._db.execute(
                    "UPDATE entity_index SET hit_count = ?, confidence = ?, "
                    "last_hit = ? WHERE entity = ? AND path = ?",
                    [new_hits, new_conf, now, entity_lower, path],
                )
            else:
                self._db.insert_data("entity_index", {
                    "entity": entity_lower,
                    "entity_type": entity_type,
                    "path": path,
                    "confidence": 0.5 if success else 0.3,
                    "hit_count": 1,
                    "last_hit": now,
                })
        except Exception as exc:
            logger.debug(f"CorpusMemory: entity_path record failed: {exc}")

    # ── JSON semantic bridge ─────────────────────────────────────────

    def _load_bridge(self) -> None:
        with self._lock:
            if not self._bridge_file.exists():
                return
            try:
                raw = json.loads(
                    self._bridge_file.read_text(encoding="utf-8")
                )
                self._bridge = {
                    k: SemanticBridgeEntry.from_dict(v)
                    for k, v in raw.items()
                }
            except Exception as exc:
                logger.warning(
                    f"CorpusMemory: bridge load failed: {exc}"
                )

    def _save_bridge(self) -> None:
        with self._lock:
            if not self._bridge:
                return
            tmp = self._bridge_file.with_suffix(".tmp")
            try:
                data = {
                    k: v.to_dict() for k, v in self._bridge.items()
                }
                tmp.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(str(tmp), str(self._bridge_file))
            except Exception as exc:
                logger.warning(f"CorpusMemory: bridge save failed: {exc}")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def expand_keywords(
        self,
        keywords: Dict[str, float],
    ) -> Dict[str, float]:
        """Expand keyword dict with learned semantic bridges.

        Only expansions meeting confidence and hit thresholds are added.
        Expansion weight = original_weight × confidence × 0.7.
        """
        expanded = dict(keywords)
        with self._lock:
            for term, weight in keywords.items():
                entry = self._bridge.get(term.lower())
                if not entry:
                    continue
                for exp in entry.expansions:
                    if (exp.confidence >= self._BRIDGE_MIN_CONFIDENCE
                            and exp.hit_count >= self._BRIDGE_MIN_HITS):
                        key = exp.target
                        if key not in expanded:
                            expanded[key] = weight * exp.confidence * 0.7
        return expanded

    def record_semantic_bridge(
        self,
        source: str,
        target: str,
        relation: str = "synonym",
        success: bool = True,
        *,
        defer_save: bool = False,
    ) -> None:
        """Record a semantic equivalence between *source* and *target*.

        When *defer_save* is True, the caller is responsible for calling
        ``_save_bridge()`` afterwards (used by batch operations).
        """
        key = source.lower()
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            entry = self._bridge.get(key)
            if not entry:
                entry = SemanticBridgeEntry(term=key, updated_at=now)
                self._bridge[key] = entry

            existing = next(
                (e for e in entry.expansions if e.target == target.lower()),
                None,
            )
            if existing:
                existing.hit_count += 1
                delta = 0.05 if success else -0.03
                existing.confidence = max(
                    0.0, min(1.0, existing.confidence + delta)
                )
            else:
                entry.expansions.append(SemanticExpansion(
                    target=target.lower(),
                    relation=relation,
                    confidence=0.5 if success else 0.3,
                    hit_count=1,
                ))
            entry.updated_at = now
        if not defer_save:
            self._save_bridge()

    # ── Keyword co-occurrence mining ─────────────────────────────────

    def record_keyword_cooccurrence(
        self,
        keywords: List[str],
        success: bool = True,
    ) -> int:
        """Mine pairwise co-occurrence from a successful keyword set.

        For every pair (k_i, k_j) in *keywords*, records a weak semantic
        bridge with relation ``cooccurrence``.  All writes are batched
        into a single disk flush at the end.
        """
        filtered = [
            k for k in keywords
            if k and len(k) > 1 and k.lower() not in _EN_STOP
        ]
        if len(filtered) < 2:
            return 0
        count = 0
        for a, b in itertools.combinations(filtered[:10], 2):
            if a.lower() == b.lower():
                continue
            self.record_semantic_bridge(
                a, b, relation="cooccurrence", success=success,
                defer_save=True,
            )
            count += 1
        if count:
            self._save_bridge()
        return count

    # ── Entity extraction (language-agnostic) ─────────────────────────

    @staticmethod
    def extract_entities(keywords: List[str]) -> List[str]:
        """Extract entities from a keyword list (supports CJK + English).

        Replaces the naive ``k[0].isupper()`` check with proper regex
        patterns for English named entities and CJK entity suffixes.
        """
        entities: List[str] = []
        seen: set = set()
        for kw in keywords:
            if not kw or len(kw) <= 1:
                continue
            kw_lower = kw.lower()
            if kw_lower in _EN_STOP or kw_lower in seen:
                continue
            if _CJK_RE.search(kw):
                entities.append(kw)
                seen.add(kw_lower)
                continue
            if _EN_ENTITY_RE.match(kw) or (len(kw) > 2 and kw[0].isupper()):
                entities.append(kw)
                seen.add(kw_lower)
        return entities[:30]
