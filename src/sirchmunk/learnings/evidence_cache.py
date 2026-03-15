# Copyright (c) ModelScope Contributors. All rights reserved.
"""Evidence sampling result cache for cross-query reuse.

Avoids redundant LLM calls when the same document is queried with
semantically similar questions by caching evaluated ROI results.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CacheEntry:
    """Cached evidence sampling results for a document+query pair."""

    doc_hash: str
    query: str
    scored_positions: List[Tuple[int, int, float]]
    summary: str
    is_found: bool
    snippets: List[Dict]
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0


class EvidenceCache:
    """LRU cache keyed on document content hash.

    Performs character-level Jaccard similarity between the incoming query
    and stored queries for the same document, returning a cached result
    when the similarity exceeds *threshold*.
    """

    _MAX_ENTRIES = 200
    _TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._store: Dict[str, List[CacheEntry]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _doc_hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _query_similarity(q1: str, q2: str) -> float:
        s1, s2 = set(q1.lower()), set(q2.lower())
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def get(
        self, doc_content: str, query: str, threshold: float = 0.7,
    ) -> Optional[CacheEntry]:
        dh = self._doc_hash(doc_content)
        with self._lock:
            entries = self._store.get(dh, [])
            now = time.time()
            best: Optional[CacheEntry] = None
            best_sim = threshold

            for entry in entries:
                if now - entry.timestamp > self._TTL_SECONDS:
                    continue
                sim = self._query_similarity(query, entry.query)
                if sim > best_sim:
                    best_sim = sim
                    best = entry

            if best is not None:
                best.hit_count += 1
            return best

    def put(
        self,
        doc_content: str,
        query: str,
        scored_positions: List[Tuple[int, int, float]],
        summary: str,
        is_found: bool,
        snippets: List[Dict],
    ) -> None:
        dh = self._doc_hash(doc_content)
        entry = CacheEntry(
            doc_hash=dh,
            query=query,
            scored_positions=scored_positions,
            summary=summary,
            is_found=is_found,
            snippets=snippets,
        )
        with self._lock:
            self._store.setdefault(dh, []).append(entry)
            self._evict()

    def _evict(self) -> None:
        now = time.time()
        total = 0
        for dh in list(self._store):
            self._store[dh] = [
                e for e in self._store[dh]
                if now - e.timestamp < self._TTL_SECONDS
            ]
            if not self._store[dh]:
                del self._store[dh]
            else:
                total += len(self._store[dh])

        if total > self._MAX_ENTRIES:
            all_entries = [
                (dh, e)
                for dh, entries in self._store.items()
                for e in entries
            ]
            all_entries.sort(key=lambda x: x[1].timestamp)
            for dh, entry in all_entries[: total - self._MAX_ENTRIES]:
                if dh in self._store:
                    self._store[dh] = [e for e in self._store[dh] if e is not entry]
                    if not self._store[dh]:
                        del self._store[dh]
