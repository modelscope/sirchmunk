# Copyright (c) ModelScope Contributors. All rights reserved.
"""QuerySimilarityIndex — semantic query similarity with optional embeddings.

Maintains a compact index of historical queries together with their
feedback summaries (mode, confidence, useful keywords, useful files).
At lookup time, the current query is compared to the index:

1. **Embedding path** (preferred): cosine similarity via ``EmbeddingUtil``.
2. **BM25 fallback**: when embeddings are unavailable, uses
   ``BM25Scorer`` from ``sirchmunk.utils.bm25_util`` for lexical
   similarity.

The index is persisted as a small JSON file (embedding vectors are
kept only in memory and recomputed on cold start to avoid bloated
JSON files).
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .schemas import SimilarQueryHint


class QuerySimilarityIndex:
    """Lightweight query similarity store with optional embedding support.

    Parameters
    ----------
    index_file : Path
        JSON file for persistent storage.
    embedding_util : optional
        An ``EmbeddingUtil`` instance.  When ``None``, falls back to BM25.
    tokenizer : optional
        A ``TokenizerUtil`` instance forwarded to ``BM25Scorer``.
    max_entries : int
        Maximum number of queries retained (FIFO eviction).
    """

    _SAVE_INTERVAL = 5.0

    def __init__(
        self,
        index_file: Path,
        *,
        embedding_util: Any = None,
        tokenizer: Any = None,
        max_entries: int = 2000,
    ):
        self._file = index_file
        self._embedding_util = embedding_util
        self._tokenizer = tokenizer
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: List[Dict[str, Any]] = []
        self._embeddings: Dict[int, List[float]] = {}
        self._dirty = False
        self._last_save: float = 0.0
        self._bm25_cache: Optional[Tuple[int, Any]] = None
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._entries = raw
        except Exception as exc:
            logger.debug(f"QuerySimilarityIndex: load failed: {exc}")

    def _save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            tmp = self._file.with_suffix(".tmp")
            try:
                tmp.write_text(
                    json.dumps(self._entries, ensure_ascii=False, indent=1),
                    encoding="utf-8",
                )
                os.replace(str(tmp), str(self._file))
                self._dirty = False
                self._last_save = time.monotonic()
            except Exception as exc:
                logger.debug(f"QuerySimilarityIndex: save failed: {exc}")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def _maybe_save(self) -> None:
        if time.monotonic() - self._last_save >= self._SAVE_INTERVAL:
            self._save()

    def close(self) -> None:
        self._save()

    # ── Recording ─────────────────────────────────────────────────────

    def _find_entry_index(self, query: str) -> int:
        """Return the index of the *best* existing entry for *query*, or -1."""
        best_idx = -1
        best_conf = -1.0
        for i, e in enumerate(self._entries):
            if e.get("query") == query:
                c = e.get("confidence", 0.0)
                if c > best_conf:
                    best_conf = c
                    best_idx = i
        return best_idx

    @staticmethod
    def _validate_useful_files(
        files: Optional[List[str]],
    ) -> List[str]:
        """Keep only entries that look like filesystem paths."""
        if not files:
            return []
        return [
            f for f in files
            if isinstance(f, str) and ("/" in f or "\\" in f)
        ][:20]

    def record(
        self,
        query: str,
        *,
        confidence: float,
        mode: str = "DEEP",
        keywords: Optional[List[str]] = None,
        useful_files: Optional[List[str]] = None,
        answer_snippet: str = "",
    ) -> None:
        """Record (or upsert) a query and its outcome summary.

        When the same query already exists, the entry with the higher
        confidence wins.  ``useful_files`` are merged (union) and
        non-path strings are filtered out.
        """
        valid_files = self._validate_useful_files(useful_files)

        with self._lock:
            existing_idx = self._find_entry_index(query)

            if existing_idx >= 0:
                old = self._entries[existing_idx]
                old_conf = old.get("confidence", 0.0)

                if confidence >= old_conf:
                    merged_files = list(dict.fromkeys(
                        valid_files
                        + self._validate_useful_files(old.get("useful_files")),
                    ))[:20]
                    merged_kw = list(dict.fromkeys(
                        (keywords or [])[:15]
                        + (old.get("keywords") or []),
                    ))[:15]
                    old.update({
                        "confidence": round(confidence, 4),
                        "mode": mode,
                        "keywords": merged_kw,
                        "useful_files": merged_files,
                        "answer_snippet": answer_snippet[:300],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                else:
                    merged_files = list(dict.fromkeys(
                        self._validate_useful_files(old.get("useful_files"))
                        + valid_files,
                    ))[:20]
                    old["useful_files"] = merged_files

                # Remove stale duplicates (keep only the best entry)
                self._remove_stale_duplicates(query, existing_idx)
            else:
                entry: Dict[str, Any] = {
                    "query": query,
                    "confidence": round(confidence, 4),
                    "mode": mode,
                    "keywords": (keywords or [])[:15],
                    "useful_files": valid_files,
                    "answer_snippet": answer_snippet[:300],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                embedding = self._encode(query)
                idx = len(self._entries)
                self._entries.append(entry)
                if embedding is not None:
                    self._embeddings[idx] = embedding

            if len(self._entries) > self._max_entries:
                drop = len(self._entries) - self._max_entries
                self._entries = self._entries[drop:]
                self._embeddings = {
                    k - drop: v for k, v in self._embeddings.items()
                    if k >= drop
                }

            self._dirty = True
            self._bm25_cache = None
        self._maybe_save()

    def _remove_stale_duplicates(self, query: str, keep_idx: int) -> None:
        """Remove all entries for *query* except *keep_idx* (caller holds lock)."""
        drop_set = {
            i for i, e in enumerate(self._entries)
            if e.get("query") == query and i != keep_idx
        }
        if not drop_set:
            return
        kept = []
        new_emb: Dict[int, List[float]] = {}
        for i, e in enumerate(self._entries):
            if i in drop_set:
                continue
            new_idx = len(kept)
            kept.append(e)
            if i in self._embeddings:
                new_emb[new_idx] = self._embeddings[i]
        self._entries = kept
        self._embeddings = new_emb

    def update_confidence(
        self,
        query: str,
        new_confidence: float,
    ) -> bool:
        """Update confidence of the matching entry in-place.

        Called by ``inject_evaluation`` after ground-truth scores are
        available.  Forces an immediate save since this is an infrequent
        but high-importance operation.
        """
        with self._lock:
            idx = self._find_entry_index(query)
            if idx < 0:
                return False
            self._entries[idx]["confidence"] = round(new_confidence, 4)
            self._dirty = True
        self._save()
        return True

    def record_avoid_files(self, query: str) -> bool:
        """Move ``useful_files`` to ``avoid_files`` for a failed query.

        Called when evaluation shows the query's result was incorrect.
        Subsequent warm_starts will inject negative beliefs for these files.
        Forces an immediate save.
        """
        with self._lock:
            idx = self._find_entry_index(query)
            if idx < 0:
                return False
            entry = self._entries[idx]
            current_useful = entry.get("useful_files", [])
            current_avoid = entry.get("avoid_files", [])
            merged = list(dict.fromkeys(current_avoid + current_useful))[:20]
            entry["avoid_files"] = merged
            entry["useful_files"] = []
            self._dirty = True
        self._save()
        return True

    # ── Lookup ────────────────────────────────────────────────────────

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.55,
        include_exact: bool = True,
    ) -> List[SimilarQueryHint]:
        """Return up to *top_k* similar historical queries.

        Parameters
        ----------
        include_exact : bool
            When ``True``, exact query matches are returned with
            similarity = 1.0 (highest priority).  Previously exact
            matches were always excluded; this default change allows
            reuse of direct experience from repeated queries.
        """
        with self._lock:
            entries = list(self._entries)
            embeddings = dict(self._embeddings)
        if not entries:
            return []

        results: List[SimilarQueryHint] = []

        if include_exact:
            for e in reversed(entries):
                if e.get("query", "") == query:
                    results.append(SimilarQueryHint(
                        query=e["query"],
                        similarity=1.0,
                        confidence=e.get("confidence", 0.0),
                        mode=e.get("mode"),
                        keywords=e.get("keywords", []),
                        useful_files=e.get("useful_files", []),
                        avoid_files=e.get("avoid_files", []),
                    ))
                    break

        remaining = top_k - len(results)
        if remaining <= 0:
            return results[:top_k]

        embedding = self._encode(query)
        if embedding is not None and embeddings:
            fuzzy = self._search_by_embedding(
                query, embedding, entries, embeddings, remaining, min_similarity,
            )
        else:
            fuzzy = self._search_by_bm25(
                query, entries, remaining, min_similarity,
            )
        results.extend(fuzzy)
        return results[:top_k]

    # ── Embedding path ────────────────────────────────────────────────

    def _encode(self, text: str) -> Optional[List[float]]:
        if self._embedding_util is None:
            return None
        try:
            vec = self._embedding_util.encode(text)
            if vec is not None:
                return vec if isinstance(vec, list) else vec.tolist()
        except Exception:
            pass
        return None

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _search_by_embedding(
        self,
        query: str,
        query_emb: List[float],
        entries: List[Dict[str, Any]],
        embeddings: Dict[int, List[float]],
        top_k: int,
        min_sim: float,
    ) -> List[SimilarQueryHint]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx, emb in embeddings.items():
            if idx >= len(entries):
                continue
            e = entries[idx]
            sim = self._cosine(query_emb, emb)
            if sim >= min_sim and e.get("query", "") != query:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SimilarQueryHint(
                query=e["query"],
                similarity=round(sim, 4),
                confidence=e.get("confidence", 0.0),
                mode=e.get("mode"),
                keywords=e.get("keywords", []),
                useful_files=e.get("useful_files", []),
                avoid_files=e.get("avoid_files", []),
            )
            for sim, e in scored[:top_k]
        ]

    # ── BM25 fallback ─────────────────────────────────────────────────

    def _get_bm25_scorer(
        self, entries: List[Dict[str, Any]],
    ) -> Optional[Any]:
        """Return a cached BM25Scorer, rebuilding only when entries change."""
        n = len(entries)
        if self._bm25_cache is not None and self._bm25_cache[0] == n:
            return self._bm25_cache[1]
        try:
            from sirchmunk.utils.bm25_util import BM25Scorer
            scorer = BM25Scorer(tokenizer=self._tokenizer)
            self._bm25_cache = (n, scorer)
            return scorer
        except Exception:
            return None

    def _search_by_bm25(
        self,
        query: str,
        entries: List[Dict[str, Any]],
        top_k: int,
        min_sim: float,
    ) -> List[SimilarQueryHint]:
        scorer = self._get_bm25_scorer(entries)
        if scorer is None:
            return []
        try:
            docs = [e.get("query", "") for e in entries]
            scores = scorer.score(query, docs)
            if not scores:
                return []
            max_s = max(scores)
            if max_s <= 0:
                return []
            normed = [s / max_s for s in scores]
            indexed = sorted(
                enumerate(normed), key=lambda x: x[1], reverse=True,
            )
            results: List[SimilarQueryHint] = []
            for idx, nscore in indexed:
                if nscore < min_sim:
                    break
                e = entries[idx]
                if e.get("query", "") == query:
                    continue
                results.append(SimilarQueryHint(
                    query=e["query"],
                    similarity=round(nscore, 4),
                    confidence=e.get("confidence", 0.0),
                    mode=e.get("mode"),
                    keywords=e.get("keywords", []),
                    useful_files=e.get("useful_files", []),
                    avoid_files=e.get("avoid_files", []),
                ))
                if len(results) >= top_k:
                    break
            return results
        except Exception:
            return []

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._entries)
            has_emb = len(self._embeddings)
        return {
            "total_queries": n,
            "with_embeddings": has_emb,
            "embedding_available": self._embedding_util is not None,
        }
