# Copyright (c) ModelScope Contributors. All rights reserved.
"""RetrievalMemory — unified orchestrator for all memory layers.

Creates, initialises, and coordinates the five memory stores plus the
optional ``QuerySimilarityIndex``.  Provides a high-level API consumed
by :class:`AgenticSearch`:

*  **Lookup** (sync, ~0 ms): ``suggest_strategy``, ``expand_keywords``,
   ``get_entity_paths``, ``filter_noise_keywords``, ``filter_dead_paths``,
   ``get_path_scores``, ``get_chain_hint``, ``get_similar_query_hints``.
*  **Record** (async): ``record_feedback`` — stores the raw signal in
   ``FeedbackMemory`` then dispatches incremental updates to the other
   layers using a *gradient confidence* (0-1) instead of binary success.
*  **Inject** (sync): ``inject_evaluation`` — back-fills EM/F1 scores
   and re-dispatches gradient updates.
*  **Maintenance** (sync): ``decay_all``, ``cleanup_all``, ``stats``.

Storage layout::

    {work_path}/memory/
    ├── pattern_memory/
    │   ├── query_patterns.json
    │   └── reasoning_chains.json
    ├── semantic_bridge.json
    ├── query_similarity.json
    ├── corpus.duckdb
    └── feedback.duckdb
"""
from __future__ import annotations

import atexit
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore
from .corpus_memory import CorpusMemory
from .failure_memory import FailureMemory
from .feedback import FeedbackMemory
from .path_memory import PathMemory
from .pattern_memory import PatternMemory
from .query_similarity import QuerySimilarityIndex
from .schemas import (
    FeedbackSignal,
    SimilarQueryHint,
    StrategyHint,
    compute_params_hash,
    compute_pattern_id,
)


class RetrievalMemory:
    """Unified facade over the retrieval-memory layers.

    Designed for graceful degradation: if any individual store fails to
    initialise, the others still function.  All lookup methods return
    safe defaults on error so the search pipeline is never blocked.

    Parameters
    ----------
    embedding_util : optional
        ``EmbeddingUtil`` instance for query-similarity embeddings.
        When ``None``, ``QuerySimilarityIndex`` falls back to BM25.
    tokenizer : optional
        ``TokenizerUtil`` instance for BM25 tokenisation inside the
        similarity index.
    """

    _MAINTENANCE_INTERVAL = 50

    def __init__(
        self,
        work_path: str,
        llm: Optional[Any] = None,
        sync_interval: int = 120,
        sync_threshold: int = 200,
        embedding_util: Any = None,
        tokenizer: Any = None,
    ):
        self._work_path = Path(work_path).expanduser().resolve()
        self._memory_dir = self._work_path / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._embedding_util = embedding_util
        self._tokenizer = tokenizer

        # ── Create backing stores ─────────────────────────────────────
        from sirchmunk.storage.duckdb import DuckDBManager

        self._corpus_db = DuckDBManager(
            persist_path=str(self._memory_dir / "corpus.duckdb"),
            sync_interval=sync_interval,
            sync_threshold=sync_threshold,
        )
        self._feedback_db = DuckDBManager(
            persist_path=str(self._memory_dir / "feedback.duckdb"),
            sync_interval=sync_interval,
            sync_threshold=sync_threshold,
        )

        self._pattern_memory = PatternMemory(
            base_dir=self._memory_dir / "pattern_memory",
        )
        self._corpus_memory = CorpusMemory(
            db=self._corpus_db,
            bridge_file=self._memory_dir / "semantic_bridge.json",
        )
        self._path_memory = PathMemory(db=self._corpus_db)
        self._failure_memory = FailureMemory(db=self._corpus_db)
        self._feedback_memory = FeedbackMemory(db=self._feedback_db)
        self._query_similarity = QuerySimilarityIndex(
            index_file=self._memory_dir / "query_similarity.json",
            embedding_util=embedding_util,
            tokenizer=tokenizer,
        )

        self._stores: List[MemoryStore] = [
            self._pattern_memory,
            self._corpus_memory,
            self._path_memory,
            self._failure_memory,
            self._feedback_memory,
        ]

        # ── Initialise all stores (tolerant) ──────────────────────────
        for store in self._stores:
            try:
                store.initialize()
            except Exception as exc:
                logger.warning(
                    f"RetrievalMemory: {store.name} init failed: {exc}"
                )

        # ── Warmup: seed default patterns + noise keywords ────────────
        try:
            self._pattern_memory.seed_defaults()
        except Exception:
            pass
        try:
            self._failure_memory.seed_noise_keywords()
        except Exception:
            pass

        self._feedback_count = 0
        self._feedback_lock = threading.Lock()

        # ── Register atexit to flush dirty state ──────────────────────
        atexit.register(self._atexit_flush)

        logger.info(
            f"RetrievalMemory initialised at {self._memory_dir} "
            f"({len(self._stores)} layers)"
        )

    def _atexit_flush(self) -> None:
        """Best-effort flush on interpreter shutdown."""
        try:
            self._pattern_memory.close()
        except Exception:
            pass
        try:
            self._query_similarity.close()
        except Exception:
            pass

    # ── Embedding util setter (for deferred warm-up) ──────────────────

    def set_embedding_util(self, embedding_util: Any) -> None:
        """Inject embedding util after background warm-up completes."""
        self._embedding_util = embedding_util
        self._query_similarity._embedding_util = embedding_util

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Inject tokenizer after background warm-up completes."""
        self._tokenizer = tokenizer
        self._query_similarity._tokenizer = tokenizer

    # ================================================================
    #  Lookup API  (sync, called on the hot path)
    # ================================================================

    def suggest_strategy(self, query: str) -> Optional[StrategyHint]:
        """PatternMemory lookup → search parameter overrides."""
        try:
            return self._pattern_memory.suggest_strategy(query)
        except Exception:
            return None

    def expand_keywords(
        self,
        keywords: Dict[str, float],
    ) -> Dict[str, float]:
        """CorpusMemory → semantic bridge expansion."""
        try:
            return self._corpus_memory.expand_keywords(keywords)
        except Exception:
            return keywords

    def get_entity_paths(self, entities: List[str]) -> List[str]:
        """CorpusMemory → file paths for known entities."""
        try:
            return self._corpus_memory.get_entity_paths(entities)
        except Exception:
            return []

    def filter_noise_keywords(self, keywords: List[str]) -> List[str]:
        """FailureMemory → remove keywords marked as noise."""
        try:
            return self._failure_memory.filter_noise_keywords(keywords)
        except Exception:
            return keywords

    def filter_dead_paths(self, paths: List[str]) -> List[str]:
        """FailureMemory → remove persistently useless paths."""
        try:
            return self._failure_memory.filter_dead_paths(paths)
        except Exception:
            return paths

    def get_path_scores(self, paths: List[str]) -> Dict[str, float]:
        """PathMemory → hot_score per path for rerank boosting."""
        try:
            return self._path_memory.get_path_scores(paths)
        except Exception:
            return {}

    def get_chain_hint(
        self,
        query: str,
    ) -> Optional[List[Dict[str, str]]]:
        """PatternMemory → reasoning chain template."""
        try:
            return self._pattern_memory.get_chain_hint(query)
        except Exception:
            return None

    def get_similar_query_hints(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.55,
    ) -> List[SimilarQueryHint]:
        """QuerySimilarityIndex → hints from similar historical queries."""
        try:
            return self._query_similarity.find_similar(
                query, top_k=top_k, min_similarity=min_similarity,
            )
        except Exception:
            return []

    def extract_entities(self, keywords: List[str]) -> List[str]:
        """Delegate to CorpusMemory's improved entity extractor."""
        return CorpusMemory.extract_entities(keywords)

    # ================================================================
    #  Recording API  (async, called after search completes)
    # ================================================================

    async def record_feedback(self, signal: FeedbackSignal) -> None:
        """Store signal and dispatch updates to all layers.

        Runs in a fire-and-forget fashion; failures are logged but
        never propagate to the caller.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self._dispatch_feedback, signal,
            )
        except Exception as exc:
            logger.debug(f"RetrievalMemory: feedback dispatch failed: {exc}")

    def _dispatch_feedback(self, signal: FeedbackSignal) -> None:
        """Synchronous dispatcher — called inside an executor thread."""
        with self._feedback_lock:
            self._feedback_count += 1
            count = self._feedback_count
        confidence = self._compute_confidence(signal)

        # 1. Raw signal storage
        try:
            self._feedback_memory.record_signal(signal)
        except Exception as exc:
            logger.debug(f"[memory:dispatch] FeedbackMemory failed: {exc}")

        # 2. PatternMemory — strategy outcome (gradient)
        try:
            params = {
                "mode": signal.mode,
                "top_k_files": 5,
                "max_loops": signal.react_loops,
            }
            self._pattern_memory.record_outcome(
                query=signal.query,
                confidence=confidence,
                mode=signal.mode,
                params=params,
                latency=signal.latency_sec,
                tokens=signal.total_tokens,
            )
        except Exception as exc:
            logger.debug(f"[memory:dispatch] PatternMemory failed: {exc}")

        # 3. CorpusMemory — entity → path mappings (improved extraction)
        try:
            entities = CorpusMemory.extract_entities(signal.keywords_used)
            if entities and signal.files_read:
                for entity in entities[:20]:
                    for fp in signal.files_read[:20]:
                        self._corpus_memory.record_entity_path(
                            entity, fp, success=(confidence >= 0.5),
                        )
        except Exception as exc:
            logger.debug(f"[memory:dispatch] CorpusMemory entity failed: {exc}")

        # 4. CorpusMemory — keyword co-occurrence + semantic bridge
        if confidence >= 0.5 and signal.keywords_used:
            try:
                self._corpus_memory.record_keyword_cooccurrence(
                    signal.keywords_used, success=True,
                )
            except Exception as exc:
                logger.debug(f"[memory:dispatch] cooccurrence failed: {exc}")

        # 5. PathMemory — per-path retrieval stats
        try:
            useful_set = set(signal.files_discovered or [])
            for fp in signal.files_read[:50]:
                self._path_memory.record_retrieval(
                    fp, useful=(fp in useful_set or confidence >= 0.5),
                )
        except Exception as exc:
            logger.debug(f"[memory:dispatch] PathMemory failed: {exc}")

        # 6. FailureMemory — noise keywords + dead paths + failed strategies
        try:
            actual_files_found = len(signal.files_discovered or [])
            for kw in signal.keywords_used[:20]:
                self._failure_memory.record_keyword_result(
                    kw, files_found=actual_files_found,
                    useful=(confidence >= 0.5),
                )
            for fp in signal.files_read[:50]:
                self._failure_memory.record_path_result(
                    fp, useful=(confidence >= 0.5),
                )

            if confidence < 0.3:
                features = PatternMemory.classify_query(signal.query)
                pid = compute_pattern_id(
                    features["query_type"],
                    features["complexity"],
                    features["entity_types"],
                    entity_count=features.get("entity_count", 0),
                    hop_hint=features.get("hop_hint", "single"),
                )
                ph = compute_params_hash({"mode": signal.mode})
                self._failure_memory.record_strategy_failure(pid, ph)
        except Exception as exc:
            logger.debug(f"[memory:dispatch] FailureMemory failed: {exc}")

        # 7. QuerySimilarityIndex — record for future similarity lookup
        try:
            self._query_similarity.record(
                query=signal.query,
                confidence=confidence,
                mode=signal.mode,
                keywords=signal.keywords_used,
                useful_files=(
                    signal.files_discovered
                    if signal.files_discovered
                    else signal.files_read
                ),
                answer_snippet=(signal.answer_text or "")[:300],
            )
        except Exception as exc:
            logger.debug(f"[memory:dispatch] QuerySimilarity failed: {exc}")

        # 8. Periodic maintenance
        if count % self._MAINTENANCE_INTERVAL == 0:
            try:
                self.decay_all()
                self.cleanup_all()
            except Exception:
                pass

    @staticmethod
    def _compute_confidence(signal: FeedbackSignal) -> float:
        """Compute a continuous confidence score (0-1) from a feedback signal.

        Priority order:
        1. Explicit EM/F1 scores (when injected by benchmark evaluation).
        2. LLM judge verdict.
        3. User verdict.
        4. Heuristic based on ``answer_found`` and ``cluster_confidence``.
        """
        if signal.em_score is not None and signal.f1_score is not None:
            return 0.5 * signal.em_score + 0.5 * min(signal.f1_score, 1.0)

        if signal.f1_score is not None:
            return min(signal.f1_score, 1.0)
        if signal.em_score is not None:
            return float(signal.em_score)

        if signal.llm_judge_verdict == "CORRECT":
            return 0.9
        if signal.llm_judge_verdict == "INCORRECT":
            return 0.1
        if signal.user_verdict == "thumbs_up":
            return 0.95
        if signal.user_verdict == "thumbs_down":
            return 0.05

        if signal.answer_found:
            base = 0.5
            if signal.cluster_confidence > 0:
                base = max(base, min(signal.cluster_confidence, 0.9))
            return base

        return 0.2

    # ================================================================
    #  Evaluation injection (for benchmark harnesses)
    # ================================================================

    def inject_evaluation(
        self,
        query: str,
        em_score: float,
        f1_score: float,
        llm_judge_verdict: Optional[str] = None,
    ) -> None:
        """Back-fill EM/F1 on the most recent signal and re-dispatch.

        Called by benchmark harnesses *after* the search completes and
        ground-truth evaluation is available.
        """
        try:
            self._feedback_memory.inject_evaluation(
                query, em_score, f1_score, llm_judge_verdict,
            )
        except Exception:
            pass

        # Re-compute confidence and update PatternMemory with better signal
        try:
            conf = 0.5 * em_score + 0.5 * min(f1_score, 1.0)
            features = PatternMemory.classify_query(query)
            pid = compute_pattern_id(
                features["query_type"],
                features["complexity"],
                features["entity_types"],
                entity_count=features.get("entity_count", 0),
                hop_hint=features.get("hop_hint", "single"),
            )
            self._pattern_memory.record_outcome(
                query=query,
                confidence=conf,
                mode="DEEP",
                params={"max_loops": 5, "top_k_files": 5},
            )
        except Exception:
            pass

    # ================================================================
    #  Maintenance API
    # ================================================================

    def decay_all(self) -> Dict[str, int]:
        """Run confidence decay on all layers."""
        results: Dict[str, int] = {}
        for store in self._stores:
            try:
                results[store.name] = store.decay()
            except Exception:
                results[store.name] = -1
        return results

    def cleanup_all(self) -> Dict[str, int]:
        """Run cleanup on all layers."""
        results: Dict[str, int] = {}
        for store in self._stores:
            try:
                results[store.name] = store.cleanup()
            except Exception:
                results[store.name] = -1
        return results

    def stats(self) -> Dict[str, Any]:
        """Combined statistics from all layers."""
        combined: Dict[str, Any] = {
            "memory_dir": str(self._memory_dir),
        }
        for store in self._stores:
            try:
                combined[store.name] = store.stats()
            except Exception:
                combined[store.name] = {"error": "unavailable"}
        try:
            combined["QuerySimilarity"] = self._query_similarity.stats()
        except Exception:
            combined["QuerySimilarity"] = {"error": "unavailable"}
        return combined

    def close(self) -> None:
        """Release all resources."""
        for store in self._stores:
            try:
                store.close()
            except Exception:
                pass
        try:
            self._query_similarity.close()
        except Exception:
            pass
        for db in (self._corpus_db, self._feedback_db):
            try:
                db.close()
            except Exception:
                pass
        logger.info("RetrievalMemory closed")
