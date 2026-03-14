# Copyright (c) ModelScope Contributors. All rights reserved.
"""RetrievalMemory — unified orchestrator for all memory layers.

Creates, initialises, and coordinates the five memory stores.  Provides
a high-level API consumed by :class:`AgenticSearch`:

*  **Lookup** (sync, ~0 ms): ``suggest_strategy``, ``expand_keywords``,
   ``get_entity_paths``, ``filter_noise_keywords``, ``filter_dead_paths``,
   ``get_path_scores``.
*  **Record** (async): ``record_feedback`` — stores the raw signal in
   ``FeedbackMemory`` then dispatches incremental updates to the other
   four layers.
*  **Maintenance** (sync): ``decay_all``, ``cleanup_all``, ``stats``.

Storage layout::

    {work_path}/memory/
    ├── pattern_memory/
    │   ├── query_patterns.json
    │   └── reasoning_chains.json
    ├── semantic_bridge.json
    ├── corpus.duckdb
    └── feedback.duckdb
"""
from __future__ import annotations

import asyncio
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
from .schemas import (
    FeedbackSignal,
    StrategyHint,
    compute_params_hash,
    compute_pattern_id,
)


class RetrievalMemory:
    """Unified facade over the five retrieval-memory layers.

    Designed for graceful degradation: if any individual store fails to
    initialise, the others still function.  All lookup methods return
    safe defaults on error so the search pipeline is never blocked.
    """

    _MAINTENANCE_INTERVAL = 50  # run decay/cleanup every N feedbacks

    def __init__(
        self,
        work_path: str,
        llm: Optional[Any] = None,
        sync_interval: int = 120,
        sync_threshold: int = 200,
    ):
        self._work_path = Path(work_path).expanduser().resolve()
        self._memory_dir = self._work_path / "memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._llm = llm

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

        self._feedback_count = 0

        logger.info(
            f"RetrievalMemory initialised at {self._memory_dir} "
            f"({len(self._stores)} layers)"
        )

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
        self._feedback_count += 1
        success = self._is_success(signal)

        # 1. Raw signal storage
        try:
            self._feedback_memory.record_signal(signal)
        except Exception:
            pass

        # 2. PatternMemory — strategy outcome
        try:
            params = {
                "mode": signal.mode,
                "top_k_files": 5,
                "max_loops": signal.react_loops,
            }
            self._pattern_memory.record_outcome(
                query=signal.query,
                success=success,
                mode=signal.mode,
                params=params,
                latency=signal.latency_sec,
                tokens=signal.total_tokens,
            )
        except Exception:
            pass

        # 3. CorpusMemory — entity → path mappings from successful reads
        if signal.files_read and success:
            try:
                entities = [
                    k for k in signal.keywords_used
                    if k and len(k) > 1 and k[0].isupper()
                ]
                for entity in entities[:20]:
                    for fp in signal.files_read[:20]:
                        self._corpus_memory.record_entity_path(
                            entity, fp, success=True,
                        )
            except Exception:
                pass

        # 4. PathMemory — per-path retrieval stats
        try:
            for fp in signal.files_read[:50]:
                self._path_memory.record_retrieval(fp, useful=success)
        except Exception:
            pass

        # 5. FailureMemory — noise keywords + dead paths + failed strategies
        try:
            if not success:
                for kw in signal.keywords_used[:20]:
                    files_found = len(signal.files_discovered)
                    self._failure_memory.record_keyword_result(
                        kw, files_found=files_found, useful=False,
                    )
                for fp in signal.files_read[:50]:
                    self._failure_memory.record_path_result(fp, useful=False)

                features = PatternMemory.classify_query(signal.query)
                pid = compute_pattern_id(
                    features["query_type"],
                    features["complexity"],
                    features["entity_types"],
                )
                ph = compute_params_hash({"mode": signal.mode})
                self._failure_memory.record_strategy_failure(pid, ph)
            else:
                actual_files_found = len(signal.files_discovered)
                for kw in signal.keywords_used[:20]:
                    self._failure_memory.record_keyword_result(
                        kw, files_found=actual_files_found, useful=True,
                    )
                for fp in signal.files_read[:50]:
                    self._failure_memory.record_path_result(fp, useful=True)
        except Exception:
            pass

        # 6. Periodic maintenance (every N feedbacks)
        if self._feedback_count % self._MAINTENANCE_INTERVAL == 0:
            try:
                self.decay_all()
                self.cleanup_all()
            except Exception:
                pass

    @staticmethod
    def _is_success(signal: FeedbackSignal) -> bool:
        """Determine whether the search was successful."""
        if signal.user_verdict == "thumbs_up":
            return True
        if signal.llm_judge_verdict == "CORRECT":
            return True
        if signal.em_score is not None and signal.em_score > 0:
            return True
        if signal.f1_score is not None and signal.f1_score > 0.5:
            return True
        return signal.answer_found

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
        return combined

    def close(self) -> None:
        """Release all resources."""
        for store in self._stores:
            try:
                store.close()
            except Exception:
                pass
        for db in (self._corpus_db, self._feedback_db):
            try:
                db.close()
            except Exception:
                pass
        logger.info("RetrievalMemory closed")
