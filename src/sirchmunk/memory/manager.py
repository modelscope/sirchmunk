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
import math
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
    compute_pattern_id_at_level,
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
        self._belief_cache: Dict[str, Dict[str, float]] = {}

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
        min_similarity: float = 0.45,
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

    def is_noise_keyword(self, keyword: str) -> bool:
        """FailureMemory → check if a keyword is known high-frequency noise."""
        try:
            return self._failure_memory.is_noise_keyword(keyword)
        except Exception:
            return False

    def record_highfreq_keyword(self, keyword: str, files_found: int) -> None:
        """FailureMemory → persist a ugrep-detected high-frequency keyword."""
        try:
            self._failure_memory.record_highfreq_keyword(keyword, files_found)
        except Exception:
            pass

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
        raw_confidence = self._compute_confidence(signal)
        confidence = self._apply_reward_shaping(signal, raw_confidence)

        # Persist heuristic confidence on the signal for inject_evaluation
        signal.heuristic_confidence = confidence

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

        # 3. CorpusMemory — entity → path mappings (batch)
        try:
            entities = CorpusMemory.extract_entities(signal.keywords_used)
            if entities and signal.files_read:
                pairs = [
                    (entity, fp)
                    for entity in entities[:20]
                    for fp in signal.files_read[:20]
                ]
                self._corpus_memory.batch_record_entity_paths(
                    pairs,
                    success=(confidence >= 0.5),
                    confidence=confidence,
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

        # 5. PathMemory — per-path retrieval stats (batch)
        #    Track processed paths to avoid double-recording with belief_snapshot
        path_recorded: set = set()
        try:
            useful_set = set(signal.files_discovered or [])
            path_useful_map = {}
            for fp in signal.files_read[:50]:
                path_useful_map[fp] = (fp in useful_set or confidence >= 0.5)
                path_recorded.add(fp)
            self._path_memory.batch_record_retrievals(path_useful_map)
        except Exception as exc:
            logger.debug(f"[memory:dispatch] PathMemory failed: {exc}")

        # 5b. PatternMemory — reasoning chain (abstracted trace)
        if confidence >= 0.5 and signal.files_read and signal.keywords_used:
            try:
                steps: List[Dict[str, str]] = []
                for kw in signal.keywords_used[:3]:
                    steps.append({"action": "keyword_search", "target": kw})
                for fp in signal.files_read[:3]:
                    steps.append({"action": "read_file", "target": fp})
                if signal.answer_found:
                    steps.append({"action": "answer", "target": "found"})
                if steps:
                    self._pattern_memory.record_chain(
                        query=signal.query,
                        steps=steps,
                        success=(confidence >= 0.5),
                    )
            except Exception as exc:
                logger.debug(f"[memory:dispatch] PatternMemory chain failed: {exc}")

        # 6. FailureMemory — noise keywords + dead paths + failed strategies (batch)
        try:
            actual_files_found = len(signal.files_discovered or [])
            is_useful = confidence >= 0.5
            self._failure_memory.batch_record_keyword_results(
                signal.keywords_used[:20], actual_files_found, is_useful,
            )
            useful_path_set = set() if not is_useful else set(signal.files_read[:50])
            self._failure_memory.batch_record_path_results(
                signal.files_read[:50],
                useful_path_set if is_useful else set(),
            )

            if confidence < 0.3:
                features = PatternMemory.classify_query(signal.query)
                pid = compute_pattern_id_at_level(
                    features["query_type"],
                    features["complexity"],
                    features["entity_types"],
                    entity_count=features.get("entity_count", 0),
                    hop_hint=features.get("hop_hint", "single"),
                    level=4,
                )
                ph = compute_params_hash({"mode": signal.mode})
                self._failure_memory.record_strategy_failure(pid, ph)
        except Exception as exc:
            logger.debug(f"[memory:dispatch] FailureMemory failed: {exc}")

        # Cache belief_snapshot for later use by inject_evaluation → record_avoid_files
        if signal.belief_snapshot:
            self._belief_cache[signal.query] = dict(signal.belief_snapshot)

        # 6b. BA-ReAct: fine-grained PathMemory from belief snapshot
        #     Skip paths already recorded in step 5 to avoid double-counting.
        if signal.belief_snapshot:
            try:
                belief_map = {
                    fp: (belief >= 0.5)
                    for fp, belief in list(signal.belief_snapshot.items())[:30]
                    if fp not in path_recorded
                }
                if belief_map:
                    self._path_memory.batch_record_retrievals(belief_map)
            except Exception as exc:
                logger.debug(
                    f"[memory:dispatch] belief→PathMemory failed: {exc}",
                )

        # 6c. BA-ReAct: dead-path candidates from belief tracking (batch)
        if signal.dead_candidates:
            try:
                self._failure_memory.batch_record_path_results(
                    signal.dead_candidates[:20], set(),
                )
            except Exception as exc:
                logger.debug(
                    f"[memory:dispatch] belief→FailureMemory failed: {exc}",
                )

        # 6d. BA-ReAct: high-value files as entity-path associations (batch)
        if signal.high_value_files and signal.keywords_used:
            try:
                entities = CorpusMemory.extract_entities(
                    signal.keywords_used,
                )
                hv_pairs = [
                    (entity, fp)
                    for entity in entities[:10]
                    for fp in signal.high_value_files[:10]
                ]
                self._corpus_memory.batch_record_entity_paths(
                    hv_pairs, success=True,
                )
            except Exception as exc:
                logger.debug(
                    f"[memory:dispatch] belief→CorpusMemory failed: {exc}",
                )

        # 7. QuerySimilarityIndex — record for future similarity lookup
        # Prefer high_value_files (belief-confirmed useful), then discovered, then read
        _useful = (
            signal.high_value_files
            or signal.files_discovered
            or signal.files_read
            or []
        )
        try:
            self._query_similarity.record(
                query=signal.query,
                confidence=confidence,
                mode=signal.mode,
                keywords=signal.keywords_used,
                useful_files=_useful,
                answer_snippet=(signal.answer_text or "")[:300],
            )
        except Exception as exc:
            logger.debug(f"[memory:dispatch] QuerySimilarity failed: {exc}")

        # 8. MCTS-Memory Integration: adapt search depth params based on convergence
        try:
            self._update_sampling_params(signal, confidence)
        except Exception as exc:
            logger.debug(f"[memory:dispatch] _update_sampling_params failed: {exc}")

        # 9. Periodic maintenance
        if count % self._MAINTENANCE_INTERVAL == 0:
            try:
                self.decay_all()
                self.cleanup_all()
            except Exception:
                pass

    def _update_sampling_params(self, signal: FeedbackSignal, confidence: float) -> None:
        """Delegate to PatternMemory's thread-safe public method."""
        self._pattern_memory.update_sampling_params(
            query=signal.query,
            react_loops=getattr(signal, 'react_loops', 0) or 0,
            convergence=getattr(signal, 'convergence_achieved', False),
            confidence=confidence,
            total_tokens=getattr(signal, 'total_tokens', 0) or 0,
        )

    @staticmethod
    def _compute_confidence(signal: FeedbackSignal) -> float:
        """Compute a continuous confidence score (0-1) from a feedback signal.

        Priority order:
        1. Explicit EM/F1 scores (when injected by benchmark evaluation).
        2. LLM judge verdict.
        3. User verdict.
        4. Enhanced heuristic based on multiple signal features.
        """
        # Priority 1: Explicit EM/F1 scores
        if signal.em_score is not None and signal.f1_score is not None:
            base = 0.5 * signal.em_score + 0.5 * min(signal.f1_score, 1.0)
        elif signal.f1_score is not None:
            base = min(signal.f1_score, 1.0)
        elif signal.em_score is not None:
            base = float(signal.em_score)
        # Priority 2: LLM judge verdict
        elif signal.llm_judge_verdict == "CORRECT":
            base = 0.9
        elif signal.llm_judge_verdict == "INCORRECT":
            base = 0.1
        # Priority 3: User verdict
        elif signal.user_verdict == "thumbs_up":
            base = 0.95
        elif signal.user_verdict == "thumbs_down":
            base = 0.05
        # Priority 4: Enhanced multi-feature heuristic
        elif signal.answer_found:
            base = 0.4  # lower base, more room for differentiation
            # cluster_confidence contribution (continuous)
            if signal.cluster_confidence is not None and signal.cluster_confidence > 0:
                base += 0.2 * min(signal.cluster_confidence, 1.0)
            else:
                base += 0.1  # default mid-value
            # Files efficiency: fewer files read = more precise retrieval
            if signal.files_read_count and signal.files_read_count > 0:
                file_efficiency = (signal.files_useful_count or 0) / signal.files_read_count
                base += 0.15 * file_efficiency
            # Loop efficiency: fewer loops = better strategy
            if signal.react_loops and signal.react_loops > 0:
                loop_efficiency = max(0, 1.0 - signal.react_loops / 10)
                base += 0.1 * loop_efficiency
            base = min(base, 0.95)
        else:
            base = 0.15  # slightly lower default for unfound

        # Efficiency shaping: bonus for token efficiency
        if signal.total_tokens and signal.total_tokens > 0:
            budget_ratio = min(signal.total_tokens / 65000, 2.0)
            efficiency_bonus = max(0, (1.0 - budget_ratio) * 0.1)  # max ±0.1
            base = min(1.0, base + efficiency_bonus)

        return base

    # ── Potential-function reward shaping ────────────────────────────

    # Weights for the potential function Φ(s) = w1·coverage + w2·diversity + w3·novelty
    _RS_W_COVERAGE = 0.05
    _RS_W_DIVERSITY = 0.03
    _RS_W_NOVELTY = 0.02

    def _apply_reward_shaping(
        self,
        signal: FeedbackSignal,
        base_confidence: float,
    ) -> float:
        """Apply potential-function–based reward shaping (Theorem 7 guarantee).

        Shaped reward = R + γΦ(s') - Φ(s).  Since we observe Φ at a single
        timestep (before dispatch), we approximate the shaping bonus as the
        *instantaneous* intrinsic motivation signal rather than a temporal
        difference.  This preserves the optimal policy (Ng et al., 1999).
        """
        bonus = 0.0

        # 1. Coverage bonus — reward visiting under-explored patterns
        try:
            stats = self._pattern_memory.get_exploration_stats(signal.query)
            if stats["pattern_sample_count"] < 3:
                bonus += self._RS_W_COVERAGE  # new/rare pattern
            coverage_frac = (
                stats["explored_count"] / max(stats["total_patterns"], 1)
            )
            bonus += self._RS_W_COVERAGE * (1.0 - coverage_frac)
        except Exception:
            pass

        # 2. Diversity bonus — entropy of mode distribution in recent feedbacks
        try:
            recent = self._feedback_memory.recent_signals(limit=20)
            if recent:
                mode_counts: Dict[str, int] = {}
                for sig in recent:
                    m = sig.get("mode", "DEEP")
                    mode_counts[m] = mode_counts.get(m, 0) + 1
                total = sum(mode_counts.values())
                if total > 1:
                    entropy = -sum(
                        (c / total) * math.log2(c / total)
                        for c in mode_counts.values() if c > 0
                    )
                    max_entropy = math.log2(max(len(mode_counts), 2))
                    norm_entropy = entropy / max(max_entropy, 1e-9)
                    bonus += self._RS_W_DIVERSITY * norm_entropy
        except Exception:
            pass

        # 3. Novelty bonus — reward queries dissimilar to recent history
        try:
            similar = self._query_similarity.find_similar(
                signal.query, top_k=1, min_similarity=0.0,
            )
            if similar:
                max_sim = similar[0].similarity
                novelty = 1.0 - max_sim
            else:
                novelty = 1.0
            bonus += self._RS_W_NOVELTY * novelty
        except Exception:
            pass

        shaped = max(0.0, min(1.0, base_confidence + bonus))
        return shaped

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
        ground-truth evaluation is available.  Propagates the refined
        confidence to **all** relevant memory layers, including the
        ``QuerySimilarityIndex`` (closing the feedback loop).

        NOTE: Does NOT call record_outcome() again to avoid double-counting
        α/β updates. Instead, applies a delta correction based on the
        difference between the heuristic confidence (from _dispatch_feedback)
        and the ground-truth confidence (from EM/F1).
        """
        ground_truth_conf = 0.5 * em_score + 0.5 * min(f1_score, 1.0)

        # 1. FeedbackMemory: persist raw scores
        try:
            self._feedback_memory.inject_evaluation(
                query, em_score, f1_score, llm_judge_verdict,
            )
        except Exception as exc:
            logger.debug(f"[memory:inject] FeedbackMemory failed: {exc}")

        # 2. PatternMemory: apply delta correction instead of full record_outcome
        # The first update happened in _dispatch_feedback with heuristic confidence.
        # Now we apply only the DELTA between ground-truth and heuristic.
        try:
            old_heuristic_conf = self._feedback_memory.get_heuristic_confidence(query)
            if old_heuristic_conf is None:
                old_heuristic_conf = 0.55  # fallback for legacy signals
            delta = ground_truth_conf - old_heuristic_conf

            self._pattern_memory.apply_confidence_delta(
                query, delta, ground_truth_conf, old_heuristic_conf,
            )
            logger.debug(
                f"[memory:inject] PatternMemory delta correction: "
                f"delta={delta:.3f}, gt={ground_truth_conf:.3f}, "
                f"heuristic={old_heuristic_conf:.3f}"
            )
        except Exception as exc:
            logger.debug(f"[memory:inject] PatternMemory delta failed: {exc}")

        # 3. QuerySimilarityIndex: propagate confidence (closes the loop)
        try:
            updated = self._query_similarity.update_confidence(query, ground_truth_conf)
            if updated:
                logger.debug(
                    "[memory:inject] QuerySimilarity confidence "
                    "updated for '{}' → {:.3f}",
                    query[:50], ground_truth_conf,
                )
        except Exception as exc:
            logger.debug(f"[memory:inject] QuerySimilarity update failed: {exc}")

        # 4. For clearly failed queries, selectively record avoid_files
        #    Pass belief_snapshot so only low-belief files are blacklisted
        if ground_truth_conf < 0.25:
            try:
                belief_snapshot = self._belief_cache.pop(query, None)
                self._query_similarity.record_avoid_files(
                    query, belief_snapshot=belief_snapshot,
                )
                logger.debug(
                    "[memory:inject] Recorded avoid_files for '{}' (conf={:.3f})",
                    query[:50], ground_truth_conf,
                )
            except Exception as exc:
                logger.debug(f"[memory:inject] avoid_files failed: {exc}")

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
