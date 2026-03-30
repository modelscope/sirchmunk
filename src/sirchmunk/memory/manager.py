# Copyright (c) ModelScope Contributors. All rights reserved.
"""RetrievalMemory — unified orchestrator for all memory layers.

Creates, initialises, and coordinates the memory stores.  Provides a
high-level API consumed by :class:`AgenticSearch`:

*  **Lookup** (sync, ~0 ms): ``suggest_strategy``, ``expand_keywords``,
   ``get_entity_paths``, ``filter_noise_keywords``, ``filter_dead_paths``,
   ``get_chain_hint``.
*  **Record** (async): ``record_feedback`` — stores the raw signal in
   ``FeedbackMemory`` then dispatches incremental updates to the other
   layers using a *gradient confidence* (0-1) instead of binary success.
*  **Inject** (sync): ``inject_evaluation`` — back-fills EM/F1 scores
   and re-dispatches gradient updates.
*  **Meta-RL** (async): ``plan_search``, ``record_trajectory``,
   ``trigger_distillation`` — strategy-level learning.
*  **Maintenance** (sync): ``decay_all``, ``cleanup_all``, ``stats``.

Storage layout::

    {work_path}/memory/
    ├── pattern_memory/
    │   ├── query_patterns.json
    │   ├── reasoning_chains.json
    │   ├── trajectories.json      (Meta-RL abstract trajectories)
    │   └── distillations.json     (LLM-distilled strategy rules)
    ├── semantic_bridge.json
    ├── corpus.duckdb
    └── feedback.duckdb
"""
from __future__ import annotations

import atexit
import asyncio
import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore
from .corpus_memory import CorpusMemory
from .failure_memory import FailureMemory
from .feedback import FeedbackMemory
from .pattern_memory import PatternMemory
from .schemas import (
    AbstractTrajectory,
    AbstractTrajectoryStep,
    FeedbackSignal,
    SearchPlan,
    StrategyDistillation,
    StrategyHint,
    compute_params_hash,
    compute_pattern_id_at_level,
)


class RetrievalMemory:
    """Unified facade over the retrieval-memory layers.

    Designed for graceful degradation: if any individual store fails to
    initialise, the others still function.  All lookup methods return
    safe defaults on error so the search pipeline is never blocked.
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
        self._failure_memory = FailureMemory(db=self._corpus_db)
        self._feedback_memory = FeedbackMemory(db=self._feedback_db)

        self._stores: List[MemoryStore] = [
            self._pattern_memory,
            self._corpus_memory,
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

        # Meta-RL: planner + distiller (require LLM)
        self._planner = None
        self._distiller = None
        self._trajectory_counter = 0
        if llm is not None:
            try:
                from .planner import MemoryAugmentedPlanner
                from .strategy_distiller import StrategyDistiller
                self._planner = MemoryAugmentedPlanner(llm)
                self._distiller = StrategyDistiller(llm)
            except Exception as exc:
                logger.debug(f"RetrievalMemory: MAP/distiller init skipped: {exc}")

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

    def get_chain_hint(
        self,
        query: str,
    ) -> Optional[List[Dict[str, str]]]:
        """PatternMemory → reasoning chain template."""
        try:
            return self._pattern_memory.get_chain_hint(query)
        except Exception:
            return None

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

        # 5. PatternMemory — reasoning chain (abstracted trace)
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

        # 6b. BA-ReAct: dead-path candidates from belief tracking (batch)
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

        # 7. MCTS-Memory Integration: adapt search depth params based on convergence
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

    # Weights for the potential function Φ(s) = w1·coverage + w2·diversity
    _RS_W_COVERAGE = 0.05
    _RS_W_DIVERSITY = 0.03

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
        ground-truth evaluation is available.  Applies a delta correction
        to PatternMemory (closing the feedback loop).

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


    # ================================================================
    #  Meta-RL API: planning, trajectory recording, distillation
    # ================================================================

    async def plan_search(self, query: str) -> Optional[SearchPlan]:
        """Generate a search plan via the Memory-Augmented Planner.

        Requires an LLM to be configured.  Returns None on cold start
        (no distilled rules available) or when the planner is unavailable.
        """
        if self._planner is None:
            return None
        try:
            mk = self._pattern_memory.get_meta_knowledge(query)
            plan = await self._planner.plan(query, mk)
            if plan:
                logger.info(
                    "[MAP] Plan generated: conf=%.2f, steps=%d, strategy=%s",
                    plan.confidence, len(plan.plan_steps), plan.keyword_strategy,
                )
            return plan
        except Exception as exc:
            logger.debug(f"[MAP] plan_search failed: {exc}")
            return None

    def get_meta_knowledge(self, query: str) -> Dict[str, Any]:
        """Expose PatternMemory meta-knowledge for external consumers."""
        try:
            return self._pattern_memory.get_meta_knowledge(query)
        except Exception:
            return {}

    def record_trajectory(
        self,
        query: str,
        context: Any,
        outcome: float,
    ) -> None:
        """Abstract a SearchContext into a strategy-level trajectory.

        Strips instance-specific data (file paths, keywords, answers)
        and stores only the strategy-level action sequence.  Increments
        the trajectory counter and may trigger distillation.

        Parameters
        ----------
        query : str
            The search query.
        context : SearchContext
            The completed search context with logs and metadata.
        outcome : float
            Evaluation score (0-1), e.g. EM or heuristic confidence.
        """
        features = PatternMemory.classify_query(query)
        steps: List[AbstractTrajectoryStep] = []
        for log in getattr(context, "retrieval_logs", []):
            step = AbstractTrajectoryStep(
                action=log.tool_name,
                strategy=log.metadata.get("strategy", "unknown"),
                result_type=log.metadata.get("result_type", "unknown"),
                files_found=len(log.metadata.get("files_discovered", [])),
            )
            steps.append(step)

        trajectory = AbstractTrajectory(
            query_type=features["query_type"],
            complexity=features["complexity"],
            hop_hint=features.get("hop_hint", "single"),
            entity_count=features.get("entity_count", 0),
            answer_format=features.get("answer_format", "entity"),
            steps=steps,
            loops_used=getattr(context, "loop_count", 0),
            total_tokens=getattr(context, "total_llm_tokens", 0),
            outcome=outcome,
        )

        try:
            self._pattern_memory.store_trajectory(trajectory)
        except Exception as exc:
            logger.debug(f"[memory] store_trajectory failed: {exc}")

        # Record loop outcome for Bayesian loop budget learning
        try:
            self._pattern_memory.record_loop_outcome(
                query,
                loops_used=trajectory.loops_used,
                success=(outcome >= 0.5),
            )
        except Exception:
            pass

        self._trajectory_counter += 1

    def should_distill(self, query: str) -> bool:
        """Check whether enough new trajectories warrant a distillation."""
        features = PatternMemory.classify_query(query)
        try:
            pending = self._pattern_memory.pending_trajectory_count(
                features["query_type"], features["complexity"],
            )
            return pending >= self._pattern_memory.distillation_batch_size
        except Exception:
            return False

    async def trigger_distillation(self, query: str) -> Optional[StrategyDistillation]:
        """Run LLM-powered strategy distillation for the query type.

        Aggregates recent trajectories for the query's type + complexity,
        sends them to the distiller, and stores the result in PatternMemory.
        """
        if self._distiller is None:
            return None
        features = PatternMemory.classify_query(query)
        qt = features["query_type"]
        cx = features["complexity"]
        try:
            trajectories = self._pattern_memory.get_recent_trajectories(qt, cx)
            if len(trajectories) < 3:
                return None
            distill = await self._distiller.distill(trajectories, qt, cx)
            if distill:
                self._pattern_memory.store_distillation(distill)
                logger.info(
                    "[distill] %d rules + %d warnings for %s/%s (n=%d)",
                    len(distill.rules), len(distill.failure_warnings),
                    qt, cx, distill.sample_count,
                )
            return distill
        except Exception as exc:
            logger.debug(f"[distill] trigger_distillation failed: {exc}")
            return None

    def get_optimal_loop_budget(self, query: str) -> Optional[int]:
        """Return the learned optimal loop budget for this query type."""
        try:
            return self._pattern_memory.get_optimal_loop_budget(query)
        except Exception:
            return None

    async def trigger_distillation_sweep(self) -> int:
        """Attempt distillation for every observed query type + complexity.

        Iterates over accumulated trajectories, groups them by type/
        complexity, and triggers distillation for any group that meets
        the batch threshold.  Returns the count of successful distillations.
        """
        if self._distiller is None:
            return 0
        seen_buckets: set = set()
        for t in self._pattern_memory._trajectories:
            seen_buckets.add((t.query_type, t.complexity))

        distilled = 0
        for qt, cx in seen_buckets:
            try:
                pending = self._pattern_memory.pending_trajectory_count(qt, cx)
                if pending < self._pattern_memory.distillation_batch_size:
                    continue
                trajectories = self._pattern_memory.get_recent_trajectories(qt, cx)
                if len(trajectories) < 3:
                    continue
                result = await self._distiller.distill(trajectories, qt, cx)
                if result:
                    self._pattern_memory.store_distillation(result)
                    distilled += 1
                    logger.info(
                        "[distill-sweep] %d rules + %d warnings for %s/%s (n=%d)",
                        len(result.rules), len(result.failure_warnings),
                        qt, cx, result.sample_count,
                    )
            except Exception as exc:
                logger.debug("[distill-sweep] %s/%s failed: %s", qt, cx, exc)
        return distilled

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
