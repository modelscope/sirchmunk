# Copyright (c) ModelScope Contributors. All rights reserved.
"""MemoryBridge — bidirectional mediator between cross-session
:class:`RetrievalMemory` and intra-session :class:`BeliefState`.

Responsibilities
~~~~~~~~~~~~~~~~

1. **extract_priors** (read path):
   Queries all memory layers to build a :class:`MemoryPrior` that the
   ``BeliefState`` can consume as warm-start information, eliminating the
   cold-start problem for new search sessions.

2. **absorb_beliefs** (write path):
   Enriches the standard :class:`FeedbackSignal` with fine-grained
   belief-state data (per-file beliefs, MCES scores, ESS convergence)
   and dispatches the enriched signal back to memory for cross-session
   learning.

Design notes
~~~~~~~~~~~~

* The bridge never imports ``BeliefState`` at runtime to avoid circular
  dependencies (``BeliefState`` imports ``MemoryPrior`` from this module).
  Instead it uses duck-typed access via ``getattr`` / ``hasattr``.
* All lookup methods are guarded with try/except so a single memory-layer
  failure never blocks the search pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger


# ------------------------------------------------------------------
# Data transfer object: Memory → BeliefState
# ------------------------------------------------------------------

@dataclass
class MemoryPrior:
    """Cross-session prior knowledge packaged for BeliefState consumption.

    All fields default to empty so callers can safely access any attribute
    without ``None`` checks.
    """

    path_scores: Dict[str, float] = field(default_factory=dict)
    """PathMemory hot_scores — files historically useful for similar queries."""

    entity_paths: Dict[str, float] = field(default_factory=dict)
    """CorpusMemory entity→file confidence scores."""

    dead_paths: Set[str] = field(default_factory=set)
    """FailureMemory-confirmed persistently useless file paths."""

    similar_query_files: Dict[str, float] = field(default_factory=dict)
    """Files that proved useful for semantically similar past queries."""

    chain_hint: Optional[List[Dict[str, str]]] = None
    """PatternMemory reasoning-chain template (e.g. search→read→compare)."""

    extra_files: Dict[str, float] = field(default_factory=dict)
    """Explicitly transferred file paths from similar-query memory hints."""

    avoid_files: Set[str] = field(default_factory=set)
    """Files that led to incorrect answers in past sessions."""

    avg_confidence: float = 0.0
    """Weighted average confidence across all memory sources (0-1)."""

    @property
    def is_empty(self) -> bool:
        return (
            not self.path_scores
            and not self.entity_paths
            and not self.dead_paths
            and not self.similar_query_files
            and self.chain_hint is None
            and not self.extra_files
            and not self.avoid_files
        )


# ------------------------------------------------------------------
# Bridge implementation
# ------------------------------------------------------------------

class MemoryBridge:
    """Bidirectional bridge between RetrievalMemory and BeliefState.

    Instantiated once per search session by ``AgenticSearch._search_deep``.
    All methods are synchronous (memory lookups are local / O(1)–O(log N)).
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    # ---- Read path: Memory → MemoryPrior ----

    def extract_priors(
        self,
        query: str,
        *,
        candidate_files: Optional[List[str]] = None,
        extra_files: Optional[List[str]] = None,
        extra_keywords: Optional[List[str]] = None,
    ) -> MemoryPrior:
        """Query all memory layers and build a warm-start prior.

        Parameters
        ----------
        query : str
            The user's search query.
        candidate_files : list[str], optional
            File paths discovered so far (e.g. from dir_scan).
        extra_files : list[str], optional
            Explicitly suggested files from similar-query memory hints.
        extra_keywords : list[str], optional
            Keywords extracted from the query (used for entity extraction).
        """
        prior = MemoryPrior()
        mem = self._memory

        # 1. PathMemory: historical file hotness
        if candidate_files:
            try:
                prior.path_scores = mem.get_path_scores(candidate_files)
            except Exception:
                pass

        # 2. CorpusMemory: entity → file path index
        try:
            entities = mem.extract_entities(extra_keywords or [])
            if entities:
                paths = mem.get_entity_paths(entities)
                for rank, p in enumerate(paths):
                    prior.entity_paths[p] = max(0.3, 0.8 - 0.05 * rank)
        except Exception:
            pass

        # 3. FailureMemory: dead paths
        if candidate_files:
            try:
                alive = mem.filter_dead_paths(candidate_files)
                prior.dead_paths = set(candidate_files) - set(alive)
            except Exception:
                pass

        # 4. QuerySimilarityIndex: transfer from similar historical queries
        _hint_confidences: list = []
        try:
            hints = mem.get_similar_query_hints(query, top_k=3)
            for hint in hints:
                _hint_confidences.append(hint.confidence)
                if hint.confidence >= 0.4:
                    for fp in hint.useful_files[:5]:
                        existing = prior.similar_query_files.get(fp, 0.0)
                        prior.similar_query_files[fp] = max(
                            existing, hint.confidence,
                        )
                # Only propagate avoid_files from near-exact matches
                # (similarity >= 0.95) to prevent over-generalization
                if hint.similarity >= 0.95:
                    for fp in getattr(hint, "avoid_files", None) or []:
                        prior.avoid_files.add(fp)
        except Exception:
            pass

        # 5. PatternMemory: reasoning chain template
        try:
            prior.chain_hint = mem.get_chain_hint(query)
        except Exception:
            pass

        # 6. Extra files from caller (similar-query hints)
        if extra_files:
            for fp in extra_files:
                prior.extra_files[fp] = 0.5

        # Compute aggregate confidence from available signals
        if _hint_confidences:
            prior.avg_confidence = sum(_hint_confidences) / len(_hint_confidences)

        if not prior.is_empty:
            n_priors = (
                len(prior.path_scores)
                + len(prior.entity_paths)
                + len(prior.similar_query_files)
                + len(prior.extra_files)
            )
            logger.info(
                "[MemoryBridge] extracted {} file priors, {} dead, "
                "{} avoid, chain_hint={}, avg_conf={:.2f}",
                n_priors,
                len(prior.dead_paths),
                len(prior.avoid_files),
                "yes" if prior.chain_hint else "no",
                prior.avg_confidence,
            )

        return prior

    # ---- Write path: BeliefState → enriched FeedbackSignal ----

    def enrich_feedback(
        self,
        signal: Any,
        belief_state: Any,
    ) -> None:
        """Enrich a ``FeedbackSignal`` with BA-ReAct belief data.

        Mutates *signal* in place, adding per-file beliefs, MCES metadata,
        ESS convergence, and dead-path candidates.  Uses duck-typed access
        to avoid importing BeliefState.

        Parameters
        ----------
        signal : FeedbackSignal
            The base feedback signal (already populated with context data).
        belief_state : BeliefState or None
            The intra-session belief tracker.  When ``None``, no enrichment
            is performed.
        """
        if belief_state is None:
            return

        try:
            bd = belief_state.to_feedback_dict()
        except Exception:
            return

        signal.belief_snapshot = bd.get("belief_snapshot")
        signal.mces_triggered_files = bd.get("mces_triggered_files")
        signal.ess_at_termination = bd.get("ess_at_termination")
        signal.convergence_achieved = bd.get("convergence_achieved", False)
        signal.high_value_files = bd.get("high_value_files")
        signal.dead_candidates = bd.get("dead_candidates")
