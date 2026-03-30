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
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .schemas import SearchPlan


# ------------------------------------------------------------------
# Data transfer object: Memory → BeliefState
# ------------------------------------------------------------------

@dataclass
class MemoryPrior:
    """Cross-session prior knowledge packaged for BeliefState consumption.

    All fields default to empty so callers can safely access any attribute
    without ``None`` checks.

    Only strategy-level and corpus-level priors are retained.  Instance-
    level priors (file hotness, similar-query files) were removed because
    they overfit to specific past queries and conflict with the Meta-RL
    architecture that learns generalizable strategies.
    """

    entity_paths: Dict[str, float] = field(default_factory=dict)
    """CorpusMemory entity→file confidence scores."""

    dead_paths: Set[str] = field(default_factory=set)
    """FailureMemory-confirmed persistently useless file paths."""

    chain_hint: Optional[List[Dict[str, str]]] = None
    """PatternMemory reasoning-chain template (e.g. search→read→compare)."""

    search_plan: Optional["SearchPlan"] = None
    """MAP-generated search plan (when meta-knowledge is available)."""

    strategy_rules: List[str] = field(default_factory=list)
    """Distilled strategy rules relevant to this query type."""

    @property
    def is_empty(self) -> bool:
        return (
            not self.entity_paths
            and not self.dead_paths
            and self.chain_hint is None
            and self.search_plan is None
            and not self.strategy_rules
        )


# ------------------------------------------------------------------
# Bridge implementation
# ------------------------------------------------------------------

class MemoryBridge:
    """Bidirectional bridge between RetrievalMemory and BeliefState.

    Instantiated once per search session by ``AgenticSearch._search_deep``.
    All methods are synchronous (memory lookups are local / O(1)–O(log N)).

    Parameters
    ----------
    memory : RetrievalMemory
        The cross-session memory manager.
    title_lookup_fn : callable, optional
        ``title_lookup_fn(title) -> list[str]`` that resolves an article
        title to corpus file paths.  Used as a fallback when CorpusMemory
        has no entity→path mappings (cold-start).
    """

    _TITLE_PRIOR_CONFIDENCE = 0.55
    _MAX_TITLE_LOOKUPS = 6

    def __init__(
        self,
        memory: Any,
        title_lookup_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._memory = memory
        self._title_lookup_fn = title_lookup_fn

    # ---- Read path: Memory → MemoryPrior ----

    def extract_priors(
        self,
        query: str,
        *,
        candidate_files: Optional[List[str]] = None,
        extra_keywords: Optional[List[str]] = None,
    ) -> MemoryPrior:
        """Query all memory layers and build a warm-start prior.

        Only strategy-level and corpus-level priors are extracted.
        Instance-level priors (file hotness, similar-query files) are
        intentionally excluded — the Meta-RL MAP planner provides
        superior guidance at the strategy level.

        Parameters
        ----------
        query : str
            The user's search query.
        candidate_files : list[str], optional
            File paths discovered so far (for dead-path filtering).
        extra_keywords : list[str], optional
            Keywords extracted from the query (used for entity extraction).
        """
        prior = MemoryPrior()
        mem = self._memory

        # 1. CorpusMemory: entity → file path index
        try:
            entities = mem.extract_entities(extra_keywords or [])
            if entities:
                paths = mem.get_entity_paths(entities)
                for rank, p in enumerate(paths):
                    prior.entity_paths[p] = max(0.3, 0.8 - 0.05 * rank)
        except Exception:
            pass

        # 1b. TitleIndex fallback: resolve entity keywords to file paths
        #     when CorpusMemory has no mappings (cold start).
        if not prior.entity_paths and self._title_lookup_fn:
            try:
                kws = extra_keywords or []
                looked = 0
                for kw in kws:
                    if looked >= self._MAX_TITLE_LOOKUPS:
                        break
                    if len(kw) <= 2 or kw[0].islower():
                        continue
                    fps = self._title_lookup_fn(kw)
                    if fps:
                        for fp in fps[:2]:
                            prior.entity_paths[str(fp)] = (
                                self._TITLE_PRIOR_CONFIDENCE
                            )
                    looked += 1
            except Exception:
                pass

        # 2. FailureMemory: dead paths
        if candidate_files:
            try:
                alive = mem.filter_dead_paths(candidate_files)
                prior.dead_paths = set(candidate_files) - set(alive)
            except Exception:
                pass

        # 3. PatternMemory: reasoning chain template
        try:
            prior.chain_hint = mem.get_chain_hint(query)
        except Exception:
            pass

        # 4. Meta-knowledge: distilled strategy rules for the ReAct agent
        try:
            mk = mem.get_meta_knowledge(query)
            prior.strategy_rules = mk.get("distilled_rules", [])
        except Exception:
            pass

        if not prior.is_empty:
            logger.info(
                "[MemoryBridge] extracted {} entity priors, {} dead, "
                "chain_hint={}, strategy_rules={}",
                len(prior.entity_paths),
                len(prior.dead_paths),
                "yes" if prior.chain_hint else "no",
                len(prior.strategy_rules),
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
