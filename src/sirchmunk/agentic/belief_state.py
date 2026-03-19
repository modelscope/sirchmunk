# Copyright (c) ModelScope Contributors. All rights reserved.
"""Belief-Augmented ReAct (BA-ReAct) state tracker.

Maintains per-file relevance beliefs and provides analytical (train-free)
decision support for the ReAct search agent:

- **Bayesian belief updates** from keyword_search and file_read results.
- **UCB-based file ranking** for exploration–exploitation balance.
- **MCES trigger decisions** for high-value large files.
- **ESS-based stopping signals** when evidence is sufficiently concentrated.
- **Advisory text** injected into the continuation prompt.
- **Memory warm-start** from cross-session priors via :class:`MemoryPrior`.

All computations are closed-form; no learned parameters or training required.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from sirchmunk.memory.bridge import MemoryPrior


class BeliefState:
    """Track and update beliefs about file relevance during a search session.

    Each discovered file is assigned a belief score in [0, 1] representing
    the system's confidence that it contains relevant evidence.  Beliefs are
    updated via lightweight Bayesian rules after each tool observation.

    The UCB (Upper Confidence Bound) ranking balances exploitation (reading
    high-belief files) with exploration (trying under-sampled files), and
    the adaptive exploration coefficient ``c_t = c_0 * budget_ratio``
    naturally reduces exploration as the token budget shrinks.
    """

    # UCB exploration base coefficient
    _UCB_C0: float = 1.0

    # Lazy MCES activation thresholds
    _MCES_BELIEF_THRESHOLD: float = 0.6
    _MCES_SIZE_THRESHOLD: int = 20_000  # characters
    _MCES_BUDGET_MIN: int = 12_000  # tokens

    # Maximum initial belief from memory priors (preserves exploration)
    _MAX_WARM_BELIEF: float = 0.5

    def __init__(self) -> None:
        self._beliefs: Dict[str, float] = {}
        self._reads: Dict[str, int] = {}
        self._sizes: Dict[str, int] = {}
        self._mces_done: Set[str] = set()
        self._actions: int = 0

        # Cross-session memory priors (populated by warm_start)
        self._dead_paths: Set[str] = set()
        self._memory_priors: Dict[str, float] = {}
        self._chain_hint: Optional[List[Dict[str, str]]] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def file_beliefs(self) -> Dict[str, float]:
        """Snapshot of current file beliefs (read-only copy)."""
        return dict(self._beliefs)

    @property
    def mces_completed_files(self) -> Set[str]:
        """Files that have already undergone deep MCES extraction."""
        return set(self._mces_done)

    @property
    def chain_hint(self) -> Optional[List[Dict[str, str]]]:
        """Reasoning chain template from PatternMemory (if available)."""
        return self._chain_hint

    # ------------------------------------------------------------------
    # Memory warm-start
    # ------------------------------------------------------------------

    # Minimum aggregate confidence to apply any warm-start priors
    _WARM_START_MIN_CONFIDENCE: float = 0.3

    # Elevated cap for high-confidence priors (confirmed correct in prior runs)
    _MAX_WARM_BELIEF_HIGH: float = 0.65

    def warm_start(self, prior: "MemoryPrior") -> None:
        """Initialize beliefs from cross-session memory priors.

        Uses confidence-scaled dampening: when ``prior.avg_confidence``
        is low, the effective cap shrinks proportionally, reducing the
        influence of uncertain or stale memory data.

        For high-confidence priors (>= 0.8), the belief cap is raised
        to ``_MAX_WARM_BELIEF_HIGH`` so that confirmed-correct files
        have a stronger initial advantage.

        Layering order (later layers can boost but not exceed the cap):
        1. PathMemory hot_scores (weakest — historical frequency)
        2. Entity-path index (moderate — content-level association)
        3. Similar-query transfer (strongest — outcome-confirmed)
        4. Extra files (explicit hint — treated as moderate prior)
        5. Avoid files (negative memory — suppressed belief)
        """
        self._dead_paths = set(prior.dead_paths) if prior.dead_paths else set()
        self._chain_hint = prior.chain_hint

        # Merge avoid_files into dead_paths for unified suppression
        avoid = getattr(prior, "avoid_files", None) or set()
        self._dead_paths |= set(avoid)

        if prior.avg_confidence < self._WARM_START_MIN_CONFIDENCE:
            return

        # High-confidence priors get a higher cap (越用越准)
        conf_scale = min(1.0, prior.avg_confidence / 0.8)
        base_cap = self._MAX_WARM_BELIEF
        if prior.avg_confidence >= 0.8:
            base_cap = self._MAX_WARM_BELIEF_HIGH
        cap = base_cap * conf_scale

        # Layer 1: path hotness (dampened by 0.4)
        for fp, score in prior.path_scores.items():
            if fp not in self._dead_paths:
                self._beliefs[fp] = min(cap, score * 0.4)
                self._memory_priors[fp] = score

        # Layer 2: entity-based priors (Bayesian combination, dampened)
        for fp, score in prior.entity_paths.items():
            if fp not in self._dead_paths:
                existing = self._beliefs.get(fp, 0.0)
                combined = 1 - (1 - existing) * (1 - score * 0.5)
                self._beliefs[fp] = min(cap, combined)
                self._memory_priors[fp] = max(
                    self._memory_priors.get(fp, 0.0), score,
                )

        # Layer 3: similar-query file transfer (confidence-proportional)
        for fp, score in prior.similar_query_files.items():
            if fp not in self._dead_paths:
                existing = self._beliefs.get(fp, 0.0)
                weight = 0.5 if prior.avg_confidence >= 0.8 else 0.3
                self._beliefs[fp] = max(existing, min(cap, score * weight))

        # Layer 4: explicitly suggested extra files
        for fp, score in prior.extra_files.items():
            if fp not in self._dead_paths:
                existing = self._beliefs.get(fp, 0.0)
                self._beliefs[fp] = max(existing, min(cap, score))

        # Layer 5: negative beliefs for avoid_files
        for fp in avoid:
            self._beliefs[fp] = 0.02

    # ------------------------------------------------------------------
    # Belief updates
    # ------------------------------------------------------------------

    def update_from_search(self, ranked_paths: List[str]) -> None:
        """Update beliefs from keyword_search discovered files.

        Uses rank-based initialization: higher-ranked files receive
        higher initial beliefs.  For files seen in a previous search,
        beliefs are combined with Bayesian-style aggregation.

        Dead paths receive a near-zero belief so they are deprioritised
        but not entirely invisible (allows recovery from false positives).
        """
        if not ranked_paths:
            return
        for rank, fp in enumerate(ranked_paths):
            if fp in self._dead_paths:
                self._beliefs[fp] = 0.05
                continue

            new_signal = max(0.2, 0.7 - 0.05 * rank)
            prior = self._beliefs.get(fp)
            if prior is not None:
                self._beliefs[fp] = 1 - (1 - prior) * (1 - new_signal)
            else:
                # First-time file: blend with memory prior if available
                memory_boost = self._memory_priors.get(fp, 0.0)
                base = new_signal
                if memory_boost > 0:
                    base = 1 - (1 - new_signal) * (1 - memory_boost * 0.3)
                self._beliefs[fp] = base
        self._actions += 1

    def update_from_read(
        self,
        file_id: str,
        content_chars: int,
        found_new_info: bool,
    ) -> None:
        """Update beliefs after a standard file_read.

        Boosts belief if new information was found; applies Bayesian
        exclusion (halves belief) otherwise.
        """
        self._reads[file_id] = self._reads.get(file_id, 0) + 1
        self._sizes[file_id] = content_chars
        if file_id in self._beliefs:
            if found_new_info:
                self._beliefs[file_id] = min(
                    1.0, self._beliefs[file_id] * 1.2,
                )
            else:
                self._beliefs[file_id] *= 0.5
        self._actions += 1

    def update_from_mces(
        self,
        file_id: str,
        best_score: float,
        is_found: bool,
    ) -> None:
        """Update beliefs with MCES LLM-evaluated evidence scores.

        MCES scores (1-10 scale) provide high-confidence relevance
        signals that override the coarser keyword-search priors.
        Also marks the file as read so it no longer appears in
        "promising unread" advisories.
        """
        if is_found and best_score >= 4.0:
            self._beliefs[file_id] = min(1.0, best_score / 10.0)
        else:
            self._beliefs[file_id] = self._beliefs.get(file_id, 0.3) * 0.1
        self._mces_done.add(file_id)
        self._reads[file_id] = self._reads.get(file_id, 0) + 1
        self._actions += 1

    # ------------------------------------------------------------------
    # Decision support
    # ------------------------------------------------------------------

    def should_trigger_mces(
        self,
        file_id: str,
        file_size: int,
        budget_remaining: int,
    ) -> bool:
        """Decide whether to activate Lazy MCES for a file.

        Conditions: (1) not already processed, (2) not a dead path,
        (3) high belief (or memory-confirmed useful), (4) large file,
        (5) sufficient token budget.
        """
        if file_id in self._mces_done:
            return False
        if file_id in self._reads:
            return False
        if file_id in self._dead_paths:
            return False

        belief = self._beliefs.get(file_id, 0.3)

        # Memory-confirmed useful files get a lower activation threshold
        threshold = self._MCES_BELIEF_THRESHOLD
        memory_score = self._memory_priors.get(file_id, 0.0)
        if memory_score >= 0.6:
            threshold *= 0.75

        return (
            belief >= threshold
            and file_size > self._MCES_SIZE_THRESHOLD
            and budget_remaining > self._MCES_BUDGET_MIN
        )

    def rank_files_ucb(
        self,
        candidates: List[str],
        budget_ratio: float,
    ) -> List[Tuple[str, float]]:
        """Rank candidate files by UCB score.

        Q_UCB(f) = mu(f) + c_t * sqrt(ln t / max(1, n(f)))

        where mu is the belief (optionally boosted by memory priors for
        unread files), c_t adapts to remaining budget, and n(f) is the
        number of reads for that file.
        """
        t = max(1, self._actions)
        c_t = self._UCB_C0 * max(0.1, budget_ratio)

        ranked: List[Tuple[str, float]] = []
        for fp in candidates:
            if fp in self._dead_paths:
                ranked.append((fp, -1.0))
                continue

            mu = self._beliefs.get(fp, 0.3)
            n_i = self._reads.get(fp, 0)

            # Blend memory prior for unread files (decays to 0 after reads)
            if n_i == 0:
                memory_score = self._memory_priors.get(fp, 0.0)
                if memory_score > 0:
                    mu = mu * 0.7 + memory_score * 0.3

            exploration = c_t * math.sqrt(math.log(t + 1) / max(1, n_i))
            ranked.append((fp, mu + exploration))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    # ------------------------------------------------------------------
    # Convergence monitoring
    # ------------------------------------------------------------------

    def compute_ess(self) -> float:
        """File-level Effective Sample Size.

        ESS = (sum b)^2 / sum(b^2).  Low ESS/N indicates beliefs are
        concentrated on few files (evidence convergence).
        """
        beliefs = [b for b in self._beliefs.values() if b > 0]
        if not beliefs:
            return 0.0
        total = sum(beliefs)
        sum_sq = sum(b * b for b in beliefs)
        if sum_sq < 1e-12:
            return 0.0
        return (total * total) / sum_sq

    def should_stop_early(self) -> bool:
        """Heuristic stopping signal based on evidence concentration.

        Returns True when beliefs are concentrated on <=2 files and
        enough files have been explored to be confident.
        """
        n = len(self._beliefs)
        if n < 3:
            return False
        ess = self.compute_ess()
        n_read = sum(1 for v in self._reads.values() if v > 0)
        return ess < 2.5 and n_read >= 3

    # ------------------------------------------------------------------
    # Advisory signals
    # ------------------------------------------------------------------

    def get_advisory(self) -> str:
        """Format advisory signals for the continuation prompt.

        Returns an empty string when there is nothing actionable to
        report, keeping the prompt clean.
        """
        parts: List[str] = []

        # Highlight high-value unread files (excluding dead paths)
        unread_high = [
            (fp, b)
            for fp, b in self._beliefs.items()
            if self._reads.get(fp, 0) == 0
            and b >= 0.4
            and fp not in self._dead_paths
        ]
        if unread_high:
            unread_high.sort(key=lambda x: x[1], reverse=True)
            names = ", ".join(
                f"{fp.rsplit('/', 1)[-1]}({b:.0%})"
                for fp, b in unread_high[:3]
            )
            parts.append(f"Promising unread: {names}")

        # ESS-based concentration signal
        n = len(self._beliefs)
        if n >= 3:
            ess = self.compute_ess()
            if ess / n < 0.3:
                parts.append("Evidence concentrated — consider answering")

        # Chain hint advisory (only useful in early rounds)
        if self._chain_hint and self._actions < 3:
            parts.append(
                "Reasoning pattern available from similar past queries",
            )

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Feedback export
    # ------------------------------------------------------------------

    def to_feedback_dict(self) -> Dict[str, Any]:
        """Export belief-state data for enriched feedback signals.

        Returns a dict suitable for merging into :class:`FeedbackSignal`
        fields.  Belief snapshot is limited to the top 20 files by belief
        to avoid serialisation bloat.
        """
        top_beliefs = sorted(
            self._beliefs.items(), key=lambda x: x[1], reverse=True,
        )[:20]

        high_value = [
            fp for fp, b in self._beliefs.items() if b >= 0.5
        ]
        dead_candidates = [
            fp
            for fp in self._reads
            if self._beliefs.get(fp, 0) < 0.1
        ]

        return {
            "belief_snapshot": dict(top_beliefs),
            "mces_triggered_files": sorted(self._mces_done),
            "ess_at_termination": self.compute_ess(),
            "convergence_achieved": self.should_stop_early(),
            "high_value_files": high_value,
            "dead_candidates": dead_candidates,
        }
