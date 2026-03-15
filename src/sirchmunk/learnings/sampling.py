# Copyright (c) ModelScope Contributors. All rights reserved.
"""Monte Carlo importance sampling strategies for evidence extraction.

Implements a faithful Monte Carlo Importance Sampling (MCIS) pipeline:

- **Round 1** (Exploration): multi-signal pre-scored chunks + stratified
  random exploration → batch LLM evaluation.
- **Round 2+** (Exploitation via IS): score-proportional seed resampling
  → Gaussian proposal distribution → data-driven adaptive sigma →
  batch LLM evaluation → ESS-monitored convergence.

Key IS concepts applied:
- Proposal q(x) = Gaussian mixture centered on resampled seeds
- Target p(x) = LLM relevance score
- Score-proportional resampling ≈ Sequential IS / Particle Filter step
- Adaptive sigma from spatial score distribution (no fixed decay)
- Effective Sample Size (ESS) for convergence monitoring
"""

import math
import random
from dataclasses import dataclass
from typing import List, Set, Tuple

from sirchmunk.learnings._types import SampleWindow
from sirchmunk.learnings.scoring import ScoredChunk


@dataclass
class SamplingConfig:
    """Dynamic configuration for one sampling session."""

    max_rounds: int
    probe_window: int
    roi_window: int
    fuzz_candidates_num: int
    random_exploration_num: int
    samples_per_round: int
    top_k_seeds: int = 2
    confidence_threshold: float = 8.5
    skip_sampling: bool = False


class AdaptiveSamplingStrategy:
    """Monte Carlo importance sampling strategy for evidence extraction.

    All methods are ``@staticmethod`` — the configuration is generated
    once via ``configure()`` and threaded through the orchestrator.
    """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def configure(doc_len: int) -> SamplingConfig:
        """Continuous log-scale adaptive configuration.

        Instead of fixed 3-tier buckets, uses ``log2(doc_len)`` for smooth
        parameter scaling across all document sizes.
        """
        if doc_len <= 3_000:
            return SamplingConfig(
                max_rounds=1,
                probe_window=doc_len,
                roi_window=doc_len,
                fuzz_candidates_num=1,
                random_exploration_num=0,
                samples_per_round=1,
                top_k_seeds=1,
                confidence_threshold=6.0,
                skip_sampling=True,
            )

        log_len = math.log2(max(doc_len, 1))

        max_rounds = min(3, max(1, int(log_len - 11)))
        probe_window = max(400, min(1500, int(doc_len * 0.08)))
        roi_window = max(1500, min(3000, int(doc_len * 0.15)))
        fuzz_candidates = max(2, min(6, int(log_len - 9)))
        random_exploration = max(1, min(3, int(log_len - 12)))
        samples_per_round = max(2, min(6, int(log_len - 10)))

        return SamplingConfig(
            max_rounds=max_rounds,
            probe_window=probe_window,
            roi_window=roi_window,
            fuzz_candidates_num=fuzz_candidates,
            random_exploration_num=random_exploration,
            samples_per_round=samples_per_round,
            confidence_threshold=8.5,
        )

    # ------------------------------------------------------------------
    # Early stopping (IS convergence monitoring)
    # ------------------------------------------------------------------

    @staticmethod
    def should_continue(
        best_score: float,
        round_num: int,
        config: SamplingConfig,
        score_variance: float = 0.0,
    ) -> bool:
        """Multi-criteria early-stop decision with ESS awareness.

        - Hard stop when best score >= confidence threshold.
        - Soft stop after round 1 with score >= 7 and low variance
          (proposal already concentrated on a high-value region).
        - Always stop when max_rounds is reached.
        """
        if best_score >= config.confidence_threshold:
            return False
        if round_num == 1 and best_score >= 7.0 and score_variance < 4.0:
            return False
        if round_num >= config.max_rounds:
            return False
        return True

    # ------------------------------------------------------------------
    # Selection from pre-scored chunks (Round 1)
    # ------------------------------------------------------------------

    @staticmethod
    def select_from_scored(
        scored_chunks: List[ScoredChunk],
        visited: Set[int],
        max_candidates: int,
        min_score: float = 0.1,
    ) -> List[ScoredChunk]:
        """Pick top pre-scored chunks, skipping visited regions."""
        selected: List[ScoredChunk] = []
        for sc in scored_chunks:
            if sc.combined_score < min_score:
                continue
            if any(abs(v - sc.start) < 200 for v in visited):
                continue
            visited.add(sc.start)
            selected.append(sc)
            if len(selected) >= max_candidates:
                break
        return selected

    # ------------------------------------------------------------------
    # Random stratified sampling (exploration)
    # ------------------------------------------------------------------

    @staticmethod
    def sample_random(
        doc_content: str,
        doc_len: int,
        probe_window: int,
        count: int,
        visited: Set[int],
    ) -> List[SampleWindow]:
        """Stratified random sampling for global exploration."""
        samples: List[SampleWindow] = []
        if count <= 0 or doc_len <= probe_window:
            return samples

        step = doc_len // count
        for i in range(count):
            section_start = i * step
            section_end = min((i + 1) * step, doc_len)
            max_start = max(section_start, section_end - probe_window)
            rand_start = random.randint(section_start, max_start)
            start = max(0, min(rand_start, doc_len - probe_window))
            end = min(start + probe_window, doc_len)

            if any(abs(v - start) < (probe_window // 2) for v in visited):
                continue

            visited.add(start)
            samples.append(
                SampleWindow(
                    start_idx=start,
                    end_idx=end,
                    content=doc_content[start:end],
                    round_num=1,
                    source="stratified",
                )
            )
        return samples

    # ------------------------------------------------------------------
    # Monte Carlo importance sampling (refinement rounds)
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_seeds(
        candidates: List[SampleWindow],
        n_seeds: int,
    ) -> List[SampleWindow]:
        """Score-proportional seed resampling (IS resampling step).

        Instead of deterministic top-k, selects seeds with probability
        proportional to their score.  This is the "resampling" step
        from Sequential Monte Carlo / Particle Filters — it prevents
        over-focusing on a single region when multiple regions have
        similar scores.

        Deduplicates by position to ensure spatial diversity.
        """
        eligible = [c for c in candidates if c.score > 0]
        if not eligible:
            return candidates[:n_seeds]
        if len(eligible) <= n_seeds:
            return eligible

        weights = [c.score for c in eligible]
        total = sum(weights)
        if total <= 0:
            return eligible[:n_seeds]
        probs = [w / total for w in weights]

        selected: List[SampleWindow] = []
        selected_positions: Set[int] = set()

        attempts = 0
        while len(selected) < n_seeds and attempts < n_seeds * 10:
            attempts += 1
            idx = _weighted_choice(probs)
            candidate = eligible[idx]
            center = (candidate.start_idx + candidate.end_idx) // 2

            if any(abs(center - p) < 200 for p in selected_positions):
                continue
            selected_positions.add(center)
            selected.append(candidate)

        return selected if selected else eligible[:n_seeds]

    @staticmethod
    def _compute_adaptive_sigma(
        seeds: List[SampleWindow],
        doc_len: int,
        current_round: int,
    ) -> float:
        """Data-driven sigma for the Gaussian proposal.

        Derives sigma from the spatial spread of high-scoring seeds
        rather than using a fixed ``doc_len/N / decay^round`` formula.

        - When seeds are clustered (small spread) → small sigma (focus).
        - When seeds are spread out → larger sigma (explore wider).
        - Falls back to ``doc_len / 15`` when only one seed exists.
        """
        if len(seeds) < 2:
            base = doc_len / 15
            return base / (1.5 ** (current_round - 1))

        centers = sorted(
            (s.start_idx + s.end_idx) // 2 for s in seeds
        )
        # Use inter-seed distance statistics
        dists = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        mean_dist = sum(dists) / len(dists)
        spread = max(centers) - min(centers)

        # Sigma ≈ half of the mean inter-seed distance, bounded
        sigma = max(
            200,
            min(mean_dist / 2, spread / 3, doc_len / 10),
        )
        # Gentle shrink across rounds (less aggressive than fixed decay)
        sigma /= 1.3 ** (current_round - 1)
        return sigma

    @staticmethod
    def _allocate_children(
        seeds: List[SampleWindow],
        total_needed: int,
    ) -> List[Tuple[SampleWindow, int]]:
        """Score-proportional child allocation across seeds.

        Higher-scoring seeds receive more children, consistent with
        importance sampling: regions where the target density p(x)
        is higher should be sampled more densely.
        """
        if not seeds:
            return []

        weights = [max(s.score, 0.1) for s in seeds]
        total_w = sum(weights)

        allocation: List[Tuple[SampleWindow, int]] = []
        assigned = 0
        for i, seed in enumerate(seeds):
            if i == len(seeds) - 1:
                n = total_needed - assigned
            else:
                n = max(1, round(total_needed * weights[i] / total_w))
            allocation.append((seed, n))
            assigned += n

        return allocation

    @staticmethod
    def sample_gaussian(
        seeds: List[SampleWindow],
        doc_len: int,
        probe_window: int,
        current_round: int,
        samples_needed: int,
        visited: Set[int],
        doc_content: str,
        all_candidates: List[SampleWindow] = None,
    ) -> List[SampleWindow]:
        """Gaussian importance sampling with IS-faithful refinements.

        Implements a proper IS proposal:
        1. Score-proportional seed resampling (avoids deterministic top-k)
        2. Data-driven adaptive sigma (no fixed exponential decay)
        3. Score-proportional child allocation per seed
        4. Gaussian proposal q(x) centered on each resampled seed
        """
        if not seeds:
            return []

        strategy = AdaptiveSamplingStrategy

        # 1. IS resampling: select seeds with probability ∝ score
        pool = all_candidates if all_candidates else seeds
        resampled = strategy._resample_seeds(pool, len(seeds))

        # 2. Adaptive sigma from spatial score distribution
        sigma = strategy._compute_adaptive_sigma(
            resampled, doc_len, current_round,
        )

        # 3. Score-proportional child allocation
        allocation = strategy._allocate_children(resampled, samples_needed)

        # 4. Gaussian sampling from proposal q(x)
        samples: List[SampleWindow] = []
        remaining = samples_needed

        for seed, n_target in allocation:
            if remaining <= 0:
                break
            center = (seed.start_idx + seed.end_idx) // 2

            for _ in range(n_target * 3):
                if remaining <= 0:
                    break

                new_center = int(random.gauss(center, sigma))
                raw_start = new_center - (probe_window // 2)
                start = max(0, min(raw_start, doc_len - probe_window))
                end = min(start + probe_window, doc_len)

                if any(abs(v - start) < (probe_window // 3) for v in visited):
                    continue

                visited.add(start)
                samples.append(
                    SampleWindow(
                        start_idx=start,
                        end_idx=end,
                        content=doc_content[start:end],
                        round_num=current_round,
                        source="gaussian",
                    )
                )
                remaining -= 1

        return samples

    @staticmethod
    def compute_ess(scores: List[float]) -> float:
        """Effective Sample Size from importance weights.

        ESS = (Σw)² / Σ(w²), where w_i = score_i.
        A low ESS relative to N indicates weight degeneracy — the
        proposal is poorly matched to the target distribution.
        Returns 0.0 for empty input.
        """
        if not scores:
            return 0.0
        positive = [s for s in scores if s > 0]
        if not positive:
            return 0.0
        total = sum(positive)
        sum_sq = sum(w * w for w in positive)
        if sum_sq == 0:
            return 0.0
        return (total * total) / sum_sq


def _weighted_choice(probs: List[float]) -> int:
    """Select an index with probability proportional to probs."""
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1
