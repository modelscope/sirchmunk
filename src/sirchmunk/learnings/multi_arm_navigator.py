# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from sirchmunk.learnings.evidence_processor import SampleWindow
from sirchmunk.learnings.lens_config import LensConfig


# --------------------------------------------------------------------------
# Arm Dataclass
# --------------------------------------------------------------------------


@dataclass
class Arm:
    """单个探索臂的状态。

    Tracks posterior statistics for a document segment using a simplified
    Normal-Normal conjugate model. Observations are individual relevance
    scores assigned by the evaluator.

    Attributes:
        arm_id: Unique string identifier for this arm.
        center: Document position center (character index).
        sigma: Current exploration radius (character distance).
        observations: Collected relevance scores from evaluated samples.
    """

    arm_id: str
    center: int = 0
    sigma: float = 5000.0
    observations: List[float] = field(default_factory=list)

    def posterior_mean(self) -> float:
        """Compute the posterior mean reward of this arm.

        Returns:
            Mean of collected observations, or 0.0 if no observations yet.
        """
        if not self.observations:
            return 0.0
        return sum(self.observations) / len(self.observations)

    def posterior_variance(self) -> float:
        """Compute the posterior variance (uncertainty) of this arm.

        Uses sample variance with a prior variance of 1.0 when observations
        are sparse (< 2). This ensures all arms maintain non-zero uncertainty
        for exploration purposes.

        Returns:
            Estimated posterior variance (always > 0).
        """
        n = len(self.observations)
        if n < 2:
            # Prior variance ensures initial exploration
            return 1.0
        mean = self.posterior_mean()
        var = sum((x - mean) ** 2 for x in self.observations) / (n - 1)
        # Add small prior to prevent zero variance
        return var + (1.0 / (n + 1))

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Compute a simplified normal-approximation confidence interval.

        Uses z-score approximation (no scipy dependency):
          - 0.90 -> z=1.645
          - 0.95 -> z=1.96
          - 0.99 -> z=2.576

        Args:
            level: Confidence level in (0, 1). Default 0.95.

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        # Z-score lookup for common confidence levels
        z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_map.get(level)
        if z is None:
            # Fallback: approximate inverse normal via rational approximation
            # For level in (0.5, 1.0), use Beasley-Springer-Moro approximation
            p = (1.0 + level) / 2.0
            t = math.sqrt(-2.0 * math.log(1.0 - p))
            z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
                1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
            )

        mean = self.posterior_mean()
        std_err = math.sqrt(self.posterior_variance())
        n = max(len(self.observations), 1)
        margin = z * std_err / math.sqrt(n)
        return (mean - margin, mean + margin)

    def add_observation(self, score: float) -> None:
        """Add a new evaluation score to this arm's observation history.

        Args:
            score: Relevance score from the evaluator.
        """
        self.observations.append(score)


# --------------------------------------------------------------------------
# MultiArmNavigator
# --------------------------------------------------------------------------


class MultiArmNavigator:
    """IDS 驱动的多臂证据导航器。

    核心思想：将文档空间分为 K 个独立探索臂，
    通过 Information-Directed Sampling 决策准则分配采样预算，
    支持多源证据的并行发现。

    Implements the ``SamplingStrategy`` protocol defined in ``lens_protocols.py``.

    Attributes:
        config: LensConfig instance with tunable parameters.
        arms: Active bandit arms tracking document segments.
        probe_window: Character size of each sampling window.
    """

    def __init__(
        self,
        config: LensConfig,
        doc_content: str,
        doc_len: int,
        log_callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Initialize the MultiArmNavigator.

        Args:
            config: LensConfig providing k_arms, min_arm_allocation, etc.
            doc_content: Full document text.
            doc_len: Length of document in characters.
            log_callback: Optional logging callback (reserved for future use).
        """
        self.config = config
        self.doc_content = doc_content
        self.doc_len = doc_len
        self.probe_window = config.probe_window
        self.arms: List[Arm] = []
        self._visited_starts: set = set()
        self._observed_sample_ids: set = set()  # Track processed samples to avoid double-counting
        self._log_callback = log_callback

    # ------------------------------------------------------------------
    # SamplingStrategy Protocol Implementation
    # ------------------------------------------------------------------

    async def next_round(
        self,
        round_num: int,
        query: str,
        keywords: Optional[Dict[str, float]] = None,
        prev_candidates: Optional[List[SampleWindow]] = None,
        doc_content: str = "",
        doc_len: int = 0,
    ) -> List[SampleWindow]:
        """SamplingStrategy 协议实现。

        Routes to initialization (round 1) or allocation (round 2+).

        Args:
            round_num: Current round number (1-indexed).
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping.
            prev_candidates: Evaluated candidates from previous rounds.
            doc_content: Full document content.
            doc_len: Length of the document in characters.

        Returns:
            A list of new candidate sample windows to evaluate.
        """
        # Allow dynamic doc updates
        if doc_content:
            self.doc_content = doc_content
        if doc_len > 0:
            self.doc_len = doc_len

        # Update arm posteriors from previous round results
        if prev_candidates:
            self.update_arms(prev_candidates)

        if round_num == 1:
            return await self._initialize_and_explore(query, keywords, doc_content, doc_len)
        else:
            return await self._allocate_and_explore(query, round_num, doc_content, doc_len)

    # ------------------------------------------------------------------
    # Round 1: Initialization
    # ------------------------------------------------------------------

    def initialize_with_anchors(
        self,
        doc_length: int,
        anchors: List[Tuple[int, int, float]],
        n_global_arms: int = 2,
    ) -> List[SampleWindow]:
        """Initialize arms around structure/snippet anchors.

        Args:
            doc_length: Full document length in characters.
            anchors: ``(center, range_width, confidence)`` tuples.  Higher-
                confidence anchors receive priority when the arm budget is
                smaller than the anchor set.
            n_global_arms: Arms reserved for uniform global exploration.

        Returns:
            One initial sample per created arm.
        """
        effective_len = max(0, doc_length or self.doc_len)
        if effective_len <= 0:
            return []
        k = max(1, self.config.k_arms)
        global_count = min(max(0, n_global_arms), max(0, k - 1))
        anchor_budget = max(1, k - global_count)

        normalized: List[Tuple[int, int, float]] = []
        seen_centers: set = set()
        for center, width, confidence in sorted(
            anchors or [], key=lambda item: item[2], reverse=True,
        ):
            bounded_center = max(0, min(int(center), effective_len - 1))
            if bounded_center in seen_centers:
                continue
            seen_centers.add(bounded_center)
            normalized.append((
                bounded_center,
                max(self.probe_window, int(width)),
                max(0.0, min(1.0, float(confidence))),
            ))
            if len(normalized) >= anchor_budget:
                break

        if not normalized:
            return []

        self.arms = []
        for index, (center, width, _confidence) in enumerate(normalized):
            self.arms.append(Arm(
                arm_id=f"anchor_{index}",
                center=center,
                sigma=max(float(self.probe_window), width / 3.0),
            ))

        remaining = k - len(self.arms)
        if remaining > 0:
            # Generate more uniform candidates than needed, then greedily pick
            # those farthest from anchored arms.  This avoids accidentally
            # dropping every global arm when anchors happen to sit at segment
            # midpoints.
            candidate_count = max(k * 2, remaining)
            candidate_centers = [
                min(
                    effective_len - 1,
                    max(0, int((index + 0.5) * effective_len / candidate_count)),
                )
                for index in range(candidate_count)
            ]
            for index in range(remaining):
                if not candidate_centers:
                    break
                center = max(
                    candidate_centers,
                    key=lambda candidate: min(
                        abs(arm.center - candidate) for arm in self.arms
                    ),
                )
                candidate_centers.remove(center)
                self.arms.append(Arm(
                    arm_id=f"global_{index}",
                    center=center,
                    sigma=max(
                        float(self.probe_window),
                        effective_len / max(candidate_count * 2, 1),
                    ),
                ))

        samples: List[SampleWindow] = []
        for arm in self.arms:
            samples.extend(
                self._sample_around_arm(arm, 1, self.doc_content, effective_len)
            )
        logger.debug(
            f"[MultiArmNavigator] Anchor initialization: "
            f"{len(normalized)} anchors, {len(self.arms)} arms"
        )
        return samples

    async def _initialize_and_explore(
        self,
        query: str,
        keywords: Optional[Dict[str, float]],
        doc_content: str,
        doc_len: int,
    ) -> List[SampleWindow]:
        """Round 1: 初始化 K 个臂并执行初始探索。

        策略：
        1. 从 Fuzz anchors 取前 K-1 个位置作为 arm centers
        2. 第 K 个臂放在文档的随机位置（保证全局覆盖）
        3. 每臂各采样 1 个候选

        Args:
            query: The user query string.
            keywords: Optional keyword-to-weight mapping.
            doc_content: Full document content.
            doc_len: Document length.

        Returns:
            Initial candidate sample windows (one per arm).
        """
        k = self.config.k_arms
        effective_doc_len = self.doc_len if self.doc_len > 0 else len(self.doc_content)

        # Determine arm centers: evenly spaced with keyword-biased offsets
        keyword_list = list((keywords or {}).keys())
        arm_centers = self._compute_initial_centers(k, effective_doc_len, keyword_list)

        # Create arms
        initial_sigma = effective_doc_len / (k * 2)
        self.arms = []
        for i, center in enumerate(arm_centers):
            arm = Arm(
                arm_id=f"arm_{i}",
                center=center,
                sigma=initial_sigma,
            )
            self.arms.append(arm)

        logger.debug(
            f"[MultiArmNavigator] Initialized {k} arms, "
            f"centers={[a.center for a in self.arms]}, sigma={initial_sigma:.0f}"
        )

        # Sample one candidate per arm
        samples: List[SampleWindow] = []
        for arm in self.arms:
            arm_samples = self._sample_around_arm(arm, 1, self.doc_content, effective_doc_len)
            samples.extend(arm_samples)

        return samples

    def _compute_initial_centers(
        self, k: int, doc_len: int, keywords: List[str]
    ) -> List[int]:
        """Compute initial arm center positions.

        First K-1 arms are evenly spaced across the document.
        The last arm is placed at a random position for global coverage.

        Args:
            k: Number of arms.
            doc_len: Document length.
            keywords: Keywords for potential position biasing (reserved).

        Returns:
            List of center positions.
        """
        if k <= 0:
            return []

        centers: List[int] = []
        if k == 1:
            centers.append(doc_len // 2)
        else:
            # K-1 evenly spaced
            segment_size = doc_len // (k - 1)
            for i in range(k - 1):
                center = segment_size * i + segment_size // 2
                centers.append(min(center, doc_len - 1))
            # Last arm: random position for exploration
            rand_center = random.randint(0, max(0, doc_len - 1))
            centers.append(rand_center)

        return centers

    # ------------------------------------------------------------------
    # Round 2+: IDS Allocation
    # ------------------------------------------------------------------

    async def _allocate_and_explore(
        self,
        query: str,
        round_num: int,
        doc_content: str,
        doc_len: int,
    ) -> List[SampleWindow]:
        """Round 2+: 使用 IDS 分配预算并探索。

        策略：
        1. 计算每臂的 IDS 分数
        2. 按 softmax(variance) 分配预算（min_allocation=1）
        3. 每臂独立执行高斯采样
        4. 检查臂合并条件

        Args:
            query: The user query string.
            round_num: Current round number.
            doc_content: Full document content.
            doc_len: Document length.

        Returns:
            Candidate sample windows distributed across arms.
        """
        effective_doc_len = self.doc_len if self.doc_len > 0 else len(self.doc_content)

        # Shrink sigma over rounds (exploration cooling)
        decay = 1.0 / (2 ** (round_num - 1))
        for arm in self.arms:
            arm.sigma = max(arm.sigma * decay, self.probe_window)

        # IDS budget allocation
        total_budget = self.config.samples_per_round
        allocation = self._ids_allocation(total_budget)

        logger.debug(
            f"[MultiArmNavigator] Round {round_num}: "
            f"allocation={allocation}, "
            f"variances=[{', '.join(f'{a.posterior_variance():.3f}' for a in self.arms)}]"
        )

        # Sample per arm
        samples: List[SampleWindow] = []
        for arm, num_samples in zip(self.arms, allocation):
            if num_samples > 0:
                arm_samples = self._sample_around_arm(
                    arm, num_samples, self.doc_content, effective_doc_len
                )
                samples.extend(arm_samples)

        # Check for arm merging
        self._check_arm_merge()

        return samples

    def _ids_allocation(self, budget: int) -> List[int]:
        """IDS 预算分配。

        allocation_i ∝ softmax(posterior_variance_i)
        确保 min_allocation=config.min_arm_allocation per arm.

        The intuition: arms with higher posterior uncertainty should be
        explored more (information-directed), as they have the greatest
        potential to reduce overall uncertainty.

        Args:
            budget: Total number of samples to allocate.

        Returns:
            List of sample counts per arm.
        """
        if not self.arms:
            return []

        min_alloc = self.config.min_arm_allocation
        num_arms = len(self.arms)

        # Ensure budget can cover minimum allocations
        if budget <= num_arms * min_alloc:
            return [min(min_alloc, budget // num_arms) for _ in self.arms]

        # Compute softmax over posterior variances
        variances = [arm.posterior_variance() for arm in self.arms]
        max_var = max(variances) if variances else 1.0
        # Temperature-scaled log variances for numerical stability
        scaled = [v / max(max_var, 1e-8) for v in variances]
        exp_vals = [math.exp(s) for s in scaled]
        total_exp = sum(exp_vals) or 1.0
        proportions = [e / total_exp for e in exp_vals]

        # Distribute budget - minimum guaranteed first
        remaining = budget - num_arms * min_alloc
        allocation = [min_alloc] * num_arms

        # Distribute remaining proportionally
        for i, prop in enumerate(proportions):
            extra = int(remaining * prop)
            allocation[i] += extra

        # Distribute any leftover from rounding
        distributed = sum(allocation)
        leftover = budget - distributed
        if leftover > 0:
            # Give to highest variance arm
            max_idx = variances.index(max(variances))
            allocation[max_idx] += leftover

        return allocation

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_around_arm(
        self, arm: Arm, num_samples: int, doc_content: str, doc_len: int
    ) -> List[SampleWindow]:
        """围绕特定臂中心的高斯采样。

        Generates sample windows by drawing positions from a Gaussian
        distribution centered at the arm's center with the arm's current sigma.

        Args:
            arm: The arm to sample around.
            num_samples: Number of samples to generate.
            doc_content: Full document content.
            doc_len: Document length in characters.

        Returns:
            List of SampleWindow instances sampled around the arm.
        """
        samples: List[SampleWindow] = []
        max_attempts = num_samples * 5  # Prevent infinite loops
        attempts = 0

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Draw from Gaussian centered on arm
            new_center = int(random.gauss(arm.center, arm.sigma))
            raw_start = new_center - (self.probe_window // 2)

            # Boundary clamp
            start = max(0, min(raw_start, doc_len - self.probe_window))
            end = min(start + self.probe_window, doc_len)

            # Deduplication check
            too_close = False
            for existing in self._visited_starts:
                if abs(existing - start) < (self.probe_window // 3):
                    too_close = True
                    break

            if not too_close:
                self._visited_starts.add(start)
                content = doc_content[start:end]
                samples.append(
                    SampleWindow(
                        start_idx=start,
                        end_idx=end,
                        content=content,
                        round_num=0,  # Will be set by caller context
                        source="multi_arm",
                    )
                )

        return samples

    # ------------------------------------------------------------------
    # Arm Update and Merge
    # ------------------------------------------------------------------

    def update_arms(self, evaluated_samples: List[SampleWindow]) -> None:
        """基于评估结果更新臂后验。

        对每个已评估样本，找到最近的臂并添加观测。
        使用 sample identity (start_idx) 去重，防止同一样本在多轮中被重复计数。

        Args:
            evaluated_samples: Samples with scores from the evaluator.
        """
        if not self.arms or not evaluated_samples:
            return

        for sample in evaluated_samples:
            if sample.score <= 0:
                continue
            # Deduplication: skip samples already processed
            sample_id = (sample.start_idx, sample.end_idx)
            if sample_id in self._observed_sample_ids:
                continue
            self._observed_sample_ids.add(sample_id)

            # Find nearest arm by center distance
            sample_center = (sample.start_idx + sample.end_idx) // 2
            nearest_arm = min(
                self.arms, key=lambda a: abs(a.center - sample_center)
            )
            nearest_arm.add_observation(sample.score)

    def _check_arm_merge(self) -> None:
        """检查是否有臂应该合并。

        条件：两臂中心距离 < 2*sigma 且 95% CI 重叠 > config.arm_merge_ci_overlap.

        When merging, the arm with fewer observations is absorbed into the arm
        with more observations (keeps the stronger posterior).
        """
        if len(self.arms) <= 1:
            return

        merged_indices: set = set()
        new_arms: List[Arm] = []

        for i in range(len(self.arms)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(self.arms)):
                if j in merged_indices:
                    continue

                arm_i = self.arms[i]
                arm_j = self.arms[j]

                # Check center distance condition
                distance = abs(arm_i.center - arm_j.center)
                avg_sigma = (arm_i.sigma + arm_j.sigma) / 2
                if distance >= 2 * avg_sigma:
                    continue

                # Check CI overlap
                ci_i = arm_i.confidence_interval(0.95)
                ci_j = arm_j.confidence_interval(0.95)
                overlap = self._ci_overlap_ratio(ci_i, ci_j)

                if overlap >= self.config.arm_merge_ci_overlap:
                    # Merge: absorb the weaker arm into the stronger
                    if len(arm_i.observations) >= len(arm_j.observations):
                        # Absorb j into i
                        arm_i.observations.extend(arm_j.observations)
                        arm_i.center = (arm_i.center + arm_j.center) // 2
                        arm_i.sigma = max(arm_i.sigma, arm_j.sigma)
                        merged_indices.add(j)
                    else:
                        # Absorb i into j
                        arm_j.observations.extend(arm_i.observations)
                        arm_j.center = (arm_i.center + arm_j.center) // 2
                        arm_j.sigma = max(arm_i.sigma, arm_j.sigma)
                        merged_indices.add(i)
                        break  # i is merged, move on

            if i not in merged_indices:
                new_arms.append(self.arms[i])

        # Collect remaining unmerged arms
        for j in range(len(self.arms)):
            if j not in merged_indices and self.arms[j] not in new_arms:
                new_arms.append(self.arms[j])

        if len(new_arms) < len(self.arms):
            logger.debug(
                f"[MultiArmNavigator] Arms merged: {len(self.arms)} -> {len(new_arms)}"
            )
        self.arms = new_arms

    @staticmethod
    def _ci_overlap_ratio(
        ci_a: Tuple[float, float], ci_b: Tuple[float, float]
    ) -> float:
        """Compute the overlap ratio of two confidence intervals.

        overlap_ratio = overlap_length / min(width_a, width_b)

        Args:
            ci_a: (lower, upper) of interval A.
            ci_b: (lower, upper) of interval B.

        Returns:
            Overlap ratio in [0, 1]. Returns 0 if no overlap.
        """
        overlap_start = max(ci_a[0], ci_b[0])
        overlap_end = min(ci_a[1], ci_b[1])
        overlap_len = max(0.0, overlap_end - overlap_start)

        width_a = ci_a[1] - ci_a[0]
        width_b = ci_b[1] - ci_b[0]
        min_width = min(width_a, width_b)

        if min_width <= 0:
            return 0.0
        return overlap_len / min_width

    # ------------------------------------------------------------------
    # Entropy (used by AdaptiveProposalMixer)
    # ------------------------------------------------------------------

    def compute_posterior_entropy(self) -> float:
        """计算所有臂的总后验熵，用于 AdaptiveProposalMixer。

        Uses the differential entropy of a Gaussian:
          H = 0.5 * ln(2πe * variance)

        Sums across all arms to measure total system uncertainty.

        Returns:
            Total posterior entropy across all arms (non-negative).
        """
        total_entropy = 0.0
        for arm in self.arms:
            var = arm.posterior_variance()
            # Differential entropy of Gaussian: 0.5 * ln(2πe * σ²)
            if var > 0:
                total_entropy += 0.5 * math.log(2 * math.pi * math.e * var)
        return max(0.0, total_entropy)


# --------------------------------------------------------------------------
# LegacySampler
# --------------------------------------------------------------------------


class LegacySampler:
    """包装现有的 Fuzz + Gaussian 采样逻辑。

    当 enable_multi_arm=False 时使用此采样器，
    保持与原 MonteCarloEvidenceSampling 完全一致的行为。

    Implements the ``SamplingStrategy`` protocol by delegating to the original
    MonteCarloEvidenceSampling instance's ``_get_fuzzy_anchors`` and
    ``_sample_gaussian`` methods.

    Attributes:
        sampler_ref: Reference to the MonteCarloEvidenceSampling instance.
        config: LensConfig for parameter access.
    """

    def __init__(self, sampler_ref: Any, config: LensConfig) -> None:
        """Initialize the LegacySampler.

        Args:
            sampler_ref: A MonteCarloEvidenceSampling instance whose sampling
                methods will be delegated to.
            config: LensConfig for reading parameters.
        """
        self.sampler_ref = sampler_ref
        self.config = config

    async def next_round(
        self,
        round_num: int,
        query: str,
        keywords: Optional[Dict[str, float]] = None,
        prev_candidates: Optional[List[SampleWindow]] = None,
        doc_content: str = "",
        doc_len: int = 0,
    ) -> List[SampleWindow]:
        """委托给原始 sampler 的 _get_fuzzy_anchors / _sample_gaussian。

        Round 1: Uses fuzzy anchor matching + stratified random exploration.
        Round 2+: Uses Gaussian importance sampling around top-K seeds.

        Args:
            round_num: Current round number (1-indexed).
            query: The user query string.
            keywords: Optional keyword-to-IDF-weight mapping.
            prev_candidates: Evaluated candidates from previous rounds.
            doc_content: Full document content (unused, sampler_ref holds doc).
            doc_len: Document length (unused, sampler_ref holds doc_len).

        Returns:
            A list of new candidate sample windows to evaluate.
        """
        sampler = self.sampler_ref

        if round_num == 1:
            # Delegate to fuzzy anchors + stratified exploration
            keyword_list = list((keywords or {}).keys())
            anchors = await sampler._get_fuzzy_anchors(query, keyword_list)
            supplements = sampler._sample_stratified_supplement(
                sampler.random_exploration_num
            )
            return anchors + supplements
        else:
            # Gaussian sampling around top-K seeds from previous round
            if not prev_candidates:
                return []
            top_k = self.config.top_k_seeds
            seeds = prev_candidates[:top_k]
            return sampler._sample_gaussian(seeds, round_num)
