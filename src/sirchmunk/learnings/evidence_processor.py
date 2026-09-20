# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, process

from sirchmunk.learnings.lens_config import LensConfig
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import EVALUATE_EVIDENCE_SAMPLE, ROI_RESULT_SUMMARY
from sirchmunk.utils import create_logger, LogCallback

# Small file threshold: skip Monte Carlo sampling and return full content as ROI
_SMALL_FILE_THRESHOLD = 100_000  # 100K chars


@dataclass
class SampleWindow:
    """
    Sampling window configuration and metadata.
    """

    start_idx: int

    end_idx: int

    content: str

    # Relevance score from LLM
    score: float = 0.0

    # Literal match score from RapidFuzz
    fuzz_score: float = 0.0

    reasoning: str = ""

    round_num: int = 0
    # 'fuzz', 'stratified', 'gaussian'

    source: str = "unknown"


@dataclass
class RoiResult:
    """
    Data class to store the final Region of Interest (ROI) result and metadata.
    """

    summary: str

    is_found: bool

    # Segments within the document (e.g., paragraph, code snippet)
    # Format: {"snippet": "xxx", "start": 7, "end": 65, "score": 9.0, "reasoning": "xxx"}
    snippets: List[Dict[str, Any]]

    def to_dict(self):
        """
        Convert RoiResult to a dictionary.
        """
        return {
            "summary": self.summary,
            "is_found": self.is_found,
            "snippets": self.snippets,
        }


class MonteCarloEvidenceSampling:
    """
    Monte Carlo Evidence Importance Sampling for Document Retrieval.

    Also available as ``LensEvidenceSampler`` (preferred alias for new code).
    """

    def __init__(
        self,
        llm: OpenAIChat,
        doc_content: str,
        verbose: bool = True,
        log_callback: LogCallback = None,
        lens_config: Optional[LensConfig] = None,
        initial_anchors: Optional[List[Tuple[int, int, float]]] = None,
    ):
        self.llm = llm
        self.doc = doc_content
        self.doc_len = len(doc_content)
        self.verbose = verbose

        self.max_rounds = 3
        # Size of each probe sampling window
        self.probe_window = 500
        # Size of the final expanded context
        self.roi_window = 2000

        # ---Sampling Configuration--- #
        # Number of anchors from Fuzz
        self.fuzz_candidates_num = 5
        # Number of random points for exploration
        self.random_exploration_num = 2
        # Samples per round for Gaussian sampling
        self.samples_per_round = 5
        # Top K samples to keep as seeds for next round
        self.top_k_seeds = 2

        self.visited_starts: Set[int] = set()
        
        # Create bound logger with callback - returns AsyncLogger instance
        self._log = create_logger(log_callback=log_callback)

        self.llm_usages: List[Dict[str, Any]] = []

        # --- LENS evaluator (strategy pattern) ---
        self.lens_config = lens_config or LensConfig.from_env()
        self.initial_anchors = list(initial_anchors or [])

        # Deferred imports to avoid circular dependency
        # (lens_protocols and batch_ranking_evaluator import SampleWindow from this module)
        from sirchmunk.learnings.batch_ranking_evaluator import (
            BatchRankingEvaluator,
            PointwiseEvaluator,
        )

        if self.lens_config.enable_batch_ranking:
            self._evaluator = BatchRankingEvaluator(
                llm=self.llm,
                config=self.lens_config,
                log_callback=log_callback,
            )
        else:
            self._evaluator = PointwiseEvaluator(
                llm=self.llm,
                log_callback=log_callback,
            )

        # --- Phase 2: Sampling Strategy ---
        # Deferred imports to prevent circular reference
        # (multi_arm_navigator imports SampleWindow from this module)
        from sirchmunk.learnings.multi_arm_navigator import (
            MultiArmNavigator,
            LegacySampler,
        )

        if self.lens_config.enable_multi_arm:
            self._sampling_strategy = MultiArmNavigator(
                config=self.lens_config,
                doc_content=doc_content,
                doc_len=self.doc_len,
                log_callback=log_callback,
            )
        else:
            self._sampling_strategy = LegacySampler(
                sampler_ref=self, config=self.lens_config
            )

        # --- Phase 2: Adaptive Proposal Mixer (only effective with multi_arm) ---
        from sirchmunk.learnings.adaptive_proposal_mixer import AdaptiveProposalMixer

        self._proposal_mixer: Any = (
            AdaptiveProposalMixer(config=self.lens_config)
            if self.lens_config.enable_adaptive_mix
            else None
        )

        # --- Phase 2: Reasoning Chain Exploiter ---
        from sirchmunk.learnings.reasoning_chain_exploiter import (
            ReasoningChainExploiter,
        )

        self._reasoning_exploiter: Any = (
            ReasoningChainExploiter(
                config=self.lens_config, log_callback=log_callback
            )
            if self.lens_config.enable_reasoning_exploit
            else None
        )

        # --- Phase 3: Stop Strategy ---
        from sirchmunk.learnings.statistical_stop_decider import (
            StatisticalStopDecider,
            ThresholdStop,
        )

        if self.lens_config.enable_statistical_stop:
            self._stop_strategy = StatisticalStopDecider(
                config=self.lens_config, log_callback=log_callback
            )
        else:
            self._stop_strategy = ThresholdStop()

        # Multi-source intent (can be set externally before calling get_roi)
        self._multi_source_intent: Optional[float] = None

    def _get_content(self, start: int) -> Tuple[int, int, str]:
        """
        Safely retrieves a document slice with boundary checks.
        """
        start = max(0, min(start, self.doc_len - self.probe_window))
        end = min(start + self.probe_window, self.doc_len)
        return start, end, self.doc[start:end]

    async def _get_fuzzy_anchors(
        self, query: str, keywords: List[str] = None, threshold: float = 10.0
    ) -> List[SampleWindow]:
        """
        Uses RapidFuzz to find heuristic anchors based on literal matching.
        Logic: Sliding window slices -> Calculate similarity with Query -> Top K.

        Args:
            query (str): The user query string.
            threshold (float): Minimum similarity score to consider between 0-100.

        Returns:
            List[SampleWindow]: List of sampled windows based on fuzzy matching.
        """
        if self.verbose:
            await self._log.info("Executing RapidFuzz heuristic pre-filtering...")

        keywords = keywords or []

        # 1. Build sliding window slices (stride = half window size)
        stride = self.probe_window // 2
        chunks = []
        for i in range(0, self.doc_len, stride):
            chunks.append(i)

        # 2. Construct text list for matching
        chunk_texts = [self.doc[i : i + self.probe_window] for i in chunks]

        # 3. Extract most similar fragments
        # TODO: try to add `fuzz.token_set_ratio` for multi-channel retrieval
        results = process.extract(
            query=f"{query} {' '.join(keywords)}".strip(),
            choices=list(chunk_texts),
            scorer=fuzz.token_set_ratio,
            limit=int(self.fuzz_candidates_num * 2),
            score_cutoff=None,
        )

        anchors = []
        for text, score, index in results:
            start_idx = chunks[index]

            # Simple deduplication
            if start_idx in self.visited_starts:
                continue

            # Threshold filtering (e.g., > 30)
            if score < threshold:
                continue

            self.visited_starts.add(start_idx)
            _, end, content = self._get_content(start_idx)

            anchors.append(
                SampleWindow(
                    start_idx=start_idx,
                    end_idx=end,
                    content=content,
                    fuzz_score=score,
                    round_num=1,
                    source="fuzz",
                )
            )

            if len(anchors) >= self.fuzz_candidates_num:
                break

        top_score = anchors[0].fuzz_score if anchors else 0.0
        if self.verbose:
            await self._log.info(
                f"   Anchors hit: {len(anchors)} (Top Fuzz Score: {top_score:.1f})"
            )

        return anchors

    def _sample_stratified_supplement(self, count: int) -> List[SampleWindow]:
        """
        Adds a small amount of global random sampling for 'Exploration',
        preventing cases where Query is semantically relevant but lacks keyword matches.

        Args:
            count (int): Number of random samples to generate.

        Returns:
            List[SampleWindow]: List of randomly sampled windows.
        """
        samples = []
        if count <= 0:
            return samples

        step = self.doc_len // count
        for i in range(count):
            section_start = i * step
            section_end = min((i + 1) * step, self.doc_len)

            # Random selection within section
            max_start = max(section_start, section_end - self.probe_window)
            rand_start = random.randint(section_start, max_start)

            start, end, content = self._get_content(rand_start)

            # Check for overlap with existing points
            is_duplicate = False
            for v in self.visited_starts:
                if abs(v - start) < (self.probe_window // 2):
                    is_duplicate = True
                    break

            if not is_duplicate:
                self.visited_starts.add(start)
                samples.append(
                    SampleWindow(
                        start_idx=start,
                        end_idx=end,
                        content=content,
                        round_num=1,
                        source="stratified",
                    )
                )

        return samples

    def _sample_gaussian(
        self, seeds: List[SampleWindow], current_round: int
    ) -> List[SampleWindow]:
        """
        [Subsequent Rounds] Gaussian Importance Sampling.

        Args:
            seeds (List[SampleWindow]): High-value seeds from previous round.
            current_round (int): Current round number.

        Returns:
            List[SampleWindow]: List of newly sampled windows.
        """
        samples = []
        # Sigma Decay: Shrink search range as rounds progress
        base_sigma = self.doc_len / 20
        sigma = base_sigma / (2 ** (current_round - 1))

        samples_needed = self.samples_per_round

        for seed in seeds:
            if samples_needed <= 0:
                break

            # Allocate children per seed
            num_children = max(1, math.ceil(samples_needed / len(seeds)))
            center = (seed.start_idx + seed.end_idx) // 2

            for _ in range(num_children):
                new_center = int(random.gauss(center, sigma))
                raw_start = new_center - (self.probe_window // 2)
                start, end, content = self._get_content(raw_start)

                # Deduplication check
                too_close = False
                for existing in self.visited_starts:
                    if abs(existing - start) < (self.probe_window // 3):
                        too_close = True
                        break

                if not too_close:
                    self.visited_starts.add(start)
                    samples.append(
                        SampleWindow(
                            start_idx=start,
                            end_idx=end,
                            content=content,
                            round_num=current_round,
                            source="gaussian",
                        )
                    )
                    samples_needed -= 1

        return samples

    async def _evaluate_sample_async(
        self, sample: SampleWindow, query: str
    ) -> SampleWindow:
        """
        Evaluates a single sample asynchronously.
        """
        prompt = EVALUATE_EVIDENCE_SAMPLE.format(
            query=query,
            sample_source=sample.source,
            sample_content=sample.content,
        )
        try:
            resp_obj = await self.llm.achat([{"role": "user", "content": prompt}])
            resp: str = resp_obj.content
            self.llm_usages.append(resp_obj.usage)

            data = self._parse_evaluation_json(resp)
            if data is not None:
                sample.score = float(data.get("score", 0))
                sample.reasoning = data.get("reasoning", "")
            else:
                await self._log.warning(
                    f"Unparseable LLM response for sample at {sample.start_idx}, "
                    f"response (first 200 chars): {resp[:200]!r}"
                )
                sample.score = 0.0
        except Exception as e:
            await self._log.warning(f"Error evaluating sample at {sample.start_idx}: {e}")
            sample.score = 0.0

        return sample

    @staticmethod
    def _parse_evaluation_json(text: str) -> Optional[dict]:
        """Extract ``{"score": ..., "reasoning": ...}`` from LLM output.

        Tries, in order: direct parse, markdown-fence stripping,
        outermost ``{...}`` extraction, and finally regex score fallback.
        """
        if not text:
            return None
        text = text.strip()

        # 1. Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Extract first {...} block (greedy, allows nested braces from reasoning)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, TypeError):
                pass
            # Try the innermost flat object (no nested braces)
            m2 = re.search(r"\{[^{}]+\}", cleaned)
            if m2:
                try:
                    return json.loads(m2.group())
                except (json.JSONDecodeError, TypeError):
                    pass

        # 4. Last resort: regex extraction of score
        score_m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text)
        if score_m:
            return {"score": float(score_m.group(1)), "reasoning": ""}

        return None

    async def _evaluate_batch(
        self, samples: List[SampleWindow], query: str
    ) -> List[SampleWindow]:
        """
        Evaluates a batch of samples concurrently.
        """
        if self.verbose:
            await self._log.info(f"   Evaluating {len(samples)} samples with LLM...")

        # Create async tasks
        tasks = [self._evaluate_sample_async(s, query) for s in samples]

        # Run concurrently
        evaluated_samples = await asyncio.gather(*tasks)
        return list(evaluated_samples)

    async def _generate_summary(
        self, top_samples: List[SampleWindow], query: str
    ) -> str:
        """
        Expands the context windows for multiple top samples and generates a summary.
        """
        combined_context = ""
        half_window = self.roi_window // 2

        # Sort by index to maintain document flow if needed, or by score
        processed_samples = sorted(top_samples, key=lambda x: x.start_idx)

        for i, sample in enumerate(processed_samples):
            center = (sample.start_idx + sample.end_idx) // 2
            start = max(0, center - half_window)
            end = min(self.doc_len, center + half_window)
            expanded_content = self.doc[start:end]
            combined_context += (
                f"\n--- Context Fragment {i + 1} ---\n...{expanded_content}...\n"
            )

        prompt = ROI_RESULT_SUMMARY.format(
            user_input=query,
            text_content=combined_context,
        )

        summary_response = await self.llm.achat([{"role": "user", "content": prompt}])
        self.llm_usages.append(summary_response.usage)
        return summary_response.content

    async def get_roi(
        self,
        query: str,
        keywords: Dict[str, float] = None,
        confidence_threshold: float = 8.5,
        top_k: int = 5,
    ) -> RoiResult:
        """
        Get the Region of Interest (ROI) for the given query.

        Args:
            query (str): The user query string.
            keywords (Dict[str, float], optional): Enhanced keywords with IDF scores for fuzzy matching.
            confidence_threshold (float): Confidence score threshold for early stopping.
            top_k (int): Number of top snippets to consider for final summary.

        Returns:
            RoiResult: The final ROI result with metadata.
        """
        # Small file fast path: skip sampling for files < 100K chars
        if self.doc_len < _SMALL_FILE_THRESHOLD:
            await self._log.info(
                f"[MC] Small file fast path: {self.doc_len} chars < {_SMALL_FILE_THRESHOLD} threshold, "
                f"returning full content as ROI"
            )
            snippet = {
                "snippet": self.doc,
                "start": 0,
                "end": self.doc_len,
                "score": 10.0,
                "reasoning": "Small file - full content returned without sampling",
            }
            return RoiResult(
                summary=self.doc,
                is_found=True,
                snippets=[snippet],
            )

        if self.verbose:
            await self._log.info(
                f"=== Starting Hybrid Adaptive Retrieval (Doc Len: {self.doc_len}) ==="
            )
            await self._log.info(f"Query: {query}, optional keywords: {keywords}")

        keywords = keywords or {}

        all_candidates: List[SampleWindow] = []

        for r in range(1, self.max_rounds + 1):
            if self.verbose:
                await self._log.info(f"--- Round {r}/{self.max_rounds} ---")

            # 1. Sampling (strategy dispatch).  On round one, compiled-tree or
            # grep anchors can warm-start the multi-arm navigator while still
            # reserving global arms for coverage outside the hinted regions.
            if (
                r == 1
                and self.initial_anchors
                and hasattr(self._sampling_strategy, "initialize_with_anchors")
            ):
                current_samples = self._sampling_strategy.initialize_with_anchors(
                    self.doc_len,
                    self.initial_anchors,
                )
            else:
                current_samples = await self._sampling_strategy.next_round(
                    round_num=r,
                    query=query,
                    keywords=keywords,
                    prev_candidates=all_candidates if r > 1 else None,
                    doc_content=self.doc,
                    doc_len=self.doc_len,
                )

            if not current_samples:
                if self.verbose:
                    await self._log.info("No new samples generated this round, skipping.")
                continue

            # 2. Evaluate (Phase 1 BatchRanking or pointwise)
            evaluated = await self._evaluator.evaluate(current_samples, query, keywords)
            all_candidates.extend(evaluated)

            for s in evaluated:
                await self._log.info(
                    f"  [Pos {s.start_idx:6d} | Src: {s.source:8s}] Score: {s.score} | {s.reasoning[:30]}..."
                )

            # 3. Reasoning chain exploitation (optional)
            if self._reasoning_exploiter:
                signals = self._reasoning_exploiter.extract_signals(evaluated, query)
                if self._reasoning_exploiter.should_exploit(signals):
                    hints = self._reasoning_exploiter.generate_sampling_hints(
                        signals, query
                    )
                    if hints:
                        # Merge hints into keywords for next round
                        keywords = {
                            **keywords,
                            **{h: 1.0 for h in hints},
                        }
                        if self.verbose:
                            await self._log.info(
                                f"   ReasoningExploiter: injected {len(hints)} hints → {hints}"
                            )

            # 4. Stop decision (Protocol dispatch)
            should_stop, reason = await self._stop_strategy.should_stop(
                all_candidates=all_candidates,
                query=query,
                round_num=r,
                max_rounds=self.max_rounds,
                confidence_threshold=confidence_threshold,
                tokens_remaining=getattr(self, '_tokens_remaining', float('inf')),
                multi_source_intent=self._multi_source_intent or 0.0,
            )
            if should_stop:
                if self.verbose and self._log:
                    await self._log.info(f"  [LENS Stop] {reason}")
                break

        # --- Final Result Processing ---
        if not all_candidates:
            await self._log.warning("Failed to retrieve any content.")
            return RoiResult(
                summary="Could not retrieve relevant content.",
                is_found=False,
                snippets=[],
            )

        # Sort all candidates by score descending for deterministic selection
        all_candidates.sort(key=lambda c: c.score, reverse=True)

        # Collect top candidates that are relevant enough
        relevance_floor = self.lens_config.relevance_floor
        relevant_candidates = [c for c in all_candidates if c.score >= relevance_floor]

        # If nothing meets the threshold, fallback to the single best candidate
        if not relevant_candidates:
            best = all_candidates[0]
            return RoiResult(
                summary="No exact answer found in the document.",
                is_found=False,
                snippets=[
                    {
                        "snippet": best.content,
                        "start": best.start_idx,
                        "end": best.end_idx,
                        "score": best.score,
                        "reasoning": best.reasoning,
                    }
                ],
            )

        # Take top_k highest-scoring candidates for summarization
        final_candidates = relevant_candidates[:top_k]
        best_score = final_candidates[0].score

        if self.verbose:
            await self._log.info(
                f"=== Final Lock: {len(final_candidates)} snippets, Top Score {best_score} ==="
            )

        # Generate summary
        summary = await self._generate_summary(final_candidates, query)

        # Construct new snippet format
        roi_snippets = []
        for c in final_candidates:
            roi_snippets.append(
                {
                    "snippet": c.content,
                    "start": c.start_idx,
                    "end": c.end_idx,
                    "score": c.score,
                    "reasoning": c.reasoning,
                }
            )

        return RoiResult(
            summary=summary,
            is_found=True,
            snippets=roi_snippets,
        )


# Preferred alias for new code — backward-compatible rename.
LensEvidenceSampler = MonteCarloEvidenceSampling
