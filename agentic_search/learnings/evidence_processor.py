import asyncio
import json
import math
import random
from dataclasses import dataclass
from typing import List, Set, Tuple

from loguru import logger
from rapidfuzz import fuzz, process

from agentic_search.llm.openai import OpenAIChat
from agentic_search.llm.prompts import EVALUATE_EVIDENCE_SAMPLE


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

    final_answer: str

    is_found: bool

    confidence_score: float

    best_content_snippet: str

    start_idx: int

    end_idx: int

    reasoning: str


class MonteCarloEvidenceSampling:
    """
    Monte Carlo Evidence Importance Sampling for Document Retrieval.
    """

    def __init__(
        self,
        llm: OpenAIChat,
        doc_content: str,
        verbose: bool = True,
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

    def _get_content(self, start: int) -> Tuple[int, int, str]:
        """
        Safely retrieves a document slice with boundary checks.
        """
        start = max(0, min(start, self.doc_len - self.probe_window))
        end = min(start + self.probe_window, self.doc_len)
        return start, end, self.doc[start:end]

    def _get_fuzzy_anchors(
        self, query: str, threshold: float = 30.0
    ) -> List[SampleWindow]:
        """
        Uses RapidFuzz to find heuristic anchors based on literal matching.
        Logic: Sliding window slices -> Calculate similarity with Query -> Top K.

        Args:
            query (str): The user query string.
            threshold (float): Minimum similarity score to consider.

        Returns:
            List[SampleWindow]: List of sampled windows based on fuzzy matching.
        """
        if self.verbose:
            logger.info(">> Executing RapidFuzz heuristic pre-filtering...")

        # 1. Build sliding window slices (stride = half window size)
        stride = self.probe_window // 2
        chunks = []
        for i in range(0, self.doc_len, stride):
            chunks.append(i)

        # 2. Construct text list for matching
        chunk_texts = [self.doc[i : i + self.probe_window] for i in chunks]

        # 3. Extract most similar fragments
        results = process.extract(
            query=query,
            choices=chunk_texts,
            scorer=fuzz.partial_token_set_ratio,
            limit=self.fuzz_candidates_num * 2,
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
            logger.info(
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
            resp: str = await self.llm.achat([{"role": "user", "content": prompt}])

            clean_resp = resp.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_resp)
            sample.score = float(data.get("score", 0))
            sample.reasoning = data.get("reasoning", "")
        except Exception as e:
            logger.warning(f"Error evaluating sample at {sample.start_idx}: {e}")
            sample.score = 0.0

        return sample

    async def _evaluate_batch(
        self, samples: List[SampleWindow], query: str
    ) -> List[SampleWindow]:
        """
        Evaluates a batch of samples concurrently.
        """
        if self.verbose:
            logger.info(f"   Evaluating {len(samples)} samples with LLM...")

        # Create async tasks
        tasks = [self._evaluate_sample_async(s, query) for s in samples]

        # Run concurrently
        evaluated_samples = await asyncio.gather(*tasks)
        return list(evaluated_samples)

    async def _expand_and_verify(self, best_sample: SampleWindow, query: str) -> str:
        """
        Expands the context window and generates the final answer.
        """
        center = (best_sample.start_idx + best_sample.end_idx) // 2
        half_window = self.roi_window // 2
        start = max(0, center - half_window)
        end = min(self.doc_len, center + half_window)
        expanded_content = self.doc[start:end]

        prompt = f"""
        Answer the question based on the context. If you don't know, say you don't know.
        Question: "{query}"
        Context:
        "...{expanded_content}..."
        """
        # Async call
        return await self.llm.achat([{"role": "user", "content": prompt}])

    async def get_roi(self, query: str, confidence_threshold: float = 8.5) -> RoiResult:
        """
        Get the Region of Interest (ROI) for the given query.

        Args:
            query (str): The user query string.
            confidence_threshold (float): Confidence score threshold for early stopping.

        Returns:
            RoiResult: The final ROI result with metadata.
        """
        if self.verbose:
            logger.info(
                f"=== Starting Hybrid Adaptive Retrieval (Doc Len: {self.doc_len}) ==="
            )
            logger.info(f"Query: {query}")

        all_candidates: List[SampleWindow] = []
        top_seeds: List[SampleWindow] = []

        for r in range(1, self.max_rounds + 1):
            if self.verbose:
                logger.info(f"--- Round {r}/{self.max_rounds} ---")
            current_samples = []

            if r == 1:
                # === Strategy: Fuzz Anchors + Random Supplement ===
                # 1. Get Fuzz Anchors (Exploitation)
                # Note: Fuzz is CPU bound, so we keep it sync
                fuzz_samples = self._get_fuzzy_anchors(query)
                current_samples.extend(fuzz_samples)

                # 2. Supplement with Random Sampling (Exploration)
                needed_random = self.random_exploration_num
                if len(fuzz_samples) == 0:
                    needed_random += 3  # Downgrade to random mode

                random_samples = self._sample_stratified_supplement(needed_random)
                current_samples.extend(random_samples)

                if self.verbose:
                    logger.info(
                        f"Sampling Distribution: Fuzz Anchors={len(fuzz_samples)}, Random Exploration={len(random_samples)}"
                    )

            else:
                # === Subsequent Rounds: Gaussian Focusing ===
                # Filter low score seeds
                valid_seeds = [s for s in top_seeds if s.score >= 4.0]

                if not valid_seeds:
                    logger.warning(
                        "No high-value regions found, attempting global random sampling again..."
                    )
                    current_samples = self._sample_stratified_supplement(
                        self.samples_per_round
                    )
                else:
                    max_score = valid_seeds[0].score
                    if self.verbose:
                        logger.info(
                            f"Focusing: Based on {len(valid_seeds)} seeds (Max Score: {max_score})"
                        )
                    current_samples = self._sample_gaussian(valid_seeds, r)

            if not current_samples and self.verbose:
                logger.info("No new samples generated this round, skipping.")
            else:
                evaluated = await self._evaluate_batch(current_samples, query)
                all_candidates.extend(evaluated)

                for s in evaluated:
                    logger.info(
                        f"  [Pos {s.start_idx:6d} | Src: {s.source:8s}] Score: {s.score} | {s.reasoning[:30]}..."
                    )

            # Sort and update seeds
            all_candidates.sort(key=lambda x: x.score, reverse=True)
            top_seeds = all_candidates[: self.top_k_seeds]

            # Early stopping check
            if top_seeds and top_seeds[0].score >= confidence_threshold:
                if self.verbose:
                    logger.info(
                        f">> High confidence target found (Score >= {confidence_threshold}), stopping early."
                    )
                break

        # --- Final Result Processing ---
        if not all_candidates:
            logger.warning("Failed to retrieve any content.")
            return RoiResult(
                final_answer="Could not retrieve relevant content.",
                is_found=False,
                confidence_score=0.0,
                best_content_snippet="",
                start_idx=-1,
                end_idx=-1,
                reasoning="No candidates generated.",
            )

        best = all_candidates[0]
        if self.verbose:
            logger.info(f"=== Final Lock: Pos {best.start_idx}, Score {best.score} ===")

        if best.score < 4.0:
            return RoiResult(
                final_answer="No exact answer found in the document.",
                is_found=False,
                confidence_score=best.score,
                best_content_snippet=best.content,
                start_idx=best.start_idx,
                end_idx=best.end_idx,
                reasoning=best.reasoning,
            )

        # Generate final answer
        final_ans = await self._expand_and_verify(best, query)

        return RoiResult(
            final_answer=final_ans,
            is_found=True,
            confidence_score=best.score,
            best_content_snippet=best.content,
            start_idx=best.start_idx,
            end_idx=best.end_idx,
            reasoning=best.reasoning,
        )
