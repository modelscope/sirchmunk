import asyncio
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

from loguru import logger
from rapidfuzz import fuzz, process

from src.agentic_search.llm.openai import OpenAIChat
from src.agentic_search.llm.prompts import EVALUATE_EVIDENCE_SAMPLE, ROI_RESULT_SUMMARY


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
            logger.info(">> Executing RapidFuzz heuristic pre-filtering...")

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

        return await self.llm.achat([{"role": "user", "content": prompt}])

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
        if self.verbose:
            logger.info(
                f"=== Starting Hybrid Adaptive Retrieval (Doc Len: {self.doc_len}) ==="
            )
            logger.info(f"Query: {query}, optional keywords: {keywords}")

        keywords = keywords or {}

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
                fuzz_samples = self._get_fuzzy_anchors(
                    query=query,
                    keywords=list(keywords.keys()),
                    threshold=10.0,
                )
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
                summary="Could not retrieve relevant content.",
                is_found=False,
                snippets=[],
            )

        # Collect top candidates that are relevant enough
        # Using 4.0 as a soft threshold for relevance inclusion
        relevant_candidates = [c for c in all_candidates if c.score >= 4.0]

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

        # Take up to top_k_seeds (e.g., 2 or 3) as the final set for summarization
        final_candidates = relevant_candidates[:top_k]
        best_score = final_candidates[0].score

        if self.verbose:
            logger.info(
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
