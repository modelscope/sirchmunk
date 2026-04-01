# Copyright (c) ModelScope Contributors. All rights reserved.
"""Evidence extraction pipeline orchestrator.

``EvidenceSampler`` coordinates the layered evidence extraction pipeline:

1. **DocumentChunker** — structure-aware segmentation
2. **SignalScorer** — RapidFuzz + BM25 + (optional) embedding fusion
3. **AdaptiveSamplingStrategy** — dynamic config, early stopping, Gaussian refinement
4. **BatchEvaluator** — batched LLM scoring (3-5× fewer calls)
5. **EvidenceCache** — cross-query result reuse

The public types ``SampleWindow`` and ``RoiResult`` are re-exported here
for full backward compatibility — existing imports continue to work.
"""

from typing import Any, Dict, List, Optional, Set

from sirchmunk.learnings._types import RoiResult, SampleWindow
from sirchmunk.learnings.chunking import DocumentChunker
from sirchmunk.learnings.evidence_cache import EvidenceCache
from sirchmunk.learnings.sampling import AdaptiveSamplingStrategy, SamplingConfig
from sirchmunk.learnings.scoring import BatchEvaluator, ScoredChunk, SignalScorer
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import ROI_RESULT_SUMMARY
from sirchmunk.utils import LogCallback, create_logger

__all__ = [
    "EvidenceSampler",
    "MonteCarloEvidenceSampling",
    "RoiResult",
    "SampleWindow",
]


class EvidenceSampler:
    """Orchestrate evidence extraction through a layered pipeline.

    Drop-in replacement for the former monolithic
    ``MonteCarloEvidenceSampling`` class.  The constructor signature is
    a strict superset — callers that do not pass the new optional
    parameters (``embedding_util``, ``tokenizer``, ``cache``) get the
    same behaviour as before, with the added benefit of batch LLM
    evaluation and structure-aware chunking.
    """

    def __init__(
        self,
        llm: OpenAIChat,
        doc_content: str,
        verbose: bool = True,
        log_callback: LogCallback = None,
        *,
        embedding_util: Any = None,
        tokenizer: Any = None,
        cache: Optional[EvidenceCache] = None,
    ):
        self.llm = llm
        self.doc = doc_content
        self.doc_len = len(doc_content)
        self.verbose = verbose
        self._log = create_logger(log_callback=log_callback)
        self.llm_usages: List[Dict[str, Any]] = []

        # --- components ---
        initial_config = AdaptiveSamplingStrategy.configure(self.doc_len)
        self._chunker = DocumentChunker(
            target_size=initial_config.probe_window,
            max_size=initial_config.probe_window * 2,
        )
        self._scorer = SignalScorer(tokenizer=tokenizer)
        self._evaluator = BatchEvaluator(llm=llm, log_callback=log_callback)
        self._strategy = AdaptiveSamplingStrategy
        self._cache = cache
        self._embedding_util = embedding_util

        self._visited: Set[int] = set()

    # ==================================================================
    # Public API — signature-compatible with the old get_roi()
    # ==================================================================

    async def get_roi(
        self,
        query: str,
        keywords: Dict[str, float] = None,
        confidence_threshold: float = 7.0,
        top_k: int = 5,
    ) -> RoiResult:
        keywords = keywords or {}

        if self.verbose:
            await self._log.info(
                f"=== Starting Hybrid Adaptive Retrieval (Doc Len: {self.doc_len}) ==="
            )
            await self._log.info(f"Query: {query}, optional keywords: {keywords}")

        # 1. Cache lookup
        if self._cache:
            cached = self._cache.get(self.doc, query)
            if cached:
                if self.verbose:
                    await self._log.info("Cache hit — returning cached result")
                return RoiResult(
                    summary=cached.summary,
                    is_found=cached.is_found,
                    snippets=cached.snippets,
                )

        # 2. Configure
        config = self._strategy.configure(self.doc_len)
        config.confidence_threshold = confidence_threshold

        # 3. Short-document fast path
        if config.skip_sampling:
            result = await self._fast_path(query, config)
            self._cache_result(query, result)
            return result

        # 4. Structure-aware chunking
        chunks = self._chunker.chunk(self.doc)

        # 5. Multi-signal pre-scoring (zero LLM cost)
        scored_chunks = await self._scorer.score(
            query=query,
            keywords=list(keywords.keys()),
            chunks=chunks,
            embedding_util=self._embedding_util,
        )

        if self.verbose:
            top3 = scored_chunks[:3]
            labels = ", ".join(
                f"[{sc.start}..{sc.end} score={sc.combined_score:.2f}]" for sc in top3
            )
            await self._log.info(
                f"Pre-scored {len(scored_chunks)} chunks — top 3: {labels}"
            )

        # 6. Iterative sampling + batch evaluation
        result = await self._iterative_sampling(
            query, scored_chunks, config, top_k,
        )
        self._cache_result(query, result)
        return result

    # ==================================================================
    # Internal pipeline stages
    # ==================================================================

    async def _fast_path(
        self, query: str, config: SamplingConfig,
    ) -> RoiResult:
        """Short doc (≤3 KB): evaluate the entire content in one LLM call."""
        if self.verbose:
            await self._log.info("Short document detected — evaluating entire content")

        sample = SampleWindow(
            start_idx=0,
            end_idx=self.doc_len,
            content=self.doc,
            round_num=1,
            source="whole_doc",
        )
        evaluated = await self._evaluator.evaluate([sample], query)
        self._collect_usages()

        if evaluated:
            s = evaluated[0]
            await self._log.info(
                f"  [Whole Doc] Score: {s.score} | {s.reasoning[:50]}..."
            )

        if evaluated and evaluated[0].score >= 4.0:
            summary = await self._generate_summary(evaluated[:1], query, config)
            return RoiResult(
                summary=summary,
                is_found=True,
                snippets=[self._snippet_dict(s) for s in evaluated[:1]],
            )

        best = evaluated[0] if evaluated else sample
        return RoiResult(
            summary="No exact answer found in the document.",
            is_found=False,
            snippets=[self._snippet_dict(best)],
        )

    async def _iterative_sampling(
        self,
        query: str,
        scored_chunks: List[ScoredChunk],
        config: SamplingConfig,
        top_k: int,
    ) -> RoiResult:
        """Multi-round sampling with batch evaluation and early stopping."""
        all_candidates: List[SampleWindow] = []
        top_seeds: List[SampleWindow] = []

        # Build pre_scores map from scored_chunks for hybrid evaluation
        pre_scores: Dict[tuple, float] = {
            (sc.start, sc.end): sc.combined_score for sc in scored_chunks
        }

        for r in range(1, config.max_rounds + 1):
            if self.verbose:
                await self._log.info(f"--- Round {r}/{config.max_rounds} ---")

            current_samples: List[SampleWindow] = []

            if r == 1:
                current_samples = self._round_one_samples(
                    scored_chunks, config,
                )
            else:
                current_samples = self._refinement_samples(
                    top_seeds, all_candidates, config, r,
                )

            if not current_samples:
                if self.verbose:
                    await self._log.info("No new samples generated this round, skipping.")
                continue

            if self.verbose:
                await self._log.info(
                    f"   Evaluating {len(current_samples)} samples with LLM (batch)..."
                )

            # Use hybrid evaluation with pre_scores for Round 1 samples
            # (these come from scored_chunks and have known pre-scores)
            evaluated = await self._evaluator.evaluate_hybrid(
                current_samples, query, pre_scores
            )
            self._collect_usages()
            all_candidates.extend(evaluated)

            for s in evaluated:
                await self._log.info(
                    f"  [Pos {s.start_idx:6d} | Src: {s.source:10s}] "
                    f"Score: {s.score} | {s.reasoning[:30]}..."
                )

            all_candidates.sort(key=lambda x: x.score, reverse=True)
            top_seeds = all_candidates[: config.top_k_seeds]

            best_score = top_seeds[0].score if top_seeds else 0.0
            scores = [c.score for c in all_candidates]
            mean = sum(scores) / len(scores) if scores else 0.0
            variance = (
                sum((s - mean) ** 2 for s in scores) / len(scores)
                if scores
                else 0.0
            )

            ess = self._strategy.compute_ess(scores)
            if self.verbose and r > 1:
                await self._log.info(
                    f"IS diagnostics: ESS={ess:.1f}/{len(scores)}, "
                    f"best={best_score:.1f}, var={variance:.1f}"
                )

            if not self._strategy.should_continue(best_score, r, config, variance):
                if self.verbose:
                    await self._log.info(
                        f"Early stop: best_score={best_score:.1f}, "
                        f"variance={variance:.1f}"
                    )
                break

        return await self._build_result(all_candidates, query, top_k, config)

    # ------------------------------------------------------------------
    # Round 1: pre-scored chunks + random exploration
    # ------------------------------------------------------------------

    def _round_one_samples(
        self,
        scored_chunks: List[ScoredChunk],
        config: SamplingConfig,
    ) -> List[SampleWindow]:
        current: List[SampleWindow] = []

        top_scored = self._strategy.select_from_scored(
            scored_chunks, self._visited, config.fuzz_candidates_num,
        )
        for sc in top_scored:
            current.append(
                SampleWindow(
                    start_idx=sc.start,
                    end_idx=sc.end,
                    content=sc.content,
                    fuzz_score=sc.combined_score * 100,
                    round_num=1,
                    source="scored",
                )
            )

        best_signal = top_scored[0].combined_score if top_scored else 0
        needed_random = config.random_exploration_num
        if not top_scored or best_signal < 0.3:
            needed_random += 3
        elif best_signal >= 0.7:
            needed_random = 0

        if needed_random > 0:
            randoms = self._strategy.sample_random(
                self.doc, self.doc_len, config.probe_window,
                needed_random, self._visited,
            )
            current.extend(randoms)

        return current

    # ------------------------------------------------------------------
    # Rounds 2+: Gaussian refinement or random fallback
    # ------------------------------------------------------------------

    def _refinement_samples(
        self,
        top_seeds: List[SampleWindow],
        all_candidates: List[SampleWindow],
        config: SamplingConfig,
        current_round: int,
    ) -> List[SampleWindow]:
        valid_seeds = [s for s in top_seeds if s.score >= 4.0]

        if valid_seeds:
            return self._strategy.sample_gaussian(
                valid_seeds,
                self.doc_len,
                config.probe_window,
                current_round,
                config.samples_per_round,
                self._visited,
                self.doc,
                all_candidates=all_candidates,
            )

        return self._strategy.sample_random(
            self.doc, self.doc_len, config.probe_window,
            config.samples_per_round, self._visited,
        )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------

    async def _build_result(
        self,
        candidates: List[SampleWindow],
        query: str,
        top_k: int,
        config: SamplingConfig,
    ) -> RoiResult:
        if not candidates:
            await self._log.warning("Failed to retrieve any content.")
            return RoiResult(
                summary="Could not retrieve relevant content.",
                is_found=False,
                snippets=[],
            )

        relevant = [c for c in candidates if c.score >= 4.0]

        if not relevant:
            best = candidates[0]
            return RoiResult(
                summary="No exact answer found in the document.",
                is_found=False,
                snippets=[self._snippet_dict(best)],
            )

        final = relevant[:top_k]

        if self.verbose:
            await self._log.info(
                f"=== Final Lock: {len(final)} snippets, Top Score {final[0].score} ==="
            )

        summary = await self._generate_summary(final, query, config)
        return RoiResult(
            summary=summary,
            is_found=True,
            snippets=[self._snippet_dict(c) for c in final],
        )

    async def _generate_summary(
        self,
        top_samples: List[SampleWindow],
        query: str,
        config: SamplingConfig,
    ) -> str:
        combined_context = ""
        half_window = config.roi_window // 2
        processed = sorted(top_samples, key=lambda x: x.start_idx)

        for i, sample in enumerate(processed):
            center = (sample.start_idx + sample.end_idx) // 2
            start = max(0, center - half_window)
            end = min(self.doc_len, center + half_window)
            expanded = self.doc[start:end]
            combined_context += (
                f"\n--- Context Fragment {i + 1} ---\n...{expanded}...\n"
            )

        prompt = ROI_RESULT_SUMMARY.format(
            user_input=query,
            text_content=combined_context,
        )

        resp = await self.llm.achat(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
        )
        self.llm_usages.append(resp.usage)
        return resp.content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_usages(self) -> None:
        self.llm_usages.extend(self._evaluator.llm_usages)
        self._evaluator.llm_usages.clear()

    def _cache_result(self, query: str, result: RoiResult) -> None:
        if self._cache and result.is_found:
            self._cache.put(
                doc_content=self.doc,
                query=query,
                scored_positions=[
                    (s["start"], s["end"], s["score"]) for s in result.snippets
                ],
                summary=result.summary,
                is_found=result.is_found,
                snippets=result.snippets,
            )

    @staticmethod
    def _snippet_dict(s: SampleWindow) -> Dict[str, Any]:
        return {
            "snippet": s.content,
            "start": s.start_idx,
            "end": s.end_idx,
            "score": s.score,
            "reasoning": s.reasoning,
        }


# Backward-compatible alias
MonteCarloEvidenceSampling = EvidenceSampler
