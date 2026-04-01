# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-signal relevance scoring and batched LLM evaluation.

Contains two complementary components:

- **SignalScorer** — fast, zero-LLM scoring that fuses RapidFuzz
  token-set-ratio, BM25 (via ``BM25Scorer``), and optional embedding
  cosine-similarity into a single ``combined_score`` per chunk.

- **BatchEvaluator** — merges N samples into a single LLM prompt per
  round, reducing evidence evaluation cost by 3-5× compared with
  per-sample calls.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz, process

from sirchmunk.learnings._types import SampleWindow
from sirchmunk.learnings.chunking import Chunk
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    BATCH_EVALUATE_EVIDENCE_SAMPLES,
    EVALUATE_EVIDENCE_SAMPLE,
)
from sirchmunk.utils import LogCallback, create_logger
from sirchmunk.utils.bm25_util import BM25Scorer

# ---------------------------------------------------------------------------
# Regex helpers for fallback JSON parsing
# ---------------------------------------------------------------------------
_SCORE_RE = re.compile(r'"score"\s*:\s*(\d+(?:\.\d+)?)')
_ID_SCORE_RE = re.compile(
    r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d+(?:\.\d+)?)'
)


# ---------------------------------------------------------------------------
# ScoredChunk
# ---------------------------------------------------------------------------
@dataclass
class ScoredChunk:
    """A chunk annotated with pre-LLM relevance scores."""

    start: int
    end: int
    content: str
    chunk_type: str = "text"
    fuzz_score: float = 0.0
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    combined_score: float = 0.0


# ===================================================================
# SignalScorer — zero-LLM multi-signal fusion
# ===================================================================
class SignalScorer:
    """Score document chunks by fusing RapidFuzz, BM25, and embedding signals.

    BM25 is handled by the shared ``BM25Scorer`` utility (backed by
    ``bm25s`` + optional ``TokenizerUtil``).  When an ``EmbeddingUtil``
    instance is provided and ready, cosine similarity is mixed in;
    otherwise the weights are redistributed between fuzz and BM25 only.
    """

    WEIGHT_FUZZ = 0.3
    WEIGHT_BM25 = 0.4
    WEIGHT_EMBEDDING = 0.3

    def __init__(self, tokenizer: Any = None):
        self._bm25 = BM25Scorer(tokenizer=tokenizer)

    # ---- embedding ----------------------------------------------------

    @staticmethod
    async def _compute_embedding(
        query: str, texts: List[str], embedding_util: Any,
    ) -> List[float]:
        import numpy as np

        all_texts = [query] + texts
        embeddings = await embedding_util.embed(all_texts)
        q_emb = np.array(embeddings[0])
        return [max(0.0, float(np.dot(q_emb, np.array(e)))) for e in embeddings[1:]]

    # ---- fuzz ---------------------------------------------------------

    @staticmethod
    def _compute_fuzz(query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        results = process.extract(
            query=query,
            choices=texts,
            scorer=fuzz.token_set_ratio,
            limit=len(texts),
            score_cutoff=None,
        )
        score_map = {idx: score / 100.0 for _, score, idx in results}
        return [score_map.get(i, 0.0) for i in range(len(texts))]

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        mn, mx = min(values), max(values)
        if mx - mn < 1e-9:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    # ---- public API ---------------------------------------------------

    async def score(
        self,
        query: str,
        keywords: List[str],
        chunks: List[Chunk],
        embedding_util: Any = None,
    ) -> List[ScoredChunk]:
        """Return chunks ranked by multi-signal combined score (desc)."""
        if not chunks:
            return []

        query_with_kw = f"{query} {' '.join(keywords)}".strip()
        contents = [c.content for c in chunks]

        fuzz_scores = self._compute_fuzz(query_with_kw, contents)

        bm25_raw = self._bm25.score(query_with_kw, contents)
        bm25_scores = self._normalize(bm25_raw)

        embedding_scores = [0.0] * len(chunks)
        has_embedding = False
        if embedding_util is not None and getattr(embedding_util, "is_ready", lambda: False)():
            try:
                embedding_scores = await self._compute_embedding(
                    query, contents, embedding_util,
                )
                has_embedding = True
            except Exception:
                pass

        w_f = self.WEIGHT_FUZZ
        w_b = self.WEIGHT_BM25
        w_e = self.WEIGHT_EMBEDDING if has_embedding else 0.0
        total_w = w_f + w_b + w_e

        scored: List[ScoredChunk] = []
        for i, chunk in enumerate(chunks):
            combined = (
                w_f * fuzz_scores[i] + w_b * bm25_scores[i] + w_e * embedding_scores[i]
            ) / total_w
            scored.append(
                ScoredChunk(
                    start=chunk.start,
                    end=chunk.end,
                    content=chunk.content,
                    chunk_type=chunk.chunk_type,
                    fuzz_score=fuzz_scores[i],
                    bm25_score=bm25_scores[i],
                    embedding_score=embedding_scores[i],
                    combined_score=combined,
                )
            )

        scored.sort(key=lambda x: x.combined_score, reverse=True)
        return scored


# ===================================================================
# BatchEvaluator — batched LLM evidence scoring
# ===================================================================

# Hybrid evaluation thresholds (configurable via environment variables)
_EVAL_HIGH_THRESHOLD = float(os.getenv("SIRCHMUNK_EVAL_HIGH_THRESHOLD", "0.75"))
_EVAL_LOW_THRESHOLD = float(os.getenv("SIRCHMUNK_EVAL_LOW_THRESHOLD", "0.25"))


class BatchEvaluator:
    """Evaluate multiple text samples in a single LLM call per batch.

    Falls back to per-sample evaluation when batch result parsing fails,
    ensuring robustness without sacrificing the 3-5× efficiency gain
    in the common case.
    """

    MAX_BATCH_SIZE = 8

    def __init__(
        self,
        llm: OpenAIChat,
        log_callback: LogCallback = None,
    ):
        self._llm = llm
        self._log = create_logger(log_callback=log_callback)
        self.llm_usages: List[Dict[str, Any]] = []

    # Maximum concurrent batch evaluations to avoid LLM API rate limits
    MAX_CONCURRENT_BATCHES = 3

    # ---- public API ---------------------------------------------------

    async def evaluate(
        self, samples: List[SampleWindow], query: str,
    ) -> List[SampleWindow]:
        """Score *samples* in batches; updates ``.score`` / ``.reasoning`` in place.

        Batches are evaluated in parallel (up to MAX_CONCURRENT_BATCHES) for
        improved throughput while respecting LLM API rate limits.
        """
        if not samples:
            return samples

        import asyncio
        import time as _time

        # Split into batches
        batches = [
            samples[i : i + self.MAX_BATCH_SIZE]
            for i in range(0, len(samples), self.MAX_BATCH_SIZE)
        ]

        # Semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_BATCHES)

        async def _eval_batch_safe(batch: List[SampleWindow]) -> None:
            """Evaluate a single batch with concurrency control and error handling."""
            async with semaphore:
                try:
                    if len(batch) == 1:
                        await self._evaluate_single(batch[0], query)
                    else:
                        await self._evaluate_one_batch(batch, query)
                except Exception as e:
                    await self._log.warning(f"Batch evaluation error: {e}")
                    # Mark failed samples with score 0 so they don't block the pipeline
                    for s in batch:
                        if s.score == 0:
                            s.reasoning = f"Evaluation failed: {e}"

        _t0 = _time.time()
        # Execute all batches in parallel (bounded by semaphore)
        await asyncio.gather(*[_eval_batch_safe(b) for b in batches])
        await self._log.info(
            f"[Timing] Batch evaluation ({len(samples)} samples, "
            f"{len(batches)} batches, parallel): {_time.time()-_t0:.2f}s"
        )
        return samples

    async def evaluate_hybrid(
        self,
        samples: List[SampleWindow],
        query: str,
        pre_scores: Optional[Dict[tuple, float]] = None,
    ) -> List[SampleWindow]:
        """Hybrid evaluation: use pre-scores to skip obvious high/low samples.

        Args:
            samples: Samples to evaluate
            query: Search query
            pre_scores: Optional dict mapping sample identifier (start_idx, end_idx)
                        to pre-computed signal score (0.0-1.0)

        Returns:
            Updated samples with scores assigned
        """
        if not samples:
            return samples

        # Fallback to standard evaluation if no pre_scores provided
        if not pre_scores:
            return await self.evaluate(samples, query)

        high_confidence: List[SampleWindow] = []
        low_confidence: List[SampleWindow] = []
        uncertain: List[SampleWindow] = []

        for sample in samples:
            key = (sample.start_idx, sample.end_idx)
            pre_score = pre_scores.get(key)

            if pre_score is None:
                # No pre-score available, send to LLM
                uncertain.append(sample)
            elif pre_score >= _EVAL_HIGH_THRESHOLD:
                # High confidence: assign high score directly
                sample.score = 8.0
                sample.reasoning = "High signal score (pre-filtered)"
                high_confidence.append(sample)
            elif pre_score <= _EVAL_LOW_THRESHOLD:
                # Low confidence: assign low score directly
                sample.score = 2.0
                sample.reasoning = "Low signal score (pre-filtered)"
                low_confidence.append(sample)
            else:
                # Uncertain: needs LLM evaluation
                uncertain.append(sample)

        await self._log.info(
            f"Hybrid eval: {len(samples)} samples -> "
            f"{len(high_confidence)} high / {len(low_confidence)} low / {len(uncertain)} LLM"
        )

        # Evaluate uncertain samples with LLM
        if uncertain:
            await self.evaluate(uncertain, query)

        # Return all samples (they've been modified in place)
        return samples

    # ---- batch evaluation ---------------------------------------------

    async def _evaluate_one_batch(
        self, batch: List[SampleWindow], query: str,
    ) -> None:
        snippets_parts = []
        for idx, s in enumerate(batch):
            snippets_parts.append(
                f"\n[{idx + 1}] (Source: {s.source})\n\"...{s.content[:2000]}...\""
            )
        snippets_block = "\n".join(snippets_parts)

        prompt = BATCH_EVALUATE_EVIDENCE_SAMPLES.format(
            query=query, snippets_block=snippets_block,
        )

        try:
            resp_obj = await self._llm.achat(
                [{"role": "user", "content": prompt}],
                enable_thinking=False,
            )
            self.llm_usages.append(resp_obj.usage)
            parsed = self._parse_batch_response(resp_obj.content, batch)
            if not parsed:
                await self._log.warning(
                    "Batch parse returned no scores, falling back to per-sample evaluation"
                )
                await self._fallback_per_sample(batch, query)
        except Exception as e:
            await self._log.warning(f"Batch evaluation failed ({e}), falling back to per-sample")
            await self._fallback_per_sample(batch, query)

    def _parse_batch_response(
        self, response: str, batch: List[SampleWindow],
    ) -> bool:
        """Parse batch response with multi-level fallback. Returns True on success."""
        clean = response.replace("```json", "").replace("```", "").strip()

        # Attempt 1: full JSON array
        try:
            arr = json.loads(clean)
            if isinstance(arr, list):
                for item in arr:
                    idx = int(item.get("id", 0)) - 1
                    if 0 <= idx < len(batch):
                        batch[idx].score = float(item.get("score", 0))
                        batch[idx].reasoning = str(item.get("reasoning", ""))
                return True
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Attempt 2: regex id+score pairs
        found_any = False
        for m in _ID_SCORE_RE.finditer(clean):
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(batch):
                batch[idx].score = float(m.group(2))
                found_any = True
        if found_any:
            return True

        # Attempt 3: single-item fallback
        m = _SCORE_RE.search(clean)
        if m and len(batch) == 1:
            batch[0].score = float(m.group(1))
            return True

        return False

    # ---- per-sample fallback ------------------------------------------

    async def _fallback_per_sample(
        self, batch: List[SampleWindow], query: str,
    ) -> None:
        import asyncio

        tasks = [self._evaluate_single(s, query) for s in batch]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _evaluate_single(
        self, sample: SampleWindow, query: str,
    ) -> SampleWindow:
        prompt = EVALUATE_EVIDENCE_SAMPLE.format(
            query=query,
            sample_source=sample.source,
            sample_content=sample.content[:2000],
        )
        try:
            resp_obj = await self._llm.achat(
                [{"role": "user", "content": prompt}],
                enable_thinking=False,
            )
            self.llm_usages.append(resp_obj.usage)
            clean = resp_obj.content.replace("```json", "").replace("```", "").strip()
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                m = _SCORE_RE.search(clean)
                if m:
                    data = {"score": float(m.group(1)), "reasoning": ""}
                else:
                    raise
            sample.score = float(data.get("score", 0))
            sample.reasoning = data.get("reasoning", "")
        except Exception as e:
            await self._log.warning(f"Error evaluating sample at {sample.start_idx}: {e}")
            sample.score = 0.0
        return sample
