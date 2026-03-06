"""VisionSearch — main orchestrator for the Sirchmunk-Vision pipeline.

Coordinates all three layers and implements the self-evolving knowledge
mechanism.  On first search (cold start) images flow through the full
Visual Grep → SigLIP2 Ranking → VLM Verification pipeline.  Verified
results are persisted so that subsequent searches short-circuit through
fast text-based retrieval over cached captions and tags.

Two speed modes, both supporting text-only / image-only / hybrid input:

    FAST (default) — greedy Phase 1+2 + VLM verification:
        Phase 0 → Phase 1 → Phase 2 → Phase 3 (VLM) → Phase 4 (persist)
        Uses greedy parameters in Phase 1+2 for speed, then sends a
        collage to the VLM for semantic verification.  Skips the online
        feedback/weight-adaptation loop.

    DEEP — full precision pipeline:
        Phase 0 → Phase 1 → Phase 2 → Phase 3 (VLM) → Phase 4 (persist + feedback)
        Thorough parameters in Phase 1+2; VLM verification; verified
        results feed the online feedback loop for operator weight tuning.

Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │ Phase 0  Knowledge cache lookup (instant short-circuit)        │
    ├────────────────────────────────────────────────────────────────┤
    │ Phase 1  Visual Grep   — pHash + colour + metadata filtering   │
    │          ↑ signature cache (DuckDB, mtime-based freshness)     │
    │          ↑ parallel operator scoring (ThreadPoolExecutor)      │
    │          ↑ learned operator weights (feedback loop)            │
    ├────────────────────────────────────────────────────────────────┤
    │ Phase 2  SigLIP2 Rank  — multi-scale + multi-query scoring     │
    │          ↑ query expansion (3-5 clip_query variants)           │
    │          ↑ 6 crops per image (global + 4 quad + centre)        │
    ├────────────────────────────────────────────────────────────────┤
    │ Phase 3  VLM Verifier  — adaptive collage verification         │
    │          ↑ 2×2 / 3×3 / multi-call based on candidate count    │
    ├────────────────────────────────────────────────────────────────┤
    │ Phase 4  Persist + result assembly                             │
    │          ↑ VLM verdicts → operator weight adaptation   [DEEP]  │
    │          ↑ hard negative mining for SigLIP threshold   [DEEP]  │
    └────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sirchmunk.memory.agent_memory import AgentMemory, SearchAdvice, SearchEpisode
from .batch_verifier import BatchVerifier
from .constraint_compiler import ConstraintCompiler
from .image_signer import ImageSigner
from sirchmunk.storage.vision_store import VisionKnowledgeStore as KnowledgeStore
from .probabilistic_scout import DEFAULT_CLIP_MODEL, ProbabilisticScout
from sirchmunk.schema.vision import (
    ImageKnowledge,
    ImageSignature,
    ScoredCandidate,
    VerificationResult,
    VisualConstraint,
)
from .visual_grep import VisualGrep
from sirchmunk.llm.vlm_chat import VLMClient

logger = logging.getLogger(__name__)


@dataclass
class VisionSearchResult:
    """Final search result combining verification and cached knowledge."""

    path: str
    caption: str = ""
    confidence: float = 0.0
    semantic_tags: List[str] = field(default_factory=list)
    source: str = "pipeline"   # "cache" | "pipeline" | "fast_pipeline"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict safe for ``json.dumps``."""
        return {
            "path": self.path,
            "caption": self.caption,
            "confidence": self.confidence,
            "semantic_tags": list(self.semantic_tags),
            "source": self.source,
        }

    def __str__(self) -> str:
        parts = [self.path]
        if self.caption:
            parts.append(f"caption={self.caption!r}")
        parts.append(f"confidence={self.confidence:.2f}")
        if self.semantic_tags:
            parts.append(f"tags={self.semantic_tags}")
        parts.append(f"source={self.source}")
        return f"VisionSearchResult({', '.join(parts)})"


class VisionSearch:
    """Three-layer vision search with self-evolving knowledge.

    The system becomes progressively faster with use: VLM-verified
    captions are stored in DuckDB and short-circuit future queries via
    text-based retrieval — no visual pipeline needed for known images.
    """

    def __init__(
        self,
        vlm: Optional[VLMClient] = None,
        work_path: str = ".sirchmunk_vision",
        max_grep_candidates: int = 5000,
        scout_top_k: int = 9,
        clip_model_id: str = DEFAULT_CLIP_MODEL,
        clip_batch_size: int = 32,
        max_scan_depth: int = 10,
        memory: Optional[AgentMemory] = None,
    ):
        self._vlm = vlm or VLMClient()
        self._work_path = work_path
        self._max_scan_depth = max_scan_depth
        os.makedirs(work_path, exist_ok=True)

        self._store = KnowledgeStore(
            db_path=os.path.join(work_path, "vision_kb.duckdb"),
        )

        self._signer = ImageSigner()
        self._compiler = ConstraintCompiler(self._vlm)
        self._grep = VisualGrep(
            signer=self._signer,
            compiler=self._compiler,
            max_candidates=max_grep_candidates,
            signature_store=self._store,
        )

        self._scout = ProbabilisticScout(
            clip_model_id=clip_model_id,
            clip_batch_size=clip_batch_size,
            cache_dir=work_path,
        )

        self._verifier = BatchVerifier(self._vlm)
        self._scout_top_k = scout_top_k

        self._memory = memory or AgentMemory(
            db_path=os.path.join(work_path, "agent_memory.duckdb"),
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str = "",
        paths: Optional[List[str]] = None,
        top_k: int = 10,
        query_images: Optional[List[Any]] = None,
        mode: str = "FAST",
    ) -> List[VisionSearchResult]:
        """Execute the vision search pipeline.

        Both ``FAST`` and ``DEEP`` modes support all three input types:

        - **Text only** (``query`` only): constraint-based pipeline.
        - **Image only** (``query_images`` only): image-similarity pipeline.
        - **Hybrid** (both ``query`` and ``query_images``): combined.

        The difference is:

        - **FAST** (default) — greedy Phase 1+2, then VLM verification:
          Phase 0 → 1 → 2 → 3 (VLM collage) → 4 (persist).
          Skips the online feedback/weight-adaptation loop.
        - **DEEP** — thorough Phase 1+2, VLM verification + feedback:
          Phase 0 → 1 → 2 → 3 (VLM collage) → 4 (persist + feedback).

        Args:
            query:        Natural language description of the target images.
            paths:        Directories or individual image files to search.
            top_k:        Maximum number of results to return.
            query_images: Reference images for similarity or hybrid search.
            mode:         ``"FAST"`` (default) or ``"DEEP"``.

        Returns:
            Ranked list of :class:`VisionSearchResult`.
        """
        if not paths:
            raise ValueError("'paths' is required and must not be empty")

        pil_images = [_normalize_to_pil(img) for img in (query_images or [])]
        has_text = bool(query and query.strip())
        has_images = bool(pil_images)

        if not has_text and not has_images:
            raise ValueError(
                "At least one of 'query' or 'query_images' must be provided",
            )

        source_paths: List[str] = []
        for img in (query_images or []):
            if isinstance(img, str) and os.path.isfile(img):
                source_paths.append(os.path.abspath(img))
            else:
                source_paths.append("")

        fast = mode.upper() == "FAST"

        # Determine input type for logging and phase dispatch
        if has_text and has_images:
            input_type = "hybrid"
        elif has_images:
            input_type = "image"
        else:
            input_type = "text"

        return await self._run_pipeline(
            query=query,
            paths=paths,
            top_k=top_k,
            fast=fast,
            input_type=input_type,
            pil_images=pil_images,
            source_paths=source_paths,
        )

    # ------------------------------------------------------------------ #
    # Unified pipeline (FAST and DEEP share the same flow)
    # ------------------------------------------------------------------ #

    async def _run_pipeline(
        self,
        query: str,
        paths: List[str],
        top_k: int,
        fast: bool,
        input_type: str,
        pil_images: Optional[List[Any]] = None,
        source_paths: Optional[List[str]] = None,
    ) -> List[VisionSearchResult]:
        """Core pipeline shared by FAST and DEEP modes.

        Both modes run the full Phase 0–3 pipeline including VLM collage
        verification.  DEEP additionally runs the online feedback loop
        for operator weight adaptation.

        Args:
            query:        Text query (may be empty for image-only).
            paths:        Search paths.
            top_k:        Max results.
            fast:         True for greedy Phase 1+2 parameters.  VLM
                          verification still runs; only the feedback
                          loop is skipped.
            input_type:   ``"text"`` | ``"image"`` | ``"hybrid"``.
            pil_images:   PIL images for image/hybrid modes.
            source_paths: Original file paths for query images.
        """
        mode_label = "FAST" if fast else "DEEP"
        t_total = time.time()
        timings: List[str] = []
        phase_stats: Dict[str, Any] = {
            "query_type": f"{input_type}_{mode_label.lower()}",
            "scout_top_k": self._scout_top_k,
        }

        print(f"\n{'=' * 60}")
        print(
            f"[VisionSearch:{mode_label}:{input_type}] "
            f"query='{(query or '')[: 60]}', "
            f"images={len(pil_images or [])}, "
            f"paths={paths}, top_k={top_k}"
        )
        print(f"{'=' * 60}")

        # ============================================================== #
        # Memory advice
        # ============================================================== #
        advice_query = query or "[image-search]"
        advice = self._memory.advise(advice_query, paths)
        if advice.memo:
            print(f"[VisionSearch:Memory] {advice.memo}")

        # ============================================================== #
        # Phase 0: Knowledge cache — exact-query short-circuit
        # ============================================================== #
        t0 = time.time()
        cached: list = []
        if query:
            print("[VisionSearch:Phase0] Checking knowledge cache...")
            cached = await self._store.search_by_exact_query(query, limit=top_k)
            if cached:
                valid = [c for c in cached if os.path.isfile(c.path)]
                if len(valid) >= top_k:
                    print(
                        f"[VisionSearch:Phase0] Full cache hit — "
                        f"{len(valid)} results ({time.time() - t0:.2f}s)"
                    )
                    return _wrap_cached(valid[:top_k])
        timings.append(f"Phase0-Cache: {time.time() - t0:.2f}s")

        # ============================================================== #
        # Phase 1: Visual Grep — filtering
        # ============================================================== #
        t1 = time.time()
        candidates, constraint, clip_query, expanded = await self._phase1_grep(
            query=query,
            paths=paths,
            input_type=input_type,
            pil_images=pil_images,
            source_paths=source_paths,
            fast=fast,
        )
        t1_elapsed = time.time() - t1
        if not candidates:
            print(f"[VisionSearch:Phase1] No candidates ({t1_elapsed:.2f}s)")
            return _wrap_cached(cached, top_k) if cached else []
        print(
            f"[VisionSearch:Phase1] {len(candidates)} candidates "
            f"({t1_elapsed:.2f}s)"
        )
        timings.append(f"Phase1-Grep: {t1_elapsed:.2f}s")
        phase_stats["phase1_candidates"] = len(candidates)

        # ============================================================== #
        # Phase 2: SigLIP2 Ranking
        # ============================================================== #
        t2 = time.time()
        scored = await self._phase2_rank(
            candidates=candidates,
            clip_query=clip_query,
            expanded=expanded,
            input_type=input_type,
            pil_images=pil_images,
            top_k=min(self._scout_top_k, top_k) if fast else self._scout_top_k,
            fast=fast,
        )
        t2_elapsed = time.time() - t2
        if not scored:
            print(f"[VisionSearch:Phase2] No scored results ({t2_elapsed:.2f}s)")
            return _wrap_cached(cached, top_k) if cached else []
        print(
            f"[VisionSearch:Phase2] {len(scored)} survivors, "
            f"top={os.path.basename(scored[0].path)} "
            f"({scored[0].score:.3f}) ({t2_elapsed:.2f}s)"
        )
        timings.append(f"Phase2-SigLIP: {t2_elapsed:.2f}s")
        phase_stats["phase2_survivors"] = len(scored)

        self._backfill_siglip_embeds(candidates)

        # ============================================================== #
        # Phase 3: VLM collage verification (both FAST and DEEP)
        # ============================================================== #
        results: List[VisionSearchResult] = []
        seen: set = set()

        verify_query = query or await self._describe_query_images(pil_images)
        verified, t3_elapsed = await self._phase3_verify(
            scored=scored,
            query=verify_query,
            cached=cached,
        )
        timings.append(f"Phase3-VLM: {t3_elapsed:.2f}s")

        # ============================================================== #
        # Phase 4: Persist verified results + assemble VisionSearchResult
        # ============================================================== #
        t4 = time.time()
        for v in verified:
            if v.verified:
                knowledge = ImageKnowledge(
                    path=v.path,
                    caption=v.caption,
                    semantic_tags=v.semantic_tags,
                    confidence=v.confidence,
                    query_history=[verify_query],
                )
                await self._store.store(knowledge)
                if query:
                    await self._store.update_query_history(v.path, query)

            results.append(VisionSearchResult(
                path=v.path,
                caption=v.caption,
                confidence=v.confidence,
                semantic_tags=v.semantic_tags,
                source="pipeline" if v.verified else "vlm_unverified",
            ))
            seen.add(v.path)

        # Feedback loop — DEEP only (online weight adaptation)
        if not fast and constraint is not None:
            self._collect_feedback(verify_query, constraint, scored, verified)
        timings.append(f"Phase4-Persist: {time.time() - t4:.2f}s")

        # Append any scored candidates that were not sent to VLM
        # (e.g. already_verified exclusion) using their SigLIP2 scores
        for s in scored[:top_k]:
            if s.path not in seen:
                results.append(VisionSearchResult(
                    path=s.path, caption="", confidence=s.score,
                    source="siglip_score",
                ))
                seen.add(s.path)

        # ============================================================== #
        # Merge cached results + sort + record episode
        # ============================================================== #
        if cached:
            for k in cached:
                if k.path not in seen and os.path.isfile(k.path):
                    results.append(VisionSearchResult(
                        path=k.path, caption=k.caption,
                        confidence=k.confidence,
                        semantic_tags=k.semantic_tags, source="cache",
                    ))
                    seen.add(k.path)

        results.sort(key=lambda r: r.confidence, reverse=True)
        final_results = results[:top_k]

        total_elapsed = time.time() - t_total
        timings.append(f"Total: {total_elapsed:.2f}s")
        print(
            f"[VisionSearch:{mode_label}] Returning "
            f"{len(final_results)} results"
        )
        print(f"[VisionSearch:Profiling] {' | '.join(timings)}")
        print(f"{'=' * 60}\n")

        self._record_episode(
            query=advice_query,
            query_type=f"{input_type}_{mode_label.lower()}",
            paths=paths,
            top_k=top_k,
            results=final_results,
            timings_raw=_parse_timings(timings),
            phase_stats=phase_stats,
        )
        return final_results

    # ------------------------------------------------------------------ #
    # Phase implementations
    # ------------------------------------------------------------------ #

    async def _phase1_grep(
        self,
        query: str,
        paths: List[str],
        input_type: str,
        pil_images: Optional[List[Any]] = None,
        source_paths: Optional[List[str]] = None,
        fast: bool = False,
    ) -> tuple:
        """Phase 1: Visual Grep — dispatch by input type.

        Returns:
            ``(candidates, constraint, clip_query, expanded)``
            For image-only mode, ``constraint`` is ``None``.
        """
        print(
            f"[VisionSearch:Phase1] Visual Grep ({input_type}) — "
            f"scanning {len(paths)} path(s)..."
        )

        if input_type == "image":
            query_sigs = self._sign_query_images(pil_images, source_paths)
            candidates = await self._grep.filter_by_image(
                query_sigs, paths, max_depth=self._max_scan_depth,
            )
            return candidates, None, "", []

        weight_overrides = self._load_learned_weights(query)
        candidates, constraint = await self._grep.filter(
            query, paths,
            max_depth=self._max_scan_depth,
            weight_overrides=weight_overrides,
            fast=fast,
        )
        clip_query = (
            constraint.clip_query
            or " ".join(constraint.semantic_tags)
            or query
        )
        expanded = constraint.expanded_clip_queries
        return candidates, constraint, clip_query, expanded

    async def _phase2_rank(
        self,
        candidates: list,
        clip_query: str,
        expanded: list,
        input_type: str,
        pil_images: Optional[List[Any]] = None,
        top_k: int = 9,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Phase 2: SigLIP2 ranking — dispatch by input type."""
        print(
            f"[VisionSearch:Phase2] SigLIP2 ranking ({input_type}) — "
            f"{len(candidates)} candidates"
        )

        if input_type == "image":
            return await self._scout.scout_by_image(
                candidates, pil_images, top_k=top_k, fast=fast,
            )
        if input_type == "hybrid":
            return await self._scout.scout_hybrid(
                candidates, clip_query, pil_images,
                top_k=top_k, expanded_queries=expanded, fast=fast,
            )
        # text-only
        return await self._scout.scout(
            candidates, clip_query,
            top_k=top_k, expanded_queries=expanded, fast=fast,
        )

    async def _phase3_verify(
        self,
        scored: List[ScoredCandidate],
        query: str,
        cached: list,
    ) -> tuple:
        """Phase 3: VLM verification (DEEP only).

        Returns:
            ``(verified_list, elapsed_seconds)``
        """
        t3 = time.time()
        already_verified = {c.path for c in cached} if cached else set()
        to_verify = [s for s in scored if s.path not in already_verified]

        if to_verify:
            print(
                f"[VisionSearch:Phase3] VLM verification — "
                f"{len(to_verify)} new candidates..."
            )
            verified = await self._verifier.verify_batch(to_verify, query)
        else:
            print("[VisionSearch:Phase3] All already verified — skipping VLM")
            verified = []

        matched = sum(1 for v in verified if v.verified)
        elapsed = time.time() - t3
        print(
            f"[VisionSearch:Phase3] VLM verified {matched}/"
            f"{len(to_verify) if to_verify else 0} ({elapsed:.2f}s)"
        )
        return verified, elapsed

    # ------------------------------------------------------------------ #
    # Feedback loop (online learning)
    # ------------------------------------------------------------------ #

    def _load_learned_weights(
        self,
        query: str,
    ) -> Optional[Dict[str, float]]:
        """Load operator weight overrides from historical feedback."""
        words = query.strip().split()[:3]
        pattern = " ".join(w.lower() for w in words) if words else ""
        if not pattern:
            return None
        weights = self._store.get_learned_weights(pattern)
        if weights:
            print(
                f"  [FeedbackLoop] Using learned weights for "
                f"pattern '{pattern}': {weights}"
            )
        return weights

    def _collect_feedback(
        self,
        query: str,
        constraint: VisualConstraint,
        scored: List[ScoredCandidate],
        verified: List[VerificationResult],
    ) -> None:
        """Record operator scores and VLM verdicts for weight adaptation."""
        if not verified:
            return

        vlm_lookup = {v.path: (v.verified, v.confidence) for v in verified}
        siglip_lookup = {s.path: s.score for s in scored}
        now = datetime.now().isoformat()
        records: List[Dict[str, Any]] = []

        for v in verified:
            vlm_match, vlm_conf = vlm_lookup.get(v.path, (False, 0.0))
            records.append({
                "path": v.path,
                "query": query,
                "operator_name": "siglip_score",
                "op_score": siglip_lookup.get(v.path, 0.0),
                "vlm_match": vlm_match,
                "vlm_confidence": vlm_conf,
                "created_at": now,
            })

        try:
            self._store.store_feedback(records)
        except Exception as e:
            logger.warning("Failed to store feedback: %s", e)

        self._update_pattern_weights(query)

    def _update_pattern_weights(self, query: str) -> None:
        """Recompute and persist operator weights for the query pattern."""
        try:
            accuracy = self._store.compute_operator_accuracy(min_samples=20)
            if not accuracy:
                return

            min_acc = min(accuracy.values())
            max_acc = max(accuracy.values())
            spread = max_acc - min_acc
            if spread < 0.01:
                return

            weights = {
                name: 0.5 + 0.5 * (acc - min_acc) / spread
                for name, acc in accuracy.items()
            }

            words = query.strip().split()[:3]
            pattern = " ".join(w.lower() for w in words) if words else ""
            if not pattern:
                return

            threshold = self._store.compute_clip_rejection_threshold(95.0)
            self._store.update_pattern_weights(
                pattern, weights, clip_threshold=threshold or 0.0,
            )
        except Exception as e:
            logger.warning("Failed to update pattern weights: %s", e)

    # ------------------------------------------------------------------ #
    # SigLIP embed backfill
    # ------------------------------------------------------------------ #

    def _backfill_siglip_embeds(self, candidates: List[Any]) -> None:
        """Write SigLIP2 global embeddings from the pickle cache to DuckDB."""
        cache = self._scout._cache
        if cache is None:
            return

        embeds_to_write: Dict[str, List[float]] = {}
        for c in candidates:
            ms = cache._data.get(c.path)
            if ms is None or ms.ndim != 2:
                continue
            embeds_to_write[c.path] = ms[0].tolist()

        if embeds_to_write:
            try:
                n = self._store.backfill_siglip_embeds(embeds_to_write)
                if n:
                    print(
                        f"  [VisionSearch] Backfilled {n} SigLIP embeddings"
                    )
            except Exception as e:
                logger.warning("SigLIP embed backfill failed: %s", e)

    # ------------------------------------------------------------------ #
    # Memory recording
    # ------------------------------------------------------------------ #

    def _record_episode(
        self,
        query: str,
        query_type: str,
        paths: List[str],
        top_k: int,
        results: List[VisionSearchResult],
        timings_raw: Dict[str, float],
        phase_stats: Dict[str, Any],
    ) -> None:
        """Persist a search episode to agent memory."""
        import uuid
        try:
            ep = SearchEpisode(
                episode_id=str(uuid.uuid4())[:12],
                query=query,
                query_type=query_type,
                paths=paths,
                top_k=top_k,
                result_count=len(results),
                hit_paths=[r.path for r in results],
                timings=timings_raw,
                phase_stats=phase_stats,
            )
            self._memory.record_episode(ep)
            print(
                f"[VisionSearch:Memory] Episode recorded "
                f"(total: {self._memory.get_episode_count()} episodes)"
            )
        except Exception as e:
            logger.warning("Memory recording failed: %s", e)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sign_query_images(
        self,
        pil_images: List[Any],
        source_paths: List[str],
    ) -> List[ImageSignature]:
        """Create visual signatures for query images."""
        sigs: List[ImageSignature] = []
        for i, (pil, spath) in enumerate(zip(pil_images, source_paths)):
            path = spath or f"<query_{i}>"
            sigs.append(self._signer.sign_image(pil, path=path))
        return sigs

    async def _describe_query_images(self, pil_images: List[Any]) -> str:
        """Ask VLM to describe the query image(s) in one sentence."""
        print(
            f"  [VisionSearch] Describing {len(pil_images)} "
            f"query image(s) via VLM..."
        )
        prompt = (
            "Describe this image in one concise English sentence. "
            "Focus on the main subject, scene, colours, and distinctive "
            "visual features."
        ) if len(pil_images) == 1 else (
            "Describe these images in one concise English sentence, "
            "focusing on their common visual theme and distinctive features."
        )
        msg = VLMClient.build_user_message(text=prompt, images=pil_images)
        resp = await self._vlm.achat(messages=[msg], temperature=0.0)
        description = resp.content.strip()
        print(f"  [VisionSearch] Image description: '{description[:120]}'")
        return description


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _normalize_to_pil(source: Any) -> Any:
    """Convert various image sources to a PIL Image.

    Accepts:
        - ``PIL.Image.Image`` — passthrough
        - ``bytes`` / ``bytearray`` — decoded
        - ``str``: file path, base64 data URI, HTTP(S) URL, or raw base64
    """
    from PIL import Image as PILImage

    if isinstance(source, PILImage.Image):
        return source.convert("RGB")

    if isinstance(source, (bytes, bytearray)):
        return PILImage.open(io.BytesIO(source)).convert("RGB")

    if isinstance(source, (str, os.PathLike)):
        s = str(source)
        if os.path.isfile(s):
            return PILImage.open(s).convert("RGB")
        if s.startswith("data:"):
            _, payload = s.split(",", 1)
            return PILImage.open(
                io.BytesIO(base64.b64decode(payload)),
            ).convert("RGB")
        if s.startswith(("http://", "https://")):
            import httpx
            resp = httpx.get(s, timeout=30)
            resp.raise_for_status()
            return PILImage.open(io.BytesIO(resp.content)).convert("RGB")
        try:
            return PILImage.open(
                io.BytesIO(base64.b64decode(s)),
            ).convert("RGB")
        except Exception:
            pass
        raise ValueError(f"Cannot resolve image source: {s[:80]!r}")

    raise TypeError(f"Unsupported query image type: {type(source)}")


def _wrap_cached(
    cached: list,
    top_k: Optional[int] = None,
) -> List[VisionSearchResult]:
    items = [
        VisionSearchResult(
            path=k.path,
            caption=k.caption,
            confidence=k.confidence,
            semantic_tags=k.semantic_tags,
            source="cache",
        )
        for k in cached
    ]
    return items[:top_k] if top_k else items


def _parse_timings(timing_strs: List[str]) -> Dict[str, float]:
    """Parse ``["Phase1: 2.34s", ...]`` into ``{"Phase1": 2.34, ...}``."""
    import re as _re
    result: Dict[str, float] = {}
    for s in timing_strs:
        m = _re.match(r"(.+?):\s*([\d.]+)s", s)
        if m:
            result[m.group(1).strip()] = float(m.group(2))
    return result
