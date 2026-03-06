"""VisualGrep — Layer 1: multi-operator image filtering.

Replaces the original monolithic scorer with a *pluggable operator fusion*
architecture.  Each operator contributes an independent [0, 1] score; the
final score is a weighted average over all applicable operators.

Built-in operators (see ``operators/`` package):
  ┌─────────────────────┬───────────────────────────────────────────┐
  │ ColorOperator        │ Global hue / saturation / brightness      │
  │ MetadataOperator     │ Filename / path keyword overlap           │
  │ BlockHashOperator    │ MSBH spatial-aware hashing                │
  │ ContrastiveColorOp   │ Centre–edge colour contrast               │
  │ GISTOperator         │ Scene-type Gabor-energy descriptor        │
  │ EntropyOperator      │ Image complexity pruning                  │
  │ NegativeOperator     │ Penalty for negative constraints          │
  └─────────────────────┴───────────────────────────────────────────┘

Post-scoring pipeline:
  1. Sort by fused score.
  2. Cap to ``max_candidates``.
  3. **Dynamic threshold (Scheme D)**: if pool > trigger, tighten to 75th
     percentile to reduce downstream CLIP cost.
  4. **Adaptive CLIP cap**: if score variance is very low (uninformative
     filtering), further reduce the pool.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constraint_compiler import ConstraintCompiler
from .image_signer import IMAGE_EXTENSIONS, ImageSigner
from .operators import DEFAULT_OPERATORS, GrepOperator
from .operators.base import hamming_distance
from sirchmunk.schema.vision import ImageCandidate, ImageSignature, VisualConstraint

# Optional — imported only when a KnowledgeStore is wired in
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sirchmunk.storage.vision_store import VisionKnowledgeStore as KnowledgeStore

_PARALLEL_SCORE_THRESHOLD = 200

# Dynamic threshold trigger: if more than this many candidates survive
# the initial cap, tighten the score cutoff.
_DYNAMIC_TIGHTEN_TRIGGER = 1000
_DYNAMIC_SURVIVAL_RATIO = 0.25

# When score variance is negligible, further cap to limit CLIP cost.
_ADAPTIVE_CLIP_CAP = 500
_ADAPTIVE_STD_THRESHOLD = 0.01

# FAST mode: more aggressive filtering to reduce Phase 2 cost.
_FAST_TIGHTEN_TRIGGER = 500
_FAST_SURVIVAL_RATIO = 0.10
_FAST_ADAPTIVE_CAP = 200


class VisualGrep:
    """Multi-operator image filtering with dynamic threshold control."""

    def __init__(
        self,
        signer: ImageSigner,
        compiler: ConstraintCompiler,
        max_candidates: int = 5000,
        operators: Optional[List[GrepOperator]] = None,
        signature_store: Optional["KnowledgeStore"] = None,
    ):
        self._signer = signer
        self._compiler = compiler
        self._max_candidates = max_candidates
        self._operators = operators if operators is not None else list(DEFAULT_OPERATORS)
        self._sig_store = signature_store

    # ------------------------------------------------------------------ #
    # Public API — text-to-image (existing)
    # ------------------------------------------------------------------ #

    async def filter(
        self,
        query: str,
        paths: List[str],
        max_depth: int = 10,
        weight_overrides: Optional[Dict[str, float]] = None,
        fast: bool = False,
    ) -> Tuple[List[ImageCandidate], VisualConstraint]:
        """Scan *paths*, compile constraints, score via operator fusion.

        Returns:
            ``(candidates, constraint)`` — ranked candidates and the
            compiled :class:`VisualConstraint` (carries ``clip_query``
            and ``expanded_clip_queries`` for downstream SigLIP2 scoring).
        """
        print(f"  [VisualGrep] Compiling visual constraints for query: '{query[:60]}'")
        constraint = await self._compiler.compile(query)
        clip_query = (
            constraint.clip_query
            or " ".join(constraint.semantic_tags)
            or query
        )
        print(
            f"  [VisualGrep] clip_query='{clip_query}', "
            f"hues={constraint.dominant_hues}, tags={constraint.semantic_tags}"
        )

        # ---- Scan & sign images (with persistent cache) ----
        signatures = self._scan_with_cache(paths, max_depth)

        print(
            f"  [VisualGrep] Scanned {len(signatures)} image signatures "
            f"from {len(paths)} path(s)"
        )
        if not signatures:
            print("  [VisualGrep] No images found — returning empty")
            return [], constraint

        # ---- Activate SigLIPEmbedOperator if cached embeds exist ----
        has_neural = any(sig.siglip_embed for sig in signatures)
        if has_neural and constraint.clip_query_embed is None:
            from .probabilistic_scout import encode_text_query
            clip_q = (
                constraint.clip_query
                or " ".join(constraint.semantic_tags)
                or query
            )
            embed = encode_text_query(clip_q)
            if embed is not None:
                constraint.clip_query_embed = embed
                n_with = sum(1 for s in signatures if s.siglip_embed)
                print(
                    f"  [VisualGrep] SigLIP embed operator activated "
                    f"({n_with}/{len(signatures)} images have cached embeds)"
                )
            else:
                print(
                    "  [VisualGrep] SigLIP model not yet loaded — "
                    "embed operator stays inactive"
                )

        # ---- Pre-filter: applicable operators ----
        active = [
            (op, op.weight) for op in self._operators
            if op.is_applicable(constraint)
        ]
        op_names = [op.name for op, _ in active]
        print(f"  [VisualGrep] Active operators ({len(active)}): {op_names}")

        # ---- Batch-compute SigLIP embed scores (SIMD-accelerated) ----
        siglip_batch_scores: Dict[str, float] = {}
        siglip_op = next(
            (op for op, _ in active if op.name == "siglip_embed"), None,
        )
        if siglip_op is not None:
            from .operators.siglip_embed_op import SigLIPEmbedOperator
            if isinstance(siglip_op, SigLIPEmbedOperator):
                siglip_batch_scores = SigLIPEmbedOperator.batch_score(
                    signatures, constraint,
                )
                if siglip_batch_scores:
                    print(
                        f"  [VisualGrep] SigLIP batch-scored "
                        f"{len(siglip_batch_scores)} images (SIMD)"
                    )

        # ---- Score via weighted operator fusion (parallelised) ----
        t_score = time.time()
        scorer = partial(
            _fused_score,
            con=constraint,
            active_ops=active,
            weight_overrides=weight_overrides,
            precomputed_siglip=siglip_batch_scores,
        )

        if len(signatures) >= _PARALLEL_SCORE_THRESHOLD:
            n_workers = min(8, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                scores = list(pool.map(
                    lambda sig: scorer(sig=sig), signatures,
                ))
            scored = [
                ImageCandidate(path=sig.path, signature=sig, grep_score=sc)
                for sig, sc in zip(signatures, scores)
            ]
        else:
            scored = [
                ImageCandidate(
                    path=sig.path,
                    signature=sig,
                    grep_score=scorer(sig=sig),
                )
                for sig in signatures
            ]

        scored.sort(key=lambda c: c.grep_score, reverse=True)
        score_elapsed = time.time() - t_score
        print(
            f"  [VisualGrep] Scored {len(scored)} images in {score_elapsed:.2f}s "
            f"({'parallel' if len(signatures) >= _PARALLEL_SCORE_THRESHOLD else 'serial'})"
        )

        # ---- Step 1: initial cap ----
        result = scored[: self._max_candidates]

        # ---- Step 2: Dynamic threshold ----
        trigger = _FAST_TIGHTEN_TRIGGER if fast else _DYNAMIC_TIGHTEN_TRIGGER
        survival = _FAST_SURVIVAL_RATIO if fast else _DYNAMIC_SURVIVAL_RATIO
        cap = _FAST_ADAPTIVE_CAP if fast else _ADAPTIVE_CLIP_CAP

        if len(result) > trigger:
            scores_arr = [c.grep_score for c in result]
            pct = max(0.0, (1.0 - survival) * 100)
            cutoff = float(np.percentile(scores_arr, pct))
            before = len(result)
            tightened = [c for c in result if c.grep_score >= cutoff]
            min_keep = max(int(before * survival), 1)
            if len(tightened) < min_keep:
                tightened = result[:min_keep]
            result = tightened
            print(
                f"  [VisualGrep] Dynamic threshold: {before} → {len(result)} "
                f"(≥ {cutoff:.3f})"
            )

        # ---- Step 3: Adaptive cap (uninformative scores) ----
        if len(result) > cap:
            score_std = float(np.std([c.grep_score for c in result]))
            if score_std < _ADAPTIVE_STD_THRESHOLD:
                result = result[:cap]
                print(
                    f"  [VisualGrep] Scores uninformative (σ={score_std:.4f}), "
                    f"capped to {len(result)}"
                )

        if result:
            print(
                f"  [VisualGrep] Returning {len(result)} candidates "
                f"(score range: {result[0].grep_score:.3f} "
                f"— {result[-1].grep_score:.3f})"
            )
        return result, constraint

    # ------------------------------------------------------------------ #
    # Public API — image-to-image
    # ------------------------------------------------------------------ #

    async def filter_by_image(
        self,
        query_signatures: List[ImageSignature],
        paths: List[str],
        max_depth: int = 10,
    ) -> List[ImageCandidate]:
        """Rank candidate images by visual similarity to query images.

        Scoring combines pHash hamming distance, colour-moment cosine
        similarity, GIST descriptor similarity, and centre-region moments.
        When multiple query signatures are provided, the maximum similarity
        across all queries is used.

        Args:
            query_signatures: Pre-computed signatures of the query image(s).
            paths:            Directories or individual image files to search.
            max_depth:        Maximum directory recursion depth.

        Returns:
            Candidates sorted by descending similarity score.
        """
        print(f"  [VisualGrep] Image-to-image filtering: {len(query_signatures)} query image(s)")

        query_paths = {qs.path for qs in query_signatures}

        signatures = self._scan_with_cache(paths, max_depth)

        print(
            f"  [VisualGrep] Scanned {len(signatures)} image signatures "
            f"from {len(paths)} path(s)"
        )
        if not signatures:
            print("  [VisualGrep] No images found — returning empty")
            return []

        scored: List[ImageCandidate] = []
        for sig in signatures:
            if sig.path in query_paths:
                continue
            sim = max(_image_similarity(qs, sig) for qs in query_signatures)
            scored.append(ImageCandidate(path=sig.path, signature=sig, grep_score=sim))

        scored.sort(key=lambda c: c.grep_score, reverse=True)
        result = scored[: self._max_candidates]

        if result:
            print(
                f"  [VisualGrep] Returning {len(result)} candidates "
                f"(score range: {result[0].grep_score:.3f} "
                f"— {result[-1].grep_score:.3f})"
            )
        return result


    # ------------------------------------------------------------------ #
    # Persistent signature cache
    # ------------------------------------------------------------------ #

    def _scan_with_cache(
        self,
        paths: List[str],
        max_depth: int,
    ) -> List[ImageSignature]:
        """Scan paths for images, using DuckDB-backed signature cache.

        If no ``signature_store`` is wired, falls back to the original
        stateless scanning flow.
        """
        from pathlib import Path as PPath

        # Discover all image files
        image_files: List[str] = []
        for p in paths:
            if os.path.isfile(p) and _is_image(p):
                image_files.append(os.path.abspath(p))
            elif os.path.isdir(p):
                root = PPath(p).resolve()
                for fp in sorted(root.rglob("*")):
                    if len(fp.relative_to(root).parts) > max_depth:
                        continue
                    if fp.is_file() and fp.suffix.lower() in IMAGE_EXTENSIONS:
                        image_files.append(str(fp))

        if not image_files:
            return []

        # Without a store, fall back to stateless signing
        if self._sig_store is None:
            signatures: List[ImageSignature] = []
            for p in paths:
                if os.path.isfile(p) and _is_image(p):
                    try:
                        signatures.append(self._signer.sign(p))
                    except Exception:
                        continue
                elif os.path.isdir(p):
                    signatures.extend(
                        self._signer.scan_directory(p, max_depth=max_depth),
                    )
            return signatures

        # Build (path, mtime, file_size) tuples
        path_info: List[tuple] = []
        for fp in image_files:
            try:
                stat = os.stat(fp)
                path_info.append((fp, stat.st_mtime, stat.st_size))
            except OSError:
                continue

        # Lookup cached signatures
        t0 = time.time()
        cached, stale = self._sig_store.get_cached_signatures(path_info)
        cache_elapsed = time.time() - t0
        print(
            f"    [VisualGrep:SigCache] {len(cached)} cached, "
            f"{len(stale)} to sign ({cache_elapsed:.2f}s lookup)"
        )

        # Sign stale/new files (parallelised)
        if stale:
            t1 = time.time()
            new_entries: list = []
            n_workers = min(8, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                future_map = {
                    pool.submit(self._signer.sign, fp): (fp, mt, sz)
                    for fp, mt, sz in stale
                }
                for future in as_completed(future_map):
                    fp, mt, sz = future_map[future]
                    try:
                        sig = future.result()
                        cached[fp] = sig
                        new_entries.append((fp, mt, sz, sig))
                    except Exception:
                        continue
            self._sig_store.store_signatures_batch(new_entries)
            sign_elapsed = time.time() - t1
            print(
                f"    [VisualGrep:SigCache] Signed {len(new_entries)} new "
                f"images in {sign_elapsed:.2f}s ({n_workers} threads)"
            )

        return list(cached.values())


# ------------------------------------------------------------------ #
# Fusion scoring
# ------------------------------------------------------------------ #

def _fused_score(
    sig: ImageSignature,
    con: VisualConstraint,
    active_ops: List[Tuple[GrepOperator, float]],
    weight_overrides: Optional[Dict[str, float]] = None,
    precomputed_siglip: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted average of all active operator scores.

    If *weight_overrides* is provided, operator weights are replaced
    by the learned values from the feedback loop.

    When *precomputed_siglip* is given, the ``siglip_embed`` operator
    uses the pre-computed batch score instead of recomputing per-image.
    """
    if not active_ops:
        return 0.5
    total_w = 0.0
    weighted_sum = 0.0
    for op, default_w in active_ops:
        w = (
            weight_overrides.get(op.name, default_w)
            if weight_overrides
            else default_w
        )
        if (
            precomputed_siglip is not None
            and op.name == "siglip_embed"
            and sig.path in precomputed_siglip
        ):
            sc = precomputed_siglip[sig.path]
        else:
            sc = op.score(sig, con)
        weighted_sum += sc * w
        total_w += w
    return weighted_sum / total_w if total_w > 0 else 0.5


def compute_per_operator_scores(
    sig: ImageSignature,
    con: VisualConstraint,
    operators: List[GrepOperator],
) -> Dict[str, float]:
    """Return per-operator scores as ``{operator_name: score}``."""
    return {
        op.name: op.score(sig, con)
        for op in operators
        if op.is_applicable(con)
    }


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS


# ------------------------------------------------------------------ #
# Image-to-image similarity scoring
# ------------------------------------------------------------------ #

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, clamped to [0, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(max(0.0, np.dot(a, b) / denom))


def _image_similarity(query: ImageSignature, candidate: ImageSignature) -> float:
    """Multi-feature similarity between two image signatures.

    Components (weighted average):
        - pHash hamming distance     (w=2)
        - Colour-moment cosine sim   (w=2)
        - GIST descriptor cosine sim (w=1.5)
        - Centre-region moment sim   (w=1)
    """
    parts: List[Tuple[float, float]] = []   # (score, weight)

    if query.phash and candidate.phash:
        h_dist = hamming_distance(query.phash, candidate.phash)
        parts.append((max(0.0, 1.0 - h_dist / 32.0), 2.0))

    if query.color_moments and candidate.color_moments:
        parts.append((_cosine_sim(
            np.array(query.color_moments), np.array(candidate.color_moments),
        ), 2.0))

    if query.gist_descriptor and candidate.gist_descriptor:
        parts.append((_cosine_sim(
            np.array(query.gist_descriptor), np.array(candidate.gist_descriptor),
        ), 1.5))

    if query.center_moments and candidate.center_moments:
        parts.append((_cosine_sim(
            np.array(query.center_moments), np.array(candidate.center_moments),
        ), 1.0))

    if not parts:
        return 0.0
    total_w = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / total_w
