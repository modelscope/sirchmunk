"""ProbabilisticScout — Layer 2: SigLIP2-based multi-scale semantic ranking.

Scoring strategies:
  1. **Text-to-image** — encode text queries (+ expanded variants from query
     expansion) via SigLIP2, rank candidates using multi-scale image
     embeddings with max-pooling across all (query, scale) pairs.
  2. **Image-to-image** — encode query images through SigLIP2 vision encoder,
     rank against multi-scale candidate embeddings.
  3. **Hybrid** — weighted combination of text and image similarity scores.

Multi-scale encoding (6 crops per image):
  - Global thumbnail       (full image context)
  - 4 quadrant crops       (TL / TR / BL / BR — local detail)
  - Centre crop            (inner 50 % — subject focus)

Performance features:
  - SigLIP2 model is lazy-loaded on first use.
  - Multi-scale image embeddings are cached to disk; the first search
    pays the encoding cost, subsequent searches only need text encoding
    + a dot product (~0.1 s regardless of image count).
  - Multiple query variants are scored via max-pooling.
"""

from __future__ import annotations

import asyncio
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from sirchmunk.schema.vision import ImageCandidate, ScoredCandidate
from sirchmunk.vision.vector_search import (
    is_accelerated as _vs_accel,
    multi_query_max_pool,
    preferred_dtype,
)

# ------------------------------------------------------------------ #
# Lazy-loaded SigLIP2 singleton (module-level for reuse across calls)
# ------------------------------------------------------------------ #
_model: Any = None
_processor: Any = None
_device: str = "cpu"

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"

_THUMB_SIZE = (224, 224)
_CACHE_VERSION = 3
_N_SCALES = 6  # global, TL, TR, BL, BR, center

# FAST early-stopping parameters
_FAST_CHUNK_SIZE = 100
_FAST_PATIENCE = 2
_FAST_MIN_EVAL_FACTOR = 5   # process at least top_k * this before stopping

# Backward-compatible alias used by external modules
DEFAULT_CLIP_MODEL = DEFAULT_MODEL_ID


def _resolve_model_path(model_id: str) -> str:
    """Resolve model to a local path via ModelScope, falling back to HuggingFace."""
    try:
        from modelscope import snapshot_download
        local_path = snapshot_download(model_id)
        print(f"    [SigLIPScout] Model downloaded via ModelScope: {model_id}")
        return local_path
    except Exception as e:
        print(
            f"    [SigLIPScout] ModelScope unavailable ({e!r}), "
            f"falling back to HuggingFace: {model_id}"
        )
        return model_id


def _ensure_model(model_id: str = DEFAULT_MODEL_ID) -> None:
    """Download and cache the SigLIP2 model on first invocation."""
    global _model, _processor, _device

    if _model is not None:
        return

    print(f"    [SigLIPScout] Loading SigLIP2 model: {model_id} ...")
    import torch
    from transformers import AutoModel, AutoProcessor

    resolved = _resolve_model_path(model_id)
    _processor = AutoProcessor.from_pretrained(resolved)
    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"
    _model = AutoModel.from_pretrained(resolved).to(_device).eval()
    print(f"    [SigLIPScout] SigLIP2 model loaded on device: {_device}")


def encode_text_query(query: str) -> Any:
    """Encode a single text query if the SigLIP2 model is already loaded.

    Returns an L2-normalised numpy vector ``(embed_dim,)`` or ``None``
    if the model hasn't been loaded yet (cold start).  This allows
    the SigLIPEmbedOperator in Visual Grep to score cached embeddings
    without forcing an early model load.
    """
    if _model is None:
        return None
    import torch
    tokens = _processor(
        text=[query], return_tensors="pt",
        padding="max_length", truncation=True,
    )
    tokens = {k: v.to(_device) for k, v in tokens.items()}
    with torch.no_grad():
        raw = _model.get_text_features(**tokens)
        feat = _extract_features(raw)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0]  # (embed_dim,)


def _extract_features(output: Any) -> Any:
    """Extract the feature tensor from a model output.

    ``get_text_features`` / ``get_image_features`` may return a raw
    ``torch.Tensor`` (newer transformers) or a ``BaseModelOutput*``
    wrapper (older versions / some architectures).  This helper
    normalises both cases to a plain tensor.
    """
    import torch
    if isinstance(output, torch.Tensor):
        return output
    # BaseModelOutputWithPooling — pooled_output is at index [1]
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output[1]


# ------------------------------------------------------------------ #
# Multi-scale crop generation
# ------------------------------------------------------------------ #

def _generate_crops(
    img: Image.Image,
    size: Tuple[int, int] = _THUMB_SIZE,
    global_only: bool = False,
) -> List[Image.Image]:
    """Generate image crops for SigLIP2 encoding.

    When *global_only* is ``True`` (FAST mode), returns a single resized
    thumbnail — 6× cheaper than the full multi-scale crop set.
    """
    if global_only:
        thumb = img.copy()
        thumb.thumbnail(size, Image.LANCZOS)
        return [thumb.convert("RGB")]

    w, h = img.size
    half_w, half_h = max(w // 2, 1), max(h // 2, 1)
    qw, qh = max(w // 4, 1), max(h // 4, 1)

    regions = [
        (0, 0, w, h),                      # global
        (0, 0, half_w, half_h),             # top-left
        (half_w, 0, w, half_h),             # top-right
        (0, half_h, half_w, h),             # bottom-left
        (half_w, half_h, w, h),             # bottom-right
        (qw, qh, w - qw, h - qh),          # centre (inner 50 %)
    ]

    crops: List[Image.Image] = []
    for box in regions:
        crop = img.crop(box)
        crop.thumbnail(size, Image.LANCZOS)
        crops.append(crop.convert("RGB"))
    return crops


# ------------------------------------------------------------------ #
# Multi-scale embedding cache
# ------------------------------------------------------------------ #

class _EmbeddingCache:
    """File-backed cache: image path → (N_SCALES, embed_dim) L2-normalised."""

    def __init__(self, cache_dir: str, model_id: str):
        os.makedirs(cache_dir, exist_ok=True)
        safe = model_id.replace("/", "_").replace("\\", "_")
        self._path = os.path.join(
            cache_dir, f"siglip_embeds_{safe}_v{_CACHE_VERSION}.pkl",
        )
        self._data: Dict[str, np.ndarray] = {}
        self._dirty = False
        if os.path.exists(self._path):
            try:
                with open(self._path, "rb") as f:
                    self._data = pickle.load(f)
                print(
                    f"    [SigLIPCache] Loaded {len(self._data)} "
                    f"cached multi-scale embeddings"
                )
            except Exception:
                self._data = {}

    def lookup(
        self,
        paths: List[str],
        expected_dim: int = 0,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Return ``(cached_dict, uncached_list)``.

        Cached values are ``(N_SCALES, embed_dim)`` matrices.  Entries
        with wrong shape or dimension are silently discarded.
        """
        cached: Dict[str, np.ndarray] = {}
        uncached: List[str] = []
        for p in paths:
            vec = self._data.get(p)
            if (
                vec is not None
                and vec.ndim == 2
                and vec.shape[0] == _N_SCALES
                and (expected_dim == 0 or vec.shape[1] == expected_dim)
            ):
                cached[p] = vec
            else:
                uncached.append(p)
        return cached, uncached

    def store(self, embeddings: Dict[str, np.ndarray]) -> None:
        self._data.update(embeddings)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        with open(self._path, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._dirty = False
        print(f"    [SigLIPCache] Persisted {len(self._data)} embeddings to disk")


# ------------------------------------------------------------------ #
# ProbabilisticScout
# ------------------------------------------------------------------ #

class ProbabilisticScout:
    """SigLIP2-based image ranking with multi-scale encoding and query expansion.

    On cold start, multi-scale image embeddings (6 crops per image) are
    computed in batches and cached to disk.  Warm searches only need text
    encoding + a matrix dot product.
    """

    def __init__(
        self,
        patch_size: int = 224,
        base_patches: int = 2,
        survival_ratio: float = 0.5,
        max_rounds: int = 4,
        clip_model_id: str = DEFAULT_MODEL_ID,
        clip_batch_size: int = 32,
        cache_dir: str = "",
    ):
        self.patch_size = patch_size
        self.base_patches = base_patches
        self.survival_ratio = survival_ratio
        self.max_rounds = max_rounds
        self._model_id = clip_model_id
        self._batch_size = clip_batch_size
        self._cache: Optional[_EmbeddingCache] = None
        if cache_dir:
            self._cache = _EmbeddingCache(cache_dir, clip_model_id)

    # ------------------------------------------------------------------ #
    # Text encoding
    # ------------------------------------------------------------------ #

    def _encode_texts(self, queries: List[str]) -> np.ndarray:
        """Encode text queries → ``(Q, embed_dim)`` L2-normalised matrix."""
        import torch

        _ensure_model(self._model_id)
        tokens = _processor(
            text=queries,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        tokens = {k: v.to(_device) for k, v in tokens.items()}
        with torch.no_grad():
            raw = _model.get_text_features(**tokens)
            text_feat = _extract_features(raw)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return text_feat.cpu().numpy()

    # ------------------------------------------------------------------ #
    # Image encoding (multi-scale)
    # ------------------------------------------------------------------ #

    def _get_candidate_multiscale(
        self,
        candidates: List[ImageCandidate],
        embed_dim: int,
        label: str = "",
        global_only: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Load cached + compute embeddings for candidates.

        When *global_only* is ``True``, only the global thumbnail is
        encoded (1 crop vs 6), yielding ``(1, embed_dim)`` matrices for
        uncached images.  Cached multi-scale entries are returned as-is.

        Returns:
            ``{path: np.ndarray(S, embed_dim)}`` where S is 1 or N_SCALES.
        """
        import torch

        all_paths = [c.path for c in candidates]
        if self._cache is not None:
            cached_embeds, uncached_paths = self._cache.lookup(
                all_paths, expected_dim=embed_dim,
            )
        else:
            cached_embeds, uncached_paths = {}, list(all_paths)

        n_crops = 1 if global_only else _N_SCALES
        tag = f" ({label})" if label else ""
        mode_tag = "global-only" if global_only else "multi-scale"
        print(
            f"    [SigLIPScout]{tag} Embeddings: {len(cached_embeds)} cached, "
            f"{len(uncached_paths)} to compute ({mode_tag})"
        )

        if not uncached_paths:
            return dict(cached_embeds)

        new_embeds: Dict[str, np.ndarray] = {}

        # Flatten all crops into one big list for efficient batching
        batch_crops: List[Image.Image] = []
        valid_paths: List[str] = []
        crop_offsets: List[int] = []

        for p in uncached_paths:
            try:
                img = Image.open(p)
                crops = _generate_crops(img, _THUMB_SIZE, global_only=global_only)
                crop_offsets.append(len(batch_crops))
                batch_crops.extend(crops)
                valid_paths.append(p)
            except Exception:
                continue

        if not batch_crops:
            return dict(cached_embeds)

        # Encode all crops in batches
        all_feats: List[np.ndarray] = []
        bs = self._batch_size
        total_batches = (len(batch_crops) + bs - 1) // bs

        for bi in range(total_batches):
            start = bi * bs
            crop_batch = batch_crops[start: start + bs]
            pixel = _processor(images=crop_batch, return_tensors="pt")
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=pixel["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat.cpu().numpy())

            if (bi + 1) % 10 == 0 or bi == total_batches - 1:
                print(
                    f"    [SigLIPScout] crop batch {bi + 1}/{total_batches}: "
                    f"{len(crop_batch)} crops"
                )

        if not all_feats:
            return dict(cached_embeds)

        feat_matrix = np.concatenate(all_feats, axis=0)  # (total_crops, D)

        for i, path in enumerate(valid_paths):
            offset = crop_offsets[i]
            end = offset + n_crops
            if end <= feat_matrix.shape[0]:
                new_embeds[path] = feat_matrix[offset:end]

        # Only persist full multi-scale embeddings (don't pollute cache
        # with global-only entries that would fail shape validation).
        if new_embeds and self._cache is not None and not global_only:
            self._cache.store(new_embeds)
            self._cache.flush()

        return {**cached_embeds, **new_embeds}

    def _encode_images_global(
        self,
        pil_images: List[Image.Image],
    ) -> np.ndarray:
        """Encode PIL Images at global scale → ``(N, embed_dim)``."""
        import torch

        _ensure_model(self._model_id)
        feats: List[np.ndarray] = []
        for img in pil_images:
            thumb = img.copy()
            thumb.thumbnail(_THUMB_SIZE, Image.LANCZOS)
            px = _processor(
                images=[thumb.convert("RGB")], return_tensors="pt",
            )
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=px["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy()[0])
        return np.stack(feats)

    # ------------------------------------------------------------------ #
    # FAST: global-only encoding helper
    # ------------------------------------------------------------------ #

    def _encode_global_batch(
        self,
        paths: List[str],
    ) -> Dict[str, np.ndarray]:
        """Encode global thumbnails for a list of image paths.

        Returns ``{path: np.ndarray(embed_dim,)}`` — one vector per image.
        Used by :meth:`_batch_rank_fast` for uncached candidates.
        """
        import torch

        imgs: List[Image.Image] = []
        valid: List[str] = []
        for p in paths:
            try:
                img = Image.open(p)
                img.thumbnail(_THUMB_SIZE, Image.LANCZOS)
                imgs.append(img.convert("RGB"))
                valid.append(p)
            except Exception:
                continue

        if not imgs:
            return {}

        result: Dict[str, np.ndarray] = {}
        bs = self._batch_size
        for bi in range(0, len(imgs), bs):
            batch = imgs[bi: bi + bs]
            batch_paths = valid[bi: bi + bs]
            px = _processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                raw = _model.get_image_features(
                    pixel_values=px["pixel_values"].to(_device),
                )
                feat = _extract_features(raw)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats = feat.cpu().numpy()
            for j, p in enumerate(batch_paths):
                result[p] = feats[j]
        return result

    # ------------------------------------------------------------------ #
    # FAST: greedy ranking with early stopping
    # ------------------------------------------------------------------ #

    def _batch_rank_fast(
        self,
        candidates: List[ImageCandidate],
        queries: List[str],
        top_k: int = 9,
    ) -> List[ScoredCandidate]:
        """FAST greedy ranking: global-crop-only + batch vector search.

        1. Bulk-score all cached embeddings with a single SIMD search.
        2. Process uncached candidates in chunks, encoding then scoring.
        3. Early-stop when ``_FAST_PATIENCE`` consecutive chunks fail to
           improve the running top-k.
        """
        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(queries)  # (Q, D)
        embed_dim = text_matrix.shape[1]
        dtype = preferred_dtype()
        backend = "USearch-SIMD" if _vs_accel() else "NumPy"
        print(
            f"    [SigLIPScout] Encoded {len(queries)} query variant(s), "
            f"dim={embed_dim}, backend={backend}"
        )

        # Bulk cache lookup — extract global row from cached multi-scale
        cached_global: Dict[str, np.ndarray] = {}
        if self._cache is not None:
            cached_ms, _ = self._cache.lookup(
                [c.path for c in candidates], expected_dim=embed_dim,
            )
            cached_global = {p: ms[0] for p, ms in cached_ms.items()}

        n_to_compute = len(candidates) - len(cached_global)
        print(
            f"    [SigLIPScout] (fast) {len(cached_global)} cached, "
            f"{n_to_compute} to compute (global-only)"
        )

        # --- Phase A: batch-score all cached embeddings at once ----------
        top_scores: Dict[str, float] = {}
        if cached_global:
            cached_paths = list(cached_global.keys())
            cached_mat = np.stack(
                [cached_global[p] for p in cached_paths],
            ).astype(dtype)
            scores = multi_query_max_pool(cached_mat, text_matrix.astype(dtype))
            for i, p in enumerate(cached_paths):
                top_scores[p] = float(scores[i])

        # --- Phase B: incremental encoding + scoring for uncached --------
        total_encoded = 0
        stale = 0
        min_before_stop = min(
            top_k * _FAST_MIN_EVAL_FACTOR, len(candidates),
        )
        processed = len(cached_global)

        uncached_cands = [
            c for c in candidates if c.path not in cached_global
        ]

        for ci in range(0, len(uncached_cands), _FAST_CHUNK_SIZE):
            chunk = uncached_cands[ci: ci + _FAST_CHUNK_SIZE]
            new = self._encode_global_batch([c.path for c in chunk])
            total_encoded += len(new)

            if new:
                chunk_paths = list(new.keys())
                chunk_mat = np.stack(
                    [new[p] for p in chunk_paths],
                ).astype(dtype)
                chunk_scores = multi_query_max_pool(
                    chunk_mat, text_matrix.astype(dtype),
                )
                improved = False
                for i, p in enumerate(chunk_paths):
                    sc = float(chunk_scores[i])
                    top_scores[p] = sc
                    improved = True
            else:
                improved = False

            processed += len(chunk)
            stale = 0 if improved else stale + 1

            if (
                stale >= _FAST_PATIENCE
                and len(top_scores) >= top_k
                and processed >= min_before_stop
            ):
                print(
                    f"    [SigLIPScout] Early stop at "
                    f"{processed}/{len(candidates)} "
                    f"(encoded {total_encoded} new)"
                )
                break

        # --- Assemble top-k from combined scores -------------------------
        sorted_items = sorted(
            top_scores.items(), key=lambda x: x[1], reverse=True,
        )[:top_k]

        elapsed = time.time() - t0
        scored = [
            ScoredCandidate(path=p, score=s, round_scores=[s])
            for p, s in sorted_items
        ]
        print(
            f"    [SigLIPScout] Ranked {processed} images (fast) "
            f"in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Shared: flatten multi-scale embeddings → batch search → max-pool
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flatten_multiscale(
        candidates: List[ImageCandidate],
        all_embeds: Dict[str, np.ndarray],
        dtype: np.dtype = np.float32,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Stack per-candidate multi-scale embeddings into one matrix.

        Returns:
            ``(matrix, paths, group_ids)`` where *matrix* is
            ``(total_rows, D)``, *paths* is the ordered candidate path
            list, and *group_ids* maps each matrix row to a candidate
            index in *paths* (for max-pooling).
        """
        rows: List[np.ndarray] = []
        paths: List[str] = []
        group_ids: List[int] = []
        cand_idx = 0
        for c in candidates:
            ms = all_embeds.get(c.path)
            if ms is None:
                continue
            if ms.ndim == 1:
                ms = ms.reshape(1, -1)
            paths.append(c.path)
            for row in ms:
                rows.append(row)
                group_ids.append(cand_idx)
            cand_idx += 1

        if not rows:
            empty = np.empty((0, 0), dtype=dtype)
            return empty, [], np.array([], dtype=np.int64)

        matrix = np.stack(rows).astype(dtype)
        return matrix, paths, np.array(group_ids, dtype=np.int64)

    @staticmethod
    def _maxpool_scores(
        flat_scores: np.ndarray,
        group_ids: np.ndarray,
        paths: List[str],
    ) -> List[ScoredCandidate]:
        """Max-pool flat similarity scores back to per-candidate level."""
        n_cands = len(paths)
        best = np.full(n_cands, -np.inf, dtype=np.float32)
        for i, gid in enumerate(group_ids):
            if flat_scores[i] > best[gid]:
                best[gid] = flat_scores[i]

        scored = [
            ScoredCandidate(path=paths[ci], score=float(best[ci]),
                            round_scores=[float(best[ci])])
            for ci in range(n_cands)
            if best[ci] > -np.inf
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    # ------------------------------------------------------------------ #
    # Text-to-image ranking (multi-query × multi-scale)
    # ------------------------------------------------------------------ #

    def _batch_rank(
        self,
        candidates: List[ImageCandidate],
        queries: List[str],
    ) -> List[ScoredCandidate]:
        """Rank candidates by max-pooled SigLIP2 similarity.

        ``score = max_{q in queries, s in scales} sim(q, image_s)``

        Uses SIMD-accelerated batch search when USearch is available;
        otherwise falls back to a vectorised NumPy implementation.
        """
        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(queries)   # (Q, D)
        embed_dim = text_matrix.shape[1]
        dtype = preferred_dtype()
        backend = "USearch-SIMD" if _vs_accel() else "NumPy"
        print(
            f"    [SigLIPScout] Encoded {len(queries)} query variant(s), "
            f"dim={embed_dim}, backend={backend}, dtype={dtype}"
        )

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="text-search",
        )

        matrix, paths, group_ids = self._flatten_multiscale(
            candidates, all_embeds, dtype=dtype,
        )

        if matrix.shape[0] == 0:
            return []

        flat_scores = multi_query_max_pool(matrix, text_matrix.astype(dtype))
        scored = self._maxpool_scores(flat_scores, group_ids, paths)

        elapsed = time.time() - t0
        print(
            f"    [SigLIPScout] Ranked {len(scored)} images "
            f"({matrix.shape[0]} vectors) in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Image-to-image ranking (multi-scale)
    # ------------------------------------------------------------------ #

    def _image_rank(
        self,
        candidates: List[ImageCandidate],
        query_images: List[Image.Image],
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank by SigLIP2 image-to-image similarity (batch vector search)."""
        _ensure_model(self._model_id)
        t0 = time.time()

        query_matrix = self._encode_images_global(query_images)  # (Nq, D)
        embed_dim = query_matrix.shape[1]
        dtype = preferred_dtype()

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="image-search",
            global_only=fast,
        )

        matrix, paths, group_ids = self._flatten_multiscale(
            candidates, all_embeds, dtype=dtype,
        )
        if matrix.shape[0] == 0:
            return []

        flat_scores = multi_query_max_pool(matrix, query_matrix.astype(dtype))
        scored = self._maxpool_scores(flat_scores, group_ids, paths)

        elapsed = time.time() - t0
        print(
            f"    [SigLIPScout] Image-to-image ranked "
            f"{len(scored)} ({matrix.shape[0]} vectors) in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Hybrid ranking (multi-scale + multi-query)
    # ------------------------------------------------------------------ #

    def _hybrid_rank(
        self,
        candidates: List[ImageCandidate],
        text_queries: List[str],
        image_queries: List[Image.Image],
        text_weight: float = 0.5,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Weighted combination of text and image SigLIP2 scores (batch search)."""
        _ensure_model(self._model_id)
        t0 = time.time()

        text_matrix = self._encode_texts(text_queries)            # (Qt, D)
        image_matrix = self._encode_images_global(image_queries)  # (Qi, D)
        embed_dim = text_matrix.shape[1]
        dtype = preferred_dtype()

        all_embeds = self._get_candidate_multiscale(
            candidates, embed_dim, label="hybrid",
            global_only=fast,
        )

        matrix, paths, group_ids = self._flatten_multiscale(
            candidates, all_embeds, dtype=dtype,
        )
        if matrix.shape[0] == 0:
            return []

        t_flat = multi_query_max_pool(matrix, text_matrix.astype(dtype))
        i_flat = multi_query_max_pool(matrix, image_matrix.astype(dtype))

        img_w = 1.0 - text_weight
        n_cands = len(paths)
        t_best = np.full(n_cands, -np.inf, dtype=np.float32)
        i_best = np.full(n_cands, -np.inf, dtype=np.float32)
        for idx, gid in enumerate(group_ids):
            if t_flat[idx] > t_best[gid]:
                t_best[gid] = t_flat[idx]
            if i_flat[idx] > i_best[gid]:
                i_best[gid] = i_flat[idx]

        scored: List[ScoredCandidate] = []
        for ci in range(n_cands):
            ts = float(t_best[ci])
            ims = float(i_best[ci])
            if ts <= -np.inf:
                continue
            combined = text_weight * ts + img_w * ims
            scored.append(ScoredCandidate(
                path=paths[ci], score=combined,
                round_scores=[ts, ims],
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        elapsed = time.time() - t0
        print(
            f"    [SigLIPScout] Hybrid ranked {len(scored)} "
            f"({matrix.shape[0]} vectors) in {elapsed:.1f}s"
        )
        return scored

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def scout(
        self,
        candidates: List[ImageCandidate],
        query: str,
        top_k: int = 10,
        expanded_queries: Optional[List[str]] = None,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank candidates using SigLIP2 scoring with query expansion.

        When *fast* is ``True``, uses global-crop-only encoding with
        batch-wise early stopping — typically 6-10× faster than the
        full multi-scale pipeline.

        Args:
            expanded_queries: Additional semantic variants of *query* for
                max-pooled scoring.  If ``None``, only the primary query
                is used.
            fast: Use greedy ranking with early stopping (FAST mode).

        Returns:
            Up to *top_k* :class:`ScoredCandidate`, sorted by score desc.
        """
        if not candidates:
            return []

        queries = [query]
        if expanded_queries:
            queries.extend(expanded_queries)

        mode = "fast (global-only + early-stop)" if fast else "full (multi-scale)"
        print(
            f"  [SigLIPScout] {mode} ranking: {len(candidates)} "
            f"candidates, {len(queries)} query variant(s) → top {top_k}"
        )

        if fast:
            ranked = await asyncio.to_thread(
                self._batch_rank_fast, candidates, queries, top_k,
            )
        else:
            ranked = await asyncio.to_thread(
                self._batch_rank, candidates, queries,
            )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top_score = ranked[0].score
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(ranked[0].path)} "
            f"({top_score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    async def scout_by_image(
        self,
        candidates: List[ImageCandidate],
        query_images: List[Image.Image],
        top_k: int = 10,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank candidates by SigLIP2 image-to-image similarity."""
        if not candidates:
            return []

        mode = "fast" if fast else "full"
        print(
            f"  [SigLIPScout] Image-to-image ranking ({mode}): "
            f"{len(candidates)} candidates → top {top_k}"
        )

        ranked = await asyncio.to_thread(
            self._image_rank, candidates, query_images, fast,
        )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top_score = ranked[0].score
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(ranked[0].path)} "
            f"({top_score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    async def scout_hybrid(
        self,
        candidates: List[ImageCandidate],
        text_query: str,
        image_queries: List[Image.Image],
        top_k: int = 10,
        text_weight: float = 0.5,
        expanded_queries: Optional[List[str]] = None,
        fast: bool = False,
    ) -> List[ScoredCandidate]:
        """Rank by combined text + image SigLIP2 similarity."""
        if not candidates:
            return []

        text_queries = [text_query]
        if expanded_queries:
            text_queries.extend(expanded_queries)

        mode = "fast" if fast else "full"
        print(
            f"  [SigLIPScout] Hybrid ranking ({mode}): "
            f"{len(candidates)} candidates, "
            f"text={len(text_queries)} variants, "
            f"images={len(image_queries)} → top {top_k}"
        )

        ranked = await asyncio.to_thread(
            self._hybrid_rank,
            candidates, text_queries, image_queries, text_weight, fast,
        )

        if not ranked:
            print("  [SigLIPScout] No candidates scored")
            return []

        top = ranked[0]
        kth = min(top_k - 1, len(ranked) - 1)
        print(
            f"  [SigLIPScout] Done: #1={os.path.basename(top.path)} "
            f"({top.score:.3f}), #{kth + 1}={ranked[kth].score:.3f}"
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------ #
    # Legacy MCS patch methods (kept for localisation tasks)
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_saliency(image: np.ndarray) -> np.ndarray:
        """Gradient-magnitude saliency map normalised to [0, 1]."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        peak = mag.max()
        return (mag / peak) if peak > 0 else mag

    def sample_patches(
        self,
        image: np.ndarray,
        saliency: np.ndarray,
        n_patches: int,
    ) -> List[np.ndarray]:
        """Sample *n_patches* from *image*, weighted by *saliency*."""
        h, w = image.shape[:2]
        ps = self.patch_size
        if h < ps or w < ps:
            return [cv2.resize(image, (ps, ps))]
        valid_h, valid_w = h - ps, w - ps
        if valid_h <= 0 or valid_w <= 0:
            return [cv2.resize(image, (ps, ps))]

        grid_h = min(valid_h, 50)
        grid_w = min(valid_w, 50)
        sal_grid = cv2.resize(
            saliency[:valid_h, :valid_w], (grid_w, grid_h),
        )
        flat = sal_grid.flatten() + 1e-8
        prob = flat / flat.sum()

        n_samples = min(n_patches, len(prob))
        indices = np.random.choice(
            len(prob), size=n_samples, replace=False, p=prob,
        )

        patches: List[np.ndarray] = []
        for idx in indices:
            gy, gx = divmod(int(idx), grid_w)
            y = int(gy * valid_h / grid_h)
            x = int(gx * valid_w / grid_w)
            patch = image[y: y + ps, x: x + ps]
            if patch.shape[0] == ps and patch.shape[1] == ps:
                patches.append(patch)
        return patches
