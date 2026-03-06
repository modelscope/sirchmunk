"""SigLIPEmbedOperator — high-level semantic scoring via cached SigLIP2 embeddings.

Activates only when the image has a pre-computed ``siglip_embed`` (backfilled
after the first search's Phase 2) and the query's text embedding is available
on the constraint.  When active, this operator dominates the fusion score
(weight=5.0), giving Visual Grep near-SigLIP semantic discrimination.

On the first search (cold start), this operator is inactive and the
pipeline falls back to the original 7 low-level operators.

Batch scoring (:meth:`batch_score`) uses SIMD-accelerated vector search
from :mod:`sirchmunk.vision.vector_search` when available, computing all
candidate scores in a single vectorised pass instead of a Python loop.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from sirchmunk.schema.vision import ImageSignature, VisualConstraint
from sirchmunk.vision.vector_search import batch_cosine
from .base import GrepOperator


class SigLIPEmbedOperator(GrepOperator):
    """Cosine similarity between cached image embedding and query text embedding."""

    @property
    def name(self) -> str:
        return "siglip_embed"

    @property
    def weight(self) -> float:
        return 5.0

    def is_applicable(self, con: VisualConstraint) -> bool:
        return con.clip_query_embed is not None

    def score(self, sig: ImageSignature, con: VisualConstraint) -> float:
        if not sig.siglip_embed or con.clip_query_embed is None:
            return 0.5

        img_vec = np.asarray(sig.siglip_embed, dtype=np.float32)
        txt_vec = np.asarray(con.clip_query_embed, dtype=np.float32)

        norm_i = np.linalg.norm(img_vec)
        norm_t = np.linalg.norm(txt_vec)
        if norm_i < 1e-8 or norm_t < 1e-8:
            return 0.5

        cosine = float(np.dot(img_vec, txt_vec) / (norm_i * norm_t))
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    # ------------------------------------------------------------------ #
    # Batch scoring — SIMD-accelerated
    # ------------------------------------------------------------------ #

    @staticmethod
    def batch_score(
        signatures: List[ImageSignature],
        con: VisualConstraint,
    ) -> Dict[str, float]:
        """Score all *signatures* against *con* in one vectorised pass.

        Returns ``{path: score}`` for every signature that has a
        ``siglip_embed``.  Signatures without embeddings are omitted
        (caller should fall back to 0.5).
        """
        if con.clip_query_embed is None:
            return {}

        txt_vec = np.asarray(con.clip_query_embed, dtype=np.float32)
        norm_t = np.linalg.norm(txt_vec)
        if norm_t < 1e-8:
            return {}
        txt_unit = txt_vec / norm_t

        paths: List[str] = []
        rows: List[np.ndarray] = []
        for sig in signatures:
            if not sig.siglip_embed:
                continue
            vec = np.asarray(sig.siglip_embed, dtype=np.float32)
            norm_v = np.linalg.norm(vec)
            if norm_v < 1e-8:
                continue
            rows.append(vec / norm_v)
            paths.append(sig.path)

        if not rows:
            return {}

        matrix = np.stack(rows)  # (N, D)
        cosines = batch_cosine(matrix, txt_unit)  # (N,)
        result: Dict[str, float] = {}
        for i, p in enumerate(paths):
            result[p] = max(0.0, min(1.0, (float(cosines[i]) + 1.0) / 2.0))
        return result
