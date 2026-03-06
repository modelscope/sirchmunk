"""SIMD-optimized brute-force vector similarity search.

Provides a thin abstraction over `USearch <https://github.com/unum-cloud/USearch>`_
``exact=True`` mode for hardware-accelerated brute-force vector search.
When USearch is not installed, falls back transparently to NumPy.

All public functions accept plain NumPy arrays and return NumPy results,
so callers never depend on USearch directly.

Typical usage::

    from sirchmunk.vision.vector_search import top_k_similar, batch_cosine

    # Find top-10 most similar rows in *matrix* to *query*
    keys, scores = top_k_similar(matrix, query, k=10)

    # Compute cosine similarity of every row in *matrix* against *query*
    similarities = batch_cosine(matrix, query)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# USearch availability probe (once at import time)
# ---------------------------------------------------------------------- #

_USEARCH_AVAILABLE = False

try:
    from usearch.index import search as _usearch_search, MetricKind
    _USEARCH_AVAILABLE = True
except ImportError:
    _usearch_search = None  # type: ignore[assignment]
    MetricKind = None  # type: ignore[assignment,misc]


class Metric(Enum):
    """Distance / similarity metric for vector search."""
    INNER_PRODUCT = "ip"
    COSINE = "cos"
    L2 = "l2sq"


# ---------------------------------------------------------------------- #
# Public helpers
# ---------------------------------------------------------------------- #

def is_accelerated() -> bool:
    """Return ``True`` if USearch SIMD acceleration is available."""
    return _USEARCH_AVAILABLE


def preferred_dtype(fp16: bool = True) -> np.dtype:
    """Return the preferred storage dtype.

    When *fp16* is ``True`` **and** USearch is available, returns
    ``float16`` for 2× memory savings and SIMD throughput.  Otherwise
    returns ``float32`` for NumPy compatibility.
    """
    if fp16 and _USEARCH_AVAILABLE:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


# ---------------------------------------------------------------------- #
# Core: top-k similarity search
# ---------------------------------------------------------------------- #

def top_k_similar(
    matrix: np.ndarray,
    query: np.ndarray,
    k: int,
    metric: Metric = Metric.INNER_PRODUCT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the *k* most similar rows in *matrix* to *query*.

    Args:
        matrix: ``(N, D)`` contiguous array of database vectors.
        query:  ``(D,)`` or ``(Q, D)`` query vector(s).  When ``(Q, D)``
                is given, each query is searched independently and the
                union of results is returned (deduplicated, best score
                wins).
        k:      Maximum number of results.
        metric: Similarity metric (default: inner product — equivalent
                to cosine for L2-normalised vectors).

    Returns:
        ``(indices, scores)`` — both ``(K,)`` int64 / float32 arrays,
        sorted by descending score.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    k = min(k, matrix.shape[0])

    if query.ndim == 1:
        query = query.reshape(1, -1)

    if _USEARCH_AVAILABLE:
        return _usearch_top_k(matrix, query, k, metric)
    return _numpy_top_k(matrix, query, k, metric)


def batch_cosine(
    matrix: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Cosine similarity of every row in *matrix* against *query*.

    Both inputs are assumed **L2-normalised** (so dot product = cosine).

    Args:
        matrix: ``(N, D)`` array.
        query:  ``(D,)`` vector.

    Returns:
        ``(N,)`` float32 similarities.
    """
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return np.array([], dtype=np.float32)
    q = np.asarray(query, dtype=np.float32).ravel()
    m = np.asarray(matrix, dtype=np.float32)
    return (m @ q).astype(np.float32)


def multi_query_max_pool(
    matrix: np.ndarray,
    queries: np.ndarray,
) -> np.ndarray:
    """Max-pooled similarity: for each row, take the best score across queries.

    Args:
        matrix:  ``(N, D)``
        queries: ``(Q, D)``

    Returns:
        ``(N,)`` float32 — ``max over Q`` of ``matrix @ queries.T``.
    """
    if matrix.ndim != 2 or queries.ndim != 2:
        return np.array([], dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    q = np.asarray(queries, dtype=np.float32)
    sim = m @ q.T  # (N, Q)
    return sim.max(axis=1).astype(np.float32)


# ---------------------------------------------------------------------- #
# USearch backend
# ---------------------------------------------------------------------- #

def _usearch_top_k(
    matrix: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: Metric,
) -> Tuple[np.ndarray, np.ndarray]:
    """USearch SIMD-accelerated exact brute-force search."""
    metric_kind = {
        Metric.INNER_PRODUCT: MetricKind.IP,
        Metric.COSINE: MetricKind.Cos,
        Metric.L2: MetricKind.L2sq,
    }[metric]

    n_queries = queries.shape[0]

    if n_queries == 1:
        matches = _usearch_search(
            matrix, queries[0], k, metric_kind, exact=True,
        )
        indices = np.asarray(matches.keys, dtype=np.int64)
        scores = np.asarray(matches.distances, dtype=np.float32)
        if metric in (Metric.INNER_PRODUCT, Metric.COSINE):
            scores = 1.0 - scores
        return indices[:k], scores[:k]

    best: dict = {}
    for qi in range(n_queries):
        matches = _usearch_search(
            matrix, queries[qi], k, metric_kind, exact=True,
        )
        keys = np.asarray(matches.keys, dtype=np.int64)
        dists = np.asarray(matches.distances, dtype=np.float32)
        if metric in (Metric.INNER_PRODUCT, Metric.COSINE):
            dists = 1.0 - dists
        for idx, sc in zip(keys, dists):
            idx_int = int(idx)
            if idx_int not in best or sc > best[idx_int]:
                best[idx_int] = float(sc)

    if not best:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    pairs = sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]
    indices = np.array([p[0] for p in pairs], dtype=np.int64)
    scores = np.array([p[1] for p in pairs], dtype=np.float32)
    return indices, scores


# ---------------------------------------------------------------------- #
# NumPy fallback
# ---------------------------------------------------------------------- #

def _numpy_top_k(
    matrix: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: Metric,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy brute-force search (fallback)."""
    m = np.asarray(matrix, dtype=np.float32)
    q = np.asarray(queries, dtype=np.float32)

    if metric == Metric.L2:
        diff = m[np.newaxis, :, :] - q[:, np.newaxis, :]  # (Q, N, D)
        dists = (diff ** 2).sum(axis=-1)                   # (Q, N)
        best_per_row = dists.min(axis=0)                   # (N,)
        top_idx = np.argpartition(best_per_row, k)[:k]
        top_idx = top_idx[np.argsort(best_per_row[top_idx])]
        return top_idx.astype(np.int64), (-best_per_row[top_idx]).astype(np.float32)

    sim = m @ q.T          # (N, Q)
    best_sim = sim.max(axis=1)  # (N,)
    if k >= len(best_sim):
        top_idx = np.argsort(-best_sim)
    else:
        top_idx = np.argpartition(-best_sim, k)[:k]
        top_idx = top_idx[np.argsort(-best_sim[top_idx])]
    return top_idx.astype(np.int64), best_sim[top_idx].astype(np.float32)
