"""Compile-time summary index for embedding + BM25 fallback retrieval.

This module provides a lightweight, file-level index that combines:
- Semantic similarity via pre-computed embeddings (384-dim MiniLM)
- Lexical matching via BM25 scoring (TokenizerUtil segmentation)

Used ONLY as a fallback when rga keyword search returns zero results.
"""

import json
import math
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SummaryIndexEntry:
    """Single file entry in the summary index."""
    file_path: str
    summary: str
    embedding: Optional[List[float]] = None     # 384-dim, pre-normalized
    tokens: Optional[List[str]] = None           # TokenizerUtil.segment() output
    token_freqs: Optional[Dict[str, int]] = None # pre-computed term frequencies


class CompileSummaryIndex:
    """Pre-computed summary index for hybrid embedding + BM25 fallback search.

    This index is built at compile time and loaded at search time.
    It provides a fallback retrieval mechanism when rga keyword search
    returns zero results, combining semantic similarity (embedding cosine)
    with lexical matching (BM25).

    The fusion algorithm uses Sigmoid Z-Score normalization:
    1. Compute raw scores from both channels
    2. Z-Score normalize each channel independently
    3. Weighted combination: alpha * z_embedding + (1-alpha) * z_bm25
    4. Sigmoid activation for final score
    """

    # BM25 parameters (Okapi BM25 standard defaults)
    _BM25_K1: float = 1.5
    _BM25_B: float = 0.75

    # Fusion parameters
    _DEFAULT_ALPHA: float = 0.5  # embedding weight; (1-alpha) = BM25 weight

    # Z-Score fallback for missing channel
    _MISSING_CHANNEL_Z: float = -3.0  # ~0.1 percentile

    def __init__(self, entries: List[SummaryIndexEntry]) -> None:
        self._entries = entries
        self._num_docs = len(entries)
        self._avg_doc_len = self._compute_avg_doc_len()
        self._doc_freqs: Dict[str, int] = self._compute_doc_freqs()

    def _compute_avg_doc_len(self) -> float:
        """Compute average document length (in tokens) across all entries."""
        lengths = [len(e.tokens or []) for e in self._entries]
        return sum(lengths) / max(1, len(lengths))

    def _compute_doc_freqs(self) -> Dict[str, int]:
        """Compute document frequency for each unique token."""
        df: Dict[str, int] = {}
        for entry in self._entries:
            if entry.token_freqs:
                for token in entry.token_freqs:
                    df[token] = df.get(token, 0) + 1
        return df

    @classmethod
    def load(cls, index_path: Path) -> Optional["CompileSummaryIndex"]:
        """Load index from JSON file. Returns None on failure."""
        try:
            if not index_path.exists():
                return None
            data = json.loads(index_path.read_text(encoding="utf-8"))
            entries = []
            for item in data.get("entries", []):
                entries.append(SummaryIndexEntry(
                    file_path=item["file_path"],
                    summary=item.get("summary", ""),
                    embedding=item.get("embedding"),
                    tokens=item.get("tokens"),
                    token_freqs=item.get("token_freqs"),
                ))
            if not entries:
                return None
            return cls(entries)
        except Exception as exc:
            logger.warning("Failed to load summary index from %s: %s", index_path, exc)
            return None

    def save(self, index_path: Path) -> None:
        """Persist index to JSON file."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "num_entries": len(self._entries),
            "entries": [
                {
                    "file_path": e.file_path,
                    "summary": e.summary,
                    "embedding": e.embedding,
                    "tokens": e.tokens,
                    "token_freqs": e.token_freqs,
                }
                for e in self._entries
            ],
        }
        index_path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Summary index saved: %d entries -> %s", len(self._entries), index_path)

    def search(
        self,
        query_embedding: Optional[List[float]],
        query_tokens: List[str],
        top_k: int = 5,
        alpha: float = _DEFAULT_ALPHA,
    ) -> List[Tuple[str, float]]:
        """Hybrid search combining embedding cosine similarity and BM25.

        Uses Sigmoid Z-Score fusion:
        1. Compute raw embedding cosine sim and BM25 score per document
        2. Z-Score normalize each channel
        3. Weighted linear combination
        4. Sigmoid activation

        Args:
            query_embedding: 384-dim query vector (None to use BM25 only).
            query_tokens: Tokenized query from TokenizerUtil.segment().
            top_k: Maximum number of results.
            alpha: Embedding weight in [0, 1]. BM25 weight = 1 - alpha.

        Returns:
            List of (file_path, fusion_score) sorted descending by score.
        """
        if not self._entries:
            return []

        # Compute raw scores
        emb_scores: List[Optional[float]] = []
        bm25_scores: List[float] = []

        has_embedding = query_embedding is not None

        for entry in self._entries:
            # Embedding channel
            if has_embedding and entry.embedding:
                emb_scores.append(self._cosine_similarity(query_embedding, entry.embedding))
            else:
                emb_scores.append(None)

            # BM25 channel
            bm25_scores.append(self._bm25_score(query_tokens, entry))

        # Z-Score normalization
        z_emb = self._z_score_normalize(emb_scores)
        z_bm25 = self._z_score_normalize(bm25_scores)

        # Sigmoid fusion
        results: List[Tuple[str, float]] = []
        for i, entry in enumerate(self._entries):
            z_e = z_emb[i] if z_emb[i] is not None else self._MISSING_CHANNEL_Z
            z_b = z_bm25[i] if z_bm25[i] is not None else self._MISSING_CHANNEL_Z

            combined = alpha * z_e + (1.0 - alpha) * z_b
            score = 1.0 / (1.0 + math.exp(-combined))
            results.append((entry.file_path, score))

        # Sort descending and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _bm25_score(self, query_tokens: List[str], entry: SummaryIndexEntry) -> float:
        """Compute BM25 score for a single document.

        Uses standard Okapi BM25 formula:
            score = sum over query terms:
                IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        """
        if not query_tokens or not entry.token_freqs:
            return 0.0

        dl = len(entry.tokens or [])
        score = 0.0

        for token in query_tokens:
            tf = entry.token_freqs.get(token, 0)
            if tf == 0:
                continue

            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = self._doc_freqs.get(token, 0)
            idf = math.log((self._num_docs - df + 0.5) / (df + 0.5) + 1.0)

            # TF component
            tf_component = (tf * (self._BM25_K1 + 1.0)) / (
                tf + self._BM25_K1 * (1.0 - self._BM25_B + self._BM25_B * dl / max(1.0, self._avg_doc_len))
            )

            score += idf * tf_component

        return score

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        When embeddings are pre-normalized (L2 norm = 1), this reduces
        to a simple dot product.
        """
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # Clamp to [-1, 1] for numerical safety
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def _z_score_normalize(scores: List[Optional[float]]) -> List[Optional[float]]:
        """Z-Score normalize a list of scores, preserving None entries.

        None entries remain None (handled as _MISSING_CHANNEL_Z at fusion).
        """
        valid = [s for s in scores if s is not None]
        if len(valid) < 2:
            # Not enough data points for meaningful normalization
            return scores

        mean = sum(valid) / len(valid)
        variance = sum((s - mean) ** 2 for s in valid) / len(valid)
        std = math.sqrt(variance) if variance > 0 else 1.0

        if std < 1e-9:
            # All scores identical — return zeros
            return [0.0 if s is not None else None for s in scores]

        return [(s - mean) / std if s is not None else None for s in scores]

    @property
    def num_entries(self) -> int:
        """Number of indexed documents."""
        return self._num_docs

    @property
    def has_embeddings(self) -> bool:
        """Whether any entry has a pre-computed embedding."""
        return any(e.embedding is not None for e in self._entries)
