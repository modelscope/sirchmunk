"""VisionKnowledgeStore — DuckDB-backed persistent image knowledge.

Stores VLM-verified image metadata (captions, semantic tags, visual
features) and supports text-based full-text retrieval.  The store
evolves with each search: newly verified images are inserted
automatically, making repeated or similar queries instantaneous.

Additional tables:
  - ``image_signatures``  — cached visual signatures with mtime-based
    freshness tracking (eliminates cold-start re-signing).
  - ``operator_feedback`` — per-operator scoring records linked to VLM
    verdicts for weight adaptation (online learning).
  - ``query_patterns``    — learned per-pattern operator weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "it",
    "and", "or", "not", "with", "from", "by", "as", "be", "this", "that",
    "all", "any", "my", "me", "we", "our", "you", "your", "its",
    "photo", "image", "picture", "photos", "images", "pictures",
    "find", "show", "get", "give", "look",
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
    "帮", "帮我", "找", "找到", "找出", "所有", "全部", "请", "把",
    "图片", "照片", "图", "张", "这", "那", "给", "出", "来",
})

_DDL = """\
CREATE TABLE IF NOT EXISTS image_knowledge (
    path          VARCHAR PRIMARY KEY,
    caption       VARCHAR,
    tags          VARCHAR,
    phash         VARCHAR,
    color_moments VARCHAR,
    verified_at   VARCHAR,
    confidence    DOUBLE,
    query_history VARCHAR
)
"""

_SIG_DDL = """\
CREATE TABLE IF NOT EXISTS image_signatures (
    path           VARCHAR PRIMARY KEY,
    mtime          DOUBLE,
    file_size      BIGINT,
    signature_json VARCHAR
)
"""

_FEEDBACK_DDL = """\
CREATE TABLE IF NOT EXISTS operator_feedback (
    path           VARCHAR,
    query          VARCHAR,
    operator_name  VARCHAR,
    op_score       DOUBLE,
    vlm_match      BOOLEAN,
    vlm_confidence DOUBLE,
    created_at     VARCHAR
)
"""

_PATTERN_DDL = """\
CREATE TABLE IF NOT EXISTS query_patterns (
    pattern          VARCHAR PRIMARY KEY,
    operator_weights VARCHAR,
    sample_count     INTEGER,
    clip_threshold   DOUBLE DEFAULT 0.0,
    updated_at       VARCHAR
)
"""


class VisionKnowledgeStore:
    """Persistent image knowledge backed by DuckDB."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(_DDL)
        self._conn.execute(_SIG_DDL)
        self._conn.execute(_FEEDBACK_DDL)
        self._conn.execute(_PATTERN_DDL)

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    async def store(self, knowledge: Any) -> None:
        """Insert or replace a knowledge entry.

        Args:
            knowledge: An ``ImageKnowledge`` dataclass instance.
        """
        print(
            f"    [VisionKnowledgeStore] Storing: {knowledge.path} "
            f"(confidence={knowledge.confidence:.2f}, "
            f"tags={knowledge.semantic_tags[:3]})"
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO image_knowledge
                (path, caption, tags, phash, color_moments,
                 verified_at, confidence, query_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                knowledge.path,
                knowledge.caption,
                json.dumps(knowledge.semantic_tags, ensure_ascii=False),
                knowledge.phash,
                json.dumps(knowledge.color_moments),
                knowledge.verified_at,
                knowledge.confidence,
                json.dumps(knowledge.query_history, ensure_ascii=False),
            ],
        )

    async def update_query_history(
        self, path: str, query: str, max_history: int = 10,
    ) -> None:
        """Append *query* to an existing entry's history (FIFO capped)."""
        existing = await self.get_by_path(path)
        if existing is None:
            return
        history = existing.query_history
        if query not in history:
            history.append(query)
        if len(history) > max_history:
            history = history[-max_history:]
        self._conn.execute(
            "UPDATE image_knowledge SET query_history = ? WHERE path = ?",
            [json.dumps(history, ensure_ascii=False), path],
        )

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    async def search_by_exact_query(
        self,
        query: str,
        limit: int = 20,
    ) -> list:
        """Return entries whose ``query_history`` contains the exact *query*."""
        print(f"    [VisionKnowledgeStore] Exact-query search: '{query[:60]}'")
        rows = self._conn.execute(
            "SELECT path, caption, tags, phash, color_moments,"
            "       verified_at, confidence, query_history "
            "FROM image_knowledge "
            "WHERE query_history LIKE ? "
            "ORDER BY confidence DESC LIMIT ?",
            [f"%{query}%", limit * 3],
        ).fetchall()
        results = []
        for r in rows:
            k = _row_to_knowledge(r)
            if query in k.query_history:
                results.append(k)
        print(f"    [VisionKnowledgeStore] Exact-query hits: {len(results)}")
        return results[:limit]

    async def search_by_vector(
        self,
        query_embed: Any,
        limit: int = 20,
        min_similarity: float = 0.3,
    ) -> list:
        """Semantic vector search over stored SigLIP embeddings.

        Loads all cached SigLIP embeddings from ``image_signatures``,
        joins with ``image_knowledge``, and ranks by cosine similarity
        against *query_embed* using SIMD-accelerated brute-force search.

        Falls back to empty results if no embeddings are stored.

        Args:
            query_embed: ``(D,)`` L2-normalised query embedding.
            limit:       Maximum number of results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of ``ImageKnowledge`` sorted by similarity (desc).
        """
        import numpy as np
        from sirchmunk.vision.vector_search import top_k_similar, Metric

        print(f"    [VisionKnowledgeStore] Semantic vector search (limit={limit})")

        rows = self._conn.execute(
            "SELECT s.path, s.signature_json, "
            "       k.caption, k.tags, k.phash, k.color_moments, "
            "       k.verified_at, k.confidence, k.query_history "
            "FROM image_signatures s "
            "INNER JOIN image_knowledge k ON s.path = k.path",
        ).fetchall()

        if not rows:
            print("    [VisionKnowledgeStore] No embeddings with knowledge — skip")
            return []

        paths: list = []
        embeds: list = []
        row_map: dict = {}
        for r in rows:
            path, sig_json = r[0], r[1]
            try:
                sig_dict = json.loads(sig_json)
            except (json.JSONDecodeError, TypeError):
                continue
            emb = sig_dict.get("siglip_embed")
            if not emb or not isinstance(emb, list):
                continue
            vec = np.array(emb, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm < 1e-8:
                continue
            embeds.append(vec / norm)
            paths.append(path)
            row_map[path] = r

        if not embeds:
            print("    [VisionKnowledgeStore] No SigLIP embeddings found")
            return []

        matrix = np.stack(embeds)  # (N, D)
        q = np.asarray(query_embed, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return []
        q = q / q_norm

        indices, scores = top_k_similar(
            matrix, q, min(limit, len(paths)), Metric.INNER_PRODUCT,
        )

        results = []
        for idx, sc in zip(indices, scores):
            if float(sc) < min_similarity:
                continue
            path = paths[int(idx)]
            r = row_map[path]
            k = _row_to_knowledge((
                r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
            ))
            results.append(k)

        print(
            f"    [VisionKnowledgeStore] Semantic search: "
            f"{len(results)} hits from {len(embeds)} indexed "
            f"(top={float(scores[0]):.3f})" if len(scores) > 0 else
            f"    [VisionKnowledgeStore] Semantic search: 0 hits"
        )
        return results

    async def search_text(
        self,
        query: str,
        limit: int = 20,
    ) -> list:
        """Full-text search over captions and tags with stopword filtering."""
        print(f"    [VisionKnowledgeStore] Text search: '{query[:60]}'")
        words = [
            w.strip().lower()
            for w in query.split()
            if len(w.strip()) > 1 and w.strip().lower() not in _STOPWORDS
        ]
        if len(words) < 2:
            print(
                f"    [VisionKnowledgeStore] Too few meaningful words "
                f"({words}) — returning empty"
            )
            return []

        conditions = []
        params: list = []
        for word in words:
            conditions.append(
                "(LOWER(caption) LIKE ? OR LOWER(tags) LIKE ?)"
            )
            params.extend([f"%{word}%", f"%{word}%"])

        sql = (
            "SELECT path, caption, tags, phash, color_moments,"
            "       verified_at, confidence, query_history "
            "FROM image_knowledge "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY confidence DESC "
            "LIMIT ?"
        )
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        print(f"    [VisionKnowledgeStore] Found {len(rows)} matching entries")
        return [_row_to_knowledge(r) for r in rows]

    async def get_by_path(self, path: str) -> Any:
        """Retrieve knowledge for a specific image file."""
        rows = self._conn.execute(
            "SELECT path, caption, tags, phash, color_moments,"
            "       verified_at, confidence, query_history "
            "FROM image_knowledge WHERE path = ?",
            [path],
        ).fetchall()
        return _row_to_knowledge(rows[0]) if rows else None

    async def get_known_paths(self) -> Set[str]:
        """Return the set of all indexed image paths."""
        rows = self._conn.execute(
            "SELECT path FROM image_knowledge",
        ).fetchall()
        return {r[0] for r in rows}

    async def count(self) -> int:
        """Return total number of stored entries."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM image_knowledge",
        ).fetchone()[0]

    # ------------------------------------------------------------------ #
    # Signature cache
    # ------------------------------------------------------------------ #

    def get_cached_signatures(
        self,
        paths_with_mtime: List[Tuple[str, float, int]],
    ) -> Tuple[dict, list]:
        """Bulk lookup cached signatures with mtime freshness check.

        Args:
            paths_with_mtime: ``[(path, mtime, file_size), ...]``

        Returns:
            ``(cached, stale)`` — cached signatures and stale entries
            that need re-signing.
        """
        if not paths_with_mtime:
            return {}, []

        cached: dict = {}
        stale: list = []

        path_map = {p: (mt, sz) for p, mt, sz in paths_with_mtime}
        placeholders = ", ".join(["?"] * len(path_map))
        rows = self._conn.execute(
            f"SELECT path, mtime, file_size, signature_json "
            f"FROM image_signatures WHERE path IN ({placeholders})",
            list(path_map.keys()),
        ).fetchall()

        found: set = set()
        for path, db_mtime, db_size, sig_json in rows:
            expected_mt, expected_sz = path_map[path]
            found.add(path)
            if abs(db_mtime - expected_mt) < 0.01 and db_size == expected_sz:
                sig = _json_to_signature(sig_json, path)
                if sig is not None:
                    cached[path] = sig
                    continue
            stale.append((path, expected_mt, expected_sz))

        for p, mt, sz in paths_with_mtime:
            if p not in found:
                stale.append((p, mt, sz))

        return cached, stale

    def store_signatures_batch(
        self,
        entries: list,
    ) -> None:
        """Persist a batch of image signatures.

        Args:
            entries: ``[(path, mtime, file_size, signature), ...]``
        """
        if not entries:
            return
        for path, mtime, file_size, sig in entries:
            sig_json = json.dumps(asdict(sig), ensure_ascii=False)
            self._conn.execute(
                "INSERT OR REPLACE INTO image_signatures "
                "(path, mtime, file_size, signature_json) "
                "VALUES (?, ?, ?, ?)",
                [path, mtime, file_size, sig_json],
            )
        print(
            f"    [VisionKnowledgeStore] Persisted {len(entries)} "
            f"image signatures"
        )

    def get_signature_count(self) -> int:
        """Return total number of cached signatures."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM image_signatures",
        ).fetchone()[0]

    def backfill_siglip_embeds(
        self,
        embeds: Dict[str, List[float]],
    ) -> int:
        """Write SigLIP2 global embeddings into cached image signatures.

        Returns the number of records updated.
        """
        if not embeds:
            return 0
        updated = 0
        for path, embed in embeds.items():
            rows = self._conn.execute(
                "SELECT signature_json FROM image_signatures WHERE path = ?",
                [path],
            ).fetchall()
            if not rows:
                continue
            try:
                sig_dict = json.loads(rows[0][0])
            except (json.JSONDecodeError, TypeError):
                continue
            sig_dict["siglip_embed"] = embed
            self._conn.execute(
                "UPDATE image_signatures SET signature_json = ? WHERE path = ?",
                [json.dumps(sig_dict, ensure_ascii=False), path],
            )
            updated += 1
        if updated:
            print(
                f"    [VisionKnowledgeStore] Backfilled SigLIP embeddings "
                f"for {updated}/{len(embeds)} signatures"
            )
        return updated

    # ------------------------------------------------------------------ #
    # Operator feedback
    # ------------------------------------------------------------------ #

    def store_feedback(
        self,
        records: List[Dict[str, Any]],
    ) -> None:
        """Store operator scoring records linked to VLM verification verdicts."""
        if not records:
            return
        for r in records:
            self._conn.execute(
                "INSERT INTO operator_feedback "
                "(path, query, operator_name, op_score, "
                " vlm_match, vlm_confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    r["path"], r["query"], r["operator_name"],
                    r["op_score"], r["vlm_match"],
                    r["vlm_confidence"], r["created_at"],
                ],
            )

    def compute_operator_accuracy(
        self,
        min_samples: int = 20,
    ) -> Dict[str, float]:
        """Compute per-operator prediction accuracy from historical feedback."""
        rows = self._conn.execute(
            "SELECT operator_name, op_score, vlm_match "
            "FROM operator_feedback",
        ).fetchall()

        from collections import defaultdict
        per_op: Dict[str, list] = defaultdict(list)
        for name, score, match in rows:
            per_op[name].append((score, bool(match)))

        result: Dict[str, float] = {}
        for name, entries in per_op.items():
            if len(entries) < min_samples:
                continue
            scores = [s for s, _ in entries]
            median = float(sorted(scores)[len(scores) // 2])
            above_median = [(s, m) for s, m in entries if s > median]
            if not above_median:
                continue
            tp = sum(1 for _, m in above_median if m)
            result[name] = tp / len(above_median)

        return result

    def get_learned_weights(
        self,
        pattern: str,
    ) -> Optional[Dict[str, float]]:
        """Retrieve learned operator weights for a query pattern."""
        rows = self._conn.execute(
            "SELECT operator_weights FROM query_patterns WHERE pattern = ?",
            [pattern],
        ).fetchall()
        if not rows:
            return None
        try:
            return json.loads(rows[0][0])
        except (json.JSONDecodeError, TypeError):
            return None

    def update_pattern_weights(
        self,
        pattern: str,
        weights: Dict[str, float],
        clip_threshold: float = 0.0,
    ) -> None:
        """Upsert learned operator weights for a query pattern."""
        from datetime import datetime
        count = self._conn.execute(
            "SELECT sample_count FROM query_patterns WHERE pattern = ?",
            [pattern],
        ).fetchall()
        cur = count[0][0] if count else 0
        self._conn.execute(
            "INSERT OR REPLACE INTO query_patterns "
            "(pattern, operator_weights, sample_count, "
            " clip_threshold, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                pattern,
                json.dumps(weights),
                cur + 1,
                clip_threshold,
                datetime.now().isoformat(),
            ],
        )

    def get_hard_negatives(
        self,
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return high-confidence VLM rejections (hard negatives)."""
        rows = self._conn.execute(
            "SELECT path, query, op_score, vlm_confidence "
            "FROM operator_feedback "
            "WHERE vlm_match = false AND vlm_confidence >= ? "
            "ORDER BY vlm_confidence DESC LIMIT ?",
            [min_confidence, limit],
        ).fetchall()
        return [
            {"path": r[0], "query": r[1], "op_score": r[2],
             "vlm_confidence": r[3]}
            for r in rows
        ]

    def compute_clip_rejection_threshold(
        self,
        percentile: float = 95.0,
    ) -> Optional[float]:
        """Compute a CLIP/SigLIP score cutoff from hard negative distribution."""
        rows = self._conn.execute(
            "SELECT op_score FROM operator_feedback "
            "WHERE vlm_match = false AND operator_name = 'siglip_score'",
        ).fetchall()
        if len(rows) < 10:
            return None
        import numpy as np
        scores = [r[0] for r in rows]
        return float(np.percentile(scores, percentile))


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _json_loads(text: Optional[str], default=None):
    if not text:
        return default if default is not None else []
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []


def _row_to_knowledge(row):
    """Convert a DB row tuple to an ImageKnowledge dataclass."""
    from sirchmunk.schema.vision import ImageKnowledge
    return ImageKnowledge(
        path=row[0],
        caption=row[1] or "",
        semantic_tags=_json_loads(row[2], []),
        phash=row[3] or "",
        color_moments=_json_loads(row[4], []),
        verified_at=row[5] or "",
        confidence=row[6] or 0.0,
        query_history=_json_loads(row[7], []),
    )


def _json_to_signature(
    sig_json: str,
    fallback_path: str,
):
    """Deserialise a JSON string back to an ImageSignature."""
    from sirchmunk.schema.vision import ImageSignature
    try:
        d = json.loads(sig_json)
        return ImageSignature(
            path=d.get("path", fallback_path),
            phash=d.get("phash", ""),
            color_moments=d.get("color_moments", []),
            file_size=d.get("file_size", 0),
            width=d.get("width", 0),
            height=d.get("height", 0),
            exif=d.get("exif", {}),
            block_hashes=d.get("block_hashes", []),
            block_hue_means=d.get("block_hue_means", []),
            center_moments=d.get("center_moments", []),
            edge_moments=d.get("edge_moments", []),
            gist_descriptor=d.get("gist_descriptor", []),
            entropy=d.get("entropy", 0.0),
            edge_density=d.get("edge_density", 0.0),
            siglip_embed=d.get("siglip_embed", []),
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
