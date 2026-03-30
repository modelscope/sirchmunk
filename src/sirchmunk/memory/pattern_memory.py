# Copyright (c) ModelScope Contributors. All rights reserved.
"""PatternMemory — query pattern → strategy mapping + reasoning chain templates.

Backed by JSON files for human-readability and easy schema evolution.
Thread-safe via a reentrant lock.

Storage layout::

    {base_dir}/query_patterns.json
    {base_dir}/reasoning_chains.json
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .base import MemoryStore
from .query_classifier import classify_query as _classify_query
from .schemas import (
    AbstractTrajectory,
    QueryPattern,
    ReasoningChain,
    SearchStrategy,
    StrategyDistillation,
    StrategyHint,
    compute_pattern_id,
    compute_pattern_id_at_level,
)

# HMRPL: minimum samples required per resolution level for confident predictions
_MIN_SAMPLES_BY_LEVEL = {4: 3, 3: 3, 2: 2, 1: 2, 0: 1}


class PatternMemory(MemoryStore):
    """Query pattern → strategy mapping and reasoning chain templates.

    Uses heuristic query classification at lookup time (zero-LLM-cost)
    and learns optimal parameters from feedback signals.
    """

    _MIN_SAMPLES_FOR_CONFIDENT = 3
    _MIN_SUCCESS_RATE = 0.4
    _SUCCESS_THRESHOLD = 0.5
    _EMA_ALPHA = 0.3
    _SAVE_MIN_INTERVAL = 5.0  # seconds between disk writes

    _MAX_TRAJECTORIES = 200
    _DISTILLATION_BATCH_SIZE = 3
    _LOOP_BUDGET_MIN_OBS = 3
    _LOOP_BUDGET_SUCCESS_RATE = 0.5

    def __init__(self, base_dir: Path, embedding_util: Any = None):
        self._base_dir = base_dir
        self._patterns_file = base_dir / "query_patterns.json"
        self._chains_file = base_dir / "reasoning_chains.json"
        self._trajectories_file = base_dir / "trajectories.json"
        self._distillations_file = base_dir / "distillations.json"
        self._patterns: Dict[str, QueryPattern] = {}
        self._chains: Dict[str, ReasoningChain] = {}
        self._trajectories: List[AbstractTrajectory] = []
        self._distillations: Dict[str, StrategyDistillation] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._last_save_time: float = 0.0
        self._embedding_util = embedding_util
        self._pattern_embeddings: Dict[str, List[float]] = {}

    # ── MemoryStore protocol ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "PatternMemory"

    def initialize(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def decay(self, now: Optional[datetime] = None) -> int:
        if not now:
            now = datetime.now(timezone.utc)
        count = 0
        with self._lock:
            for p in self._patterns.values():
                try:
                    updated = datetime.fromisoformat(p.updated_at)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    if (now - updated).days > 30 and p.success_rate > 0.1:
                        p.success_rate *= 0.95
                        count += 1
                except (ValueError, TypeError):
                    continue
        if count:
            self._mark_dirty()
        return count

    def cleanup(self, max_entries: Optional[int] = None) -> int:
        max_entries = max_entries or 500
        removed = 0
        with self._lock:
            if len(self._patterns) <= max_entries:
                return 0
            ranked = sorted(
                self._patterns.items(),
                key=lambda x: (x[1].success_rate, x[1].sample_count),
            )
            while len(self._patterns) > max_entries and ranked:
                pid, _ = ranked.pop(0)
                del self._patterns[pid]
                self._chains.pop(f"{pid}_chain", None)
                removed += 1
        if removed:
            self._mark_dirty()
        return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._patterns)
            return {
                "name": self.name,
                "patterns_count": total,
                "chains_count": len(self._chains),
                "avg_success_rate": (
                    sum(p.success_rate for p in self._patterns.values())
                    / max(total, 1)
                ),
            }

    def close(self) -> None:
        self._flush()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        with self._lock:
            self._patterns = self._load_json(
                self._patterns_file, QueryPattern.from_dict,
            )
            self._chains = self._load_json(
                self._chains_file, ReasoningChain.from_dict,
            )
            self._trajectories = self._load_json_list(
                self._trajectories_file, AbstractTrajectory.from_dict,
            )
            self._distillations = self._load_json(
                self._distillations_file, StrategyDistillation.from_dict,
            )
            self._dirty = False

    @staticmethod
    def _load_json(path: Path, factory):
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {k: factory(v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning(f"PatternMemory: failed to load {path.name}: {exc}")
            return {}

    @staticmethod
    def _load_json_list(path: Path, factory):
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return [factory(item) for item in raw]
        except Exception as exc:
            logger.warning(f"PatternMemory: failed to load {path.name}: {exc}")
            return []

    def _mark_dirty(self) -> None:
        """Mark data as changed and flush if enough time has elapsed."""
        self._dirty = True
        now = time.monotonic()
        if now - self._last_save_time >= self._SAVE_MIN_INTERVAL:
            self._flush()

    def _flush(self) -> None:
        """Unconditionally write dirty data to disk."""
        with self._lock:
            if not self._dirty:
                return
            self._atomic_write(
                self._patterns_file,
                {k: v.to_dict() for k, v in self._patterns.items()},
            )
            self._atomic_write(
                self._chains_file,
                {k: v.to_dict() for k, v in self._chains.items()},
            )
            if self._trajectories:
                self._atomic_write(
                    self._trajectories_file,
                    [t.to_dict() for t in self._trajectories[-self._MAX_TRAJECTORIES:]],
                )
            if self._distillations:
                self._atomic_write(
                    self._distillations_file,
                    {k: v.to_dict() for k, v in self._distillations.items()},
                )
            self._dirty = False
            self._last_save_time = time.monotonic()

    @staticmethod
    def _atomic_write(path: Path, data: Any) -> None:
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(path))
        except Exception as exc:
            logger.warning(f"PatternMemory: write failed for {path.name}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    # ── Query classification (heuristic, zero-LLM) ───────────────────

    @staticmethod
    def classify_query(query: str) -> Dict[str, Any]:
        """Extract lightweight feature vector from a query string.

        Delegates to :func:`sirchmunk.memory.query_classifier.classify_query`.
        """
        return _classify_query(query)

    # ── Public API ────────────────────────────────────────────────────

    def suggest_strategy(self, query: str) -> Optional[StrategyHint]:
        """Return a strategy hint via hierarchical Thompson Sampling (HMRPL).

        Performs fine-to-coarse fallback (L4 → L0). Fine-grained matches (L2+)
        return immediately; coarse matches (L0/L1) trigger warm transfer first.
        When no level matches, falls back to soft-match or warm transfer.
        """
        features = self.classify_query(query)

        # Extract feature components for compute_pattern_id_at_level
        query_type = features["query_type"]
        complexity = features["complexity"]
        entity_types = features["entity_types"]
        entity_count = features.get("entity_count", 0)
        hop_hint = features.get("hop_hint", "single")

        # Track best coarse match for warm transfer comparison
        coarse_match: Optional[Tuple[QueryPattern, float, int]] = None

        with self._lock:
            # HMRPL: fine-to-coarse hierarchical lookup (L4 → L0)
            for level in range(4, -1, -1):
                pid = compute_pattern_id_at_level(
                    query_type, complexity, entity_types,
                    entity_count, hop_hint, level=level,
                )
                pattern = self._patterns.get(pid)
                if not pattern:
                    continue

                min_samples = _MIN_SAMPLES_BY_LEVEL.get(level, 3)
                if pattern.sample_count < min_samples:
                    continue

                # Thompson Sampling: draw from Beta posterior
                sampled = random.betavariate(
                    max(pattern.alpha, 0.01), max(pattern.beta_param, 0.01),
                )
                if sampled >= self._MIN_SUCCESS_RATE:
                    if level >= 2:
                        # Fine-grained match (L2+) — trust it, return immediately
                        token_budget = self._estimate_token_budget(pattern, features)
                        return StrategyHint(
                            mode=pattern.optimal_mode,
                            top_k_files=pattern.optimal_params.get("top_k_files"),
                            max_loops=pattern.optimal_params.get("max_loops"),
                            enable_dir_scan=pattern.optimal_params.get("enable_dir_scan"),
                            keyword_strategy=pattern.optimal_params.get("keyword_strategy"),
                            confidence=sampled,
                            source_pattern_id=pattern.pattern_id,
                            resolution_level=level,
                            token_budget=token_budget,
                        )
                    else:
                        # Coarse match (L0 or L1) — save it but try warm transfer first
                        coarse_match = (pattern, sampled, level)
                        break  # Don't need to check coarser levels

            # Hierarchical lookup found nothing or only coarse match; try legacy soft-match fallback
            soft_pattern = self._soft_match(features) if coarse_match is None else None

        # Try warm transfer for cold-start or coarse-match scenarios
        warm_hint = self._suggest_warm_transfer(features, query=query)

        if warm_hint:
            # If we have a coarse match, only use warm transfer if it's more confident
            if coarse_match is None or warm_hint.confidence > coarse_match[1]:
                if coarse_match is not None:
                    logger.debug(
                        f"[memory] warm transfer preferred over L{coarse_match[2]} coarse match "
                        f"(warm_conf={warm_hint.confidence:.3f} > coarse_conf={coarse_match[1]:.3f})"
                    )
                return warm_hint

        # Fall back to coarse match if warm transfer wasn't better
        if coarse_match:
            pattern, sampled, level = coarse_match
            token_budget = self._estimate_token_budget(pattern, features)
            return StrategyHint(
                mode=pattern.optimal_mode,
                top_k_files=pattern.optimal_params.get("top_k_files"),
                max_loops=pattern.optimal_params.get("max_loops"),
                enable_dir_scan=pattern.optimal_params.get("enable_dir_scan"),
                keyword_strategy=pattern.optimal_params.get("keyword_strategy"),
                confidence=sampled,
                source_pattern_id=pattern.pattern_id,
                resolution_level=level,
                token_budget=token_budget,
            )

        # Legacy soft-match fallback (only when no coarse match was found)
        if not soft_pattern:
            return None
        if soft_pattern.sample_count < self._MIN_SAMPLES_FOR_CONFIDENT:
            return None

        # Thompson Sampling on soft-matched pattern
        sampled = random.betavariate(
            max(soft_pattern.alpha, 0.01), max(soft_pattern.beta_param, 0.01),
        )
        if sampled < self._MIN_SUCCESS_RATE:
            return None

        token_budget = self._estimate_token_budget(soft_pattern, features)
        return StrategyHint(
            mode=soft_pattern.optimal_mode,
            top_k_files=soft_pattern.optimal_params.get("top_k_files"),
            max_loops=soft_pattern.optimal_params.get("max_loops"),
            enable_dir_scan=soft_pattern.optimal_params.get("enable_dir_scan"),
            keyword_strategy=soft_pattern.optimal_params.get("keyword_strategy"),
            confidence=sampled,
            source_pattern_id=soft_pattern.pattern_id,
            resolution_level=soft_pattern.resolution_level,
            token_budget=token_budget,
        )

    def _soft_match(self, features: Dict[str, Any]) -> Optional[QueryPattern]:
        """Find the nearest pattern when exact pid is absent.

        Uses feature-level overlap (Jaccard on entity_types + exact match
        on query_type and complexity).  Returns the best match only when
        it has sufficient samples.
        """
        target_et = set(features.get("entity_types", []))
        best: Optional[Tuple[float, QueryPattern]] = None
        for p in self._patterns.values():
            if p.sample_count < self._MIN_SAMPLES_FOR_CONFIDENT:
                continue
            score = 0.0
            if p.query_type == features["query_type"]:
                score += 0.4
            if p.complexity == features["complexity"]:
                score += 0.2
            if p.hop_hint == features.get("hop_hint", "single"):
                score += 0.1
            pet = set(p.entity_types)
            if target_et or pet:
                jaccard = len(target_et & pet) / max(len(target_et | pet), 1)
                score += 0.3 * jaccard
            else:
                score += 0.3
            if best is None or score > best[0]:
                best = (score, p)
        if best and best[0] >= 0.6:
            return best[1]
        return None

    def _estimate_token_budget(
        self, pattern: QueryPattern, features: Dict[str, Any]
    ) -> Optional[int]:
        """Estimate optimal token budget from historical pattern data.

        Uses Pareto principle: budget = avg_tokens * complexity_factor,
        clamped to reasonable bounds. Returns None if insufficient data.
        """
        if pattern.sample_count < 3:
            return None
        avg_tokens = pattern.optimal_params.get("avg_tokens", 0)
        if avg_tokens <= 0:
            return None

        complexity_factor = {"simple": 0.8, "moderate": 1.0, "complex": 1.3}
        factor = complexity_factor.get(features.get("complexity", "moderate"), 1.0)
        budget = int(avg_tokens * factor)

        # Clamp to reasonable bounds
        return max(10000, min(budget, 150000))

    def _find_similar_patterns(
        self,
        features: Dict[str, Any],
        k: int = 5,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Tuple[float, QueryPattern]]:
        """Find k most similar patterns by feature + optional embedding similarity.

        When ``query_embedding`` is provided (from ``EmbeddingUtil``), a
        weighted combination of heuristic feature score (60%) and cosine
        similarity (40%) is used.  Otherwise falls back to pure heuristic.
        """
        candidates: List[Tuple[float, QueryPattern]] = []
        for pattern in self._patterns.values():
            if pattern.sample_count < 1:
                continue
            # Heuristic feature score
            hscore = 0.0
            if pattern.query_type == features.get("query_type"):
                hscore += 0.4
            if pattern.complexity == features.get("complexity"):
                hscore += 0.2
            if pattern.hop_hint == features.get("hop_hint"):
                hscore += 0.2
            pat_types = set(pattern.entity_types or [])
            feat_types = set(features.get("entity_types", []))
            if pat_types and feat_types:
                overlap = len(pat_types & feat_types) / max(len(pat_types | feat_types), 1)
                hscore += 0.2 * overlap

            # Embedding cosine similarity (if available)
            if query_embedding and pattern.pattern_id in self._pattern_embeddings:
                pat_emb = self._pattern_embeddings[pattern.pattern_id]
                csim = self._cosine_sim(query_embedding, pat_emb)
                score = 0.6 * hscore + 0.4 * max(csim, 0.0)
            else:
                score = hscore

            if score >= 0.3:
                candidates.append((score, pattern))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:k]

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _encode_query(self, query: str) -> Optional[List[float]]:
        """Encode query text into an embedding vector (returns None if unavailable)."""
        if self._embedding_util is None:
            return None
        try:
            vec = self._embedding_util.encode(query)
            if vec is not None:
                return vec if isinstance(vec, list) else vec.tolist()
        except Exception:
            pass
        return None

    def set_embedding_util(self, embedding_util: Any) -> None:
        """Inject embedding util after deferred warm-up."""
        self._embedding_util = embedding_util

    def _suggest_warm_transfer(
        self,
        features: Dict[str, Any],
        query: str = "",
    ) -> Optional[StrategyHint]:
        """CTS-WT: Warm transfer from similar patterns for cold-start queries.

        Aggregates Beta posteriors from k nearest patterns weighted by
        feature similarity (+ optional embedding similarity), then samples
        from the aggregate distribution.
        """
        query_embedding = self._encode_query(query) if query else None
        with self._lock:
            similar = self._find_similar_patterns(
                features, k=5, query_embedding=query_embedding,
            )

        if not similar:
            return None

        # Weighted aggregation of Beta parameters
        alpha_warm = 0.0
        beta_warm = 0.0
        total_weight = 0.0
        best_mode: Optional[str] = None
        best_params: Dict[str, Any] = {}
        best_pattern: Optional[QueryPattern] = None
        best_weight = 0.0

        for score, pattern in similar:
            w = score  # similarity score as weight
            alpha_warm += w * pattern.alpha
            beta_warm += w * pattern.beta_param
            total_weight += w
            if w > best_weight:
                best_weight = w
                best_mode = pattern.optimal_mode
                best_params = pattern.optimal_params
                best_pattern = pattern

        if total_weight < 1e-9:
            return None

        # Normalize
        alpha_warm /= total_weight
        beta_warm /= total_weight

        # Sample from warm-transferred Beta
        sampled = random.betavariate(max(alpha_warm, 0.01), max(beta_warm, 0.01))

        if sampled >= self._MIN_SUCCESS_RATE and best_mode:
            token_budget = None
            if best_pattern:
                token_budget = self._estimate_token_budget(best_pattern, features)
            return StrategyHint(
                mode=best_mode,
                top_k_files=best_params.get("top_k_files") if best_params else None,
                max_loops=best_params.get("max_loops") if best_params else None,
                enable_dir_scan=best_params.get("enable_dir_scan") if best_params else None,
                keyword_strategy=best_params.get("keyword_strategy") if best_params else None,
                confidence=round(sampled, 3),
                source_pattern_id="warm_transfer",
                resolution_level=0,  # warm transfer is lowest confidence level
                token_budget=token_budget,
            )
        return None

    def _update_pattern_stats(
        self,
        pattern: QueryPattern,
        confidence: float,
        mode: str,
        params: Dict[str, Any],
        latency: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Update statistics on a pattern (shared by primary and ancestor updates).

        This is a helper for record_outcome() to avoid code duplication when
        propagating updates to ancestor patterns in the HMRPL hierarchy.
        """
        success = confidence >= self._SUCCESS_THRESHOLD
        now = datetime.now(timezone.utc).isoformat()

        pattern.sample_count += 1
        if success:
            pattern.success_count += 1
        pattern.success_rate = (
            pattern.success_count / max(pattern.sample_count, 1)
        )

        # Thompson Sampling: update Beta posterior
        pattern.alpha += confidence
        pattern.beta_param += (1.0 - confidence)

        # HMRPL: update total_visits and avg_reward
        pattern.total_visits += 1
        ema_reward = 0.1  # slower EMA for hierarchical reward
        pattern.avg_reward = (
            ema_reward * confidence + (1 - ema_reward) * pattern.avg_reward
        )

        ema = self._EMA_ALPHA
        pattern.avg_latency = (
            ema * latency + (1 - ema) * pattern.avg_latency
        )
        pattern.avg_tokens = int(
            ema * tokens + (1 - ema) * pattern.avg_tokens
        )

        # Also track avg_tokens in optimal_params for token budget estimation
        if tokens and tokens > 0:
            old_avg = pattern.optimal_params.get("avg_tokens", 0)
            if old_avg > 0:
                new_avg_tokens = int(0.3 * tokens + 0.7 * old_avg)
            else:
                new_avg_tokens = tokens
            pattern.optimal_params["avg_tokens"] = new_avg_tokens
        else:
            new_avg_tokens = pattern.optimal_params.get("avg_tokens")

        if success and pattern.success_rate >= self._SUCCESS_THRESHOLD:
            pattern.optimal_mode = mode
            pattern.optimal_params = params
            # Preserve avg_tokens in new optimal_params
            if new_avg_tokens:
                pattern.optimal_params["avg_tokens"] = new_avg_tokens

        pattern.updated_at = now

    def record_outcome(
        self,
        query: str,
        confidence: float,
        mode: str,
        params: Dict[str, Any],
        latency: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Record a search outcome with continuous *confidence* (0-1).

        The old binary ``success`` is replaced by a gradient signal:
        ``confidence >= 0.5`` counts as a success for rate tracking,
        and the Beta distribution is updated proportionally.

        HMRPL: Also propagates updates to all ancestor levels and performs
        auto-split/merge when conditions are met.
        """
        features = self.classify_query(query)

        # Extract feature components for hierarchical pattern IDs
        query_type = features["query_type"]
        complexity = features["complexity"]
        entity_types = features["entity_types"]
        entity_count = features.get("entity_count", 0)
        hop_hint = features.get("hop_hint", "single")

        # Primary pattern at finest level (L4)
        pid = compute_pattern_id_at_level(
            query_type, complexity, entity_types,
            entity_count, hop_hint, level=4,
        )
        now = datetime.now(timezone.utc).isoformat()

        # Cache embedding for CTS-WT (non-blocking, best-effort)
        if pid not in self._pattern_embeddings:
            emb = self._encode_query(query)
            if emb is not None:
                self._pattern_embeddings[pid] = emb

        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                pattern = QueryPattern(
                    pattern_id=pid,
                    query_type=query_type,
                    entity_types=entity_types,
                    complexity=complexity,
                    entity_count=entity_count,
                    hop_hint=hop_hint,
                    resolution_level=4,
                    optimal_mode=mode,
                    optimal_params=params,
                    created_at=now,
                    updated_at=now,
                )
                self._patterns[pid] = pattern

            # Update primary pattern stats using helper
            self._update_pattern_stats(
                pattern, confidence, mode, params, latency, tokens,
            )
            current_level = pattern.resolution_level

            # HMRPL: Propagate update to all ancestor levels (L3 → L0)
            for ancestor_level in range(current_level - 1, -1, -1):
                ancestor_pid = compute_pattern_id_at_level(
                    query_type, complexity, entity_types,
                    entity_count, hop_hint, level=ancestor_level,
                )
                ancestor = self._patterns.get(ancestor_pid)
                if ancestor is None:
                    # Create ancestor pattern with minimal initialization
                    ancestor = QueryPattern(
                        pattern_id=ancestor_pid,
                        query_type=query_type,
                        resolution_level=ancestor_level,
                        complexity=complexity if ancestor_level >= 1 else "moderate",
                        hop_hint=hop_hint if ancestor_level >= 2 else "single",
                        entity_count=entity_count if ancestor_level >= 3 else 0,
                        entity_types=entity_types if ancestor_level >= 4 else [],
                        optimal_mode=mode,
                        optimal_params=params,
                        created_at=now,
                        updated_at=now,
                    )
                    self._patterns[ancestor_pid] = ancestor
                # Update ancestor's stats (same update logic)
                self._update_pattern_stats(
                    ancestor, confidence, mode, params, latency, tokens,
                )

                # HMRPL: Auto-split on ancestor if it has enough samples
                if (ancestor.sample_count > 20
                        and ancestor.resolution_level < 4
                        and not ancestor.children_ids):
                    self._try_split_pattern(ancestor, mode, params, now)

                # HMRPL: Auto-merge children (rate-limited to every 10 visits)
                if ancestor.total_visits % 10 == 0:
                    self._try_merge_children(ancestor)

                # Link parent-child relationship
                if ancestor_level == current_level - 1:
                    pattern.parent_id = ancestor_pid
                    if pid not in ancestor.children_ids:
                        ancestor.children_ids.append(pid)

        self._mark_dirty()

    def _try_split_pattern(
        self,
        pattern: QueryPattern,
        mode: str,
        params: Dict[str, Any],
        now: str,
    ) -> None:
        """Attempt to create child patterns at finer resolution level.

        Children inherit half of the parent's Beta priors (α/2, β/2) to
        bootstrap with some prior knowledge while allowing divergence.
        """
        child_level = pattern.resolution_level + 1
        if child_level > 4:
            return

        # Create a single child pattern at the finer level
        # (actual child patterns are created lazily when queries arrive)
        child_pid = compute_pattern_id_at_level(
            pattern.query_type, pattern.complexity,
            pattern.entity_types, pattern.entity_count,
            pattern.hop_hint, level=child_level,
        )

        if child_pid not in self._patterns:
            child = QueryPattern(
                pattern_id=child_pid,
                query_type=pattern.query_type,
                resolution_level=child_level,
                complexity=pattern.complexity,
                hop_hint=pattern.hop_hint,
                entity_count=pattern.entity_count,
                entity_types=pattern.entity_types,
                optimal_mode=mode,
                optimal_params=params,
                # Inherited Beta priors (halved for uncertainty)
                alpha=max(pattern.alpha / 2, 1.0),
                beta_param=max(pattern.beta_param / 2, 1.0),
                parent_id=pattern.pattern_id,
                created_at=now,
                updated_at=now,
            )
            self._patterns[child_pid] = child
            pattern.children_ids.append(child_pid)

    def _try_merge_children(self, pattern: QueryPattern) -> None:
        """Merge children stats back to parent if they have similar success rates.

        Merges when all children exist and have success rates within epsilon.
        Correctly syncs **all** statistics (sample_count, success_count, α, β)
        so that derived invariants are maintained.
        """
        if not pattern.children_ids or len(pattern.children_ids) < 2:
            return

        epsilon = 0.05
        children = [self._patterns.get(cid) for cid in pattern.children_ids]
        children = [c for c in children if c is not None and c.sample_count > 0]

        if len(children) < 2:
            return

        success_rates = [c.success_rate for c in children]
        rate_range = max(success_rates) - min(success_rates)

        if rate_range <= epsilon:
            total_samples = sum(c.sample_count for c in children)
            total_success = sum(c.success_count for c in children)
            total_alpha = sum(c.alpha for c in children)
            total_beta = sum(c.beta_param for c in children)
            n = len(children)

            if total_samples > 0:
                pattern.sample_count = total_samples
                pattern.success_count = total_success
                pattern.success_rate = total_success / total_samples
                pattern.alpha = total_alpha / n
                pattern.beta_param = total_beta / n
                pattern.total_visits = sum(c.total_visits for c in children)
                pattern.avg_reward = (
                    sum(c.avg_reward * c.total_visits for c in children)
                    / max(pattern.total_visits, 1)
                )

    # ── Public mutation API (lock-safe, for use by RetrievalMemory) ────

    def apply_confidence_delta(
        self,
        query: str,
        delta: float,
        ground_truth_conf: float,
        old_heuristic_conf: float,
    ) -> None:
        """Apply a delta correction to a pattern's Beta posterior.

        Called by ``inject_evaluation`` when ground-truth replaces the
        heuristic confidence.  Thread-safe.
        """
        features = self.classify_query(query)
        pid = compute_pattern_id_at_level(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
            features.get("entity_count", 0),
            features.get("hop_hint", "single"),
            level=4,
        )
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                return
            if delta > 0:
                pattern.alpha += delta
            else:
                pattern.beta_param += abs(delta)
            # Correct success tracking if the outcome class flipped
            if ground_truth_conf >= 0.5 > old_heuristic_conf:
                pattern.success_count += 1
                pattern.success_rate = (
                    pattern.success_count / max(pattern.sample_count, 1)
                )
            elif ground_truth_conf < 0.5 <= old_heuristic_conf:
                pattern.success_count = max(0, pattern.success_count - 1)
                pattern.success_rate = (
                    pattern.success_count / max(pattern.sample_count, 1)
                )
        self._mark_dirty()

    def update_sampling_params(
        self,
        query: str,
        react_loops: int,
        convergence: bool,
        confidence: float,
        total_tokens: int,
    ) -> None:
        """Adapt search depth params based on convergence signals.

        Connects memory to MCTS by adjusting recommended ``max_loops``.
        Thread-safe (acquires internal lock).
        """
        if react_loops <= 0:
            return

        features = self.classify_query(query)
        pid = compute_pattern_id_at_level(
            features.get("query_type", "unknown"),
            features.get("complexity", "moderate"),
            features.get("entity_types", []),
            features.get("entity_count", 0),
            features.get("hop_hint", "single"),
            level=4,
        )
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                return
            current_loops = pattern.optimal_params.get("max_loops", 5)
            if convergence and confidence >= 0.5:
                suggested = max(2, min(react_loops + 1, current_loops))
                pattern.optimal_params["max_loops"] = int(
                    0.3 * suggested + 0.7 * current_loops
                )
            elif not convergence and total_tokens > 100_000:
                suggested = min(10, current_loops + 1)
                pattern.optimal_params["max_loops"] = int(
                    0.3 * suggested + 0.7 * current_loops
                )
        self._mark_dirty()

    def get_exploration_stats(self, query: str) -> Dict[str, Any]:
        """Return exploration statistics for reward shaping.

        Thread-safe.  Returns dict with ``pattern_sample_count``,
        ``total_patterns``, and ``explored_count``.
        """
        features = self.classify_query(query)
        pid = compute_pattern_id_at_level(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
            features.get("entity_count", 0),
            features.get("hop_hint", "single"),
            level=4,
        )
        with self._lock:
            pattern = self._patterns.get(pid)
            total = len(self._patterns)
            explored = sum(
                1 for p in self._patterns.values()
                if p.sample_count >= self._MIN_SAMPLES_FOR_CONFIDENT
            )
        return {
            "pattern_sample_count": pattern.sample_count if pattern else 0,
            "total_patterns": total,
            "explored_count": explored,
        }

    def _chain_pid(self, query: str) -> str:
        """Compute the canonical pattern ID used for chain lookup/storage.

        Uses ``compute_pattern_id_at_level`` (same as ``record_outcome``) so
        that chains are co-located with the patterns they belong to.
        """
        features = self.classify_query(query)
        return compute_pattern_id_at_level(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
            features.get("entity_count", 0),
            features.get("hop_hint", "single"),
            level=4,
        )

    def record_chain(
        self,
        query: str,
        steps: List[Dict[str, str]],
        success: bool,
    ) -> None:
        """Record (or update) a reasoning chain trace."""
        pid = self._chain_pid(query)
        now = datetime.now(timezone.utc).isoformat()
        tid = f"{pid}_chain"

        with self._lock:
            chain = self._chains.get(tid)
            if not chain:
                chain = ReasoningChain(
                    template_id=tid,
                    pattern_id=pid,
                    steps=steps,
                    created_at=now,
                    updated_at=now,
                )
                self._chains[tid] = chain

            chain.total_count += 1
            if success:
                chain.success_count += 1
                chain.steps = steps
            chain.updated_at = now

        self._mark_dirty()

    def get_chain_hint(self, query: str) -> Optional[List[Dict[str, str]]]:
        """Return a reasoning chain template if available and reliable."""
        pid = self._chain_pid(query)
        tid = f"{pid}_chain"

        with self._lock:
            chain = self._chains.get(tid)
            if (chain and chain.success_count >= 2
                    and chain.total_count > 0
                    and chain.success_count / chain.total_count >= 0.4):
                return chain.steps
        return None

    @property
    def distillation_batch_size(self) -> int:
        """Minimum trajectories needed before distillation is triggered."""
        return self._DISTILLATION_BATCH_SIZE

    def _classify_and_pid(self, query: str) -> Tuple[Dict[str, Any], str]:
        """Classify a query and compute its L4 pattern ID.

        Reduces boilerplate across Meta-RL methods that all need the same
        (features, pattern_id) pair.
        """
        features = self.classify_query(query)
        pid = compute_pattern_id_at_level(
            features["query_type"], features["complexity"],
            features["entity_types"],
            features.get("entity_count", 0),
            features.get("hop_hint", "single"),
            level=4,
        )
        return features, pid

    # ── Meta-RL: trajectory storage + distillation + loop budget ─────

    def store_trajectory(self, trajectory: AbstractTrajectory) -> None:
        """Append an abstracted trajectory for future distillation."""
        with self._lock:
            self._trajectories.append(trajectory)
            if len(self._trajectories) > self._MAX_TRAJECTORIES:
                self._trajectories = self._trajectories[-self._MAX_TRAJECTORIES:]
        self._mark_dirty()

    def get_recent_trajectories(
        self,
        query_type: str,
        complexity: str,
        limit: int = 20,
    ) -> List[AbstractTrajectory]:
        """Return recent trajectories matching the given type + complexity.

        When the exact (type, complexity) bucket has fewer than
        ``_DISTILLATION_BATCH_SIZE`` entries, trajectories from the
        same *query_type* (any complexity) are included to reach the
        minimum count.  This cross-complexity sharing accelerates cold
        start without mixing fundamentally different reasoning types.
        """
        with self._lock:
            exact = [
                t for t in reversed(self._trajectories)
                if t.query_type == query_type and t.complexity == complexity
            ]
            if len(exact) >= self._DISTILLATION_BATCH_SIZE:
                return exact[:limit]
            same_type = [
                t for t in reversed(self._trajectories)
                if t.query_type == query_type and t.complexity != complexity
            ]
        merged = exact + same_type
        return merged[:limit]

    def pending_trajectory_count(self, query_type: str, complexity: str) -> int:
        """Count trajectories accumulated since last distillation.

        Uses cross-complexity sharing: counts all trajectories of the
        same *query_type* so distillation triggers faster on cold start.
        """
        with self._lock:
            key = f"{query_type}|{complexity}"
            distill = self._distillations.get(key)
            if not distill:
                return sum(
                    1 for t in self._trajectories
                    if t.query_type == query_type
                )
            return sum(
                1 for t in self._trajectories
                if (t.query_type == query_type
                    and t.timestamp > distill.distilled_at)
            )

    def store_distillation(self, distill: StrategyDistillation) -> None:
        """Persist a distillation and propagate rules to matching patterns."""
        key = f"{distill.query_type}|{distill.complexity}"
        with self._lock:
            self._distillations[key] = distill
            for p in self._patterns.values():
                if p.query_type == distill.query_type:
                    p.distilled_rules = list(distill.rules)
                    p.failure_warnings = list(distill.failure_warnings)
                    if distill.best_keyword_strategy:
                        p.best_keyword_strategy = distill.best_keyword_strategy
        self._mark_dirty()

    def get_distillation(
        self,
        query_type: str,
        complexity: str,
    ) -> Optional[StrategyDistillation]:
        """Return the most recent distillation for a query type."""
        key = f"{query_type}|{complexity}"
        with self._lock:
            return self._distillations.get(key)

    def record_loop_outcome(
        self,
        query: str,
        loops_used: int,
        success: bool,
    ) -> None:
        """Update Bayesian loop budget statistics for the pattern.

        Tracks success/total counts per loop depth.  The optimal
        loop budget is the smallest depth with success_rate >= threshold
        (minimum effective depth principle).

        Creates the pattern if it doesn't exist, because the async
        ``record_outcome`` (which normally creates patterns) may not
        have completed yet when this method is called synchronously
        from ``record_trajectory``.
        """
        features, pid = self._classify_and_pid(query)
        key = str(loops_used)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                pattern = QueryPattern(
                    pattern_id=pid,
                    query_type=features["query_type"],
                    complexity=features["complexity"],
                    entity_types=features["entity_types"],
                    entity_count=features.get("entity_count", 0),
                    hop_hint=features.get("hop_hint", "single"),
                    resolution_level=4,
                    optimal_mode="DEEP",
                    optimal_params={"max_loops": 5, "top_k_files": 5},
                    created_at=now,
                    updated_at=now,
                )
                self._patterns[pid] = pattern
            stats = pattern.loop_budget_stats.get(key, [0, 0])
            stats[1] += 1
            if success:
                stats[0] += 1
            pattern.loop_budget_stats[key] = stats
        self._mark_dirty()

    def get_optimal_loop_budget(self, query: str) -> Optional[int]:
        """Return the minimum effective loop depth for this query type.

        Uses Bayesian estimation: finds the smallest loop count where
        success_rate >= _LOOP_BUDGET_SUCCESS_RATE with at least
        _LOOP_BUDGET_MIN_OBS observations.
        """
        _, pid = self._classify_and_pid(query)
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern or not pattern.loop_budget_stats:
                return None
            for loops in sorted(pattern.loop_budget_stats.keys(), key=int):
                s, t = pattern.loop_budget_stats[loops]
                if (t >= self._LOOP_BUDGET_MIN_OBS
                        and s / t >= self._LOOP_BUDGET_SUCCESS_RATE):
                    return int(loops)
        return None

    def record_strategy_arm(
        self,
        query: str,
        strategy: str,
        success: bool,
    ) -> None:
        """Update the contextual bandit arm for a strategy.

        Each (pattern, strategy) pair maintains independent Thompson
        Sampling parameters (alpha, beta) for strategy selection.

        Creates the pattern if it doesn't exist (same async-safety
        rationale as ``record_loop_outcome``).
        """
        if strategy not in SearchStrategy.ALL:
            return
        features, pid = self._classify_and_pid(query)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                pattern = QueryPattern(
                    pattern_id=pid,
                    query_type=features["query_type"],
                    complexity=features["complexity"],
                    entity_types=features["entity_types"],
                    entity_count=features.get("entity_count", 0),
                    hop_hint=features.get("hop_hint", "single"),
                    resolution_level=4,
                    optimal_mode="DEEP",
                    optimal_params={"max_loops": 5, "top_k_files": 5},
                    created_at=now,
                    updated_at=now,
                )
                self._patterns[pid] = pattern
            arm = pattern.strategy_arms.get(strategy, [1.0, 1.0])
            if success:
                arm[0] += 1.0
            else:
                arm[1] += 1.0
            pattern.strategy_arms[strategy] = arm
        self._mark_dirty()

    def suggest_strategy_arm(self, query: str) -> Optional[str]:
        """Thompson Sampling over strategy arms for the query's pattern.

        Returns the strategy with the highest sampled value, or None
        if no arms have been observed.
        """
        _, pid = self._classify_and_pid(query)
        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern or not pattern.strategy_arms:
                return None
            best_strategy = None
            best_sample = -1.0
            for strategy, (alpha, beta) in pattern.strategy_arms.items():
                sample = random.betavariate(max(alpha, 0.01), max(beta, 0.01))
                if sample > best_sample:
                    best_sample = sample
                    best_strategy = strategy
        return best_strategy

    def get_meta_knowledge(self, query: str) -> Dict[str, Any]:
        """Aggregate all meta-knowledge for the MAP planner.

        Combines pattern statistics, distilled rules, loop budget,
        and strategy arm data into a single dict consumed by the
        :class:`MemoryAugmentedPlanner`.
        """
        features, pid = self._classify_and_pid(query)
        qt = features["query_type"]
        cx = features["complexity"]

        mk: Dict[str, Any] = {
            "query_type": qt,
            "complexity": cx,
            "entity_count": features.get("entity_count", 0),
            "hop_hint": features.get("hop_hint", "single"),
            "distilled_rules": [],
            "failure_warnings": [],
            "success_rate": 0.0,
            "avg_loops": 4.0,
            "avg_tokens": 0,
            "best_keyword_strategy": "",
        }

        with self._lock:
            pattern = self._patterns.get(pid)
            if pattern:
                mk["distilled_rules"] = list(pattern.distilled_rules)
                mk["failure_warnings"] = list(pattern.failure_warnings)
                mk["success_rate"] = pattern.success_rate
                mk["avg_tokens"] = pattern.avg_tokens
                mk["best_keyword_strategy"] = pattern.best_keyword_strategy

                # Compute avg_loops from loop_budget_stats if available
                if pattern.loop_budget_stats:
                    total_loops = 0
                    total_count = 0
                    for loops_str, (s, t) in pattern.loop_budget_stats.items():
                        total_loops += int(loops_str) * t
                        total_count += t
                    if total_count > 0:
                        mk["avg_loops"] = total_loops / total_count

            # Enrich from distillation if pattern-level rules are empty
            key = f"{qt}|{cx}"
            distill = self._distillations.get(key)
            if distill and not mk["distilled_rules"]:
                mk["distilled_rules"] = list(distill.rules)
                mk["failure_warnings"] = list(distill.failure_warnings)
                mk["success_rate"] = distill.success_rate
                mk["avg_loops"] = distill.avg_loops
                mk["avg_tokens"] = distill.avg_tokens
                mk["best_keyword_strategy"] = distill.best_keyword_strategy

        return mk

    # ── Warmup seeding ────────────────────────────────────────────────

    _DEFAULT_STRATEGIES: List[Dict[str, Any]] = [
        {
            "query_type": "factual", "complexity": "simple",
            "entity_types": ["named_entity"], "entity_count": 1,
            "hop_hint": "single",
            "mode": "DEEP", "params": {"max_loops": 3, "top_k_files": 5},
        },
        {
            "query_type": "bridge", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 2,
            "hop_hint": "multi",
            "mode": "DEEP", "params": {"max_loops": 5, "top_k_files": 5},
        },
        {
            "query_type": "comparison", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 2,
            "hop_hint": "multi",
            "mode": "DEEP", "params": {"max_loops": 5, "top_k_files": 5},
        },
        {
            "query_type": "factual", "complexity": "moderate",
            "entity_types": ["named_entity"], "entity_count": 1,
            "hop_hint": "single",
            "mode": "DEEP", "params": {"max_loops": 4, "top_k_files": 5},
        },
        {
            "query_type": "definition", "complexity": "simple",
            "entity_types": [], "entity_count": 0,
            "hop_hint": "single",
            "mode": "FAST", "params": {"max_loops": 2, "top_k_files": 3},
        },
    ]

    def seed_defaults(self) -> int:
        """Insert default strategy patterns if absent (idempotent).

        Returns the number of newly inserted patterns.
        """
        seeded = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            for spec in self._DEFAULT_STRATEGIES:
                pid = compute_pattern_id(
                    spec["query_type"], spec["complexity"],
                    spec["entity_types"],
                    entity_count=spec.get("entity_count", 0),
                    hop_hint=spec.get("hop_hint", "single"),
                )
                if pid in self._patterns:
                    continue
                self._patterns[pid] = QueryPattern(
                    pattern_id=pid,
                    query_type=spec["query_type"],
                    complexity=spec["complexity"],
                    entity_types=spec["entity_types"],
                    entity_count=spec.get("entity_count", 0),
                    hop_hint=spec.get("hop_hint", "single"),
                    optimal_mode=spec["mode"],
                    optimal_params=spec["params"],
                    created_at=now,
                    updated_at=now,
                )
                seeded += 1
        if seeded:
            self._mark_dirty()
            logger.debug(f"PatternMemory: seeded {seeded} default patterns")
        return seeded
