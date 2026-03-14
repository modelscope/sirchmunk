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
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .base import MemoryStore
from .schemas import (
    QueryPattern,
    ReasoningChain,
    StrategyHint,
    compute_pattern_id,
)

# Heuristic word sets for lightweight query classification (no LLM needed)
_COMPARISON_WORDS = frozenset({
    "which", "both", "difference", "compare", "versus", "vs",
    "better", "more", "less", "rather", "prefer", "between",
})
_FACTUAL_WORDS = frozenset({
    "when", "where", "who", "what", "how many", "how much",
})
_DEFINITION_WORDS = frozenset({
    "what is", "what are", "define", "definition", "meaning",
})
_PROCEDURAL_WORDS = frozenset({
    "how to", "how do", "steps", "process", "procedure",
})

# Chinese query classification patterns
_ZH_COMPARISON_RE = re.compile(r"哪个更|对比|区别|比较|和.+哪个|还是.+好")
_ZH_FACTUAL_RE = re.compile(r"谁|什么时候|哪里|多少|几个|在哪")
_ZH_DEFINITION_RE = re.compile(r"什么是|是什么|定义|含义|意思是")
_ZH_PROCEDURAL_RE = re.compile(r"如何|怎么|怎样|步骤|流程|方法是")
_ZH_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class PatternMemory(MemoryStore):
    """Query pattern → strategy mapping and reasoning chain templates.

    Uses heuristic query classification at lookup time (zero-LLM-cost)
    and learns optimal parameters from feedback signals.
    """

    _MIN_SAMPLES_FOR_CONFIDENT = 3
    _MIN_SUCCESS_RATE = 0.4
    _EMA_ALPHA = 0.3
    _SAVE_MIN_INTERVAL = 5.0  # seconds between disk writes

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._patterns_file = base_dir / "query_patterns.json"
        self._chains_file = base_dir / "reasoning_chains.json"
        self._patterns: Dict[str, QueryPattern] = {}
        self._chains: Dict[str, ReasoningChain] = {}
        self._lock = threading.RLock()
        self._dirty = False
        self._last_save_time: float = 0.0

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

        Supports both English and Chinese queries.

        Returns dict with ``query_type``, ``complexity``, ``entity_types``.
        """
        q_lower = query.lower().strip()
        words = q_lower.split()
        has_cjk = bool(_ZH_CJK_RE.search(query))

        # --- Query type ---
        query_type = "factual"
        if has_cjk:
            if _ZH_DEFINITION_RE.search(query):
                query_type = "definition"
            elif _ZH_PROCEDURAL_RE.search(query):
                query_type = "procedural"
            elif _ZH_COMPARISON_RE.search(query):
                query_type = "comparison"
            elif _ZH_FACTUAL_RE.search(query):
                query_type = "factual"
            else:
                query_type = "bridge"
        else:
            if any(w in q_lower for w in _DEFINITION_WORDS):
                query_type = "definition"
            elif any(w in q_lower for w in _PROCEDURAL_WORDS):
                query_type = "procedural"
            elif any(w in words for w in _COMPARISON_WORDS):
                query_type = "comparison"
            elif any(w in q_lower for w in _FACTUAL_WORDS):
                query_type = "factual"
            else:
                query_type = "bridge"

        # --- Complexity ---
        if has_cjk:
            char_count = len(query.strip())
            if char_count <= 10:
                complexity = "simple"
            elif char_count <= 30:
                complexity = "moderate"
            else:
                complexity = "complex"
        else:
            if len(words) <= 6:
                complexity = "simple"
            elif len(words) <= 15:
                complexity = "moderate"
            else:
                complexity = "complex"

        # --- Entity types ---
        entity_types: List[str] = []
        if re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query):
            entity_types.append("named_entity")
        if re.search(r"\b\d{4}\b", query):
            entity_types.append("date")
        if re.search(r"\b(?:city|country|state|river|mountain)\b", q_lower):
            entity_types.append("location")
        if has_cjk and re.search(
            r"[\u4e00-\u9fff]{2,}(?:市|省|县|国|河|山|湖)", query,
        ):
            entity_types.append("location")

        return {
            "query_type": query_type,
            "complexity": complexity,
            "entity_types": entity_types,
        }

    # ── Public API ────────────────────────────────────────────────────

    def suggest_strategy(self, query: str) -> Optional[StrategyHint]:
        """Return a strategy hint if a confident pattern exists."""
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
        )
        with self._lock:
            pattern = self._patterns.get(pid)

        if not pattern:
            return None
        if (pattern.sample_count < self._MIN_SAMPLES_FOR_CONFIDENT
                or pattern.success_rate < self._MIN_SUCCESS_RATE):
            return None

        return StrategyHint(
            mode=pattern.optimal_mode,
            top_k_files=pattern.optimal_params.get("top_k_files"),
            max_loops=pattern.optimal_params.get("max_loops"),
            enable_dir_scan=pattern.optimal_params.get("enable_dir_scan"),
            keyword_strategy=pattern.optimal_params.get("keyword_strategy"),
            confidence=pattern.success_rate,
            source_pattern_id=pid,
        )

    def record_outcome(
        self,
        query: str,
        success: bool,
        mode: str,
        params: Dict[str, Any],
        latency: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Record a search outcome to update pattern statistics."""
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
        )
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            pattern = self._patterns.get(pid)
            if not pattern:
                pattern = QueryPattern(
                    pattern_id=pid,
                    query_type=features["query_type"],
                    entity_types=features["entity_types"],
                    complexity=features["complexity"],
                    optimal_mode=mode,
                    optimal_params=params,
                    created_at=now,
                    updated_at=now,
                )
                self._patterns[pid] = pattern

            pattern.sample_count += 1
            if success:
                pattern.success_count += 1
            pattern.success_rate = (
                pattern.success_count / max(pattern.sample_count, 1)
            )

            alpha = self._EMA_ALPHA
            pattern.avg_latency = (
                alpha * latency + (1 - alpha) * pattern.avg_latency
            )
            pattern.avg_tokens = int(
                alpha * tokens + (1 - alpha) * pattern.avg_tokens
            )

            if success and pattern.success_rate >= 0.5:
                pattern.optimal_mode = mode
                pattern.optimal_params = params

            pattern.updated_at = now

        self._mark_dirty()

    def record_chain(
        self,
        query: str,
        steps: List[Dict[str, str]],
        success: bool,
    ) -> None:
        """Record (or update) a reasoning chain trace."""
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
        )
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
        features = self.classify_query(query)
        pid = compute_pattern_id(
            features["query_type"],
            features["complexity"],
            features["entity_types"],
        )
        tid = f"{pid}_chain"

        with self._lock:
            chain = self._chains.get(tid)
            if (chain and chain.success_count >= 2
                    and chain.total_count > 0
                    and chain.success_count / chain.total_count >= 0.4):
                return chain.steps
        return None
