# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared data models for the retrieval memory system.

All models are plain dataclasses — no external validation dependency.
They are serialisable to/from dicts and JSON natively.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ────────────────────────────────────────────────────────────────────
#  Strategy hint (output of PatternMemory lookup)
# ────────────────────────────────────────────────────────────────────

@dataclass
class StrategyHint:
    """Search parameter overrides suggested by PatternMemory."""

    mode: Optional[str] = None
    top_k_files: Optional[int] = None
    max_loops: Optional[int] = None
    enable_dir_scan: Optional[bool] = None
    keyword_strategy: Optional[str] = None
    confidence: float = 0.0
    source_pattern_id: Optional[str] = None


# ────────────────────────────────────────────────────────────────────
#  PatternMemory models
# ────────────────────────────────────────────────────────────────────

@dataclass
class QueryPattern:
    """Learned mapping: query feature signature → optimal strategy params.

    Thompson Sampling fields (*alpha*, *beta*) model the Beta distribution
    over the success probability for this pattern.  Each observed outcome
    shifts the distribution:  ``alpha += confidence`` on success,
    ``beta += (1 - confidence)`` on failure.
    """

    pattern_id: str
    query_type: str = "factual"
    entity_types: List[str] = field(default_factory=list)
    complexity: str = "moderate"
    entity_count: int = 0
    hop_hint: str = "single"
    optimal_mode: str = "DEEP"
    optimal_params: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    success_count: int = 0
    success_rate: float = 0.0
    avg_latency: float = 0.0
    avg_tokens: int = 0
    # Thompson Sampling Beta-distribution priors
    alpha: float = 1.0
    beta_param: float = 1.0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> QueryPattern:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ReasoningChain:
    """Abstracted ReAct trace template linked to a query pattern."""

    template_id: str
    pattern_id: str
    steps: List[Dict[str, str]] = field(default_factory=list)
    success_count: int = 0
    total_count: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReasoningChain:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ────────────────────────────────────────────────────────────────────
#  CorpusMemory models
# ────────────────────────────────────────────────────────────────────

@dataclass
class SemanticExpansion:
    """A single semantic expansion rule (synonym / alias / related)."""

    target: str
    relation: str = "synonym"
    confidence: float = 0.5
    hit_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SemanticBridgeEntry:
    """Term → expansion mapping for keyword generalisation."""

    term: str
    expansions: List[SemanticExpansion] = field(default_factory=list)
    domain: str = "general"
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "expansions": [e.to_dict() for e in self.expansions],
            "domain": self.domain,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SemanticBridgeEntry:
        expansions = [
            SemanticExpansion(**e) if isinstance(e, dict) else e
            for e in d.get("expansions", [])
        ]
        return cls(
            term=d["term"],
            expansions=expansions,
            domain=d.get("domain", "general"),
            updated_at=d.get("updated_at", ""),
        )


# ────────────────────────────────────────────────────────────────────
#  FeedbackMemory model
# ────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackSignal:
    """Unified feedback signal emitted after a completed search."""

    signal_type: str = "implicit"
    query: str = ""
    mode: str = "FAST"
    answer_found: bool = False
    answer_text: str = ""
    cluster_confidence: float = 0.0
    react_loops: int = 0
    files_read_count: int = 0
    files_useful_count: int = 0
    total_tokens: int = 0
    latency_sec: float = 0.0
    keywords_used: List[str] = field(default_factory=list)
    paths_searched: List[str] = field(default_factory=list)
    files_read: List[str] = field(default_factory=list)
    files_discovered: List[str] = field(default_factory=list)
    # Explicit evaluation signals (optional)
    user_verdict: Optional[str] = None
    em_score: Optional[float] = None
    f1_score: Optional[float] = None
    llm_judge_verdict: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # BA-ReAct enrichment (populated when belief tracking is active)
    belief_snapshot: Optional[Dict[str, float]] = None
    mces_triggered_files: Optional[List[str]] = None
    ess_at_termination: Optional[float] = None
    convergence_achieved: bool = False
    high_value_files: Optional[List[str]] = None
    dead_candidates: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ────────────────────────────────────────────────────────────────────
#  Similar query hint (output of QuerySimilarityIndex lookup)
# ────────────────────────────────────────────────────────────────────

@dataclass
class SimilarQueryHint:
    """Hints transferred from a semantically similar historical query."""

    query: str
    similarity: float = 0.0
    confidence: float = 0.0
    mode: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    useful_files: List[str] = field(default_factory=list)
    avoid_files: List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
#  Utility helpers
# ────────────────────────────────────────────────────────────────────

def compute_pattern_id(
    query_type: str,
    complexity: str,
    entity_types: List[str],
    *,
    entity_count: int = 0,
    hop_hint: str = "single",
) -> str:
    """Deterministic ID from query feature signature.

    The *entity_count* and *hop_hint* parameters add finer granularity
    while remaining backward-compatible (default values produce the
    same hash as the old 3-field version when callers omit them).
    """
    key = json.dumps(
        {
            "type": query_type,
            "complexity": complexity,
            "entities": sorted(entity_types),
            "entity_count": entity_count,
            "hop_hint": hop_hint,
        },
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def compute_params_hash(params: Dict[str, Any]) -> str:
    """Stable hash of strategy parameters for failed_strategies dedup."""
    key = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(key.encode()).hexdigest()[:16]
