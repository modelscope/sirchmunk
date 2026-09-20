# Copyright (c) ModelScope Contributors. All rights reserved.
"""Confidence-aware fusion and fast-track policy for heterogeneous retrieval."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence


_ROUTE_NAMES = ("lexical", "entity", "structure", "directory", "existing")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ConfidenceConfig:
    """Centralized confidence and fast-track thresholds."""

    mode: str = "soft"
    rrf_k: int = 60
    global_threshold: float = 0.90
    margin_threshold: float = 0.10
    route_support_threshold: float = 0.35
    directory_high_threshold: float = 0.85
    entity_high_threshold: float = 0.90
    min_supporting_routes: int = 2
    fast_track_max_files: int = 1
    fast_track_max_loops: int = 3
    route_weights: Mapping[str, float] = field(default_factory=lambda: {
        "lexical": 1.0,
        "entity": 1.2,
        "structure": 0.8,
        "directory": 0.5,
        "existing": 0.7,
    })
    route_reliability: Mapping[str, float] = field(default_factory=lambda: {
        "lexical": 0.75,
        "entity": 0.90,
        "structure": 0.85,
        "directory": 1.00,
        "existing": 0.55,
    })

    @classmethod
    def from_env(cls) -> "ConfidenceConfig":
        """Load policy from environment, falling back to conservative defaults."""
        mode = os.getenv("LENS_CONFIDENCE_MODE", "soft").strip().lower()
        if mode not in {"off", "monitor", "soft"}:
            mode = "soft"
        return cls(
            mode=mode,
            rrf_k=max(1, _env_int("LENS_CONFIDENCE_RRF_K", 60)),
            global_threshold=max(0.0, min(1.0, _env_float(
                "LENS_CONFIDENCE_GLOBAL_THRESHOLD", 0.90,
            ))),
            margin_threshold=max(0.0, min(1.0, _env_float(
                "LENS_CONFIDENCE_MARGIN", 0.10,
            ))),
            route_support_threshold=max(0.0, min(1.0, _env_float(
                "LENS_CONFIDENCE_ROUTE_THRESHOLD", 0.35,
            ))),
            directory_high_threshold=max(0.0, min(1.0, _env_float(
                "LENS_CONFIDENCE_DIR_THRESHOLD", 0.85,
            ))),
            entity_high_threshold=max(0.0, min(1.0, _env_float(
                "LENS_CONFIDENCE_ENTITY_THRESHOLD", 0.90,
            ))),
            min_supporting_routes=max(1, _env_int(
                "LENS_CONFIDENCE_MIN_SUPPORT", 2,
            )),
            fast_track_max_files=max(1, _env_int(
                "LENS_FAST_TRACK_MAX_FILES", 1,
            )),
            fast_track_max_loops=max(1, _env_int(
                "LENS_FAST_TRACK_MAX_LOOPS", 3,
            )),
            route_weights={
                route: max(0.0, _env_float(
                    f"LENS_CONFIDENCE_WEIGHT_{route.upper()}", default,
                ))
                for route, default in {
                    "lexical": 1.0,
                    "entity": 1.2,
                    "structure": 0.8,
                    "directory": 0.5,
                    "existing": 0.7,
                }.items()
            },
            route_reliability={
                route: max(0.0, min(1.0, _env_float(
                    f"LENS_CONFIDENCE_RELIABILITY_{route.upper()}", default,
                )))
                for route, default in {
                    "lexical": 0.75,
                    "entity": 0.90,
                    "structure": 0.85,
                    "directory": 1.00,
                    "existing": 0.55,
                }.items()
            },
        )


@dataclass(frozen=True)
class LexicalConfidenceFeatures:
    """Query-local lexical signals whose scales remain comparable."""

    term_coverage: float
    group_coverage: float
    rarity: float
    density: float
    exact_phrase: bool = False


def calibrate_lexical_features(features: LexicalConfidenceFeatures) -> float:
    """Calibrate lexical relevance without letting raw TF dominate."""
    confidence = (
        0.35 * max(0.0, min(1.0, features.term_coverage))
        + 0.25 * max(0.0, min(1.0, features.group_coverage))
        + 0.30 * max(0.0, min(1.0, features.rarity))
        + 0.10 * max(0.0, min(1.0, features.density))
    )
    if features.exact_phrase:
        confidence += 0.08
    # High frequency with weak coverage is evidence of corpus prevalence, not
    # query relevance.  Such candidates may rank, but cannot dominate.
    if features.term_coverage < 0.5 and not features.exact_phrase:
        confidence = min(confidence, 0.74)
    return round(max(0.0, min(0.95, confidence)), 4)


@dataclass
class ConfidenceDecision:
    """Result of confidence-aware rank fusion and route selection."""

    ranked_files: List[str]
    fusion_scores: Dict[str, float]
    global_confidence: Dict[str, float]
    route_confidence: Dict[str, Dict[str, float]]
    supporting_routes: Dict[str, int]
    dominant_file: Optional[str] = None
    dominant_evidence_kind: Optional[str] = None
    route_collapse: bool = False
    reasoning_profile: str = "standard"
    fast_track: bool = False
    reason: str = "no_candidates"
    margin: float = 0.0

    def telemetry(self) -> Dict[str, object]:
        visible = set(self.ranked_files[:20])
        return {
            "retrieval_fusion_method": "confidence_weighted_rrf",
            "retrieval_fusion_scores": self.fusion_scores,
            "route_confidence": {
                path: scores for path, scores in self.route_confidence.items()
                if path in visible
            },
            "global_confidence": {
                path: score for path, score in self.global_confidence.items()
                if path in visible
            },
            "confidence_supporting_routes": {
                path: count for path, count in self.supporting_routes.items()
                if path in visible
            },
            "confidence_margin": round(self.margin, 6),
            "dominant_route_file": self.dominant_file,
            "dominant_evidence_kind": self.dominant_evidence_kind,
            "dominant_route_collapse": self.route_collapse,
            "dominant_reasoning_profile": self.reasoning_profile,
            "dominant_route_fast_track": self.fast_track,
            "dominant_route_reason": self.reason,
        }


class ConfidenceFusionPolicy:
    """Fuse route rankings while preserving calibrated route confidence."""

    def __init__(self, config: Optional[ConfidenceConfig] = None) -> None:
        self.config = config or ConfidenceConfig.from_env()

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def fuse(
        self,
        rankings: Mapping[str, Sequence[str]],
        route_scores: Optional[Mapping[str, Mapping[str, float]]] = None,
        *,
        fast_track_eligible: bool = False,
        dominant_evidence: Optional[Mapping[str, str]] = None,
        allow_loop_reduction: bool = True,
    ) -> ConfidenceDecision:
        """Fuse route ranks and calibrated confidences into one decision."""
        raw_scores = route_scores or {}
        dominant_evidence = dominant_evidence or {}
        rrf_scores: Dict[str, float] = {}
        route_confidence: Dict[str, Dict[str, float]] = {}
        first_seen: Dict[str, int] = {}
        ordinal = 0

        for route in _ROUTE_NAMES:
            ranking = list(rankings.get(route, []))
            weight = float(self.config.route_weights.get(route, 1.0))
            seen: set[str] = set()
            rank = 0
            for file_path in ranking:
                if not file_path or file_path in seen:
                    continue
                seen.add(file_path)
                rank += 1
                if file_path not in first_seen:
                    first_seen[file_path] = ordinal
                    ordinal += 1
                rrf_scores[file_path] = rrf_scores.get(file_path, 0.0) + (
                    weight / (self.config.rrf_k + rank)
                )

                route_raw = raw_scores.get(route, {}).get(file_path)
                if route_raw is None:
                    # Rank-only routes carry a bounded weak prior; they can
                    # corroborate but cannot independently dominate.
                    confidence = min(0.30, 0.30 / (1.0 + 0.15 * (rank - 1)))
                else:
                    confidence = self._clamp(route_raw)
                route_confidence.setdefault(file_path, {})[route] = confidence

        if not rrf_scores:
            return ConfidenceDecision([], {}, {}, {})

        max_rrf = max(rrf_scores.values()) or 1.0
        global_confidence: Dict[str, float] = {}
        supporting_routes: Dict[str, int] = {}
        composite_scores: Dict[str, float] = {}
        for file_path, rrf_score in rrf_scores.items():
            confidences = route_confidence.get(file_path, {})
            miss_probability = 1.0
            supports = 0
            for route, confidence in confidences.items():
                reliability = float(self.config.route_reliability.get(route, 0.5))
                effective = self._clamp(confidence * reliability)
                miss_probability *= 1.0 - effective
                if confidence >= self.config.route_support_threshold:
                    supports += 1
            global_score = self._clamp(1.0 - miss_probability)
            rrf_normalized = rrf_score / max_rrf
            composite = 0.65 * global_score + 0.35 * rrf_normalized
            global_confidence[file_path] = round(global_score, 6)
            supporting_routes[file_path] = supports
            composite_scores[file_path] = round(composite, 8)

        ranking_scores = (
            {path: rrf_scores[path] / max_rrf for path in rrf_scores}
            if self.config.mode in {"off", "monitor"}
            else composite_scores
        )
        ranked_files = sorted(
            ranking_scores,
            key=lambda path: (
                -ranking_scores[path],
                first_seen.get(path, 0),
            ),
        )
        top_file = ranked_files[0]
        top_confidence = global_confidence[top_file]
        second_confidence = (
            global_confidence[ranked_files[1]] if len(ranked_files) > 1 else 0.0
        )
        margin = top_confidence - second_confidence
        top_routes = route_confidence.get(top_file, {})
        evidence_kind = dominant_evidence.get(top_file)
        unique_exact = evidence_kind == "unique_exact_entity"
        directory_direct = evidence_kind == "directory_direct_match"
        has_strong_route = (
            top_routes.get("directory", 0.0) >= self.config.directory_high_threshold
            or top_routes.get("entity", 0.0) >= self.config.entity_high_threshold
        )
        strong_single_evidence = unique_exact or directory_direct
        required_global = (
            min(
                self.config.global_threshold,
                self.config.entity_high_threshold
                * float(self.config.route_reliability.get("entity", 0.9)),
            )
            if unique_exact else self.config.global_threshold
        )
        support_ok = (
            supporting_routes[top_file] >= self.config.min_supporting_routes
            or strong_single_evidence
        )
        eligible = (
            fast_track_eligible
            and top_confidence >= required_global
            and margin >= self.config.margin_threshold
            and support_ok
            and has_strong_route
        )
        route_collapse = eligible and self.config.mode == "soft"
        reasoning_profile = (
            "reduced" if route_collapse and allow_loop_reduction else "standard"
        )
        fast_track = route_collapse

        if self.config.mode == "off":
            reason = "confidence_mode_off"
        elif not fast_track_eligible:
            reason = "query_not_fast_track_eligible"
        elif not has_strong_route:
            reason = "no_strong_precision_route"
        elif not support_ok:
            reason = "insufficient_route_consensus"
        elif top_confidence < required_global:
            reason = "global_confidence_below_threshold"
        elif margin < self.config.margin_threshold:
            reason = "confidence_margin_too_small"
        elif self.config.mode == "monitor":
            reason = "monitor_only"
        elif evidence_kind:
            reason = f"dominant_{evidence_kind}"
        else:
            reason = "dominant_route_consensus"

        return ConfidenceDecision(
            ranked_files=ranked_files,
            fusion_scores={
                path: round(ranking_scores[path], 8) for path in ranked_files[:50]
            },
            global_confidence=global_confidence,
            route_confidence=route_confidence,
            supporting_routes=supporting_routes,
            dominant_file=top_file if eligible else None,
            dominant_evidence_kind=evidence_kind if eligible else None,
            route_collapse=route_collapse,
            reasoning_profile=reasoning_profile,
            fast_track=fast_track,
            reason=reason,
            margin=margin,
        )


def calibrate_match_score(raw_score: float) -> float:
    """Map an unbounded positive retrieval score to a conservative [0, 1]."""
    if raw_score <= 0:
        return 0.0
    return min(0.95, 1.0 - math.exp(-float(raw_score) / 3.5))
