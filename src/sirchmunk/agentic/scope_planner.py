# Copyright (c) ModelScope Contributors. All rights reserved.
"""Deterministic scope planning for large heterogeneous search roots."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from sirchmunk.agentic.file_system_tools import DirectoryProfile, DirectoryProfiler
from sirchmunk.utils.constants import (
    SCOPE_PLANNER_CONFIDENCE,
    SCOPE_PLANNER_MAX_PATHS,
)

_TOKEN_RE = re.compile(r"[a-z0-9\u4e00-\u9fff]+", re.IGNORECASE)
_SUMMARY_MARKERS = {
    "summarize",
    "summary",
    "overview",
    "explain",
    "analyze",
    "review",
    "总结",
    "概述",
    "分析",
    "解读",
}
_FORMAT_ALIASES = {
    ".pdf": {"pdf", "paper", "papers", "论文", "文献"},
    ".doc": {"doc", "word", "document", "文档"},
    ".docx": {"docx", "word", "document", "文档"},
    ".xls": {"xls", "excel", "spreadsheet", "表格"},
    ".xlsx": {"xlsx", "excel", "spreadsheet", "表格"},
    ".ppt": {"ppt", "powerpoint", "slides", "幻灯片"},
    ".pptx": {"pptx", "powerpoint", "slides", "幻灯片"},
    ".md": {"md", "markdown", "文档"},
    ".txt": {"txt", "text", "文本"},
    ".parquet": {"parquet", "dataset", "table", "数据集"},
    "<extensionless>": {"jsonl", "shard", "wiki", "raw", "语料", "分片"},
}


@dataclass
class ScopePlan:
    """Search-range recommendation with an auditable confidence score."""

    original_paths: List[str]
    effective_paths: List[str]
    confidence: float
    narrowed: bool
    recommended_tools: List[str]
    reason: str
    profile: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "original_paths": self.original_paths,
            "effective_paths": self.effective_paths,
            "confidence": round(self.confidence, 4),
            "narrowed": self.narrowed,
            "recommended_tools": self.recommended_tools,
            "reason": self.reason,
            "profile": self.profile,
        }


class ScopePlanner:
    """Plan a conservative search scope from filesystem metadata only."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        confidence_threshold: float = SCOPE_PLANNER_CONFIDENCE,
        max_paths: int = SCOPE_PLANNER_MAX_PATHS,
    ) -> None:
        self.roots = [str(Path(root).expanduser().resolve()) for root in roots]
        self.confidence_threshold = min(1.0, max(0.0, confidence_threshold))
        self.max_paths = max(1, max_paths)
        self.profiler = DirectoryProfiler(self.roots)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(value or "") if len(token) > 1}

    @staticmethod
    def _normalise_name(value: str) -> str:
        return " ".join(
            sorted(
                ScopePlanner._tokens(value.replace("_", " ").replace("-", " "))
            )
        )

    @staticmethod
    def _candidate_directories(profile: DirectoryProfile) -> Dict[Path, int]:
        counts: Counter[Path] = Counter()
        root = Path(profile.roots[0])
        for record in profile.files:
            relative = Path(record.relative_path)
            parent_parts = relative.parts[:-1]
            for depth in range(1, min(3, len(parent_parts)) + 1):
                counts[root.joinpath(*parent_parts[:depth])] += 1
        return dict(counts)

    def _score_directory(
        self,
        query: str,
        query_tokens: set[str],
        directory: Path,
        profile: DirectoryProfile,
    ) -> float:
        root = Path(profile.roots[0])
        try:
            relative = directory.relative_to(root).as_posix()
        except ValueError:
            relative = directory.name
        name = self._normalise_name(relative)
        name_tokens = self._tokens(name)
        score = 0.0
        query_lower = query.lower()
        compact_name = relative.lower().replace("_", " ").replace("-", " ")
        if compact_name and compact_name in query_lower:
            score += 0.9
        if directory.name.lower() in query_lower and len(directory.name) >= 3:
            score += 0.8
        if name_tokens:
            overlap = len(name_tokens & query_tokens) / len(name_tokens)
            score += 0.65 * overlap

        records = [
            record
            for record in profile.files
            if directory == Path(record.path).parent or directory in Path(record.path).parents
        ]
        if records:
            extensions = Counter(record.extension or "<extensionless>" for record in records)
            for extension, aliases in _FORMAT_ALIASES.items():
                if query_tokens & aliases and extensions.get(extension, 0):
                    score += 0.45 * (extensions[extension] / len(records))
        return min(1.0, score)

    def plan(self, query: str, *, max_depth: int = 4) -> ScopePlan:
        """Return a high-confidence narrowed scope or preserve the original roots."""
        query_tokens = self._tokens(query)
        profiles = [self.profiler.profile(root, max_depth=max_depth) for root in self.roots]
        candidates: List[tuple[float, int, str]] = []
        total_files = 0
        total_bytes = 0
        extension_counts: Counter[str] = Counter()
        truncated = False

        for profile in profiles:
            total_files += len(profile.files)
            total_bytes += profile.total_size_bytes
            extension_counts.update(profile.extension_counts)
            truncated = truncated or profile.truncated
            for directory, file_count in self._candidate_directories(profile).items():
                score = self._score_directory(query, query_tokens, directory, profile)
                if score > 0:
                    candidates.append((score, file_count, str(directory)))

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected: List[str] = []
        top_score = candidates[0][0] if candidates else 0.0
        if top_score >= self.confidence_threshold:
            floor = max(self.confidence_threshold, top_score - 0.12)
            for score, _, path in candidates:
                if score < floor or len(selected) >= self.max_paths:
                    break
                if any(Path(existing) in Path(path).parents for existing in selected):
                    continue
                selected.append(path)

        narrowed = bool(selected)
        effective_paths = selected or list(self.roots)
        summary_intent = bool(query_tokens & _SUMMARY_MARKERS)
        extensionless_count = extension_counts.get("<extensionless>", 0)
        shard_ratio = extensionless_count / max(total_files, 1)
        recommended_tools = ["keyword_search"]
        if not narrowed or len(effective_paths) > 1:
            recommended_tools.insert(0, "file_list")
        if summary_intent or (total_files <= 20 and total_bytes <= 2_000_000):
            recommended_tools.append("file_read")
        if shard_ratio >= 0.5 and total_files >= 100 and not narrowed:
            recommended_tools = ["file_list", "keyword_search"]

        reason = (
            f"Selected {len(selected)} directory scope(s) from deterministic name/type "
            f"signals (confidence={top_score:.2f})."
            if narrowed
            else "No high-confidence directory match; preserved all configured roots."
        )
        return ScopePlan(
            original_paths=list(self.roots),
            effective_paths=effective_paths,
            confidence=top_score,
            narrowed=narrowed,
            recommended_tools=list(dict.fromkeys(recommended_tools)),
            reason=reason,
            profile={
                "file_count": total_files,
                "total_size_bytes": total_bytes,
                "extension_counts": dict(extension_counts),
                "extensionless_shard_ratio": round(shard_ratio, 4),
                "truncated": truncated,
            },
        )
