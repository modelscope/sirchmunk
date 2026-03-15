# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared data types for the evidence extraction pipeline."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SampleWindow:
    """Sampling window with position, content, and evaluation metadata."""

    start_idx: int
    end_idx: int
    content: str

    score: float = 0.0
    fuzz_score: float = 0.0
    reasoning: str = ""
    round_num: int = 0
    source: str = "unknown"


@dataclass
class RoiResult:
    """Region of Interest result from evidence sampling."""

    summary: str
    is_found: bool
    # Format: {"snippet": "xxx", "start": 7, "end": 65, "score": 9.0, "reasoning": "xxx"}
    snippets: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "is_found": self.is_found,
            "snippets": self.snippets,
        }
