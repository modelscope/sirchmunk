# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract base for retrieval memory stores.

Every memory layer (PatternMemory, CorpusMemory, etc.) implements this
protocol so the :class:`RetrievalMemory` manager can uniformly initialise,
maintain, and shut down all layers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


class MemoryStore(ABC):
    """Lifecycle contract for all retrieval memory layers.

    Subclasses choose their own backing storage (JSON files, DuckDB, etc.)
    and expose domain-specific query/record APIs in addition to this base.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this memory store."""

    @abstractmethod
    def initialize(self) -> None:
        """Create storage structures (tables / files) if absent."""

    @abstractmethod
    def decay(self, now: Optional[datetime] = None) -> int:
        """Apply time-based confidence decay.  Returns entries affected."""

    @abstractmethod
    def cleanup(self, max_entries: Optional[int] = None) -> int:
        """Evict low-value or expired entries.  Returns entries removed."""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Diagnostic summary of this store's current state."""

    @abstractmethod
    def close(self) -> None:
        """Release any resources (file handles, DB connections)."""
