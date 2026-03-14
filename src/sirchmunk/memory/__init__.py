# Copyright (c) ModelScope Contributors. All rights reserved.
"""Self-evolving retrieval memory system.

Provides a layered memory architecture that learns from each search session:

- **PatternMemory**: query type → optimal search strategy mapping
- **CorpusMemory**: entity-path index + semantic keyword expansion
- **PathMemory**: file path hotness and utility statistics
- **FailureMemory**: noise keywords, dead paths, failed strategies
- **FeedbackMemory**: implicit/explicit signal collection and dispatch

Usage::

    from sirchmunk.memory import RetrievalMemory, FeedbackSignal

    memory = RetrievalMemory(work_path="/path/to/workspace")
    hint = memory.suggest_strategy("Who invented the telephone?")
"""

from .manager import RetrievalMemory
from .schemas import FeedbackSignal, StrategyHint

__all__ = ["RetrievalMemory", "FeedbackSignal", "StrategyHint"]
