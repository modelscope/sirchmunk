# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Search context for tracking state across agentic retrieval loops.

Provides LLM token budget enforcement, file-level deduplication, and
structured logging of all retrieval operations within a single
search session.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class RetrievalLog:
    """Single retrieval operation record.

    Attributes:
        tool_name: Name of the tool that performed the retrieval.
        tokens: Approximate tokens consumed by this operation.
        timestamp: When the operation occurred.
        metadata: Additional context (e.g., keywords used, files returned).
    """

    tool_name: str
    tokens: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tokens": self.tokens,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SearchContext:
    """Stateful context for a single agentic search session.

    Tracks LLM token consumption, file deduplication, and retrieval logs
    across multiple tool calls within one ReAct loop execution.

    The **token budget** is enforced against ``total_llm_tokens`` — the
    cumulative token count reported by the LLM API (prompt + completion).
    This is the only reliable metric; approximate "retrieval tokens"
    (char-count / 4) proved unreliable and have been removed.

    Attributes:
        max_token_budget: Maximum total LLM tokens allowed per session.
        total_llm_tokens: Running sum of LLM tokens (from API usage).
        llm_usages: Raw usage dicts from each LLM call.
        read_file_ids: Set of file paths already fully read (for dedup).
        retrieval_logs: Chronological list of all retrieval operations.
        search_history: Raw search queries issued during this session.
        loop_count: Number of ReAct loops executed so far.
        max_loops: Maximum number of ReAct loops allowed.
        start_time: Session start timestamp.
    """

    max_token_budget: int = 64000
    max_loops: int = 10

    total_llm_tokens: int = field(default=0, init=False)
    llm_usages: List[Dict[str, Any]] = field(default_factory=list, init=False)
    read_file_ids: Set[str] = field(default_factory=set, init=False)
    retrieval_logs: List[RetrievalLog] = field(default_factory=list, init=False)
    search_history: List[str] = field(default_factory=list, init=False)
    loop_count: int = field(default=0, init=False)
    start_time: datetime = field(default_factory=datetime.now, init=False)

    # ---- Token accounting ----

    def add_llm_tokens(self, tokens: int, usage: Optional[Dict[str, Any]] = None) -> None:
        """Record tokens consumed by an LLM generation call.

        Args:
            tokens: Total tokens from the LLM response.
            usage: Raw usage dict from ``OpenAIChatResponse.usage``
                   (e.g. ``{"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}``).
        """
        self.total_llm_tokens += tokens
        if usage:
            self.llm_usages.append(usage)

    def is_budget_exceeded(self) -> bool:
        """Check whether the LLM token budget has been exhausted."""
        return self.total_llm_tokens > self.max_token_budget

    @property
    def budget_remaining(self) -> int:
        """LLM tokens remaining in the budget."""
        return max(0, self.max_token_budget - self.total_llm_tokens)

    # ---- File deduplication ----

    def mark_file_read(self, file_path: str) -> None:
        """Mark a file as fully read to prevent redundant reads."""
        self.read_file_ids.add(str(file_path))

    def is_file_read(self, file_path: str) -> bool:
        """Check whether a file has already been fully read."""
        return str(file_path) in self.read_file_ids

    # ---- Logging ----

    def add_log(
        self,
        tool_name: str,
        tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a retrieval operation.

        The ``tokens`` parameter is an approximate count for diagnostic
        purposes only — it does NOT affect the budget (which is purely
        LLM-token based).

        Args:
            tool_name: Which tool performed the operation.
            tokens: Approximate tokens consumed (for diagnostics).
            metadata: Arbitrary context for debugging / analytics.
        """
        self.retrieval_logs.append(
            RetrievalLog(
                tool_name=tool_name,
                tokens=tokens,
                metadata=metadata or {},
            )
        )

    def add_search(self, query: str) -> None:
        """Record a search query issued during this session."""
        self.search_history.append(query)

    # ---- Loop management ----

    def increment_loop(self) -> None:
        """Advance the loop counter by one."""
        self.loop_count += 1

    def is_loop_limit_reached(self) -> bool:
        """Check whether the maximum loop count has been reached."""
        return self.loop_count >= self.max_loops

    # ---- Serialization ----

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context state for logging / diagnostics."""
        return {
            "max_token_budget": self.max_token_budget,
            "max_loops": self.max_loops,
            "total_llm_tokens": self.total_llm_tokens,
            "budget_remaining": self.budget_remaining,
            "llm_call_count": len(self.llm_usages),
            "read_file_count": len(self.read_file_ids),
            "retrieval_log_count": len(self.retrieval_logs),
            "search_history": self.search_history,
            "loop_count": self.loop_count,
            "start_time": self.start_time.isoformat(),
        }

    def summary(self) -> str:
        """Human-readable one-line summary of context state."""
        return (
            f"phases={self.loop_count}/{self.max_loops} "
            f"llm_tokens={self.total_llm_tokens}/{self.max_token_budget} "
            f"llm_calls={len(self.llm_usages)} "
            f"files_read={len(self.read_file_ids)} "
            f"searches={len(self.search_history)}"
        )
