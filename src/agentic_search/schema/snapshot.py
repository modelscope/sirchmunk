import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from loguru import logger

from src.agentic_search.insight.text_insights import TextInsights
from src.agentic_search.llm.openai import OpenAIChat
from src.agentic_search.utils.tokenizer_util import TokenizerUtil


@dataclass
class SnapshotInfo:
    """
    Data class to hold snapshot information of the specified file.
    """

    title: str = field(default="")

    description: str = field(default="")

    keywords: List[str] = field(default_factory=list)

    contents: List[str] = field(default_factory=list)

    # Additional resources (e.g., images, tables, or others associated with current file)
    resources: List[Any] = field(default_factory=list)

    def to_dict(self):
        """Convert SnapshotInfo to a dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "contents": self.contents,
            "resources": self.resources,
        }


class Snapshot(ABC):
    """
    Base class for file snapshotting strategies.
    """

    def __init__(
        self,
        llm: Optional[OpenAIChat] = None,
        **kwargs,
    ):

        self.llm: Optional[OpenAIChat] = llm
        self.kwargs = kwargs

    @abstractmethod
    def sampling(self, **kwargs) -> List[str]:
        """
        Abstract method to perform sampling on a file.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class TextSnapshot(Snapshot):
    """
    Text file sampling strategies for snapshotting large text files.
    """

    # TODO: sampling支持多次采样，尤其是针对中间的chunks；后续LLM可以自适应动态调用sampling来做内容增强；
    # Staring mode（凝视模式，参考Radar中的概念）：该过程在search阶段由LLM启发式调度；越复杂的文档，中间chunks采样越多；同时需要多次迭代summary，并回写metadata的snapshot
    # 对于contents，采用多阶段summary
    # def resampling(): ...

    MAX_SNAPSHOT_TOKENS = 2048  # Can be adaptive based on file size

    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

    def __init__(self, llm: Optional[OpenAIChat] = None, **kwargs):

        super().__init__(llm=llm, **kwargs)

        self.text_insights = TextInsights(llm=llm)
        self.tokenizer_util = TokenizerUtil()

    @staticmethod
    def filter_line(line: str) -> Optional[str]:
        """Filter out lines with low information content or noise.

        Filters out:
        - Empty lines or whitespace-only lines
        - Markdown formatting noise (horizontal rules, excessive headings, etc.)
        - Lines consisting mostly of symbols/punctuation
        - URLs, email addresses, and common noise patterns
        - Very short lines (typically < 3 characters after cleaning)
        - Lines with excessive repeated characters
        - Common boilerplate/footer patterns

        Args:
            line: Input line string.

        Returns:
            Cleaned line string if it passes filters, None otherwise.
        """
        if not line:
            return None

        # Strip whitespace and check for empty lines
        stripped = line.strip()
        if not stripped:
            return None

        # Remove leading/trailing whitespace but preserve internal structure
        cleaned = line.rstrip(
            "\n\r"
        )  # Keep original indentation, only remove line endings

        # Check line length (after stripping)
        if len(stripped) < 10:
            return None

        # Markdown-specific noise patterns
        markdown_noise_patterns = [
            r"^\s*-{3,}\s*$",  # Horizontal rules (---, --- , etc.)
            r"^\s*\*{3,}\s*$",  # Horizontal rules (***)
            r"^\s*_{3,}\s*$",  # Horizontal rules (___)
            r"^\s*#{6,}.*$",  # Excessive headings (6+ #)
            r"^\s*>+\s*$",  # Empty blockquotes
            r"^\s*[-*+]\s*$",  # Empty list items
            r"^\s*\d+\.\s*$",  # Empty numbered list items
            r"^\s*!\[.*\]\(.*\)\s*$",  # Standalone images without context
            r"^\s*\[.*\]:\s*https?://\S+\s*$",  # Reference-style link definitions
            r"^\s*```\s*\w*\s*$",  # Code block delimiters (``` or ```python)
            r"^\s*~~~\s*\w*\s*$",  # Alternative code block delimiters
            r"^\s*\|\s*[-:]+\s*\|\s*$",  # Table separator rows
            r"^\s*\|(\s*:?-+:?\s*\|)+\s*$",  # Alternative table separator format
        ]

        # Check for Markdown noise patterns
        for pattern in markdown_noise_patterns:
            if re.match(pattern, stripped):
                return None

        # Remove common noise patterns and check if line becomes empty
        noise_patterns = [
            r"^https?://\S+",  # URLs at start of line
            r"\bhttps?://\S+\b",  # URLs anywhere
            r"\bwww\.\S+\b",  # www URLs
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # emails
            r"^[\s\*\-\_\=\#\+]+$",  # lines with only symbols
        ]

        # Check for excessive symbols ratio
        alphanumeric_chars = sum(c.isalnum() for c in stripped)
        if len(stripped) > 0 and alphanumeric_chars / len(stripped) < 0.3:
            return None

        # Check for excessive repeated characters (e.g., "..........." or "----------")
        if TextSnapshot._has_excessive_repetition(stripped):
            return None

        # Check for common boilerplate patterns
        boilerplate_patterns = [
            r"^(copyright|©|\(c\))",
            r"^all rights reserved",
            r"^confidential",
            r"^page \d+",
            r"^\d+\s+of\s+\d+$",
            r"^file:|^path:",
            r"^created:|^modified:",
            r"^author:",
            r"^version:",
            r"^build:",
            r"^generated by",
            r"^this document",
            r"^last updated",
            r"^table of contents",
            r"^toc:",
            r"^contents",
            r"^index",
        ]

        stripped_lower = stripped.lower()
        for pattern in boilerplate_patterns:
            if re.search(pattern, stripped_lower):
                return None

        # Apply noise pattern removal and check if meaningful content remains
        temp_line = stripped
        for pattern in noise_patterns + markdown_noise_patterns:
            temp_line = re.sub(pattern, "", temp_line)

        # After removing noise, check if we still have meaningful content
        temp_stripped = temp_line.strip()
        if (
            len(temp_stripped) < 3
            or (
                sum(c.isalnum() for c in temp_stripped) / len(temp_stripped)
                if len(temp_stripped) > 0
                else 0
            )
            < 0.4
        ):
            return None

        # Additional check: reject lines that are mostly Markdown syntax characters
        markdown_chars = sum(1 for c in stripped if c in "#*_-~`>![](){}|:")
        if len(stripped) > 0 and markdown_chars / len(stripped) > 0.6:
            return None

        return cleaned

    @staticmethod
    def _has_excessive_repetition(text: str) -> bool:
        """Check if text has excessive character repetition.

        Args:
            text: Input text string.

        Returns:
            True if excessive repetition detected.
        """
        if len(text) < 10:
            return False

        # Check for sequences of same character (e.g., "------", "......")
        for i in range(len(text) - 4):
            if text[i] == text[i + 1] == text[i + 2] == text[i + 3] == text[i + 4]:
                # Allow some legitimate cases like "hello....." but not full line
                if text.count(text[i]) / len(text) > 0.6:
                    return True

        # Check for alternating patterns (e.g., "- - - -", ". . . .")
        if re.search(r"([^\w\s])\s*\1\s*\1\s*\1", text):
            return True

        return False

    def sampling(
        self,
        file_path: Union[str, Path],
        max_snapshot_tokens: int = MAX_SNAPSHOT_TOKENS,
        max_file_size: int = MAX_FILE_SIZE,
    ) -> Union[SnapshotInfo, None]:

        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        if file_size > max_file_size:  # TODO: add more strategies for large files
            logger.warning(
                f"File size {file_size} exceeds maximum allowed size of {max_file_size} bytes, skipping snapshot."
            )
            return None

        snapshot_info = SnapshotInfo()

        selected_lines = []
        accumulated_tokens = 0
        line_count = 0

        # Stream through file once
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = self.filter_line(line)
                if not line:
                    continue

                line_count += 1
                line_tokens = self.tokenizer_util.count_tokens(contents=line.strip())

                # Adaptive sampling strategy:
                # - Early in file: higher acceptance probability
                # - Near token limit: lower probability
                # - Always accept if budget allows and line is small

                # Calculate acceptance probability
                token_fill_ratio = (
                    accumulated_tokens / max_snapshot_tokens
                    if max_snapshot_tokens > 0
                    else 0
                )
                line_token_ratio = (
                    line_tokens / max_snapshot_tokens if max_snapshot_tokens > 0 else 0
                )

                # Base probability decreases as we fill token budget
                base_prob = max(0.2, 1.0 - token_fill_ratio * 2.0)

                # Adjust for line size (prefer smaller lines when budget is tight)
                size_penalty = (
                    min(1.0, 0.8 + (1.0 - line_token_ratio) * 0.4)
                    if token_fill_ratio > 0.5
                    else 1.0
                )

                # Add some randomness to avoid deterministic patterns
                noise = random.uniform(0.8, 1.2)
                acceptance_prob = min(1.0, base_prob * size_penalty * noise)

                # Force accept conditions
                if (accumulated_tokens == 0 and line_tokens > 0) or (
                    accumulated_tokens + line_tokens <= max_snapshot_tokens * 0.7
                ):
                    acceptance_prob = 1.0

                # Sampling decision
                if random.random() < acceptance_prob:
                    # Check if adding this line keeps us within reasonable bounds
                    if (
                        accumulated_tokens + line_tokens <= max_snapshot_tokens * 1.5
                    ):  # Allow 50% overflow max
                        selected_lines.append((line, line_count))
                        accumulated_tokens += line_tokens

                        # Early termination if we've significantly exceeded target
                        if accumulated_tokens >= max_snapshot_tokens * 1.2:
                            break

        # Sort by original line number to preserve document structure
        selected_lines.sort(key=lambda x: x[1])
        logger.info(
            f"Got {len(selected_lines)} selected lines, total tokens: {accumulated_tokens}"
        )

        snapshot_info.contents = [line for line, _ in selected_lines]

        # Get keywords/key phrase
        try:
            snapshot_info.keywords = self.text_insights.extract_phrase(
                contents=snapshot_info.contents
            )
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            snapshot_info.keywords = []

        try:
            # TODO: 与phrase一起放到同一个llm calling中
            snapshot_info.description = self.text_insights.extract_toc(
                contents=snapshot_info.contents
            )
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            snapshot_info.description = ""

        return snapshot_info
