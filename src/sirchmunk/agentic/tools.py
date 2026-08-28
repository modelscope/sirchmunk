# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Hierarchical retrieval tools for the ReAct search agent.

Provides a tool abstraction layer and four concrete tools that operate
at different granularities — from lightweight keyword search to deep
file reading and knowledge base querying.  All tools are stateless;
side-effects (token accounting, dedup) are recorded via SearchContext.
"""
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.constants import (
    FILE_READ_MAX_CHARS,
    FILE_READ_SMALL_FILE_CHARS,
    FILE_READ_WINDOW_LINES,
    GREP_KEYWORD_CONCURRENT_LIMIT,
    GREP_TIMEOUT,
)
from sirchmunk.utils.file_utils import fast_extract, looks_like_plain_text_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseTool(ABC):
    """Abstract base for all ReAct retrieval tools.

    Each tool exposes:
    - ``name``: unique identifier used by the LLM to invoke it.
    - ``get_schema()``: OpenAI function-calling schema.
    - ``execute()``: run the tool and return (result_text, metadata).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must be unique within a ToolRegistry)."""

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI function-calling schema for this tool."""

    @abstractmethod
    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute the tool.

        Args:
            context: Shared search context for token accounting and dedup.
            **kwargs: Tool-specific arguments (match schema properties).

        Returns:
            Tuple of (result_text_for_llm, metadata_dict_for_logging).
        """


class ToolRegistry:
    """Registry that manages a set of tools and dispatches execution.

    Usage::

        registry = ToolRegistry()
        registry.register(KeywordSearchTool(retriever))
        schemas = registry.get_all_schemas()
        result, meta = await registry.execute("keyword_search", context, keywords=["foo"])
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance (overwrites if name exists)."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by name."""
        return self._tools.get(name)

    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling schemas for all registered tools."""
        return [
            {"type": "function", "function": tool.get_schema()}
            for tool in self._tools.values()
        ]

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def execute(
        self,
        tool_name: str,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Dispatch execution to the named tool.

        Raises:
            KeyError: If tool_name is not registered.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' is not registered. Available: {self.tool_names}")
        try:
            return await tool.execute(context=context, **kwargs)
        except Exception as exc:
            error_msg = f"[{tool_name}] execution error: {exc}"
            logger.error(error_msg)
            return error_msg, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 1: Keyword Search (lightweight — returns snippets only)
# ---------------------------------------------------------------------------

class KeywordSearchTool(BaseTool):
    """Lexical keyword search via ripgrep-all.

    Returns matching **line snippets** (not full file content) ranked by
    TF-IDF relevance.  Cheapest tool in terms of token cost.

    Uses ``literal=True`` by default so that keywords containing regex
    metacharacters (``+``, ``(``, ``.``, CJK punctuation, etc.) are
    matched verbatim.  If the literal search returns no results, a
    fallback regex search is attempted with escaped metacharacters.
    """

    # Default patterns that should always be excluded from keyword search
    _DEFAULT_EXCLUDE: List[str] = ["*.pyc", "*.log", "__pycache__", ".DS_Store", "._*"]

    def __init__(
        self,
        retriever: GrepRetriever,
        paths: Union[str, Path, List[str], List[Path]],
        max_depth: int = 5,
        max_results: int = 10,
        max_snippet_lines: int = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        timeout: float = GREP_TIMEOUT,
        max_concurrent_terms: int = GREP_KEYWORD_CONCURRENT_LIMIT,
    ) -> None:
        self._retriever = retriever
        self._paths = paths
        self._max_depth = max_depth
        self._max_results = max_results
        self._max_snippet_lines = max_snippet_lines
        self._include = include
        self._timeout = max(1.0, timeout)
        self._max_concurrent_terms = max(1, max_concurrent_terms)
        # Merge caller-provided excludes with sensible defaults (deduped)
        self._exclude = list(set(self._DEFAULT_EXCLUDE) | set(exclude or []))

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Search files using keywords. Returns ranked file snippets "
                "with matching lines. Best for known entities, names, codes, "
                "and technical terms. Low token cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of keywords or short phrases to search for.",
                    },
                },
                "required": ["keywords"],
            },
        }

    async def _do_search_per_term(
        self,
        keywords: List[str],
        *,
        literal: bool,
        regex: bool,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Search each keyword individually and return results plus failures.

        ripgrep's ``-F`` (fixed-string / literal) mode does NOT support
        ``|`` alternation — it treats ``|`` as a literal character.
        So ``keyword1|keyword2`` with ``-F`` searches for the six-char
        string "keyword1|keyword2" literally, missing both individual
        terms.

        This method works around that by issuing one rga call per
        keyword and then merging the per-file results.  Each match is
        tagged with ``_keyword`` so the formatter can ensure keyword
        diversity in the output snippets.
        """
        import asyncio as _aio

        term_semaphore = _aio.Semaphore(self._max_concurrent_terms)

        async def _single(term: str) -> List[Dict[str, Any]]:
            async with term_semaphore:
                return await self._retriever.retrieve(
                    terms=term,
                    path=self._paths,
                    logic="or",
                    case_sensitive=False,
                    literal=literal,
                    regex=regex,
                    max_depth=self._max_depth,
                    include=self._include,
                    exclude=self._exclude,
                    timeout=self._timeout,
                )

        raw_lists = await _aio.gather(
            *[_single(keyword) for keyword in keywords],
            return_exceptions=True,
        )

        # Preserve successful partial results even if another term exhausted both
        # the rga and rg backends. Cancellation still propagates immediately.
        combined: List[Dict[str, Any]] = []
        failures: List[str] = []
        for keyword, raw in zip(keywords, raw_lists):
            if isinstance(raw, _aio.CancelledError):
                raise raw
            if isinstance(raw, Exception):
                failures.append(f"{keyword}: {raw}")
                logger.warning(
                    "[keyword_search] Search failed for term %r: %s", keyword, raw
                )
                continue
            for item in raw:
                if item.get("type") == "match":
                    item["_keyword"] = keyword
                combined.append(item)

        return (
            self._retriever.merge_results(combined, limit=self._max_results * 2),
            failures,
        )

    async def _do_search_regex(
        self,
        keywords: List[str],
    ) -> List[Dict[str, Any]]:
        """Search using escaped-regex OR alternation (single rga call).

        Wraps each keyword in ``(?:re.escape(k))`` and joins with ``|``
        so that ripgrep handles the alternation natively in regex mode.
        """
        import re as _re

        escaped = [_re.escape(k) for k in keywords]
        pattern = "|".join(f"(?:{e})" for e in escaped)

        raw = await self._retriever.retrieve(
            terms=pattern,
            path=self._paths,
            logic="or",
            case_sensitive=False,
            literal=False,
            regex=True,
            max_depth=self._max_depth,
            include=self._include,
            exclude=self._exclude,
            timeout=self._timeout,
        )
        return self._retriever.merge_results(raw, limit=self._max_results)

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        keywords: List[str] = kwargs.get("keywords", [])
        if not keywords:
            return "No keywords provided.", {}

        context.add_search(" ".join(keywords))

        # Strategy 1: per-keyword literal search (safe for metacharacters,
        # avoids the `-F` + `|` bug where rga treats `|` literally)
        results, search_failures = await self._do_search_per_term(
            keywords, literal=True, regex=False
        )

        # Strategy 2: escaped-regex OR search (single rga call, handles
        # adapters that only work in regex mode — e.g. some PDF/DOCX)
        if not results:
            logger.info("[keyword_search] Literal per-term empty, trying escaped regex OR")
            try:
                results = await self._do_search_regex(keywords)
            except Exception as exc:
                search_failures.append(f"regex fallback: {exc}")
                logger.warning("[keyword_search] Regex fallback failed: %s", exc)

        if not results:
            context.add_log(
                tool_name=self.name,
                metadata={
                    "keywords": keywords,
                    "files_found": 0,
                    "files_discovered": [],
                    "search_failures": search_failures,
                    "degraded": bool(search_failures),
                },
            )
            return "No results found for the given keywords.", {
                "keywords": keywords,
                "count": 0,
                "search_failures": search_failures,
                "degraded": bool(search_failures),
            }

        # Deduplicate: per-term literal search may return the same file
        # from multiple keyword passes.  Merge matches by file path.
        deduped: Dict[str, List[Dict]] = {}
        for item in results:
            path = item.get("path", "unknown")
            if path not in deduped:
                deduped[path] = []
            deduped[path].extend(item.get("matches", []))

        # Format as concise snippets (low token cost).
        # Use keyword-diverse selection: group matches by _keyword tag
        # and round-robin so each keyword contributes at least one
        # snippet when possible.
        output_lines: List[str] = []
        total_chars = 0
        snippets_by_file: Dict[str, List[str]] = {}
        for path, matches in list(deduped.items())[: self._max_results]:
            selected = self._select_diverse_snippets(
                matches, max_lines=self._max_snippet_lines,
            )
            if selected:
                block = f"[{path}]\n" + "\n".join(selected)
                output_lines.append(block)
                total_chars += len(block)
                snippets_by_file[path] = list(selected)

        result_text = "\n\n".join(output_lines)

        # Approximate token count (~4 chars per token)
        approx_tokens = total_chars // 4
        discovered_paths = list(deduped.keys())
        search_backends = sorted({
            str(match.get("_search_backend", "rga"))
            for item in results
            for match in item.get("matches", [])
            if isinstance(match, dict)
        })
        fallback_reasons = sorted({
            str(match["_fallback_reason"])
            for item in results
            for match in item.get("matches", [])
            if isinstance(match, dict) and match.get("_fallback_reason")
        })
        context.add_log(
            tool_name=self.name,
            tokens=approx_tokens,
            metadata={
                "keywords": keywords,
                "files_found": len(discovered_paths),
                "files_discovered": discovered_paths,
                "search_failures": search_failures,
                "search_backends": search_backends,
                "fallback_reasons": fallback_reasons,
                "fallback_used": "rg" in search_backends,
                "degraded": bool(search_failures),
            },
        )

        # ``snippets_by_file`` exposes the matched lines per file in structured
        # form. Callers that only need file discovery can ignore it, while
        # retrieval pipelines can carry the match locations forward instead of
        # rediscovering where in each file the keywords actually hit.
        return result_text, {
            "keywords": keywords,
            "files_found": len(deduped),
            "tokens": approx_tokens,
            "snippets_by_file": snippets_by_file,
            "search_failures": search_failures,
            "search_backends": search_backends,
            "fallback_reasons": fallback_reasons,
            "fallback_used": "rg" in search_backends,
            "degraded": bool(search_failures),
        }

    @staticmethod
    def _select_diverse_snippets(
        matches: List[Dict],
        max_lines: int = 5,
    ) -> List[str]:
        """Select diverse snippet lines ensuring each keyword contributes.

        Groups matches by their ``_keyword`` tag (set by ``_do_search_per_term``)
        and round-robins across groups so that every keyword is represented
        in the output.  Falls back to score-based ordering when no tags exist.

        Args:
            matches: List of rga match dicts, optionally tagged with ``_keyword``.
            max_lines: Maximum number of snippet lines to return.

        Returns:
            List of formatted snippet strings.
        """
        from collections import defaultdict

        # Group by keyword tag
        by_keyword: Dict[str, List[Dict]] = defaultdict(list)
        for m in matches:
            tag = m.get("_keyword", "_default")
            by_keyword[tag].append(m)

        # Sort each group by score descending
        for group in by_keyword.values():
            group.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Round-robin across keyword groups
        selected: List[str] = []
        seen_texts: set = set()
        iterators = {k: iter(v) for k, v in by_keyword.items()}
        exhausted: set = set()

        while len(selected) < max_lines and len(exhausted) < len(iterators):
            for tag, it in iterators.items():
                if tag in exhausted:
                    continue
                while True:
                    m = next(it, None)
                    if m is None:
                        exhausted.add(tag)
                        break
                    line_text = m.get("data", {}).get("lines", {}).get("text", "").strip()
                    line_no = m.get("data", {}).get("line_number")
                    if line_text and line_text not in seen_texts:
                        seen_texts.add(line_text)
                        prefix = f"  L{line_no}: " if line_no else "  "
                        selected.append(f"{prefix}{line_text[:200]}")
                        break
                if len(selected) >= max_lines:
                    break

        return selected


# ---------------------------------------------------------------------------
# Tool 2: File Read (medium cost — returns full file content)
# ---------------------------------------------------------------------------

class FileReadTool(BaseTool):
    """Read approved files using adaptive full, window, range, or section modes."""

    _TEXT_EXTENSIONS = {
        ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml",
        ".yml", ".xml", ".csv", ".log", ".rst", ".html", ".css",
        ".sh", ".bash", ".toml", ".cfg", ".ini", ".conf",
    }

    def __init__(
        self,
        max_chars_per_file: int = FILE_READ_MAX_CHARS,
        *,
        allowed_roots: Optional[List[Union[str, Path]]] = None,
        small_file_chars: int = FILE_READ_SMALL_FILE_CHARS,
        default_window_lines: int = FILE_READ_WINDOW_LINES,
    ) -> None:
        self._max_chars = max(1000, max_chars_per_file)
        self._small_file_chars = max(1000, small_file_chars)
        self._default_window_lines = max(1, default_window_lines)
        self._allowed_roots = [
            Path(root).expanduser().resolve() for root in (allowed_roots or [])
        ]

    @property
    def name(self) -> str:
        return "file_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Read approved files with adaptive evidence extraction. Use auto for "
                "normal operation, full for small files or summaries, window around "
                "query terms, range for known line/character bounds, and section for "
                "a named Markdown/text section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths inside the configured search roots.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "full", "window", "range", "section"],
                        "default": "auto",
                    },
                    "query": {
                        "type": "string",
                        "description": "Terms used to select the best context window.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Section heading to extract in section mode.",
                    },
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "start_char": {"type": "integer", "minimum": 0},
                    "end_char": {"type": "integer", "minimum": 0},
                    "context_lines": {
                        "type": "integer",
                        "default": self._default_window_lines,
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": self._max_chars,
                        "minimum": 1000,
                    },
                },
                "required": ["file_paths"],
            },
        }

    def _resolve_path(self, value: Union[str, Path]) -> Path:
        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [
            root / raw for root in self._allowed_roots
        ]
        if not candidates and not raw.is_absolute():
            candidates = [raw]
        for candidate in candidates:
            resolved = candidate.resolve()
            if not resolved.is_file():
                continue
            if not self._allowed_roots or any(
                resolved == root or root in resolved.parents
                for root in self._allowed_roots
            ):
                return resolved
        raise ValueError("file path must exist within the configured search roots")

    @classmethod
    def _is_text_file(cls, path: Path) -> bool:
        return path.suffix.lower() in cls._TEXT_EXTENSIONS or looks_like_plain_text_file(path)

    async def _extract_content(self, path: Path) -> str:
        if self._is_text_file(path):
            return path.read_text(encoding="utf-8", errors="replace")
        extraction = await fast_extract(path)
        return extraction.content if extraction else ""

    @staticmethod
    def _line_range(content: str, start_line: int, end_line: int) -> str:
        lines = content.splitlines()
        start = max(1, start_line)
        end = max(start, end_line)
        return "\n".join(
            f"L{index}: {lines[index - 1]}"
            for index in range(start, min(end, len(lines)) + 1)
        )

    @staticmethod
    def _query_window(content: str, query: str, context_lines: int) -> str:
        lines = content.splitlines()
        terms = {
            token.lower()
            for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", query or "")
            if len(token) > 1
        }
        if not lines or not terms:
            return "\n".join(lines[:context_lines])
        scores = [
            sum(term in line.lower() for term in terms)
            for line in lines
        ]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] == 0:
            return "\n".join(lines[:context_lines])
        half = max(1, context_lines // 2)
        start = max(0, best - half)
        end = min(len(lines), start + context_lines)
        return "\n".join(
            f"L{index + 1}: {lines[index]}" for index in range(start, end)
        )

    @staticmethod
    def _section(content: str, section: str) -> str:
        if not section.strip():
            return ""
        lines = content.splitlines()
        target = section.strip().lower()
        start: Optional[int] = None
        heading_level = 0
        for index, line in enumerate(lines):
            stripped = line.strip()
            heading = stripped.lstrip("#").strip()
            if target in heading.lower():
                start = index
                heading_level = len(stripped) - len(stripped.lstrip("#"))
                break
        if start is None:
            return ""
        end = len(lines)
        for index in range(start + 1, len(lines)):
            stripped = lines[index].strip()
            level = len(stripped) - len(stripped.lstrip("#"))
            if heading_level and level and level <= heading_level:
                end = index
                break
        return "\n".join(lines[start:end])

    def _select_content(
        self,
        content: str,
        *,
        mode: str,
        query: str,
        section: str,
        start_line: Optional[int],
        end_line: Optional[int],
        start_char: Optional[int],
        end_char: Optional[int],
        context_lines: int,
        max_chars: int,
    ) -> Tuple[str, str, bool]:
        effective_mode = mode
        if mode == "auto":
            if section:
                effective_mode = "section"
            elif len(content) <= self._small_file_chars:
                effective_mode = "full"
            elif query:
                effective_mode = "window"
            else:
                effective_mode = "range"

        if effective_mode == "section":
            selected = self._section(content, section)
            if not selected:
                selected = self._query_window(content, query or section, context_lines)
                effective_mode = "window"
        elif effective_mode == "window":
            selected = self._query_window(content, query, context_lines)
        elif effective_mode == "range":
            if start_line is not None or end_line is not None:
                selected = self._line_range(
                    content,
                    start_line or 1,
                    end_line or (start_line or 1) + context_lines - 1,
                )
            else:
                start = max(0, start_char or 0)
                end = max(start, end_char or start + max_chars)
                selected = content[start:end]
        else:
            selected = content
            effective_mode = "full"

        truncated = len(selected) > max_chars
        if truncated:
            selected = selected[:max_chars] + "\n... [truncated]"
        return selected, effective_mode, truncated

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        raw_paths = kwargs.get("file_paths", [])
        file_paths: List[str]
        if isinstance(raw_paths, (str, Path)):
            file_paths = [str(raw_paths)]
        else:
            file_paths = [str(item) for item in (raw_paths or [])]
        if not file_paths:
            return "No file paths provided.", {}
        mode = str(kwargs.get("mode", "auto") or "auto").lower()
        if mode not in {"auto", "full", "window", "range", "section"}:
            return "Invalid read mode.", {"error": "invalid_mode"}
        max_chars = min(
            self._max_chars,
            max(1000, int(kwargs.get("max_chars", self._max_chars))),
        )
        start_line = (
            int(kwargs["start_line"]) if kwargs.get("start_line") is not None else None
        )
        end_line = (
            int(kwargs["end_line"]) if kwargs.get("end_line") is not None else None
        )
        start_char = (
            int(kwargs["start_char"]) if kwargs.get("start_char") is not None else None
        )
        end_char = (
            int(kwargs["end_char"]) if kwargs.get("end_char") is not None else None
        )
        context_lines = min(
            500,
            max(1, int(kwargs.get("context_lines", self._default_window_lines))),
        )

        outputs: List[str] = []
        files_read: List[str] = []
        read_modes: Dict[str, str] = {}
        total_chars = 0

        for value in file_paths:
            if context.is_budget_exceeded():
                outputs.append(f"[{value}] (skipped — token budget exceeded)")
                break
            try:
                path = self._resolve_path(value)
                content = await self._extract_content(path)
                selected, effective_mode, truncated = self._select_content(
                    content,
                    mode=mode,
                    query=str(kwargs.get("query", "") or ""),
                    section=str(kwargs.get("section", "") or ""),
                    start_line=start_line,
                    end_line=end_line,
                    start_char=start_char,
                    end_char=end_char,
                    context_lines=context_lines,
                    max_chars=max_chars,
                )
                outputs.append(f"[{path}] mode={effective_mode}\n{selected}")
                total_chars += len(selected)
                files_read.append(str(path))
                read_modes[str(path)] = effective_mode
                if effective_mode == "full" and not truncated:
                    context.mark_file_read(str(path))
            except Exception as exc:
                outputs.append(f"[{value}] Read error: {exc}")

        approx_tokens = total_chars // 4
        context.add_log(
            tool_name=self.name,
            tokens=approx_tokens,
            metadata={
                "files_read": files_read,
                "files_requested": file_paths,
                "read_modes": read_modes,
                "chars_read": total_chars,
            },
        )
        return "\n\n---\n\n".join(outputs), {
            "files_read": files_read,
            "read_modes": read_modes,
            "chars_read": total_chars,
            "tokens": approx_tokens,
        }


# ---------------------------------------------------------------------------
# Tool 3: Knowledge Query (free — queries cached clusters)
# ---------------------------------------------------------------------------

class KnowledgeQueryTool(BaseTool):
    """Query the persistent knowledge cluster cache.

    Searches previously-built KnowledgeClusters by fuzzy text match.
    Zero retrieval-token cost (data is already in memory).
    """

    def __init__(self, storage: KnowledgeStorage) -> None:
        self._storage = storage

    @property
    def name(self) -> str:
        return "knowledge_query"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Search the knowledge cache for previously extracted information. "
                "Returns cached knowledge clusters matching the query. "
                "Zero token cost — use this first before searching files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query to search cached knowledge.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of clusters to return (default: 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        }

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        query: str = kwargs.get("query", "")
        limit: int = kwargs.get("limit", 3)
        if not query:
            return "No query provided.", {}

        try:
            clusters = await self._storage.find(query, limit=limit)
        except Exception as exc:
            return f"Knowledge query failed: {exc}", {"error": str(exc)}

        if not clusters:
            return "No matching knowledge clusters found.", {"query": query, "count": 0}

        output_parts: List[str] = []
        for c in clusters:
            content = c.content if isinstance(c.content, str) else "\n".join(c.content) if c.content else ""
            desc = c.description if isinstance(c.description, str) else "\n".join(c.description) if c.description else ""
            part = (
                f"### {c.name} (id: {c.id})\n"
                f"{desc}\n\n"
                f"{content}"
            )
            output_parts.append(part)

        result_text = "\n\n---\n\n".join(output_parts)

        # Knowledge queries are free (already cached)
        context.add_log(
            tool_name=self.name,
            tokens=0,
            metadata={"query": query, "clusters_found": len(clusters)},
        )

        return result_text, {"query": query, "clusters_found": len(clusters)}


# ---------------------------------------------------------------------------
# Tool 5: Tree Navigation (medium cost — LLM-guided tree index navigation)
# ---------------------------------------------------------------------------

class TreeNavigationTool(BaseTool):
    """Navigate a document's compiled tree index to extract targeted evidence.

    Uses an LLM-driven tree navigation strategy: the model selects
    the most relevant branches/sections from a hierarchical document
    index, then extracts the corresponding page or char-range content.

    This tool bridges the gap between keyword search (which finds
    *where* a term appears) and file read (which returns *everything*).
    Tree navigation returns the most relevant *sections* of a document
    without reading the whole file.

    Requires compile artifacts (tree indices) to be available for the
    target files.
    """

    def __init__(
        self,
        navigate_fn: Any,
        available_paths: Optional[set] = None,
        max_chars: int = 15_000,
    ) -> None:
        self._navigate_fn = navigate_fn
        self._available_paths = available_paths or set()
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "tree_navigate"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Navigate a document's compiled tree index to extract "
                "targeted sections relevant to the query. More precise "
                "than file_read — returns only relevant sections instead "
                "of the entire file. Works with PDF, DOCX, and other "
                "compiled document types. Medium token cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "Absolute path of the document to navigate."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "What information to look for in the document."
                        ),
                    },
                },
                "required": ["file_path", "query"],
            },
        }

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        file_path: str = kwargs.get("file_path", "")
        query: str = kwargs.get("query", "")
        if not file_path or not query:
            return "file_path and query are required.", {}

        if (
            self._available_paths
            and file_path not in self._available_paths
        ):
            return (
                f"No tree index available for {Path(file_path).name}. "
                "Use file_read instead."
            ), {"file_path": file_path, "indexed": False}

        try:
            result = await self._navigate_fn(
                file_path, query, max_chars=self._max_chars,
            )
        except Exception as exc:
            return (
                f"Tree navigation failed: {exc}"
            ), {"file_path": file_path, "error": str(exc)}

        if not result:
            return (
                f"No relevant sections found in "
                f"{Path(file_path).name} for this query."
            ), {"file_path": file_path, "chars": 0}

        total_chars = len(result)
        approx_tokens = total_chars // 4
        context.add_log(
            tool_name=self.name,
            tokens=approx_tokens,
            metadata={
                "file_path": file_path,
                "chars": total_chars,
            },
        )

        header = f"[Tree navigation: {Path(file_path).name}]"
        return f"{header}\n{result}", {
            "file_path": file_path,
            "chars": total_chars,
            "tokens": approx_tokens,
        }
