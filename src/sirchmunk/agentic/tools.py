# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Hierarchical retrieval tools for the ReAct search agent.

Provides a tool abstraction layer and four concrete tools that operate
at different granularities — from lightweight keyword search to deep
file reading and knowledge base querying.  All tools are stateless;
side-effects (token accounting, dedup) are recorded via SearchContext.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.constants import GREP_KEYWORD_CONCURRENT_LIMIT, GREP_TIMEOUT
from sirchmunk.utils.file_utils import fast_extract

logger = logging.getLogger(__name__)


def _looks_like_plain_text_file(path: Path, *, sample_bytes: int = 4096) -> bool:
    """Return True for extensionless/raw files that are likely UTF-8 text."""
    try:
        raw = path.read_bytes()[:sample_bytes]
    except OSError:
        return False
    if not raw:
        return True
    if b"\x00" in raw:
        return False
    decoded = raw.decode("utf-8", errors="replace")
    if not decoded:
        return False
    replacement_count = decoded.count("\ufffd")
    if replacement_count > max(1, len(decoded) // 100):
        return False
    textish_count = sum(1 for ch in decoded if ch.isprintable() or ch in "\r\n\t")
    return textish_count / max(len(decoded), 1) >= 0.85


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

        # Strategy 1: literal rga and targeted native-format fallback run in
        # parallel.  Native extraction is restricted to discriminative terms
        # and formats with known rga adapter gaps, then cached by file signature.
        literal_result, native_result = await asyncio.gather(
            self._do_search_per_term(keywords, literal=True, regex=False),
            self._retriever.retrieve_native_formats(
                keywords,
                path=self._paths,
                max_depth=self._max_depth,
                include=self._include,
                exclude=self._exclude,
                max_results=self._max_results * 2,
                max_lines=self._max_snippet_lines,
            ),
            return_exceptions=True,
        )
        if isinstance(literal_result, Exception):
            results, search_failures = [], [f"literal search: {literal_result}"]
        else:
            results, search_failures = literal_result
        if isinstance(native_result, Exception):
            search_failures.append(f"native format fallback: {native_result}")
        else:
            results.extend(native_result)

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
        all_matches = [
            match
            for item in results
            for match in item.get("matches", [])
            if isinstance(match, dict)
        ]
        search_backends = sorted({
            str(match.get("_search_backend", "rga"))
            for match in all_matches
        })
        fallback_reasons = sorted({
            str(match["_fallback_reason"])
            for match in all_matches if match.get("_fallback_reason")
        })
        search_elapsed_ms = max([
            float(match.get("_search_elapsed_ms", 0.0) or 0.0)
            for match in all_matches
        ] or [0.0])
        queue_wait_ms = max([
            float(match.get("_search_queue_wait_ms", 0.0) or 0.0)
            for match in all_matches
        ] or [0.0])
        adapter_execution_ms = max([
            float(match.get("_search_execution_ms", 0.0) or 0.0)
            for match in all_matches
        ] or [0.0])
        fallback_elapsed_ms = max([
            float(match.get("_fallback_elapsed_ms", 0.0) or 0.0)
            for match in all_matches
        ] or [0.0])
        native_cache_states = [
            bool(match["_search_cache_hit"])
            for match in all_matches if "_search_cache_hit" in match
        ]
        primary_failure_stages = sorted({
            str(match["_primary_failure_stage"])
            for match in all_matches if match.get("_primary_failure_stage")
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
                "fallback_used": any(
                    backend in {"rg", "native_extract"} for backend in search_backends
                ),
                "search_elapsed_ms": round(search_elapsed_ms, 3),
                "queue_wait_ms": round(queue_wait_ms, 3),
                "adapter_execution_ms": round(adapter_execution_ms, 3),
                "fallback_elapsed_ms": round(fallback_elapsed_ms, 3),
                "native_cache_hits": sum(native_cache_states),
                "native_cache_misses": len(native_cache_states) - sum(native_cache_states),
                "primary_failure_stages": primary_failure_stages,
                "degraded": bool(search_failures or fallback_reasons),
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
            "fallback_used": any(
                backend in {"rg", "native_extract"} for backend in search_backends
            ),
            "search_elapsed_ms": round(search_elapsed_ms, 3),
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
    """Read full content of specified files.

    Supports all formats via kreuzberg extraction (PDF, DOCX, XLSX, etc.).
    Tracks read files in SearchContext to prevent redundant reads.
    """

    def __init__(self, max_chars_per_file: int = 30000) -> None:
        self._max_chars = max_chars_per_file

    @property
    def name(self) -> str:
        return "file_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Read the full content of one or more files. Supports PDF, "
                "DOCX, XLSX, TXT, MD, and other formats. Use this after "
                "keyword_search identifies promising files. Higher token cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute paths of files to read.",
                    },
                },
                "required": ["file_paths"],
            },
        }

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        file_paths: List[str] = kwargs.get("file_paths", [])
        if not file_paths:
            return "No file paths provided.", {}

        outputs: List[str] = []
        files_read: List[str] = []
        total_chars = 0

        for fp in file_paths:
            fp_str = str(fp)

            # Dedup: skip already-read files
            if context.is_file_read(fp_str):
                outputs.append(f"[{fp_str}] (already read, skipped)")
                continue

            # Budget check
            if context.is_budget_exceeded():
                outputs.append(f"[{fp_str}] (skipped — token budget exceeded)")
                break

            try:
                path = Path(fp_str)
                if not path.exists():
                    outputs.append(f"[{fp_str}] File not found.")
                    continue

                # Text-like files: read directly; others: use kreuzberg
                text_extensions = {
                    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml",
                    ".yml", ".xml", ".csv", ".log", ".rst", ".html", ".css",
                    ".sh", ".bash", ".toml", ".cfg", ".ini", ".conf",
                }
                if path.suffix.lower() in text_extensions or _looks_like_plain_text_file(path):
                    content = path.read_text(encoding="utf-8", errors="replace")
                else:
                    extraction = await fast_extract(path)
                    content = extraction.content if extraction else ""

                # Truncate if needed
                if len(content) > self._max_chars:
                    content = content[: self._max_chars] + "\n... [truncated]"

                outputs.append(f"[{fp_str}]\n{content}")
                total_chars += len(content)
                context.mark_file_read(fp_str)
                files_read.append(fp_str)

            except Exception as exc:
                outputs.append(f"[{fp_str}] Read error: {exc}")

        result_text = "\n\n---\n\n".join(outputs)
        approx_tokens = total_chars // 4
        context.add_log(
            tool_name=self.name,
            tokens=approx_tokens,
            metadata={"files_read": files_read, "files_requested": file_paths},
        )

        return result_text, {"files_read": files_read, "tokens": approx_tokens}


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
