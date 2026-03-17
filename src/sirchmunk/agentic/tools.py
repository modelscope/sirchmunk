# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Hierarchical retrieval tools for the ReAct search agent.

Provides a tool abstraction layer and four concrete tools that operate
at different granularities — from lightweight keyword search to deep
file reading and knowledge base querying.  All tools are stateless;
side-effects (token accounting, dedup) are recorded via SearchContext.
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.file_utils import fast_extract

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
    _DEFAULT_EXCLUDE: List[str] = ["*.pyc", "*.log", "__pycache__"]

    def __init__(
        self,
        retriever: GrepRetriever,
        paths: Union[str, Path, List[str], List[Path]],
        max_depth: int = 5,
        max_results: int = 10,
        max_snippet_lines: int = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        bm25_scorer: Any = None,
        max_count: Optional[int] = None,
        rga_no_cache: bool = False,
    ) -> None:
        self._retriever = retriever
        self._paths = paths
        self._max_depth = max_depth
        self._max_results = max_results
        self._max_snippet_lines = max_snippet_lines
        self._include = include
        self._exclude = list(set(self._DEFAULT_EXCLUDE) | set(exclude or []))
        self._bm25_scorer = bm25_scorer
        self._max_count = max_count
        self._rga_no_cache = rga_no_cache

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
    ) -> List[Dict[str, Any]]:
        """Search each keyword individually and merge all results.

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

        # Fire one search per keyword concurrently
        async def _single(term: str) -> List[Dict[str, Any]]:
            return await self._retriever.retrieve(
                terms=term,
                path=self._paths,
                logic="or",
                case_sensitive=False,
                literal=literal,
                regex=regex,
                max_depth=self._max_depth,
                max_count=self._max_count,
                include=self._include,
                exclude=self._exclude,
                rga_no_cache=self._rga_no_cache,
                timeout=30.0,
            )

        raw_lists = await _aio.gather(*[_single(k) for k in keywords])

        # Flatten all raw rga JSON events into one list, then merge
        # Tag each match event with the keyword that produced it so
        # the formatter can guarantee keyword diversity in snippets.
        combined: List[Dict[str, Any]] = []
        for keyword, raw in zip(keywords, raw_lists):
            for item in raw:
                if item.get("type") == "match":
                    item["_keyword"] = keyword
                combined.append(item)

        return self._retriever.merge_results(
            combined,
            limit=self._max_results * 2,
            max_files=self._retriever._merge_max_files,
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
            max_count=self._max_count,
            include=self._include,
            exclude=self._exclude,
            rga_no_cache=self._rga_no_cache,
            timeout=30.0,
        )
        return self._retriever.merge_results(
            raw,
            limit=self._max_results,
            max_files=self._retriever._merge_max_files,
        )

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        import time as _time

        keywords: List[str] = kwargs.get("keywords", [])
        if not keywords:
            return "No keywords provided.", {}

        _t_start = _time.perf_counter()
        context.add_search(" ".join(keywords))

        phrases = [k for k in keywords if " " in k.strip()]
        singles = [k for k in keywords if " " not in k.strip()]

        results: List[Dict[str, Any]] = []
        _phase_hit = ""

        # Phase A: search compound phrases first (e.g. "Tivoli Gardens")
        if phrases:
            _ta = _time.perf_counter()
            results = await self._do_search_per_term(phrases, literal=True, regex=False)
            logger.info(
                "[keyword_search:profile] Phase-A phrases={}: {:.2f}s, {} file groups",
                phrases, _time.perf_counter() - _ta, len(results),
            )
            if results:
                _phase_hit = "A"

        # Phase B: add single-word terms only when phrase hits are insufficient.
        if len(results) < 3 and singles:
            _tb = _time.perf_counter()
            extra = await self._do_search_per_term(singles, literal=True, regex=False)
            logger.info(
                "[keyword_search:profile] Phase-B singles={}: {:.2f}s, {} file groups",
                singles, _time.perf_counter() - _tb, len(extra) if extra else 0,
            )
            if extra:
                seen = {r.get("path") for r in results}
                for item in extra:
                    if item.get("path") not in seen:
                        results.append(item)
                        seen.add(item.get("path"))
                _phase_hit = _phase_hit or "B"

        # Phase C: if nothing found at all, try all keywords together
        if not results:
            _tc = _time.perf_counter()
            results = await self._do_search_per_term(keywords, literal=True, regex=False)
            logger.info(
                "[keyword_search:profile] Phase-C all={}: {:.2f}s, {} file groups",
                keywords, _time.perf_counter() - _tc, len(results),
            )
            if results:
                _phase_hit = "C"

        # Phase D: escaped-regex OR search
        if not results:
            _td = _time.perf_counter()
            results = await self._do_search_regex(keywords)
            logger.info(
                "[keyword_search:profile] Phase-D regex: {:.2f}s, {} file groups",
                _time.perf_counter() - _td, len(results),
            )
            if results:
                _phase_hit = "D"

        if not results:
            _elapsed = _time.perf_counter() - _t_start
            logger.info(
                "[keyword_search:profile] TOTAL {:.2f}s — no results | keywords={}",
                _elapsed, keywords,
            )
            return "No results found for the given keywords.", {"keywords": keywords, "count": 0}

        # Deduplicate
        _t_dedup = _time.perf_counter()
        deduped: Dict[str, List[Dict]] = {}
        for item in results:
            path = item.get("path", "unknown")
            if path not in deduped:
                deduped[path] = []
            deduped[path].extend(item.get("matches", []))
        _dedup_ms = (_time.perf_counter() - _t_dedup) * 1000

        # BM25-rerank files when significantly more candidates than limit
        _rerank_ms = 0.0
        _rerank_input = len(deduped)
        _BM25_RERANK_RATIO = 1.5
        if (
            self._bm25_scorer
            and len(deduped) > self._max_results * _BM25_RERANK_RATIO
        ):
            _tr = _time.perf_counter()
            ranked = self._bm25_rerank_results(keywords, deduped, self._max_results)
            _rerank_ms = (_time.perf_counter() - _tr) * 1000
            if ranked is not None:
                deduped = ranked
            logger.info(
                "[keyword_search:profile] BM25 rerank {}→{} files: {:.0f}ms",
                _rerank_input, len(deduped), _rerank_ms,
            )

        # Format as concise snippets
        output_lines: List[str] = []
        total_chars = 0
        for path, matches in list(deduped.items())[: self._max_results]:
            selected = self._select_diverse_snippets(
                matches, max_lines=self._max_snippet_lines,
            )
            if selected:
                block = f"[{path}]\n" + "\n".join(selected)
                output_lines.append(block)
                total_chars += len(block)

        result_text = "\n\n".join(output_lines)

        approx_tokens = total_chars // 4
        discovered_paths = list(deduped.keys())[: self._max_results]
        context.add_discovered_files(discovered_paths)

        _elapsed = _time.perf_counter() - _t_start
        timing_meta = {
            "total_sec": round(_elapsed, 2),
            "phase_hit": _phase_hit,
            "dedup_ms": round(_dedup_ms, 1),
            "rerank_ms": round(_rerank_ms, 1),
            "files_before_dedup": len(results),
            "files_after_dedup": _rerank_input,
        }
        logger.info(
            "[keyword_search:profile] TOTAL {:.2f}s | phase={} | raw={} dedup={} "
            "dedup_ms={:.0f} rerank_ms={:.0f} | keywords={}",
            _elapsed, _phase_hit, len(results), _rerank_input,
            _dedup_ms, _rerank_ms, keywords,
        )

        context.add_log(
            tool_name=self.name,
            tokens=approx_tokens,
            metadata={
                "keywords": keywords,
                "files_found": len(deduped),
                "files_discovered": discovered_paths,
                "timing": timing_meta,
            },
        )

        return result_text, {"keywords": keywords, "files_found": len(deduped), "tokens": approx_tokens}

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

    def _bm25_rerank_results(
        self,
        keywords: List[str],
        deduped: Dict[str, List[Dict]],
        top_k: int,
    ) -> Optional[Dict[str, List[Dict]]]:
        """Rerank deduplicated file results by BM25 relevance to *keywords*."""
        file_paths = list(deduped.keys())
        docs = []
        for matches in deduped.values():
            text = " ".join(
                m.get("data", {}).get("lines", {}).get("text", "")
                for m in matches
            ).strip()
            docs.append(text or " ")
        query = " ".join(keywords)
        try:
            indices = self._bm25_scorer.rerank(query, docs, top_k=top_k)
            if indices:
                ranked: Dict[str, List[Dict]] = {}
                for idx in indices:
                    if 0 <= idx < len(file_paths):
                        key = file_paths[idx]
                        ranked[key] = deduped[key]
                return ranked if ranked else None
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Tool 2: File Read (medium cost — returns full file content)
# ---------------------------------------------------------------------------

class FileReadTool(BaseTool):
    """Read full content of specified files.

    Supports all formats via kreuzberg extraction (PDF, DOCX, XLSX, etc.).
    Tracks read files in SearchContext to prevent redundant reads.
    For large files, optional keyword-guided extraction via BM25 ranking
    returns only the most relevant sections.
    """

    _LARGE_FILE_THRESHOLD = 30_000
    _CHUNK_CACHE_MAX = 32

    def __init__(
        self,
        max_chars_per_file: int = 30000,
        bm25_scorer: Any = None,
        base_paths: Optional[List[str]] = None,
    ) -> None:
        self._max_chars = max_chars_per_file
        self._bm25_scorer = bm25_scorer
        # Base paths for relative path resolution (e.g., wiki corpus dir)
        self._base_paths = [Path(p).resolve() for p in (base_paths or [])]
        self._chunk_cache: Dict[str, List[str]] = {}

    @staticmethod
    def _normalize_path(fp: str, base_paths: Optional[List[Path]] = None) -> Optional[Path]:
        """Normalize and resolve file path with multiple fallback strategies.

        Handles:
        - Absolute paths (returned as-is if they exist)
        - Relative paths (resolved against base_paths)
        - Paths with symlinks (resolved to real path)
        - Paths with case mismatches on case-insensitive systems

        Args:
            fp: File path string to normalize.
            base_paths: List of base directories to try for relative path resolution.

        Returns:
            Resolved Path if file exists, None otherwise.
        """
        # Strategy 1: Direct absolute/relative path
        path = Path(fp)
        if path.is_absolute():
            if path.exists():
                return path.resolve()
            # Try resolving symlinks
            try:
                resolved = path.resolve(strict=False)
                if resolved.exists():
                    return resolved
            except (OSError, RuntimeError):
                pass
        else:
            # Relative path: try resolving against base_paths
            for base in (base_paths or []):
                candidate = base / fp
                if candidate.exists():
                    return candidate.resolve()

        # Strategy 2: Basename matching against base_paths
        # (handles cases where only filename is provided)
        basename = Path(fp).name
        if basename and base_paths:
            for base in base_paths:
                if base.is_dir():
                    # Look for the file in immediate children or subdirs
                    for candidate in base.rglob(basename):
                        if candidate.is_file():
                            return candidate.resolve()

        return None

    @property
    def name(self) -> str:
        return "file_read"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Read the full content of one or more files. Supports PDF, "
                "DOCX, XLSX, TXT, MD, and other formats. Use this after "
                "keyword_search identifies promising files. For large files, "
                "provide keywords to extract only the most relevant sections. "
                "Higher token cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Absolute paths of files to read.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional focus keywords for large files. When "
                            "provided, only sections relevant to these keywords "
                            "are returned instead of the full content."
                        ),
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
        keywords: List[str] = kwargs.get("keywords", [])
        if not file_paths:
            return "No file paths provided.", {}

        outputs: List[str] = []
        files_read: List[str] = []
        total_chars = 0

        for fp in file_paths:
            fp_str = str(fp)

            # Normalize path with fallback strategies
            resolved = self._normalize_path(fp_str, self._base_paths)
            if resolved:
                fp_str = str(resolved)

            if context.is_file_read(fp_str):
                outputs.append(f"[{fp_str}] (already read, skipped)")
                continue

            if context.is_budget_exceeded():
                outputs.append(f"[{fp_str}] (skipped — token budget exceeded)")
                break

            try:
                path = Path(fp_str)
                if not path.exists():
                    # Log the failed path for debugging
                    outputs.append(f"[{fp_str}] File not found. (original: {fp})")
                    continue

                text_extensions = {
                    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml",
                    ".yml", ".xml", ".csv", ".log", ".rst", ".html", ".css",
                    ".sh", ".bash", ".toml", ".cfg", ".ini", ".conf",
                    ".jsonl",
                }
                if path.suffix.lower() in text_extensions:
                    content = path.read_text(encoding="utf-8", errors="replace")
                else:
                    extraction = await fast_extract(path)
                    content = extraction.content if extraction else ""

                # Focused extraction for large files when keywords provided
                if len(content) > self._LARGE_FILE_THRESHOLD and keywords:
                    content = self._extract_focused_content(content, keywords, file_path=fp_str)

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

    def _make_chunks(self, content: str) -> List[str]:
        """Split content into ~1000-char chunks, keeping lines intact."""
        lines = content.splitlines()
        if not lines:
            return []
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for line in lines:
            line_len = len(line) + 1
            if buf_len + line_len > 1000 and buf:
                chunks.append("\n".join(buf))
                buf = [line]
                buf_len = line_len
            else:
                buf.append(line)
                buf_len += line_len
        if buf:
            chunks.append("\n".join(buf))
        return chunks

    def _get_chunks(self, file_path: str, content: str) -> List[str]:
        """Return cached chunks for *file_path*, building them if needed."""
        if file_path in self._chunk_cache:
            return self._chunk_cache[file_path]
        chunks = self._make_chunks(content)
        if len(self._chunk_cache) >= self._CHUNK_CACHE_MAX:
            oldest = next(iter(self._chunk_cache))
            del self._chunk_cache[oldest]
        self._chunk_cache[file_path] = chunks
        return chunks

    def _extract_focused_content(
        self,
        content: str,
        keywords: List[str],
        max_chunks: int = 20,
        file_path: str = "",
    ) -> str:
        """Extract keyword-relevant chunks using BM25 ranking.

        Splits content into line-based chunks (preserving structural
        boundaries for JSONL and similar formats), then selects the
        most relevant chunks via BM25 scoring.  Falls back to simple
        keyword matching when no scorer is available.
        """
        chunks = self._get_chunks(file_path, content) if file_path else self._make_chunks(content)
        if not chunks or len(chunks) <= max_chunks:
            return content

        query = " ".join(keywords)

        # Only invoke BM25 when chunk count is significantly above limit
        if self._bm25_scorer and len(chunks) > max_chunks * 1.5:
            try:
                self._bm25_scorer.index_corpus(chunks)
                indices = self._bm25_scorer.rerank(
                    query, chunks, top_k=max_chunks,
                )
                if indices:
                    indices.sort()
                    return "\n\n[...]\n\n".join(chunks[i] for i in indices)
            except Exception:
                pass

        # Fallback: keyword matching
        kw_lower = [k.lower() for k in keywords]
        matched = [c for c in chunks if any(k in c.lower() for k in kw_lower)]
        if matched:
            return "\n\n[...]\n\n".join(matched[:max_chunks])

        return content[: self._max_chars]


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
