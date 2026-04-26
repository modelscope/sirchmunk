# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Document tree indexer — PageIndex-inspired hierarchical structure analysis.

Builds a JSON tree index for structured long documents (PDF, DOCX, MD, HTML)
so that downstream search can navigate via LLM reasoning instead of brute-force
Monte Carlo sampling.
"""

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.utils import LogCallback, create_logger
from sirchmunk.utils.file_utils import get_fast_hash

# File-size threshold: skip tree indexing for small files
_TREE_MIN_CHARS = 10_000  # 10 K characters (lowered from 20K for broader coverage)

# Adaptive depth thresholds: (min_chars, max_depth) — evaluated top-down;
# **must** be sorted by min_chars descending so the first match wins.
_TREE_ADAPTIVE_DEPTH_THRESHOLDS: tuple = (
    (100_000, 4),
    (50_000, 3),
    (20_000, 2),
)

# Summary snippet length extracted from section content (chars)
_TOC_NODE_SUMMARY_MAX_CHARS = 300

# Marker substring length for fuzzy fallback matching in _resolve_positions
_MARKER_SUBSTRING_LEN = 32

# Maximum span ratio: filter out overly large spans (>80% of document)
_MAX_SPAN_RATIO = 0.8

# Adaptive preview window for LLM structure analysis
_TREE_PREVIEW_MIN = 12_000    # Minimum preview window (chars)
_TREE_PREVIEW_MAX = 50_000    # Maximum preview window (~12K tokens)
_TREE_PREVIEW_RATIO = 0.15    # Fraction of document to preview

# Extensions eligible for tree indexing
_TREE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".md", ".markdown",
    ".html", ".htm", ".rst", ".tex", ".txt",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """Single node in the document tree."""

    node_id: str
    title: str
    summary: str
    char_range: Tuple[int, int]  # [start, end) in the extracted text
    level: int = 0
    page_range: Optional[Tuple[int, int]] = None
    children: List["TreeNode"] = field(default_factory=list)
    table_count: int = 0  # Number of tables associated with this node's page range
    content_type: str = "text"  # "text" | "table"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "char_range": list(self.char_range),
            "level": self.level,
            "page_range": list(self.page_range) if self.page_range else None,
            "children": [c.to_dict() for c in self.children],
            "table_count": self.table_count,
            "content_type": self.content_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TreeNode":
        children = [cls.from_dict(c) for c in data.get("children", [])]
        pr = data.get("page_range")
        return cls(
            node_id=data["node_id"],
            title=data["title"],
            summary=data["summary"],
            char_range=tuple(data["char_range"]),
            level=data.get("level", 0),
            page_range=tuple(pr) if pr else None,
            children=children,
            table_count=data.get("table_count", 0),
            content_type=data.get("content_type", "text"),
        )

    @property
    def leaf(self) -> bool:
        return len(self.children) == 0

    def all_leaves(self) -> List["TreeNode"]:
        """Return all leaf nodes under this subtree."""
        if self.leaf:
            return [self]
        leaves: List["TreeNode"] = []
        for c in self.children:
            leaves.extend(c.all_leaves())
        return leaves


@dataclass
class DocumentTree:
    """Complete tree index for a single document."""

    file_path: str
    file_hash: str
    created_at: str
    total_chars: int
    total_pages: Optional[int] = None
    root: Optional[TreeNode] = None

    def to_json(self) -> str:
        return json.dumps({
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "created_at": self.created_at,
            "total_chars": self.total_chars,
            "total_pages": self.total_pages,
            "root": self.root.to_dict() if self.root else None,
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentTree":
        data = json.loads(json_str)
        root = TreeNode.from_dict(data["root"]) if data.get("root") else None
        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            created_at=data["created_at"],
            total_chars=data["total_chars"],
            total_pages=data.get("total_pages"),
            root=root,
        )


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

class DocumentTreeIndexer:
    """Build and cache PageIndex-style hierarchical tree indices for documents."""

    # Maximum child nodes before switching to paginated LLM selection.
    # Balance: lower = more LLM calls, higher = more tokens per call.
    _PAGE_SIZE_THRESHOLD: int = 15

    # Number of nodes per group in paginated selection.
    _GROUP_PAGE_SIZE: int = 15

    def __init__(
        self,
        llm: OpenAIChat,
        cache_dir: Union[str, Path],
        log_callback: LogCallback = None,
    ):
        self._llm = llm
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._log = create_logger(log_callback=log_callback)

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def build_tree(
        self,
        file_path: str,
        content: str,
        *,
        max_depth: int = 4,
        force_rebuild: bool = False,
        total_pages: Optional[int] = None,
        toc_entries: Optional[List[Any]] = None,
    ) -> Optional[DocumentTree]:
        """Build a tree index for a document.

        When *toc_entries* are provided (from TOCExtractor), uses the
        TOC-accelerated path that skips recursive LLM analysis and builds
        the tree directly from extracted headings.

        Returns None when the document is too small or unstructured.
        """
        file_hash = get_fast_hash(file_path)
        if file_hash is None:
            return None

        if not force_rebuild:
            cached = self._load_cache(file_hash)
            if cached is not None:
                await self._log.info(f"[TreeIndexer] Cache hit for {Path(file_path).name}")
                return cached

        if len(content) < _TREE_MIN_CHARS:
            return None

        ext = Path(file_path).suffix.lower()
        if ext not in _TREE_EXTENSIONS:
            return None

        # Use adaptive depth based on document length
        effective_depth = self._compute_adaptive_depth(len(content))

        await self._log.info(
            f"[TreeIndexer] Building tree for {Path(file_path).name} "
            f"({len(content)} chars, depth={effective_depth})"
        )

        # TOC-accelerated path: skip recursive LLM analysis
        if toc_entries:
            root = await self._build_tree_from_toc(
                toc_entries, content, total_pages=total_pages,
            )
            if root is not None:
                # NOTE: _deepen_large_leaves disabled - char_range anchoring via LLM start_text
                # is unreliable, causing overlapping ranges and search failures.
                # TODO: Re-enable when robust char_range calculation is implemented.
                # await self._deepen_large_leaves(root, content, max_depth=effective_depth)
                # Node summary enrichment: controlled by SIRCHMUNK_SKIP_NODE_SUMMARIES env var.
                # Set to "true" to skip during debugging / performance testing.
                _skip_summaries = os.getenv("SIRCHMUNK_SKIP_NODE_SUMMARIES", "").lower() in ("true", "1", "yes")
                print(f"SEARCH_WIKI_DEBUG [T1] enrich_node_summaries (TOC path): skip={_skip_summaries}, env={os.getenv('SIRCHMUNK_SKIP_NODE_SUMMARIES', '')}", flush=True)
                if not _skip_summaries:
                    await self._enrich_node_summaries(root, content)
                tree = DocumentTree(
                    file_path=file_path,
                    file_hash=file_hash,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    total_chars=len(content),
                    total_pages=total_pages,
                    root=root,
                )
                self._save_cache(file_hash, tree)
                await self._log.info(
                    f"[TreeIndexer] Built tree from TOC: {self._count_nodes(root)} nodes"
                )
                return tree

        # Fallback: existing recursive LLM path (with adaptive depth)
        root = await self._build_node(content, level=0, max_depth=effective_depth)
        if root is None:
            return None

        # NOTE: _deepen_large_leaves disabled - char_range anchoring via LLM start_text
        # is unreliable, causing overlapping ranges and search failures.
        # TODO: Re-enable when robust char_range calculation is implemented.
        # await self._deepen_large_leaves(root, content, max_depth=effective_depth)
        # Node summary enrichment: controlled by SIRCHMUNK_SKIP_NODE_SUMMARIES env var.
        # Set to "true" to skip during debugging / performance testing.
        _skip_summaries = os.getenv("SIRCHMUNK_SKIP_NODE_SUMMARIES", "").lower() in ("true", "1", "yes")
        print(f"SEARCH_WIKI_DEBUG [T1] enrich_node_summaries (recursive path): skip={_skip_summaries}, env={os.getenv('SIRCHMUNK_SKIP_NODE_SUMMARIES', '')}", flush=True)
        if not _skip_summaries:
            await self._enrich_node_summaries(root, content)

        tree = DocumentTree(
            file_path=file_path,
            file_hash=file_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_chars=len(content),
            total_pages=total_pages,
            root=root,
        )
        self._save_cache(file_hash, tree)
        await self._log.info(
            f"[TreeIndexer] Built tree: {self._count_nodes(root)} nodes, "
            f"depth={self._max_node_depth(root)}"
        )
        return tree

    async def navigate(
        self,
        tree: DocumentTree,
        query: str,
        *,
        max_results: int = 3,
        max_depth: int = 4,
    ) -> List[TreeNode]:
        """Adaptive-depth LLM-driven tree navigation.

        Iteratively descends the tree using _select_children() at each level,
        collecting leaf nodes until *max_results* are found or *max_depth* is
        reached.

        Args:
            tree: DocumentTree with a root node.
            query: Search query for relevance selection.
            max_results: Maximum number of leaf nodes to return.
            max_depth: Maximum descent depth (default 4).

        Returns:
            List of the most relevant leaf TreeNodes.
        """
        if tree.root is None:
            return []

        print(f"SEARCH_WIKI_DEBUG [T2] navigate: query={query[:80]}, total_nodes={self._count_nodes(tree.root)}", flush=True)

        candidates = tree.root.children if tree.root.children else [tree.root]
        if not candidates:
            return [tree.root]

        result_leaves: List[TreeNode] = []
        visited: set = set()  # prevent cycles
        frontier = candidates
        selected: List[TreeNode] = []

        depth = 0
        while depth < max_depth and frontier:
            selected = await self._select_children(
                frontier, query, max_selections=max_results,
            )
            print(f"SEARCH_WIKI_DEBUG [T3] navigate layer: depth={depth}, selected={len(selected)}, names={[n.title[:30] for n in selected][:5]}", flush=True)
            if not selected:
                break

            next_frontier: List[TreeNode] = []
            for node in selected:
                node_id = id(node)
                if node_id in visited:
                    continue
                visited.add(node_id)

                if node.leaf or not node.children:
                    result_leaves.append(node)
                else:
                    next_frontier.extend(node.children)

            if len(result_leaves) >= max_results:
                break

            frontier = next_frontier
            depth += 1

        # Fallback: if no leaves found, expand last selected nodes
        if not result_leaves and selected:
            for node in selected:
                result_leaves.extend(node.all_leaves()[:max_results])

        # Deduplicate and cap
        seen_ids: set = set()
        unique: List[TreeNode] = []
        for n in result_leaves:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                unique.append(n)
        leaves = unique[:max_results]
        _page_valid = sum(1 for l in leaves if getattr(l, 'page_range', None) and len(l.page_range) == 2 and l.page_range[0])
        print(f"SEARCH_WIKI_DEBUG [T4] navigate result: leaves={len(leaves)}, page_range_valid={_page_valid}", flush=True)
        return leaves

    def load_tree(self, file_path: str) -> Optional[DocumentTree]:
        """Load a cached tree index for the given file (sync)."""
        file_hash = get_fast_hash(file_path)
        if file_hash is None:
            return None
        return self._load_cache(file_hash)

    def has_tree(self, file_path: str) -> bool:
        """Check whether a cached tree index exists for the file."""
        file_hash = get_fast_hash(file_path)
        if file_hash is None:
            return False
        return self._cache_path(file_hash).exists()

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    async def _build_tree_from_toc(
        self,
        toc_entries: List[Any],
        content: str,
        *,
        total_pages: Optional[int] = None,
    ) -> Optional[TreeNode]:
        """Build tree directly from extracted TOC entries, avoiding recursive LLM.

        Each TOCEntry becomes a TreeNode with char_range from the entry positions.
        Only the root summary requires an LLM call (_synthesize_root_summary).

        Args:
            toc_entries: List of TOCEntry from toc_extractor.
            content: Full extracted text of the document.
            total_pages: Total page count for page_range calculation.

        Returns:
            Root TreeNode, or None if no children could be created.
        """
        # Infer hierarchy when TOC entries are flat (all same level)
        toc_entries = self._infer_hierarchy(toc_entries)

        seen_ids: set = set()
        children = self._toc_entries_to_nodes(
            toc_entries, content, len(content), seen_ids,
            fallback_level=1, total_pages=total_pages,
        )

        if not children:
            return None

        root_summary = await self._synthesize_root_summary(children)
        root_page_range = (1, total_pages) if total_pages and total_pages > 0 else None
        return TreeNode(
            node_id=self._unique_node_id(0, seen_ids),
            title="Document",
            summary=root_summary,
            char_range=(0, len(content)),
            level=0,
            page_range=root_page_range,
            children=children,
        )

    @staticmethod
    def _toc_entries_to_nodes(
        entries: List[Any],
        content: str,
        parent_end: int,
        seen_ids: set,
        fallback_level: int,
        total_pages: Optional[int] = None,
    ) -> List["TreeNode"]:
        """Recursively convert TOCEntry trees into TreeNode trees.

        Handles arbitrary nesting depth and guards against invalid
        char_start / char_end values.  Computes ``page_range`` using a
        look-ahead algorithm when ``page_start`` is available on entries.

        Args:
            entries: List of TOCEntry objects (may have children).
            content: Full extracted text.
            parent_end: End offset inherited from the parent node.
            seen_ids: Set for unique node-id generation.
            fallback_level: Default level when entry.level is 0.
            total_pages: Total page count for page_range look-ahead.
        """
        nodes: List[TreeNode] = []
        content_len = len(content)
        for i, entry in enumerate(entries):
            start = max(0, min(entry.char_start, content_len))
            end = entry.char_end if entry.char_end and entry.char_end > start else parent_end
            end = min(end, content_len)

            section_text = content[start:min(start + _TOC_NODE_SUMMARY_MAX_CHARS, end)]
            nid = DocumentTreeIndexer._unique_node_id(start, seen_ids)
            level = entry.level if entry.level > 0 else fallback_level

            # page_range: look-ahead algorithm
            page_range = None
            if hasattr(entry, 'page_start') and entry.page_start is not None:
                # Find next sibling with page_start to determine page_end
                page_end = total_pages or entry.page_start
                for j in range(i + 1, len(entries)):
                    if hasattr(entries[j], 'page_start') and entries[j].page_start is not None:
                        page_end = entries[j].page_start
                        break
                page_range = (entry.page_start, max(entry.page_start, page_end))

            child_nodes: List[TreeNode] = []
            if entry.children:
                child_nodes = DocumentTreeIndexer._toc_entries_to_nodes(
                    entry.children, content, end, seen_ids,
                    fallback_level=level + 1,
                    total_pages=total_pages,
                )

            node = TreeNode(
                node_id=nid,
                title=entry.title,
                summary=section_text.strip(),
                char_range=(start, end),
                level=level,
                page_range=page_range,
                children=child_nodes,
            )
            nodes.append(node)
        return nodes

    @staticmethod
    def _unique_node_id(start: int, seen_ids: set) -> str:
        """Generate a unique node_id based on char offset, appending a
        disambiguator when collisions occur."""
        base = f"N{start:06d}"
        if base not in seen_ids:
            seen_ids.add(base)
            return base
        suffix = 1
        while f"{base}_{suffix}" in seen_ids:
            suffix += 1
        nid = f"{base}_{suffix}"
        seen_ids.add(nid)
        return nid

    @staticmethod
    def _compute_adaptive_depth(content_length: int) -> int:
        """Compute max tree depth based on document length.

        Longer documents get deeper trees for finer-grained navigation.
        Uses _TREE_ADAPTIVE_DEPTH_THRESHOLDS for threshold-based selection.

        Args:
            content_length: Character count of the document.

        Returns:
            Maximum tree depth (2-4).
        """
        for threshold, depth in _TREE_ADAPTIVE_DEPTH_THRESHOLDS:
            if content_length >= threshold:
                return depth
        return 2  # minimum depth

    async def _build_node(
        self, text: str, level: int, max_depth: int,
        offset: int = 0,
    ) -> Optional[TreeNode]:
        """Recursively build tree nodes via LLM structure analysis."""
        from sirchmunk.llm.prompts import COMPILE_TREE_STRUCTURE

        preview_size = self._compute_preview_size(len(text))
        preview = text[:preview_size]
        prompt = COMPILE_TREE_STRUCTURE.format(
            document_content=preview,
            max_sections=8,
        )

        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        sections = self._parse_sections(resp.content, text)

        if not sections:
            return TreeNode(
                node_id=f"N{offset:06d}",
                title="Document",
                summary=text[:300],
                char_range=(offset, offset + len(text)),
                level=level,
            )

        children: List[TreeNode] = []
        for i, sec in enumerate(sections):
            child = TreeNode(
                node_id=f"N{sec['start'] + offset:06d}",
                title=sec["title"],
                summary=sec["summary"],
                char_range=(sec["start"] + offset, sec["end"] + offset),
                level=level + 1,
            )
            section_text = text[sec["start"]:sec["end"]]
            if level + 1 < max_depth and len(section_text) > _TREE_MIN_CHARS:
                deeper = await self._build_node(
                    section_text, level + 1, max_depth, offset=sec["start"] + offset,
                )
                if deeper and deeper.children:
                    child.children = deeper.children
            children.append(child)

        root_summary = await self._synthesize_root_summary(children)

        return TreeNode(
            node_id=f"N{offset:06d}",
            title="Document",
            summary=root_summary,
            char_range=(offset, offset + len(text)),
            level=level,
            children=children,
        )

    async def _synthesize_root_summary(self, children: List[TreeNode]) -> str:
        """Synthesize a document-level summary from children's section summaries."""
        if not children:
            return ""
        from sirchmunk.llm.prompts import COMPILE_SYNTHESIZE_SUMMARY
        sections_text = "\n".join(
            f"- {c.title}: {c.summary}" for c in children
        )
        prompt = COMPILE_SYNTHESIZE_SUMMARY.format(sections=sections_text)
        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        return resp.content.strip()

    def _parse_sections(
        self, llm_output: str, full_text: str,
    ) -> List[Dict[str, Any]]:
        """Parse LLM section output into [{title, summary, start, end}, ...]."""
        # Try JSON array first
        try:
            raw = llm_output
            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                items = json.loads(m.group())
                return self._resolve_positions(items, full_text)
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    @staticmethod
    def _resolve_positions(
        items: List[Dict[str, Any]], full_text: str,
    ) -> List[Dict[str, Any]]:
        """Resolve section start/end character offsets from marker text.

        Two-pass algorithm:
          Pass 1 — determine all start positions with tiered fallback:
                   exact match from prev_end -> substring match -> full-text fallback.
          Pass 2 — set end[i] = start[i+1]; last end = text_len.

        Filters out invalid spans and overly large spans (> ``_MAX_SPAN_RATIO``
        of the document) to prevent accumulated positioning errors.
        """
        text_lower = full_text.lower()
        text_len = len(full_text)
        resolved: List[Dict[str, Any]] = []

        # Pass 1: determine all start positions
        prev_end = 0
        for item in items:
            title = item.get("title", "")
            marker = item.get("start_marker", title)

            pos = -1
            if marker:
                marker_lower = marker.lower()
                # Level 1: exact match from prev_end
                pos = text_lower.find(marker_lower, prev_end)
                # Level 2: substring match (first N chars) from prev_end
                if pos < 0 and len(marker_lower) > _MARKER_SUBSTRING_LEN:
                    pos = text_lower.find(
                        marker_lower[:_MARKER_SUBSTRING_LEN], prev_end,
                    )
                # Level 3: full text fallback from start
                if pos < 0:
                    pos = text_lower.find(marker_lower, 0)

            start = pos if pos >= 0 else prev_end
            resolved.append({
                "title": title,
                "summary": item.get("summary", ""),
                "start": start,
                "end": text_len,  # placeholder
            })
            prev_end = (
                start + max(1, len(marker))
                if pos >= 0
                else prev_end
            )

        # Pass 2: set end[i] = start[i+1], last end = text_len
        for i in range(len(resolved) - 1):
            resolved[i]["end"] = resolved[i + 1]["start"]
        if resolved:
            resolved[-1]["end"] = text_len

        # Filter out invalid spans and overly large spans
        return [
            s for s in resolved
            if s["end"] > s["start"]
            and (s["end"] - s["start"]) / max(text_len, 1) < _MAX_SPAN_RATIO
        ]

    async def _select_children(
        self, nodes: List[TreeNode], query: str, *, max_selections: int = 3,
    ) -> List[TreeNode]:
        """LLM-driven branch selection: pick the most relevant children.

        Dispatches to paginated selection when *nodes* exceeds
        ``_PAGE_SIZE_THRESHOLD`` to avoid overwhelming the LLM.
        """
        if len(nodes) <= 2:
            return nodes

        if len(nodes) > self._PAGE_SIZE_THRESHOLD:
            return await self._select_children_paginated(
                nodes, query, max_selections=max_selections,
            )

        listing = "\n".join(
            f"[{i}] {n.title}{self._format_page_range(n.page_range)}"
            f"{' [' + str(n.table_count) + ' tables]' if n.table_count > 0 else ''}"
            f": {n.summary[:150]}"
            for i, n in enumerate(nodes)
        )

        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Select the 1-2 most relevant sections (by index number):\n{listing}\n\n"
            f"Return ONLY a JSON array of index numbers, e.g. [0, 2]"
        )
        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        try:
            raw = resp.content.strip()
            m = re.search(r"\[[\d\s,]+\]", raw)
            if m:
                indices = json.loads(m.group())
                selected = [nodes[i] for i in indices if 0 <= i < len(nodes)]
                return selected if selected else nodes[:max_selections]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
        return nodes[:max_selections]

    async def _select_children_paginated(
        self,
        nodes: List[TreeNode],
        query: str,
        *,
        page_size: int = 15,
        max_selections: int = 3,
    ) -> List[TreeNode]:
        """Two-phase paginated selection for large node sets.

        Phase 1: partition *nodes* into sequential groups of *page_size*,
                 present group summaries to LLM, and select 1-2 groups.
        Phase 2: run fine-grained selection within each chosen group.

        Falls back to the first *max_selections* nodes on any LLM failure.
        """
        page_size = max(page_size, self._GROUP_PAGE_SIZE)

        # --- Phase 0: build groups ---
        groups: List[List[TreeNode]] = []
        for start in range(0, len(nodes), page_size):
            groups.append(nodes[start:start + page_size])

        if len(groups) <= 1:
            # Only one group — skip directly to fine-grained selection
            return await self._select_from_group(nodes, query, max_selections)

        # --- Phase 1: group-level selection ---
        group_listing = "\n".join(
            f"[{i}] {g[0].title} ... {g[-1].title} ({len(g)} sections)"
            for i, g in enumerate(groups)
        )
        group_prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"The document has {len(nodes)} sections organized into "
            f"{len(groups)} groups.\n"
            f"Select the 1-2 most relevant groups (by index number):\n"
            f"{group_listing}\n\n"
            f"Return ONLY a JSON array of group index numbers, e.g. [0, 2]"
        )

        selected_groups: List[List[TreeNode]] = []
        try:
            resp = await self._llm.achat(
                [{"role": "user", "content": group_prompt}],
            )
            raw = resp.content.strip()
            m = re.search(r"\[[\d\s,]+\]", raw)
            if m:
                g_indices = json.loads(m.group())
                selected_groups = [
                    groups[i] for i in g_indices if 0 <= i < len(groups)
                ]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass

        if not selected_groups:
            # Fallback: take the first group
            selected_groups = [groups[0]]

        # --- Phase 2: fine-grained selection within chosen groups ---
        results: List[TreeNode] = []
        for group in selected_groups:
            picked = await self._select_from_group(group, query, max_selections)
            results.extend(picked)

        # Deduplicate by node_id and cap
        seen: set = set()
        unique: List[TreeNode] = []
        for n in results:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique.append(n)
        return unique[:max_selections] if unique else nodes[:max_selections]

    async def _select_from_group(
        self,
        group: List[TreeNode],
        query: str,
        max_selections: int,
    ) -> List[TreeNode]:
        """Select the most relevant nodes within a single group via LLM."""
        if len(group) <= 2:
            return group

        listing = "\n".join(
            f"[{i}] {n.title}{self._format_page_range(n.page_range)}"
            f"{' [' + str(n.table_count) + ' tables]' if n.table_count > 0 else ''}"
            f": {n.summary[:150]}"
            for i, n in enumerate(group)
        )
        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Select the 1-2 most relevant sections (by index number):\n{listing}\n\n"
            f"Return ONLY a JSON array of index numbers, e.g. [0, 2]"
        )
        try:
            resp = await self._llm.achat([{"role": "user", "content": prompt}])
            raw = resp.content.strip()
            m = re.search(r"\[[\d\s,]+\]", raw)
            if m:
                indices = json.loads(m.group())
                selected = [group[i] for i in indices if 0 <= i < len(group)]
                if selected:
                    return selected[:max_selections]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
        return group[:max_selections]

    # ------------------------------------------------------------------ #
    #  Cache I/O                                                          #
    # ------------------------------------------------------------------ #

    def _cache_path(self, file_hash: str) -> Path:
        return self._cache_dir / f"{file_hash}.json"

    def _save_cache(self, file_hash: str, tree: DocumentTree) -> None:
        path = self._cache_path(file_hash)
        path.write_text(tree.to_json(), encoding="utf-8")
        print(f"SEARCH_WIKI_DEBUG [C5] tree_json_saved: path={path}", flush=True)

    def _load_cache(self, file_hash: str) -> Optional[DocumentTree]:
        path = self._cache_path(file_hash)
        if not path.exists():
            return None
        try:
            return DocumentTree.from_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_preview_size(text_len: int) -> int:
        """Compute adaptive preview window size for LLM structure analysis.

        Scales with document length: at least *_TREE_PREVIEW_MIN* chars,
        up to *_TREE_PREVIEW_MAX*, using *_TREE_PREVIEW_RATIO* of the
        document length as the baseline.
        """
        return max(
            _TREE_PREVIEW_MIN,
            min(int(text_len * _TREE_PREVIEW_RATIO), _TREE_PREVIEW_MAX),
        )

    @staticmethod
    def _count_nodes(node: TreeNode) -> int:
        return 1 + sum(DocumentTreeIndexer._count_nodes(c) for c in node.children)

    @staticmethod
    def _max_node_depth(node: TreeNode) -> int:
        if not node.children:
            return node.level
        return max(DocumentTreeIndexer._max_node_depth(c) for c in node.children)

    @staticmethod
    def _format_page_range(
        page_range: "Optional[Tuple[int, int]]",
    ) -> str:
        """Format a page_range tuple into a human-readable string for prompts."""
        if not page_range:
            return ""
        ps, pe = page_range
        return f" [pages {ps}-{pe}]" if ps != pe else f" [page {ps}]"

    # ------------------------------------------------------------------ #
    #  Leaf deepening & summary enrichment                                #
    # ------------------------------------------------------------------ #

    async def _deepen_large_leaves(
        self,
        node: TreeNode,
        content: str,
        *,
        max_leaf_chars: int = 5000,
        max_depth: int = 4,
        _seen_ids: Optional[set] = None,
    ) -> None:
        """Recursively deepen leaf nodes whose char_range exceeds *max_leaf_chars* using LLM decomposition."""
        if _seen_ids is None:
            _seen_ids = self._collect_node_ids(node)

        if not node.leaf:
            for child in node.children:
                await self._deepen_large_leaves(
                    child, content,
                    max_leaf_chars=max_leaf_chars,
                    max_depth=max_depth,
                    _seen_ids=_seen_ids,
                )
            return

        start, end = node.char_range
        span = end - start
        if span <= max_leaf_chars or node.level >= max_depth:
            return

        snippet = self._truncate_snippet(content[start:end])

        prompt = (
            "Analyze this document section and identify 3-8 logical sub-sections.\n"
            "For each sub-section, provide:\n"
            '- "title": descriptive heading (concise)\n'
            '- "start_text": the first 8-15 words that mark where this sub-section '
            "begins (must be exact text from the content)\n"
            '- "content_type": "text" or "table"\n\n'
            f'Section: "{node.title}"\n---\n{snippet}\n---\n\n'
            'Return ONLY a JSON array, e.g. '
            '[{"title": "...", "start_text": "...", "content_type": "text"}, ...]'
        )

        try:
            resp = await self._llm.achat([{"role": "user", "content": prompt}])
            sub_sections = self._parse_json_array(resp.content)
            if not sub_sections or len(sub_sections) < 2:
                return
        except Exception:
            return

        sub_nodes = self._build_sub_nodes_from_llm(
            sub_sections, node, content, _seen_ids,
        )
        if not sub_nodes:
            return

        node.children = sub_nodes
        await self._log.info(
            f"[TreeIndexer] Deepened '{node.title}' into {len(sub_nodes)} sub-nodes"
        )

        # Recurse into newly created children
        for child in node.children:
            await self._deepen_large_leaves(
                child, content,
                max_leaf_chars=max_leaf_chars,
                max_depth=max_depth,
                _seen_ids=_seen_ids,
            )

    def _build_sub_nodes_from_llm(
        self,
        sub_sections: List[Dict[str, Any]],
        parent: TreeNode,
        content: str,
        seen_ids: set,
    ) -> List[TreeNode]:
        """Create child TreeNodes from LLM-decomposed sub-sections."""
        parent_start, parent_end = parent.char_range
        parent_span = max(parent_end - parent_start, 1)
        parent_ps, parent_pe = parent.page_range if parent.page_range else (0, 0)
        page_span = parent_pe - parent_ps
        child_level = parent.level + 1

        # Resolve char_start for each sub-section
        positions: List[int] = []
        search_from = parent_start
        for sec in sub_sections:
            start_text = sec.get("start_text", "")
            pos = content.find(start_text, search_from) if start_text else -1
            if pos < 0 or pos >= parent_end:
                pos = search_from
            positions.append(pos)
            search_from = pos + 1

        nodes: List[TreeNode] = []
        for i, sec in enumerate(sub_sections):
            char_start = positions[i]
            char_end = positions[i + 1] if i + 1 < len(positions) else parent_end

            # Estimate page_range proportionally from parent
            page_range = None
            if parent.page_range and parent_span > 0:
                p_start = parent_ps + (char_start - parent_start) / parent_span * page_span
                p_end = parent_ps + (char_end - parent_start) / parent_span * page_span
                page_range = (int(p_start), max(int(p_start), int(p_end)))

            content_type = sec.get("content_type", "text")
            if content_type not in ("text", "table"):
                content_type = "text"

            nodes.append(TreeNode(
                node_id=self._unique_node_id(char_start, seen_ids),
                title=sec.get("title", f"Sub-section {i + 1}"),
                summary="",
                char_range=(char_start, char_end),
                level=child_level,
                page_range=page_range,
                content_type=content_type,
            ))
        return nodes

    async def _enrich_node_summaries(
        self,
        node: TreeNode,
        content: str,
        *,
        max_summary_len: int = 200,
    ) -> None:
        """Post-order traversal to enrich empty summaries: leaf from content, non-leaf via LLM."""
        # Post-order: process children first
        for child in node.children:
            await self._enrich_node_summaries(
                child, content, max_summary_len=max_summary_len,
            )

        if self._summary_needs_enrichment(node.summary):
            if node.leaf:
                node.summary = self._extract_leaf_summary(
                    content, node.char_range, max_summary_len,
                )
            else:
                node.summary = await self._generate_nonleaf_summary(
                    node, max_summary_len,
                )

    @staticmethod
    def _summary_needs_enrichment(summary: str) -> bool:
        """Check whether a summary is empty or too short to be useful."""
        return not summary or len(summary.strip()) < 10

    @staticmethod
    def _extract_leaf_summary(
        content: str,
        char_range: Tuple[int, int],
        max_len: int,
    ) -> str:
        """Extract a concise summary for a leaf node from its content slice."""
        start, end = char_range
        raw = content[start:end][:500]
        # Clean to single line
        return " ".join(raw.split())[:max_len]

    async def _generate_nonleaf_summary(
        self,
        node: TreeNode,
        max_summary_len: int,
    ) -> str:
        """Generate a summary for a non-leaf node via LLM, with fallback."""
        children_listing = "\n".join(
            f"- {c.title}: {c.summary[:100]}" for c in node.children
        )
        prompt = (
            "Summarize this document section in 1-2 concise sentences.\n"
            f'Section: "{node.title}"\n'
            f"Sub-sections:\n{children_listing}\n\n"
            "Return ONLY the summary text."
        )
        try:
            resp = await self._llm.achat([{"role": "user", "content": prompt}])
            return resp.content.strip()[:max_summary_len]
        except Exception:
            # Fallback: concatenate children titles
            return ", ".join(c.title for c in node.children)[:max_summary_len]

    # ------------------------------------------------------------------ #
    #  Parsing / snippet helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _truncate_snippet(
        text: str,
        *,
        head_chars: int = 3000,
        tail_chars: int = 1000,
    ) -> str:
        """Truncate a long text snippet keeping head and tail with an ellipsis marker."""
        if len(text) <= head_chars + tail_chars:
            return text
        return text[:head_chars] + "\n...[truncated]...\n" + text[-tail_chars:]

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """Extract and parse a JSON array from LLM output."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []

    @staticmethod
    def _collect_node_ids(node: TreeNode) -> set:
        """Collect all existing node_ids in the subtree."""
        ids = {node.node_id}
        for c in node.children:
            ids.update(DocumentTreeIndexer._collect_node_ids(c))
        return ids

    @staticmethod
    def should_build_tree(file_path: str, content_length: int) -> bool:
        """Determine whether a file is eligible for tree indexing."""
        ext = Path(file_path).suffix.lower()
        return ext in _TREE_EXTENSIONS and content_length >= _TREE_MIN_CHARS

    # ------------------------------------------------------------------ #
    #  Hierarchy inference for flat TOC entries                            #
    # ------------------------------------------------------------------ #

    # Minimum number of TOC entries to trigger hierarchy inference.
    # Documents with fewer entries are typically already well-structured.
    _FLAT_ENTRY_THRESHOLD = 20

    # If this fraction of entries share the same level, consider it "flat"
    # and apply hierarchy inference. Real hierarchies typically have
    # varied level distribution.
    _FLAT_LEVEL_RATIO = 0.9

    # Number of entries per virtual group when using uniform grouping fallback.
    _GROUP_SIZE = 15

    @staticmethod
    def _infer_hierarchy(entries: List[Any]) -> List[Any]:
        """When all entries share the same level, infer hierarchy from title patterns.

        Applies three strategies in priority order:
          A. Keyword groups — detect repeated structural prefixes (generic)
          B. Generic numbering patterns (1., 1.1, I., A., etc.)
          C. Uniform grouping fallback (virtual parent nodes)

        Only activates when >90% of entries share the same level and
        the total count exceeds ``_FLAT_ENTRY_THRESHOLD``.

        Args:
            entries: List of TOCEntry (may be nested).

        Returns:
            Possibly restructured list of TOCEntry with updated levels
            and rebuilt hierarchy.
        """
        if not entries:
            return entries or []

        try:
            from sirchmunk.learnings.toc_extractor import TOCExtractor
            flat: List[Any] = []
            TOCExtractor._flatten_entries(entries, flat)
        except Exception:
            return entries  # Cannot flatten; return original entries

        if not flat:
            return entries

        if len(flat) <= DocumentTreeIndexer._FLAT_ENTRY_THRESHOLD:
            return entries

        # Validate level field: skip entries with invalid levels
        valid_flat = [e for e in flat if hasattr(e, 'level') and isinstance(e.level, (int, float))]
        if not valid_flat:
            return entries

        # Check if >90% share the same level
        level_counts = Counter(e.level for e in valid_flat)
        dominant_level, dominant_count = level_counts.most_common(1)[0]
        if dominant_count / len(flat) <= DocumentTreeIndexer._FLAT_LEVEL_RATIO:
            return entries  # Already has meaningful hierarchy

        # Try strategies in priority order
        modified = DocumentTreeIndexer._strategy_keyword_groups(flat, dominant_level)
        if modified is None:
            modified = DocumentTreeIndexer._strategy_numbering(flat, dominant_level)
        if modified is None:
            modified = DocumentTreeIndexer._strategy_uniform_grouping(
                flat, dominant_level,
            )
        if modified is None:
            return entries

        # Rebuild hierarchy from the re-leveled flat list
        return TOCExtractor._build_hierarchy(modified)

    # -- Strategy A: keyword groups (generic structural prefix detection) #

    # Pattern: title starts with a capitalized word optionally followed by
    # a Roman numeral or Arabic number (e.g. "PART IV", "Item 1A",
    # "Section 3", "Chapter 12", "Article II").
    _RE_STRUCTURAL_PREFIX = re.compile(
        r'^([A-Z][A-Za-z]*(?:\s+[IVXLCDM\d]+[A-Za-z]?)?)\b',
    )

    @staticmethod
    def _extract_structural_prefix(title: str) -> Optional[str]:
        """Extract a structural prefix from a title.

        Matches leading capitalized words optionally followed by a number
        or Roman numeral (e.g. "PART IV", "Item 1A", "Section 3").
        Returns the normalized (uppercased) prefix, or None.
        """
        if not title or not title.strip():
            return None
        m = DocumentTreeIndexer._RE_STRUCTURAL_PREFIX.match(title.strip())
        if m:
            prefix = m.group(1).strip()
            # Prefix must not be too long (avoid capturing entire title)
            if len(prefix) <= 20:
                return prefix.upper()
        return None

    @staticmethod
    def _strategy_keyword_groups(
        flat: List[Any],
        dominant_level: int,
    ) -> Optional[List[Any]]:
        """Strategy A — detect repeated structural prefixes and infer levels.

        Works for any document with repetitive heading patterns (SEC filings,
        legal contracts, technical specs, etc.).  Automatically discovers
        prefix groups and assigns hierarchical levels based on frequency:
        lower-frequency prefixes become higher-level parents.

        Returns re-leveled flat list, or None if coverage is insufficient.
        """
        # 1. Extract prefix for each entry
        prefix_map: Dict[str, List[int]] = {}  # prefix -> [entry indices]
        for i, e in enumerate(flat):
            prefix = DocumentTreeIndexer._extract_structural_prefix(e.title)
            if prefix:
                prefix_map.setdefault(prefix, []).append(i)

        # 2. Keep only prefixes appearing >= 2 times
        repeated_prefixes = {k: v for k, v in prefix_map.items() if len(v) >= 2}
        if not repeated_prefixes:
            return None

        # 3. Check coverage: at least 30% of entries must be covered
        covered = sum(len(indices) for indices in repeated_prefixes.values())
        if covered < len(flat) * 0.3:
            return None

        # 4. Sort prefixes by frequency (ascending) then by first appearance
        #    Low frequency = higher level (parent), high frequency = lower level
        sorted_prefixes = sorted(
            repeated_prefixes.items(),
            key=lambda x: (len(x[1]), min(x[1])),
        )

        # 5. Assign level per prefix group
        prefix_to_level: Dict[str, int] = {}
        for level_idx, (prefix, _) in enumerate(sorted_prefixes):
            prefix_to_level[prefix] = level_idx + 1

        # 6. Determine the "other" level for entries without a known prefix
        max_level = max(prefix_to_level.values()) + 1

        # 7. Apply levels
        for i, e in enumerate(flat):
            prefix = DocumentTreeIndexer._extract_structural_prefix(e.title)
            if prefix and prefix in prefix_to_level:
                e.level = prefix_to_level[prefix]
            else:
                e.level = max_level
            e.children = []

        return flat

    # -- Strategy B: generic numbering --------------------------------- #

    # Three-level numbering: 1.1.1, (a), (i), (1)
    _RE_NUM_LEVEL3 = re.compile(
        r"^\s*(?:\d+\.\d+\.\d+|\([a-z]\)|\([ivx]+\)|\(\d+\))\s",
        re.IGNORECASE,
    )
    # Two-level numbering: 1.1, A., B., a., b.
    _RE_NUM_LEVEL2 = re.compile(
        r"^\s*(?:\d+\.\d+(?!\.)\b|[A-Z]\.\s|[a-z]\.\s)",
    )
    # Top-level numbering: 1., 2., I., II.
    _RE_NUM_LEVEL1 = re.compile(
        r"^\s*(?:\d+\.\s|[IVXLC]+\.\s)",
    )

    @staticmethod
    def _strategy_numbering(
        flat: List[Any],
        dominant_level: int,
    ) -> Optional[List[Any]]:
        """Strategy B — detect generic numbering patterns.

        Returns re-leveled flat list, or None if fewer than 30% of
        entries match any numbering pattern.
        """
        matched = 0
        assignments: List[Optional[int]] = []

        for e in flat:
            title = e.title
            if DocumentTreeIndexer._RE_NUM_LEVEL3.match(title):
                assignments.append(3)
                matched += 1
            elif DocumentTreeIndexer._RE_NUM_LEVEL2.match(title):
                assignments.append(2)
                matched += 1
            elif DocumentTreeIndexer._RE_NUM_LEVEL1.match(title):
                assignments.append(1)
                matched += 1
            else:
                assignments.append(None)

        if matched < len(flat) * 0.3:
            return None

        # Apply assignments; entries without a pattern get the level of
        # the previous entry + 1 (capped at 3)
        prev_level = 1
        for i, e in enumerate(flat):
            if assignments[i] is not None:
                e.level = assignments[i]
            else:
                e.level = min(prev_level + 1, 3)
            prev_level = e.level
            e.children = []
        return flat

    # -- Strategy C: uniform grouping fallback ------------------------- #

    @staticmethod
    def _strategy_uniform_grouping(
        flat: List[Any],
        dominant_level: int,
    ) -> Optional[List[Any]]:
        """Strategy C — group entries into fixed-size buckets with virtual parents.

        Creates synthetic parent TOCEntry nodes whose char_start/char_end
        and page_start/page_end are derived from the first and last child
        in each group.

        Returns the re-leveled flat list including virtual parents, or None
        on error.
        """
        from sirchmunk.learnings.toc_extractor import TOCEntry

        group_size = DocumentTreeIndexer._GROUP_SIZE
        num_groups = math.ceil(len(flat) / group_size)
        if num_groups <= 1:
            return None  # Grouping would not improve anything

        parent_level = max(1, dominant_level - 1) if dominant_level > 1 else 1
        child_level = parent_level + 1

        result: List[Any] = []
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min((g + 1) * group_size, len(flat))
            group = flat[start_idx:end_idx]

            first = group[0]
            last = group[-1]

            # Derive positions from children
            char_start = first.char_start
            char_end = last.char_end if last.char_end else None
            page_start = first.page_start
            page_end = last.page_start  # Best available estimate

            virtual_parent = TOCEntry(
                title=f"{first.title} \u2013 {last.title}",
                level=parent_level,
                char_start=char_start,
                char_end=char_end,
                page_start=page_start,
                page_end=page_end,
                children=[],
                source="inferred",
            )
            result.append(virtual_parent)

            # Set child level
            for e in group:
                e.level = child_level
                e.children = []
            result.extend(group)

        return result
