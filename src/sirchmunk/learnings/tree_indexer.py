# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Document tree indexer — PageIndex-inspired hierarchical structure analysis.

Builds a JSON tree index for structured long documents (PDF, DOCX, MD, HTML)
so that downstream search can navigate via LLM reasoning instead of brute-force
Monte Carlo sampling.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.utils import LogCallback, create_logger
from sirchmunk.utils.file_utils import get_fast_hash

# File-size threshold: skip tree indexing for small files
_TREE_MIN_CHARS = 50_000  # 50 K characters

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
            "char_range": list(self.char_range),
            "level": self.level,
            "page_range": list(self.page_range) if self.page_range else None,
            "children": [c.to_dict() for c in self.children],
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
    ) -> Optional[DocumentTree]:
        """Build a tree index for a document.

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

        await self._log.info(
            f"[TreeIndexer] Building tree for {Path(file_path).name} "
            f"({len(content)} chars, depth={max_depth})"
        )

        root = await self._build_node(content, level=0, max_depth=max_depth)
        if root is None:
            return None

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
    ) -> List[TreeNode]:
        """Reasoning-based tree navigation: LLM selects the most relevant branches.

        Returns up to *max_results* leaf nodes with their char_range for
        precise evidence extraction.
        """
        if tree.root is None:
            return []

        candidates = tree.root.children if tree.root.children else [tree.root]
        if not candidates:
            return [tree.root]

        selected = await self._select_children(candidates, query)
        if not selected:
            return []

        result_leaves: List[TreeNode] = []
        for node in selected:
            if node.leaf:
                result_leaves.append(node)
            else:
                deeper = await self._select_children(node.children, query)
                for d in (deeper or node.children[:1]):
                    result_leaves.extend(d.all_leaves()[:max_results])

        # Deduplicate and cap
        seen_ids = set()
        unique: List[TreeNode] = []
        for n in result_leaves:
            if n.node_id not in seen_ids:
                seen_ids.add(n.node_id)
                unique.append(n)
        return unique[:max_results]

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
        """Resolve section start/end character offsets from marker text."""
        resolved: List[Dict[str, Any]] = []
        prev_end = 0
        text_lower = full_text.lower()
        for item in items:
            title = item.get("title", "")
            summary = item.get("summary", "")
            marker = item.get("start_marker", title)

            pos = text_lower.find(marker.lower(), prev_end) if marker else -1
            start = pos if pos >= 0 else prev_end

            end_marker = item.get("end_marker", "")
            if end_marker:
                epos = text_lower.find(end_marker.lower(), start + 1)
                end = epos if epos > start else min(start + 50000, len(full_text))
            else:
                end = min(start + 50000, len(full_text))

            resolved.append({
                "title": title,
                "summary": summary,
                "start": start,
                "end": end,
            })
            prev_end = end

        # Fix gaps: each section ends where the next begins
        for i in range(len(resolved) - 1):
            resolved[i]["end"] = resolved[i + 1]["start"]
        if resolved:
            resolved[-1]["end"] = len(full_text)

        return [s for s in resolved if s["end"] > s["start"]]

    async def _select_children(
        self, nodes: List[TreeNode], query: str,
    ) -> List[TreeNode]:
        """LLM-driven branch selection: pick the most relevant children."""
        if len(nodes) <= 2:
            return nodes

        listing = "\n".join(
            f"[{i}] {n.title}: {n.summary[:150]}"
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
                return [nodes[i] for i in indices if 0 <= i < len(nodes)]
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
        return nodes[:2]

    # ------------------------------------------------------------------ #
    #  Cache I/O                                                          #
    # ------------------------------------------------------------------ #

    def _cache_path(self, file_hash: str) -> Path:
        return self._cache_dir / f"{file_hash}.json"

    def _save_cache(self, file_hash: str, tree: DocumentTree) -> None:
        path = self._cache_path(file_hash)
        path.write_text(tree.to_json(), encoding="utf-8")

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
    def should_build_tree(file_path: str, content_length: int) -> bool:
        """Determine whether a file is eligible for tree indexing."""
        ext = Path(file_path).suffix.lower()
        return ext in _TREE_EXTENSIONS and content_length >= _TREE_MIN_CHARS
