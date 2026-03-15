# Copyright (c) ModelScope Contributors. All rights reserved.
"""Structure-aware document chunking for evidence extraction.

Splits documents along natural boundaries (paragraphs, headings, code
fences, JSON lines) instead of fixed-size sliding windows, producing
semantically coherent chunks for downstream scoring and evaluation.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """A semantically coherent document segment."""

    start: int
    end: int
    content: str
    chunk_type: str = "text"  # text | code | json_line | heading

    @property
    def length(self) -> int:
        return self.end - self.start


class DocumentChunker:
    """Split documents along natural boundaries.

    Detects document structure (paragraphs, code blocks, JSON lines,
    markdown headings) and produces chunks whose sizes are balanced
    around *target_size* by merging undersize and splitting oversize
    segments.
    """

    _CODE_FENCE = re.compile(r"^```", re.MULTILINE)
    _HEADING = re.compile(r"^#{1,6}\s", re.MULTILINE)
    _BLANK_LINE = re.compile(r"\n\s*\n")

    def __init__(
        self,
        target_size: int = 800,
        max_size: int = 2000,
        min_size: int = 100,
    ):
        self.target_size = target_size
        self.max_size = max_size
        self.min_size = min_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: str) -> List[Chunk]:
        """Split *doc* into balanced, boundary-respecting chunks."""
        if not doc or not doc.strip():
            return []

        doc_type = self._detect_type(doc)

        if doc_type == "jsonl":
            raw = self._split_jsonl(doc)
        elif doc_type == "markdown":
            raw = self._split_markdown(doc)
        elif doc_type == "code":
            raw = self._split_code(doc)
        else:
            raw = self._split_paragraphs(doc)

        return self._balance(raw, doc)

    # ------------------------------------------------------------------
    # Heuristic document-type detection
    # ------------------------------------------------------------------

    def _detect_type(self, doc: str) -> str:
        head = doc.lstrip()[:2000]
        if head.startswith("{") and "\n{" in head:
            return "jsonl"
        fence_count = len(self._CODE_FENCE.findall(head))
        if fence_count > 4:
            return "code"
        if self._HEADING.search(head):
            return "markdown"
        return "text"

    # ------------------------------------------------------------------
    # Type-specific splitters
    # ------------------------------------------------------------------

    def _split_jsonl(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        pos = 0
        for line in doc.split("\n"):
            end = pos + len(line)
            stripped = line.strip()
            if stripped:
                chunks.append(
                    Chunk(start=pos, end=end, content=stripped, chunk_type="json_line")
                )
            pos = end + 1
        return chunks

    def _split_markdown(self, doc: str) -> List[Chunk]:
        positions = [m.start() for m in self._HEADING.finditer(doc)]
        if not positions:
            return self._split_paragraphs(doc)

        boundaries = sorted(set([0] + positions + [len(doc)]))
        sections: List[Chunk] = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            text = doc[s:e].strip()
            if text:
                ctype = "heading" if s in positions else "text"
                sections.append(Chunk(start=s, end=e, content=text, chunk_type=ctype))
        return sections

    def _split_code(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        in_code = False
        code_start = 0
        text_start = 0
        pos = 0

        for line in doc.split("\n"):
            line_end = pos + len(line)
            if line.strip().startswith("```"):
                if not in_code:
                    preceding = doc[text_start:pos].strip()
                    if preceding:
                        chunks.append(
                            Chunk(start=text_start, end=pos, content=preceding, chunk_type="text")
                        )
                    code_start = pos
                    in_code = True
                else:
                    code_content = doc[code_start:line_end].strip()
                    if code_content:
                        chunks.append(
                            Chunk(start=code_start, end=line_end, content=code_content, chunk_type="code")
                        )
                    in_code = False
                    text_start = line_end + 1
            pos = line_end + 1

        remaining_start = code_start if in_code else text_start
        remaining = doc[remaining_start:].strip()
        if remaining:
            rtype = "code" if in_code else "text"
            chunks.append(
                Chunk(start=remaining_start, end=len(doc), content=remaining, chunk_type=rtype)
            )
        return chunks if chunks else self._split_paragraphs(doc)

    def _split_paragraphs(self, doc: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        parts = self._BLANK_LINE.split(doc)
        pos = 0
        for part in parts:
            idx = doc.find(part, pos)
            if idx == -1:
                idx = pos
            stripped = part.strip()
            if stripped:
                chunks.append(
                    Chunk(start=idx, end=idx + len(part), content=stripped, chunk_type="text")
                )
            pos = idx + len(part)
        return chunks if chunks else [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

    # ------------------------------------------------------------------
    # Post-processing: merge / split to target size
    # ------------------------------------------------------------------

    def _balance(self, raw_chunks: List[Chunk], doc: str) -> List[Chunk]:
        if not raw_chunks:
            return [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

        balanced: List[Chunk] = []
        buf: Optional[Chunk] = None

        for chunk in raw_chunks:
            if chunk.length > self.max_size:
                if buf:
                    balanced.append(buf)
                    buf = None
                balanced.extend(self._split_fixed(chunk))
            elif chunk.length < self.min_size and buf is not None:
                buf = Chunk(
                    start=buf.start,
                    end=chunk.end,
                    content=doc[buf.start : chunk.end],
                    chunk_type=buf.chunk_type,
                )
                if buf.length >= self.target_size:
                    balanced.append(buf)
                    buf = None
            else:
                if buf:
                    balanced.append(buf)
                buf = chunk

        if buf:
            balanced.append(buf)

        return balanced if balanced else [Chunk(start=0, end=len(doc), content=doc, chunk_type="text")]

    def _split_fixed(self, chunk: Chunk) -> List[Chunk]:
        parts: List[Chunk] = []
        text = chunk.content
        offset = chunk.start
        step = self.target_size

        i = 0
        while i < len(text):
            end = min(i + step, len(text))
            if end < len(text):
                search_start = max(i, end - step // 5)
                for sep in ("\n", ". ", "。", "；", "; "):
                    idx = text.rfind(sep, search_start, end)
                    if idx > i:
                        end = idx + len(sep)
                        break
            segment = text[i:end]
            if segment.strip():
                parts.append(
                    Chunk(
                        start=offset + i,
                        end=offset + end,
                        content=segment,
                        chunk_type=chunk.chunk_type,
                    )
                )
            i = end

        return parts
