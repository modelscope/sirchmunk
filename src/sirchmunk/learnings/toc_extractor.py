# Copyright (c) ModelScope Contributors. All rights reserved.
"""
TOC (Table of Contents) extractor — pure local operations, zero LLM calls.

Extracts hierarchical table-of-contents structures from various document
formats (PDF, Markdown, DOCX, HTML) using native format features (bookmarks,
heading styles, heading tags).  The extracted TOCEntry list is consumed by
the tree indexer to accelerate tree construction.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Minimum number of TOC entries required to form a meaningful structure
_MIN_TOC_ENTRIES = 3

# Known heading-style prefixes across locales (English, Chinese, etc.)
_HEADING_STYLE_PREFIXES = ("Heading", "heading", "\u6807\u9898")  # "标题" = Chinese


@dataclass
class TOCEntry:
    """Single entry in an extracted table of contents."""

    title: str
    level: int  # 0=root, 1=section, 2=subsection
    char_start: int  # Character offset in extracted text
    char_end: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    children: List["TOCEntry"] = field(default_factory=list)


class TOCExtractor:
    """Extract TOC structure from documents using native format features.

    All methods are static — no instance state required.  Each extraction
    method handles one file format and returns a flat or nested list of
    ``TOCEntry`` objects.  The main ``extract()`` entry point dispatches
    by file extension and resolves character positions against the
    extracted text content.

    Design constraints:
    - Pure local operations, zero LLM calls
    - Exceptions handled internally; failure returns None
    """

    @staticmethod
    def extract(file_path: str, content: str) -> Optional[List[TOCEntry]]:
        """Main entry point: extract TOC entries from a file.

        Dispatches to format-specific extractors based on file extension,
        then resolves character positions in the extracted text content.

        Args:
            file_path: Absolute path to the source file.
            content: Extracted text content of the file.

        Returns:
            List of TOCEntry with resolved char positions, or None if
            the file format is unsupported or fewer than _MIN_TOC_ENTRIES
            entries are found.
        """
        ext = Path(file_path).suffix.lower()

        entries: Optional[List[TOCEntry]] = None
        if ext == ".pdf":
            entries = TOCExtractor._extract_pdf_toc(file_path)
        elif ext in (".md", ".markdown"):
            entries = TOCExtractor._extract_markdown_toc(content)
        elif ext in (".docx",):
            entries = TOCExtractor._extract_docx_toc(file_path)
        elif ext in (".html", ".htm"):
            entries = TOCExtractor._extract_html_toc(content)
        else:
            return None

        if not entries:
            return None

        # Flatten nested children for total count check
        total = TOCExtractor._count_entries(entries)
        if total < _MIN_TOC_ENTRIES:
            return None

        # Resolve character positions in extracted text
        entries = TOCExtractor._resolve_char_positions(entries, content)
        return entries

    @staticmethod
    def _extract_pdf_toc(file_path: str) -> Optional[List[TOCEntry]]:
        """Extract TOC from PDF bookmarks/outline using pypdf.

        Recursively parses the nested bookmark structure from
        ``PdfReader.outline``.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of TOCEntry with page_start populated, or None on failure.
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            outline = reader.outline
            if not outline:
                return None

            entries: List[TOCEntry] = []
            TOCExtractor._parse_pdf_outline(reader, outline, entries, level=1)
            return entries if entries else None
        except Exception:
            return None

    @staticmethod
    def _parse_pdf_outline(
        reader: "PdfReader",
        outline_items: List,
        entries: List[TOCEntry],
        level: int,
    ) -> None:
        """Recursively parse pypdf outline items into TOCEntry list.

        Args:
            reader: PdfReader instance for page number resolution.
            outline_items: Nested list of outline Destination objects.
            entries: Accumulator list to append entries to.
            level: Current nesting level (1=top-level section).
        """
        for item in outline_items:
            if isinstance(item, list):
                # Nested list means sub-bookmarks — attach to last entry
                if entries:
                    sub_entries: List[TOCEntry] = []
                    TOCExtractor._parse_pdf_outline(
                        reader, item, sub_entries, level=level + 1,
                    )
                    entries[-1].children.extend(sub_entries)
                else:
                    TOCExtractor._parse_pdf_outline(
                        reader, item, entries, level=level,
                    )
            else:
                # Single bookmark destination
                try:
                    title = item.title if hasattr(item, "title") else str(item)
                    page_num = None
                    try:
                        page_num = reader.get_destination_page_number(item)
                    except Exception:
                        pass
                    entry = TOCEntry(
                        title=title.strip(),
                        level=level,
                        char_start=0,
                        page_start=page_num,
                    )
                    entries.append(entry)
                except Exception:
                    continue

    @staticmethod
    def _extract_markdown_toc(content: str) -> Optional[List[TOCEntry]]:
        """Extract TOC from Markdown heading syntax (# / ## / ###).

        Matches ATX-style headings: lines beginning with 1-6 '#' characters
        followed by whitespace and the heading text.

        Args:
            content: Markdown text content.

        Returns:
            List of TOCEntry with level derived from '#' count, or None.
        """
        try:
            pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
            matches = pattern.findall(content)
            if not matches:
                return None

            entries: List[TOCEntry] = []
            for hashes, title in matches:
                entries.append(TOCEntry(
                    title=title.strip(),
                    level=len(hashes),
                    char_start=0,
                ))
            return entries if entries else None
        except Exception:
            return None

    @staticmethod
    def _extract_docx_toc(file_path: str) -> Optional[List[TOCEntry]]:
        """Extract TOC from DOCX heading styles using python-docx.

        Reads paragraphs with heading style names (English ``Heading``,
        Chinese ``\u6807\u9898``, etc.), extracting the heading level from the style
        name suffix (e.g., ``Heading 1`` -> level 1).

        Args:
            file_path: Path to the DOCX file.

        Returns:
            List of TOCEntry with level from heading style, or None.
        """
        try:
            import docx

            doc = docx.Document(file_path)
            entries: List[TOCEntry] = []
            for para in doc.paragraphs:
                style_name = para.style.name or ""
                # Match heading styles across locales ("Heading 1", "标题 1", etc.)
                matched_prefix = ""
                for prefix in _HEADING_STYLE_PREFIXES:
                    if style_name.startswith(prefix):
                        matched_prefix = prefix
                        break
                if not matched_prefix:
                    continue
                level_str = style_name[len(matched_prefix):].strip()
                try:
                    level = int(level_str) if level_str else 1
                except ValueError:
                    level = 1
                title = para.text.strip()
                if title:
                    entries.append(TOCEntry(
                        title=title,
                        level=level,
                        char_start=0,
                    ))
            return entries if entries else None
        except Exception:
            return None

    @staticmethod
    def _extract_html_toc(content: str) -> Optional[List[TOCEntry]]:
        """Extract TOC from HTML heading tags (<h1> through <h6>).

        Uses regex to match heading tags and strips inner HTML tags
        from the title text.

        Args:
            content: HTML text content.

        Returns:
            List of TOCEntry with level from tag number, or None.
        """
        try:
            pattern = re.compile(
                r"<h([1-6])[^>]*>(.*?)</h\1>",
                re.IGNORECASE | re.DOTALL,
            )
            matches = pattern.findall(content)
            if not matches:
                return None

            entries: List[TOCEntry] = []
            for level_str, raw_title in matches:
                # Strip HTML tags from title
                title = re.sub(r"<[^>]+>", "", raw_title).strip()
                if title:
                    entries.append(TOCEntry(
                        title=title,
                        level=int(level_str),
                        char_start=0,
                    ))
            return entries if entries else None
        except Exception:
            return None

    @staticmethod
    def _resolve_char_positions(
        entries: List[TOCEntry],
        content: str,
    ) -> List[TOCEntry]:
        """Resolve character start/end positions for TOC entries in content.

        Searches for each entry's title in the content text using
        case-insensitive matching, progressing forward to avoid duplicate
        matches.  Sets char_end to the start of the next entry (or
        len(content) for the last entry).

        Also recurses into children to resolve their positions.

        Args:
            entries: Flat list of TOCEntry to resolve.
            content: Full extracted text to search within.

        Returns:
            The same list with char_start and char_end populated.
        """
        if not content or not entries:
            return entries

        content_lower = content.lower()
        search_from = 0

        # Collect all entries in document order (top-level + children)
        flat: List[TOCEntry] = []
        TOCExtractor._flatten_entries(entries, flat)

        # Pass 1: resolve char_start for each entry
        for entry in flat:
            title_lower = entry.title.lower().strip()
            if not title_lower:
                entry.char_start = search_from
                continue
            # Normalise whitespace for fuzzy matching (PDF extracts may
            # insert extra spaces inside headings).
            title_normalised = re.sub(r"\s+", " ", title_lower)
            pos = content_lower.find(title_normalised, search_from)
            if pos < 0:
                # Retry with the original (un-normalised) title
                pos = content_lower.find(title_lower, search_from)
            if pos >= 0:
                entry.char_start = pos
                search_from = pos + len(title_lower)
            else:
                # Title not found after search_from; try from beginning
                pos = content_lower.find(title_normalised)
                if pos < 0:
                    pos = content_lower.find(title_lower)
                if pos >= 0:
                    entry.char_start = pos
                    # Do NOT reset search_from to avoid breaking order
                else:
                    # Last resort: place at current search frontier
                    entry.char_start = search_from

        # Pass 2: resolve char_end as start of next entry (or len(content))
        for i in range(len(flat) - 1):
            flat[i].char_end = flat[i + 1].char_start
        if flat:
            flat[-1].char_end = len(content)

        return entries

    @staticmethod
    def _flatten_entries(
        entries: List[TOCEntry],
        flat: List[TOCEntry],
    ) -> None:
        """Flatten nested TOCEntry tree into document-order list.

        Args:
            entries: Nested entry list.
            flat: Accumulator for flattened output.
        """
        for entry in entries:
            flat.append(entry)
            if entry.children:
                TOCExtractor._flatten_entries(entry.children, flat)

    @staticmethod
    def _count_entries(entries: List[TOCEntry]) -> int:
        """Count total entries including nested children.

        Args:
            entries: Nested entry list.

        Returns:
            Total number of entries in the tree.
        """
        count = 0
        for entry in entries:
            count += 1
            if entry.children:
                count += TOCExtractor._count_entries(entry.children)
        return count
    @staticmethod
    def _count_entries(entries: List[TOCEntry]) -> int:
        """Count total entries including nested children.

        Args:
            entries: Nested entry list.

        Returns:
            Total number of entries in the tree.
        """
        count = 0
        for entry in entries:
            count += 1
            if entry.children:
                count += TOCExtractor._count_entries(entry.children)
        return count
