# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unified document extraction facade over kreuzberg.

Centralizes all kreuzberg interaction into a single module, providing a clean,
configurable interface for document text extraction with support for tables,
metadata, language detection, OCR, and page-range filtering.

All other modules should import from here rather than from kreuzberg directly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Optional, Sequence, Union

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionProfile:
    """Immutable extraction configuration profile.

    Controls which kreuzberg features are enabled during document extraction.
    Default values align with the legacy ``fast_extract()`` behavior
    (plain text only, no extras).
    """

    output_format: str = "plain"
    """Output format: ``plain`` | ``markdown`` | ``html`` | ``djot``."""

    extract_tables: bool = False
    """Whether to extract and return tables."""

    extract_metadata: bool = False
    """Whether to return document metadata."""

    detect_language: bool = False
    """Whether to detect document language."""

    ocr_enabled: bool = False
    """Whether to enable OCR fallback."""

    ocr_backend: str = "tesseract"
    """OCR engine: ``tesseract`` | ``easyocr`` | ``paddleocr``."""

    ocr_language: str = "eng"
    """OCR language code (e.g. ``eng``, ``chi_sim``)."""

    page_start: Optional[int] = None
    """Page range start (0-indexed). ``None`` means first page."""

    page_end: Optional[int] = None
    """Page range end (inclusive). ``None`` means last page."""

    pdf_extract_images: bool = False
    """Extract images embedded in PDF pages."""

    pdf_extract_metadata: bool = False
    """Extract PDF-level metadata (author, title, etc.)."""

    force_ocr: bool = False
    """Force OCR for all pages, bypassing native text extraction.

    Maps directly to kreuzberg's ``ExtractionConfig.force_ocr``.
    Note: kreuzberg does not offer a "fallback" OCR mode —
    when set, OCR is always applied regardless of text layer presence.
    """

    pdf_password: Optional[str] = None
    """Password for encrypted PDFs."""

    max_concurrent: Optional[int] = None
    """Max concurrency for batch extraction."""


# ---------------------------------------------------------------------------
# Extraction output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionOutput:
    """Structured extraction result.

    Always contains ``content``.  Other fields are populated based on the
    :class:`ExtractionProfile` settings used during extraction.
    """

    content: str
    """Extracted text content."""

    mime_type: str = ""
    """MIME type of the source document."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Document metadata (empty when ``extract_metadata`` is disabled)."""

    tables: list[dict[str, Any]] = field(default_factory=list)
    """Extracted tables (empty when ``extract_tables`` is disabled)."""

    detected_languages: dict[str, float] = field(default_factory=dict)
    """Language → confidence mapping (empty when ``detect_language`` is disabled)."""

    page_count: Optional[int] = None
    """Number of pages in the source document (if available)."""


# ---------------------------------------------------------------------------
# Document extractor facade
# ---------------------------------------------------------------------------

class DocumentExtractor:
    """Unified document extraction facade over kreuzberg.

    Provides a clean, configurable interface for document text extraction,
    centralizing all kreuzberg interaction within a single module.

    Usage::

        # Basic extraction (identical to legacy fast_extract)
        result = await DocumentExtractor.extract(path)

        # Enhanced extraction with tables and metadata
        result = await DocumentExtractor.extract(path, DocumentExtractor.ENHANCED)

        # Custom profile
        profile = ExtractionProfile(output_format="markdown", extract_tables=True)
        result = await DocumentExtractor.extract(path, profile)
    """

    # Pre-defined profiles -------------------------------------------------

    BASIC: ClassVar[ExtractionProfile] = ExtractionProfile()
    """Plain-text extraction only — equivalent to legacy ``fast_extract()``."""

    ENHANCED: ClassVar[ExtractionProfile] = ExtractionProfile(
        output_format="markdown",
        extract_tables=True,
        extract_metadata=True,
        pdf_extract_metadata=True,
        force_ocr=True,
    )
    """Rich extraction with tables, metadata, and OCR fallback."""

    # Public API -----------------------------------------------------------

    @staticmethod
    async def extract(
        file_path: Union[str, Path],
        profile: Optional[ExtractionProfile] = None,
    ) -> ExtractionOutput:
        """Extract content from a single file.

        Args:
            file_path: Path to the document.
            profile:   Extraction profile.  Defaults to :attr:`BASIC`.

        Returns:
            :class:`ExtractionOutput` with at least ``content`` populated.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            Exception: Propagates kreuzberg extraction errors after logging.
        """
        from kreuzberg import extract_file

        profile = profile or DocumentExtractor.BASIC
        config = DocumentExtractor._build_config(profile)

        try:
            result = await extract_file(file_path=file_path, config=config)
            return DocumentExtractor._convert_result(result, profile)
        except Exception as exc:
            logger.error(
                "Document extraction failed for {}: {}",
                file_path,
                exc,
            )
            raise

    @staticmethod
    async def extract_bytes(
        data: bytes,
        mime_type: str,
        profile: Optional[ExtractionProfile] = None,
    ) -> ExtractionOutput:
        """Extract content from raw bytes.

        Args:
            data:      File content as bytes.
            mime_type: MIME type of the data (required for format detection).
            profile:   Extraction profile.  Defaults to :attr:`BASIC`.

        Returns:
            :class:`ExtractionOutput`.
        """
        from kreuzberg import extract_bytes as _extract_bytes

        profile = profile or DocumentExtractor.BASIC
        config = DocumentExtractor._build_config(profile)

        try:
            result = await _extract_bytes(data=data, mime_type=mime_type, config=config)
            return DocumentExtractor._convert_result(result, profile)
        except Exception:
            logger.error("Byte extraction failed for mime_type={}", mime_type)
            raise

    @staticmethod
    async def batch_extract(
        file_paths: Sequence[Union[str, Path]],
        profile: Optional[ExtractionProfile] = None,
    ) -> List[ExtractionOutput]:
        """Extract content from multiple files in parallel.

        Args:
            file_paths: Sequence of document paths.
            profile:    Extraction profile.  Defaults to :attr:`BASIC`.

        Returns:
            List of :class:`ExtractionOutput`, one per input path.
        """
        from kreuzberg import batch_extract_files

        profile = profile or DocumentExtractor.BASIC
        config = DocumentExtractor._build_config(profile)

        try:
            results = await batch_extract_files(paths=list(file_paths), config=config)
            return [
                DocumentExtractor._convert_result(r, profile) for r in results
            ]
        except Exception:
            logger.error("Batch extraction failed for {} files", len(file_paths))
            raise

    # Internal helpers -----------------------------------------------------

    @staticmethod
    def _build_config(profile: ExtractionProfile):
        """Build a kreuzberg ``ExtractionConfig`` from an :class:`ExtractionProfile`.

        Maps profile fields to the kreuzberg configuration objects that are
        actually available in the installed version.
        """
        from kreuzberg import (
            ExtractionConfig,
            OcrConfig,
            OutputFormat,
            PageConfig,
            PdfConfig,
        )

        # --- Output format ---
        format_map = {
            "plain": OutputFormat.PLAIN,
            "markdown": OutputFormat.MARKDOWN,
            "html": OutputFormat.HTML,
            "djot": OutputFormat.DJOT,
        }
        output_format = format_map.get(profile.output_format, OutputFormat.PLAIN)

        # --- OCR config ---
        ocr_config: Optional[OcrConfig] = None
        if profile.ocr_enabled:
            ocr_config = OcrConfig(
                backend=profile.ocr_backend,
                language=profile.ocr_language,
            )

        # --- Page config ---
        page_config: Optional[PageConfig] = None
        if profile.page_start is not None or profile.page_end is not None:
            # kreuzberg PageConfig.extract_pages expects a list of page indices
            pages: Optional[list[int]] = None
            if profile.page_start is not None:
                end = profile.page_end if profile.page_end is not None else profile.page_start
                pages = list(range(profile.page_start, end + 1))
            page_config = PageConfig(extract_pages=pages)

        # --- PDF config ---
        pdf_config: Optional[PdfConfig] = None
        if any([
            profile.pdf_extract_images,
            profile.pdf_extract_metadata,
            profile.pdf_password,
        ]):
            passwords = [profile.pdf_password] if profile.pdf_password else None
            pdf_config = PdfConfig(
                extract_images=profile.pdf_extract_images,
                extract_metadata=profile.pdf_extract_metadata,
                passwords=passwords,
            )

        # --- Language detection ---
        lang_config = None
        if profile.detect_language:
            from kreuzberg import LanguageDetectionConfig
            lang_config = LanguageDetectionConfig(enabled=True)

        # --- Layout detection for table extraction ---
        layout_config = None
        if profile.extract_tables:
            try:
                from kreuzberg import LayoutDetectionConfig
                layout_config = LayoutDetectionConfig()
            except ImportError:
                # kreuzberg <= 4.2.x extracts tables by default;
                # filtering is handled in _convert_result().
                pass

        # --- Assemble ExtractionConfig ---
        kwargs: dict[str, Any] = {
            "output_format": output_format,
        }
        if ocr_config is not None:
            kwargs["ocr"] = ocr_config
        if profile.force_ocr:
            kwargs["force_ocr"] = True
        if page_config is not None:
            kwargs["pages"] = page_config
        if pdf_config is not None:
            kwargs["pdf_options"] = pdf_config
        if lang_config is not None:
            kwargs["language_detection"] = lang_config
        if profile.max_concurrent is not None:
            kwargs["max_concurrent_extractions"] = profile.max_concurrent
        if layout_config is not None:
            kwargs["layout"] = layout_config

        return ExtractionConfig(**kwargs)

    @staticmethod
    def _convert_result(
        result: "ExtractionResult",
        profile: ExtractionProfile,
    ) -> ExtractionOutput:
        """Convert a kreuzberg ``ExtractionResult`` to :class:`ExtractionOutput`.

        Only populates optional fields when the corresponding profile flag is
        enabled, keeping the output lean for basic extraction.
        """
        content: str = result.content or ""
        mime_type: str = getattr(result, "mime_type", "") or ""

        # Metadata
        metadata: dict[str, Any] = {}
        if profile.extract_metadata:
            raw_meta = getattr(result, "metadata", None)
            if raw_meta is not None:
                if isinstance(raw_meta, dict):
                    metadata = dict(raw_meta)
                else:
                    # kreuzberg may return a non-dict metadata object
                    try:
                        metadata = dict(raw_meta)
                    except (TypeError, ValueError):
                        metadata = {"raw": str(raw_meta)}

        # Tables
        tables: list[dict[str, Any]] = []
        if profile.extract_tables:
            raw_tables = getattr(result, "tables", None) or []
            for t in raw_tables:
                if isinstance(t, dict):
                    tables.append(t)
                else:
                    # kreuzberg ExtractedTable has: cells, markdown, page_number
                    tables.append({
                        "markdown": getattr(t, "markdown", ""),
                        "cells": getattr(t, "cells", []),
                        "page_number": getattr(t, "page_number", None),
                    })

        # Language detection
        detected_languages: dict[str, float] = {}
        if profile.detect_language:
            raw_langs = getattr(result, "detected_languages", None)
            if raw_langs:
                for entry in raw_langs:
                    if isinstance(entry, dict):
                        lang = entry.get("language", "")
                        conf = entry.get("confidence", 0.0)
                    else:
                        # kreuzberg DetectedLanguage object
                        lang = getattr(entry, "language", "")
                        conf = getattr(entry, "confidence", 0.0)
                    if lang:
                        detected_languages[lang] = float(conf)

        # Page count — prefer get_page_count() over get_chunk_count()
        page_count: Optional[int] = None
        get_page_count = getattr(result, "get_page_count", None)
        if get_page_count and callable(get_page_count):
            cnt = get_page_count()
            if cnt is not None and cnt > 0:
                page_count = cnt

        return ExtractionOutput(
            content=content,
            mime_type=mime_type,
            metadata=metadata,
            tables=tables,
            detected_languages=detected_languages,
            page_count=page_count,
        )
