# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import os
from pathlib import Path
from typing import Union

from kreuzberg import ExtractionResult, extract_file
from loguru import logger


def _infer_mime_for_path(file_path: Union[str, Path]) -> str | None:
    """Return MIME type for path-based detection, or None if unknown/extensionless.

    When the path has no extension (e.g. HotpotQA wiki_82, wiki_96), kreuzberg
    raises ValidationError. Callers should pass mime_type="text/plain" for
    those cases.
    """
    p = Path(file_path)
    suffix = (p.suffix or "").lower()
    # Map known extensions to MIME; extensionless or unknown -> None
    _MIME_MAP = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".jsonl": "application/jsonlines",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".html": "text/html",
        ".htm": "text/html",
        ".xml": "application/xml",
    }
    return _MIME_MAP.get(suffix)


async def fast_extract(file_path: Union[str, Path]) -> ExtractionResult:
    """
    Automatically detects and extracts text content from various file formats
    (docx, pptx, pdf, xlsx, etc.). For paths with no extension or unknown
    extension (e.g. HotpotQA wiki_82), treats content as text/plain to avoid
    ValidationError from MIME inference.
    """
    path = Path(file_path).resolve()
    mime_type: str | None = _infer_mime_for_path(path)

    if mime_type is None:
        # Extensionless or unknown: force text/plain so kreuzberg does not raise
        # "Could not determine MIME type from file path"
        mime_type = "text/plain"

    result: ExtractionResult = await extract_file(
        file_path=path,
        mime_type=mime_type,
    )
    return result


def get_fast_hash(file_path: Union[str, Path], sample_size: int = 8192):
    """
    Computes a partial hash (fingerprint) by combining:
    File Size + Head Chunk + Tail Chunk.
    This is extremely efficient for large-scale file hash calculation.
    """
    file_path = Path(file_path)
    try:
        # Get metadata first (O(1) operation)
        file_size = file_path.stat().st_size

        # If the file is smaller than the combined sample size, read it entirely
        if file_size <= sample_size * 2:
            with open(file_path, "rb") as f:
                return f"{hashlib.md5(f.read()).hexdigest()}_{file_size}"

        # Large file sampling: Read head and tail to avoid full disk I/O
        hash_content = hashlib.md5()
        with open(file_path, "rb") as f:
            hash_content.update(f.read(sample_size))
            f.seek(-sample_size, os.SEEK_END)
            hash_content.update(f.read(sample_size))

        # Mix the file size into the hash string to minimize collisions
        return f"{hash_content.hexdigest()}_{file_size}"
    except (FileNotFoundError, PermissionError):
        # Handle cases where files are deleted during scan or access is denied
        logger.warning("File not found or inaccessible: {}", file_path)
        return None


class StorageStructure:
    """
    Standardized directory and file naming conventions for caching and storage.
    """

    CACHE_DIR = ".cache"

    METADATA_DIR = "metadata"

    GREP_DIR = "rga"

    KNOWLEDGE_DIR = "knowledge"

    COGNITION_DIR = "cognition"

    # `.idx` -> Index file for fast lookup of cluster content
    CLUSTER_INDEX_FILE = "cluster.idx"

    # `.mpk` -> MessagePack serialized cluster content
    CLUSTER_CONTENT_FILE = "cluster.mpk"
