"""File storage service for managing uploaded files organized by collections.

Provides the FileStorageService class and associated Pydantic data models
for file metadata, collection info, and disk usage tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class FileMetadata(BaseModel):
    file_id: str
    name: str
    size_bytes: int
    sha256: str
    upload_time: str  # ISO 8601
    collection: str


class CollectionInfo(BaseModel):
    name: str
    file_count: int
    total_bytes: int
    path: str
    created_at: str  # ISO 8601


class DiskUsageInfo(BaseModel):
    used_bytes: int
    max_bytes: int
    used_pct: float


# ---------------------------------------------------------------------------
# FileStorageService
# ---------------------------------------------------------------------------


class FileStorageService:
    """Manages file uploads organized by collections.

    Storage layout::

        {upload_root}/
            {collection}/
                .manifest.json   <- metadata index
                file1.pdf
                file2.docx
    """

    _ALLOWED_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
        ".txt", ".md", ".csv", ".json", ".html", ".xml",
        ".rtf", ".epub", ".yaml", ".yml", ".log", ".tsv",
    }

    _COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

    _MANIFEST_FILE = ".manifest.json"

    def __init__(
        self,
        upload_root: Optional[Path] = None,
        max_file_size: Optional[int] = None,
        max_total_size: Optional[int] = None,
    ):
        # Read from env with defaults
        if upload_root is None:
            work_path = Path(
                os.getenv("SIRCHMUNK_WORK_PATH", os.path.expanduser("~/.sirchmunk"))
            ).expanduser().resolve()
            upload_root = work_path / "uploads"

        self._upload_root = Path(upload_root).resolve()

        # Values are in MB; convert to bytes
        self._max_file_size = (max_file_size or int(
            os.getenv("SIRCHMUNK_UPLOAD_MAX_FILE_SIZE", "1024")
        )) * 1024 * 1024  # Convert MB to bytes

        self._max_total_size = (max_total_size or int(
            os.getenv("SIRCHMUNK_UPLOAD_MAX_TOTAL", "10240")
        )) * 1024 * 1024  # Convert MB to bytes

        self._upload_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("SIRCHMUNK_UPLOAD_ENABLED", "true").lower() in (
            "true", "1", "yes",
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_collection_name(self, name: str) -> None:
        if not self._COLLECTION_PATTERN.match(name):
            raise ValueError(
                f"Invalid collection name '{name}': "
                "must match [a-zA-Z0-9][a-zA-Z0-9_-]{0,63}"
            )

    def validate_file(self, filename: str, size: int) -> None:
        ext = Path(filename).suffix.lower()
        if ext not in self._ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{ext}' not allowed. "
                f"Allowed: {sorted(self._ALLOWED_EXTENSIONS)}"
            )
        if size > self._max_file_size:
            raise ValueError(
                f"File size {size / (1024*1024):.1f} MB exceeds limit of "
                f"{self._max_file_size / (1024*1024):.0f} MB"
            )

    # ------------------------------------------------------------------
    # Quota
    # ------------------------------------------------------------------

    def _check_total_quota(self, additional_bytes: int) -> None:
        current = self._compute_total_usage()
        if current + additional_bytes > self._max_total_size:
            raise ValueError(
                f"Upload would exceed total quota: "
                f"{(current + additional_bytes) / (1024*1024):.1f} MB > {self._max_total_size / (1024*1024):.0f} MB"
            )

    def _compute_total_usage(self) -> int:
        total = 0
        if self._upload_root.exists():
            for f in self._upload_root.rglob("*"):
                if f.is_file() and f.name != self._MANIFEST_FILE:
                    total += f.stat().st_size
        return total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection_dir(self, collection: str) -> Path:
        # Path traversal prevention
        safe = Path(collection).name
        return self._upload_root / safe

    def _read_manifest(self, collection: str) -> List[dict]:
        manifest_path = self._collection_dir(collection) / self._MANIFEST_FILE
        if not manifest_path.exists():
            return []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _write_manifest(self, collection: str, entries: List[dict]) -> None:
        manifest_path = self._collection_dir(collection) / self._MANIFEST_FILE
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    async def save_file(
        self,
        collection: str,
        filename: str,
        content: bytes,
    ) -> FileMetadata:
        """Save an uploaded file into *collection*, returning its metadata."""
        self.validate_collection_name(collection)
        self.validate_file(filename, len(content))
        self._check_total_quota(len(content))

        col_dir = self._collection_dir(collection)
        col_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate filename: if exists, append _1, _2, etc.
        stem = Path(filename).stem
        ext = Path(filename).suffix
        target = col_dir / filename
        counter = 1
        while target.exists():
            target = col_dir / f"{stem}_{counter}{ext}"
            counter += 1

        # Write file
        target.write_bytes(content)

        # Compute hash
        sha256 = hashlib.sha256(content).hexdigest()

        # Create metadata
        file_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        meta = FileMetadata(
            file_id=file_id,
            name=target.name,
            size_bytes=len(content),
            sha256=sha256,
            upload_time=now,
            collection=collection,
        )

        # Update manifest
        entries = self._read_manifest(collection)
        entries.append(meta.model_dump())
        self._write_manifest(collection, entries)

        return meta

    def list_collections(self) -> List[CollectionInfo]:
        """Return information about every collection under the upload root."""
        collections: List[CollectionInfo] = []
        if not self._upload_root.exists():
            return collections
        for d in sorted(self._upload_root.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                entries = self._read_manifest(d.name)
                total_bytes = sum(e.get("size_bytes", 0) for e in entries)
                # Use dir creation time as created_at
                try:
                    created_at = datetime.fromtimestamp(
                        d.stat().st_ctime, tz=timezone.utc
                    ).isoformat()
                except OSError:
                    created_at = ""
                collections.append(CollectionInfo(
                    name=d.name,
                    file_count=len(entries),
                    total_bytes=total_bytes,
                    path=str(d),
                    created_at=created_at,
                ))
        return collections

    def list_files(self, collection: str) -> List[FileMetadata]:
        """List every file in *collection*."""
        self.validate_collection_name(collection)
        col_dir = self._collection_dir(collection)
        if not col_dir.exists():
            raise FileNotFoundError(f"Collection '{collection}' not found")
        entries = self._read_manifest(collection)
        return [FileMetadata(**e) for e in entries]

    def get_collection_path(self, collection: str) -> Path:
        """Return the resolved filesystem path for *collection*."""
        self.validate_collection_name(collection)
        col_dir = self._collection_dir(collection)
        if not col_dir.exists():
            raise FileNotFoundError(f"Collection '{collection}' not found")
        return col_dir

    def delete_file(self, file_id: str) -> Tuple[str, str]:
        """Delete a file by *file_id* across all collections.

        Returns ``(collection_name, filename)`` on success.
        """
        for d in self._upload_root.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            entries = self._read_manifest(d.name)
            for i, e in enumerate(entries):
                if e.get("file_id") == file_id:
                    # Remove physical file
                    file_path = d / e["name"]
                    if file_path.exists():
                        file_path.unlink()
                    # Remove from manifest
                    entries.pop(i)
                    self._write_manifest(d.name, entries)
                    return d.name, e["name"]
        raise FileNotFoundError(f"File with id '{file_id}' not found")

    def delete_collection(self, collection: str) -> int:
        """Delete an entire collection. Returns the number of files deleted."""
        self.validate_collection_name(collection)
        col_dir = self._collection_dir(collection)
        if not col_dir.exists():
            raise FileNotFoundError(f"Collection '{collection}' not found")
        entries = self._read_manifest(collection)
        count = len(entries)
        shutil.rmtree(col_dir)
        return count

    def get_disk_usage(self) -> DiskUsageInfo:
        """Return current disk usage statistics."""
        used = self._compute_total_usage()
        return DiskUsageInfo(
            used_bytes=used,
            max_bytes=self._max_total_size,
            used_pct=(
                round(used / self._max_total_size * 100, 2)
                if self._max_total_size > 0
                else 0.0
            ),
        )
