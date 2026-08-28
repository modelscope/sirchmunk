# Copyright (c) ModelScope Contributors. All rights reserved.
"""Deterministic, read-only filesystem discovery tools for agentic search."""

from __future__ import annotations

import fnmatch
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sirchmunk.agentic.tools import BaseTool
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.utils.constants import (
    DIRECTORY_PROFILE_MAX_DEPTH,
    DIRECTORY_PROFILE_MAX_ENTRIES,
    FILE_LIST_PAGE_SIZE,
)


@dataclass(frozen=True)
class FileRecord:
    """Lightweight metadata for one file without reading its content."""

    path: str
    relative_path: str
    name: str
    extension: str
    size_bytes: int
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "name": self.name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "depth": self.depth,
        }


@dataclass
class DirectoryProfile:
    """Bounded snapshot of files and directories under approved roots."""

    roots: List[str]
    files: List[FileRecord] = field(default_factory=list)
    directory_count: int = 0
    inaccessible_count: int = 0
    truncated: bool = False

    @property
    def total_size_bytes(self) -> int:
        return sum(record.size_bytes for record in self.files)

    @property
    def extension_counts(self) -> Dict[str, int]:
        counts = Counter(record.extension or "<extensionless>" for record in self.files)
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def directory_file_counts(self) -> Dict[str, int]:
        counts = Counter(
            "<root>"
            if str(Path(record.relative_path).parent) == "."
            else Path(record.relative_path).parent.as_posix()
            for record in self.files
        )
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def to_dict(self, *, top_directories: int = 20) -> Dict[str, Any]:
        return {
            "roots": self.roots,
            "file_count": len(self.files),
            "directory_count": self.directory_count,
            "total_size_bytes": self.total_size_bytes,
            "extension_counts": self.extension_counts,
            "top_directories": dict(
                list(self.directory_file_counts.items())[:top_directories]
            ),
            "inaccessible_count": self.inaccessible_count,
            "truncated": self.truncated,
        }


class DirectoryProfiler:
    """Walk approved roots deterministically without following symbolic links."""

    DEFAULT_EXCLUDE = {
        ".git",
        ".idea",
        ".next",
        ".cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
        "Thumbs.db",
        ".DS_Store",
    }

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        max_entries: int = DIRECTORY_PROFILE_MAX_ENTRIES,
        default_max_depth: int = DIRECTORY_PROFILE_MAX_DEPTH,
    ) -> None:
        resolved = [Path(root).expanduser().resolve() for root in roots]
        self.roots = [root for root in resolved if root.exists()]
        self.max_entries = max(1, max_entries)
        self.default_max_depth = max(0, default_max_depth)

    @staticmethod
    def _matches_patterns(path: Path, patterns: Iterable[str]) -> bool:
        value = path.as_posix()
        return any(
            path.match(pattern)
            or fnmatch.fnmatch(path.name, pattern)
            or fnmatch.fnmatch(value, pattern)
            for pattern in patterns
        )

    def _is_allowed(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.roots)

    def resolve_target(self, value: Optional[str | Path]) -> Path:
        """Resolve one user-supplied path while enforcing configured roots."""
        if value is None or not str(value).strip():
            if len(self.roots) != 1:
                raise ValueError("path is required when multiple search roots are configured")
            return self.roots[0]

        raw = Path(value).expanduser()
        candidates = [raw] if raw.is_absolute() else [root / raw for root in self.roots]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists() and self._is_allowed(resolved):
                return resolved
        raise ValueError("path must exist within the configured search roots")

    def profile(
        self,
        path: Optional[str | Path] = None,
        *,
        max_depth: Optional[int] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> DirectoryProfile:
        """Create a bounded metadata-only directory snapshot."""
        target = self.resolve_target(path)
        depth_limit = self.default_max_depth if max_depth is None else max(0, max_depth)
        excluded = set(self.DEFAULT_EXCLUDE)
        excluded.update(exclude or [])
        included = tuple(include or ())
        profile = DirectoryProfile(roots=[str(target)])

        if target.is_file():
            stat = target.stat()
            profile.files.append(
                FileRecord(
                    path=str(target),
                    relative_path=target.name,
                    name=target.name,
                    extension=target.suffix.lower(),
                    size_bytes=stat.st_size,
                    depth=0,
                )
            )
            return profile

        queue = deque([(target, 0)])
        while queue:
            directory, depth = queue.popleft()
            profile.directory_count += 1
            try:
                entries = sorted(directory.iterdir(), key=lambda item: item.name.lower())
            except (OSError, PermissionError):
                profile.inaccessible_count += 1
                continue

            for entry in entries:
                if entry.name.startswith(".") or self._matches_patterns(entry, excluded):
                    continue
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        if depth < depth_limit:
                            queue.append((entry, depth + 1))
                        continue
                    if not entry.is_file():
                        continue
                    relative = entry.relative_to(target)
                    if included and not self._matches_patterns(relative, included):
                        continue
                    stat = entry.stat()
                    profile.files.append(
                        FileRecord(
                            path=str(entry.resolve()),
                            relative_path=relative.as_posix(),
                            name=entry.name,
                            extension=entry.suffix.lower(),
                            size_bytes=stat.st_size,
                            depth=depth + 1,
                        )
                    )
                    if len(profile.files) >= self.max_entries:
                        profile.truncated = True
                        return profile
                except (OSError, PermissionError):
                    profile.inaccessible_count += 1
        return profile

    def profile_all(
        self,
        *,
        max_depth: Optional[int] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
    ) -> DirectoryProfile:
        """Create one combined profile across every configured search root."""
        combined = DirectoryProfile(roots=[str(root) for root in self.roots])
        for root in self.roots:
            profile = self.profile(
                root,
                max_depth=max_depth,
                include=include,
                exclude=exclude,
            )
            combined.files.extend(profile.files)
            combined.directory_count += profile.directory_count
            combined.inaccessible_count += profile.inaccessible_count
            combined.truncated = combined.truncated or profile.truncated
            if len(combined.files) >= self.max_entries:
                combined.files = combined.files[: self.max_entries]
                combined.truncated = True
                break
        return combined

    @staticmethod
    def render_tree(profile: DirectoryProfile, *, max_nodes: int = 200) -> List[str]:
        """Render a compact deterministic tree from a directory profile."""
        paths = sorted(record.relative_path for record in profile.files)
        counts: Counter[str] = Counter()
        for value in paths:
            parts = Path(value).parts
            for index in range(1, len(parts)):
                counts[Path(*parts[:index]).as_posix()] += 1

        lines: List[str] = []
        for directory, count in sorted(
            counts.items(), key=lambda item: (len(Path(item[0]).parts), item[0])
        ):
            depth = len(Path(directory).parts) - 1
            lines.append(f"{'  ' * depth}{Path(directory).name}/ ({count} files)")
            if len(lines) >= max_nodes:
                break
        return lines


class FileListTool(BaseTool):
    """Expose safe, paginated filesystem discovery to the ReAct agent."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        max_entries: int = DIRECTORY_PROFILE_MAX_ENTRIES,
        default_page_size: int = FILE_LIST_PAGE_SIZE,
    ) -> None:
        self._profiler = DirectoryProfiler(roots, max_entries=max_entries)
        self._default_page_size = max(1, default_page_size)

    @property
    def name(self) -> str:
        return "file_list"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Inspect approved search roots without reading file contents. Use "
                "view=profile or tree before guessing paths in an unfamiliar directory; "
                "use view=files for a paginated file list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory under the configured search roots.",
                    },
                    "view": {
                        "type": "string",
                        "enum": ["profile", "tree", "files"],
                        "default": "profile",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional case-insensitive filename/path filter.",
                    },
                    "offset": {"type": "integer", "default": 0, "minimum": 0},
                    "limit": {
                        "type": "integer",
                        "default": self._default_page_size,
                        "minimum": 1,
                        "maximum": 500,
                    },
                    "max_depth": {
                        "type": "integer",
                        "default": DIRECTORY_PROFILE_MAX_DEPTH,
                        "minimum": 0,
                        "maximum": 32,
                    },
                },
            },
        }

    async def execute(
        self,
        context: SearchContext,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        view = str(kwargs.get("view", "profile") or "profile").lower()
        if view not in {"profile", "tree", "files"}:
            return "view must be profile, tree, or files.", {"error": "invalid_view"}
        try:
            profile = (
                self._profiler.profile_all(
                    max_depth=int(
                        kwargs.get("max_depth", DIRECTORY_PROFILE_MAX_DEPTH)
                    )
                )
                if not kwargs.get("path") and len(self._profiler.roots) > 1
                else self._profiler.profile(
                    kwargs.get("path"),
                    max_depth=int(
                        kwargs.get("max_depth", DIRECTORY_PROFILE_MAX_DEPTH)
                    ),
                )
            )
        except (OSError, ValueError) as exc:
            return f"File listing failed: {exc}", {"error": str(exc)}

        query = str(kwargs.get("query", "") or "").strip().lower()
        records = profile.files
        if query:
            records = [
                record
                for record in records
                if query in record.relative_path.lower() or query in record.name.lower()
            ]

        offset = max(0, int(kwargs.get("offset", 0) or 0))
        limit = min(500, max(1, int(kwargs.get("limit", self._default_page_size) or 1)))
        page = records[offset : offset + limit]
        has_more = offset + limit < len(records)
        payload: Dict[str, Any] = profile.to_dict()
        payload.update(
            {
                "view": view,
                "matched_count": len(records),
                "offset": offset,
                "limit": limit,
                "has_more": has_more,
            }
        )

        if view == "files":
            payload["files"] = [record.to_dict() for record in page]
        elif view == "tree":
            payload["tree"] = self._profiler.render_tree(profile)

        context.add_log(
            tool_name=self.name,
            metadata={
                "view": view,
                "roots": profile.roots,
                "files_listed": len(page) if view == "files" else 0,
                "files_profiled": len(profile.files),
                "directories_profiled": profile.directory_count,
                "truncated": profile.truncated,
            },
        )
        return json.dumps(payload, ensure_ascii=False, indent=2), payload
