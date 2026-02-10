# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Directory scan tool for the ReAct agent.

Wraps ``DirectoryScanner`` as a ``BaseTool`` so the LLM can discover
files in unknown directories during a ReAct search loop.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from sirchmunk.agentic.tools import BaseTool
from sirchmunk.scan.dir_scanner import DirectoryScanner, ScanResult
from sirchmunk.schema.search_context import SearchContext

logger = logging.getLogger(__name__)


class DirScanTool(BaseTool):
    """Scan directories to discover document candidates.

    Performs a fast recursive scan and optionally uses LLM to rank
    discovered files by relevance.  Low-to-medium token cost depending
    on the number of files discovered.
    """

    def __init__(
        self,
        scanner: DirectoryScanner,
        paths: Union[str, List[str]],
    ) -> None:
        self._scanner = scanner
        self._paths = paths if isinstance(paths, list) else [paths]
        self._cached_result: Optional[ScanResult] = None

    @property
    def name(self) -> str:
        return "dir_scan"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": (
                "Scan directories to discover available documents. Returns file "
                "names, types, sizes, titles, and keyword previews. Use this "
                "when you don't know what files exist in the search paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to rank files by relevance.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of candidates to rank (default: 20).",
                        "default": 20,
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
        top_k: int = kwargs.get("top_k", 20)

        if not query:
            return "No query provided for directory scan.", {}

        try:
            # Cache scan results (filesystem walk is expensive)
            if self._cached_result is None:
                self._cached_result = await self._scanner.scan(self._paths)

            # Rank with LLM
            result = await self._scanner.rank(
                query=query,
                scan_result=self._cached_result,
                top_k=top_k,
            )

            if not result.ranked_candidates:
                return "No files found in the specified directories.", {
                    "total_files": result.total_files,
                }

            # Format output — include full content for high-relevance small files
            lines: List[str] = [
                f"Scanned {result.total_files} files in {result.total_dirs} directories.\n"
            ]

            for c in result.ranked_candidates:
                tag = f"[{c.relevance or '?'}]" if c.relevance else ""
                meta_parts = [
                    f"{tag} {c.path}",
                    f"  Type: {c.extension} | Size: {c._human_size()}",
                    f"  Title: {c.title or '(none)'}",
                ]
                if c.author:
                    meta_parts.append(f"  Author: {c.author}")
                if c.page_count > 0:
                    meta_parts.append(f"  Pages: {c.page_count}")
                if c.keywords:
                    meta_parts.append(f"  Keywords: {', '.join(c.keywords[:5])}")
                meta_parts.append(f"  Reason: {c.reason or 'N/A'}")

                # For high-relevance files with loaded content, include it
                if c.relevance == "high" and c.content_loaded and c.full_content:
                    content_preview = c.full_content[:3000]
                    if len(c.full_content) > 3000:
                        content_preview += "\n... [truncated]"
                    meta_parts.append(f"  --- Content ---\n{content_preview}")

                lines.append("\n".join(meta_parts))

            result_text = "\n\n".join(lines)

            approx_tokens = len(result_text) // 4
            context.add_log(
                tool_name=self.name,
                tokens=approx_tokens,
                metadata={
                    "total_files": result.total_files,
                    "ranked_count": len(result.ranked_candidates),
                    "high_count": len(result.high_relevance),
                },
            )

            return result_text, {
                "total_files": result.total_files,
                "ranked_count": len(result.ranked_candidates),
                "high_relevance_paths": [c.path for c in result.high_relevance],
            }

        except Exception as exc:
            error_msg = f"Directory scan failed: {exc}"
            logger.error(error_msg)
            return error_msg, {"error": str(exc)}
