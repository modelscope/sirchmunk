# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge lint — health checks and auto-fixes for the knowledge network.

Inspired by LLM Wiki's Lint operation: validates cluster integrity,
detects stale evidence, and cleans orphaned tree indices.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from sirchmunk.schema.knowledge import KnowledgeCluster, Lifecycle
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils import LogCallback, create_logger


@dataclass
class LintIssue:
    """A single lint finding."""

    severity: str  # "error", "warning", "info"
    category: str  # "stale_evidence", "orphan_tree", "empty_cluster", etc.
    message: str
    cluster_id: Optional[str] = None
    file_path: Optional[str] = None
    auto_fixed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "cluster_id": self.cluster_id,
            "file_path": self.file_path,
            "auto_fixed": self.auto_fixed,
        }


@dataclass
class LintReport:
    """Summary of a lint run."""

    total_clusters_checked: int = 0
    total_trees_checked: int = 0
    issues: List[LintIssue] = field(default_factory=list)
    auto_fixes_applied: int = 0

    @property
    def errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_clusters_checked": self.total_clusters_checked,
            "total_trees_checked": self.total_trees_checked,
            "errors": self.errors,
            "warnings": self.warnings,
            "auto_fixes_applied": self.auto_fixes_applied,
            "issues": [i.to_dict() for i in self.issues],
        }


class KnowledgeLint:
    """Validate the health of the knowledge network and apply auto-fixes."""

    def __init__(
        self,
        knowledge_storage: KnowledgeStorage,
        work_path: Union[str, Path],
        log_callback: LogCallback = None,
    ):
        self._storage = knowledge_storage
        self._work_path = Path(work_path).expanduser().resolve()
        self._tree_dir = self._work_path / ".cache" / "compile" / "trees"
        self._manifest_path = self._work_path / ".cache" / "compile" / "manifest.json"
        self._log = create_logger(log_callback=log_callback)

    async def run(self, *, auto_fix: bool = False) -> LintReport:
        """Execute all lint checks and optionally apply auto-fixes."""
        report = LintReport()

        await self._log.info("[Lint] Starting knowledge health check")

        # Check clusters
        await self._check_clusters(report, auto_fix=auto_fix)

        # Check orphaned tree caches
        await self._check_orphan_trees(report, auto_fix=auto_fix)

        # Check manifest consistency
        await self._check_manifest(report)

        await self._log.info(
            f"[Lint] Done — clusters={report.total_clusters_checked}, "
            f"trees={report.total_trees_checked}, "
            f"errors={report.errors}, warnings={report.warnings}, "
            f"fixes={report.auto_fixes_applied}"
        )
        return report

    async def _check_clusters(self, report: LintReport, auto_fix: bool) -> None:
        """Validate each knowledge cluster."""
        all_clusters = await self._storage.find("", limit=10000)
        report.total_clusters_checked = len(all_clusters)

        for cluster in all_clusters:
            # Check: empty content
            if not cluster.content or (
                isinstance(cluster.content, str) and len(cluster.content.strip()) < 10
            ):
                report.issues.append(LintIssue(
                    severity="warning",
                    category="empty_cluster",
                    message=f"Cluster has empty or minimal content",
                    cluster_id=cluster.id,
                ))

            # Check: stale evidence (source files no longer exist)
            stale_count = 0
            for ev in cluster.evidences:
                fp = str(ev.file_or_url)
                if fp.startswith("/") and not Path(fp).exists():
                    stale_count += 1

            if stale_count > 0:
                report.issues.append(LintIssue(
                    severity="warning",
                    category="stale_evidence",
                    message=f"{stale_count} evidence source(s) no longer exist",
                    cluster_id=cluster.id,
                ))

                if auto_fix and stale_count == len(cluster.evidences):
                    cluster.lifecycle = Lifecycle.DEPRECATED
                    await self._storage.update(cluster)
                    report.auto_fixes_applied += 1
                    report.issues[-1].auto_fixed = True

            # Check: no queries and no evidences (orphan cluster)
            if not cluster.evidences and not cluster.queries:
                report.issues.append(LintIssue(
                    severity="info",
                    category="orphan_cluster",
                    message="Cluster has no evidence and no queries",
                    cluster_id=cluster.id,
                ))

            # Check: isolated cluster (no WeakSemanticEdge connections)
            if not cluster.related_clusters and cluster.evidences:
                report.issues.append(LintIssue(
                    severity="info",
                    category="isolated_cluster",
                    message="Cluster has no cross-references to other clusters",
                    cluster_id=cluster.id,
                ))

    async def _check_orphan_trees(self, report: LintReport, auto_fix: bool) -> None:
        """Find tree cache files whose source documents no longer exist."""
        if not self._tree_dir.exists():
            return

        manifest = self._load_manifest()
        # Build set of valid file hashes from the manifest
        valid_hashes: Set[str] = set()
        for entry_data in manifest.get("files", {}).values():
            fh = entry_data.get("file_hash", "")
            if fh:
                valid_hashes.add(fh)

        tree_files = list(self._tree_dir.glob("*.json"))
        report.total_trees_checked = len(tree_files)

        for tf in tree_files:
            tree_hash = tf.stem
            if tree_hash not in valid_hashes:
                report.issues.append(LintIssue(
                    severity="info",
                    category="orphan_tree",
                    message=f"Tree cache has no matching manifest entry",
                    file_path=str(tf),
                ))
                if auto_fix:
                    tf.unlink(missing_ok=True)
                    report.auto_fixes_applied += 1
                    report.issues[-1].auto_fixed = True

    async def _check_manifest(self, report: LintReport) -> None:
        """Validate manifest references."""
        manifest = self._load_manifest()
        files = manifest.get("files", {})

        for fp, entry_data in files.items():
            if not Path(fp).exists():
                report.issues.append(LintIssue(
                    severity="warning",
                    category="stale_manifest",
                    message=f"Manifest references non-existent file",
                    file_path=fp,
                ))

    def _load_manifest(self) -> Dict[str, Any]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
