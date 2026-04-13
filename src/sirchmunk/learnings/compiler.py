# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Knowledge compiler — orchestrates offline compile of document collections.

Fuses PageIndex (tree indexing) and LLM Wiki (knowledge compilation network)
into a single compile pipeline that produces structured tree indices and
knowledge clusters for downstream search acceleration.
"""

import asyncio
import json
import math
import os
import random
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sirchmunk.learnings.tree_indexer import (
    DocumentTree,
    DocumentTreeIndexer,
)
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.schema.knowledge import (
    AbstractionLevel,
    EvidenceUnit,
    KnowledgeCluster,
    Lifecycle,
    WeakSemanticEdge,
)
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils import LogCallback, create_logger
from sirchmunk.utils.file_utils import fast_extract, get_fast_hash

# Concurrency cap for LLM-heavy file processing
_DEFAULT_CONCURRENCY = 3

# Similarity threshold for merging into existing clusters during compile
_MERGE_SIMILARITY_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileManifestEntry:
    """State of a single file in the compile manifest."""

    file_hash: str
    compiled_at: str
    has_tree: bool
    cluster_ids: List[str]
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_hash": self.file_hash,
            "compiled_at": self.compiled_at,
            "has_tree": self.has_tree,
            "cluster_ids": self.cluster_ids,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileManifestEntry":
        return cls(
            file_hash=data["file_hash"],
            compiled_at=data["compiled_at"],
            has_tree=data.get("has_tree", False),
            cluster_ids=data.get("cluster_ids", []),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class CompileManifest:
    """Tracks compiled file states for incremental processing."""

    version: str = "1.0"
    last_compile_at: Optional[str] = None
    files: Dict[str, FileManifestEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "last_compile_at": self.last_compile_at,
            "files": {k: v.to_dict() for k, v in self.files.items()},
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CompileManifest":
        data = json.loads(json_str)
        files = {
            k: FileManifestEntry.from_dict(v)
            for k, v in data.get("files", {}).items()
        }
        return cls(
            version=data.get("version", "1.0"),
            last_compile_at=data.get("last_compile_at"),
            files=files,
        )


@dataclass
class FileEntry:
    """Discovered file pending compilation."""

    path: str
    size_bytes: int
    file_hash: str


@dataclass
class ChangeSet:
    """Delta between discovered files and the manifest."""

    added: List[FileEntry] = field(default_factory=list)
    modified: List[FileEntry] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)


@dataclass
class FileCompileResult:
    """Result of compiling a single file."""

    path: str
    tree: Optional[DocumentTree] = None
    summary: str = ""
    topics: List[str] = field(default_factory=list)
    evidence: Optional[EvidenceUnit] = None
    cluster_ids: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class CompileReport:
    """Summary report of a compile run."""

    total_files: int = 0
    files_added: int = 0
    files_modified: int = 0
    files_skipped: int = 0
    files_deleted: int = 0
    files_sampled: int = 0
    trees_built: int = 0
    clusters_created: int = 0
    clusters_merged: int = 0
    cross_refs_built: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_skipped": self.files_skipped,
            "files_deleted": self.files_deleted,
            "files_sampled": self.files_sampled,
            "trees_built": self.trees_built,
            "clusters_created": self.clusters_created,
            "clusters_merged": self.clusters_merged,
            "cross_refs_built": self.cross_refs_built,
            "errors": self.errors,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class CompileStatus:
    """Status snapshot of the compile state."""

    total_compiled_files: int = 0
    total_clusters: int = 0
    total_trees: int = 0
    last_compile_at: Optional[str] = None
    manifest_path: str = ""


# ---------------------------------------------------------------------------
# Importance probability sampler
# ---------------------------------------------------------------------------

class ImportanceSampler:
    """Select a representative subset of files using importance-based probability.

    Sampling strategy for large datasets:
    - Larger files get higher probability (they contain more information).
    - Uncompiled (new) files are prioritised over previously compiled ones.
    - Files with rare extensions get a mild boost (diversity signal).
    - The final probability is proportional to a composite importance score.
    """

    def __init__(self, max_files: int, seed: Optional[int] = None):
        self._max_files = max_files
        self._rng = random.Random(seed)

    def sample(self, files: List[FileEntry], manifest: CompileManifest) -> List[FileEntry]:
        """Return up to *max_files* entries sampled by importance."""
        if len(files) <= self._max_files:
            return files

        scores = [self._score(f, manifest) for f in files]
        total = sum(scores) or 1.0
        probs = [s / total for s in scores]

        selected_indices = set()
        attempts = 0
        while len(selected_indices) < self._max_files and attempts < len(files) * 3:
            idx = self._weighted_choice(probs)
            selected_indices.add(idx)
            attempts += 1

        return [files[i] for i in sorted(selected_indices)]

    def _score(self, entry: FileEntry, manifest: CompileManifest) -> float:
        """Compute composite importance score."""
        # Size factor: log-scaled, bounded
        size_score = math.log2(max(entry.size_bytes, 1024)) / 20.0

        # Novelty factor: new files are more important
        novelty = 2.0 if entry.path not in manifest.files else 0.5

        # Extension diversity: rare extensions get a mild boost
        ext = Path(entry.path).suffix.lower()
        diversity = 1.5 if ext in {".pdf", ".docx", ".doc", ".tex"} else 1.0

        return size_score * novelty * diversity

    def _weighted_choice(self, probs: List[float]) -> int:
        r = self._rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return len(probs) - 1


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

class KnowledgeCompiler:
    """Orchestrate compile pipeline: file discovery -> tree indexing -> knowledge aggregation."""

    # File extensions eligible for compilation
    _ELIGIBLE_EXTENSIONS = {
        ".pdf", ".docx", ".doc", ".md", ".markdown", ".html", ".htm",
        ".rst", ".tex", ".txt", ".pptx", ".xlsx",
    }

    def __init__(
        self,
        llm: OpenAIChat,
        embedding_client: Optional[Any],
        knowledge_storage: KnowledgeStorage,
        tree_indexer: DocumentTreeIndexer,
        work_path: Union[str, Path],
        log_callback: LogCallback = None,
    ):
        self._llm = llm
        self._embedding = embedding_client
        self._storage = knowledge_storage
        self._tree_indexer = tree_indexer
        self._work_path = Path(work_path).expanduser().resolve()
        self._log = create_logger(log_callback=log_callback)

        self._compile_dir = self._work_path / ".cache" / "compile"
        self._compile_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._compile_dir / "manifest.json"

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    async def compile(
        self,
        paths: List[str],
        *,
        incremental: bool = True,
        shallow: bool = False,
        max_files: Optional[int] = None,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> CompileReport:
        """Execute the unified knowledge compile pipeline.

        Args:
            paths: Directories or files to compile.
            incremental: Skip unchanged files.
            shallow: Skip tree building even for eligible files — use direct
                     LLM summarisation only (faster, lower quality).
            max_files: Cap on files to process (triggers importance sampling).
            concurrency: Max parallel file compilations.
        """
        import time
        t0 = time.monotonic()
        report = CompileReport()

        # Phase 1: discover and diff
        await self._log.info("[Compile] Phase 1: File discovery & change detection")
        manifest = self._load_manifest()
        discovered = await self._discover_files(paths)
        report.total_files = len(discovered)
        await self._log.info(f"[Compile] Discovered {len(discovered)} eligible files")

        if incremental:
            changes = self._detect_changes(discovered, manifest)
            to_compile = changes.added + changes.modified
            report.files_skipped = len(changes.unchanged)
            report.files_deleted = len(changes.deleted)
            for deleted_path in changes.deleted:
                manifest.files.pop(deleted_path, None)
        else:
            to_compile = discovered
            report.files_skipped = 0

        report.files_added = len([f for f in to_compile if f.path not in manifest.files])
        report.files_modified = len(to_compile) - report.files_added

        # Phase 1.5: importance sampling for large datasets
        if max_files and len(to_compile) > max_files:
            await self._log.info(
                f"[Compile] Applying importance sampling: {len(to_compile)} -> {max_files} files"
            )
            sampler = ImportanceSampler(max_files=max_files)
            to_compile = sampler.sample(to_compile, manifest)
            report.files_sampled = len(to_compile)

        if not to_compile:
            await self._log.info("[Compile] No files to compile (all up-to-date)")
            report.elapsed_seconds = time.monotonic() - t0
            return report

        await self._log.info(
            f"[Compile] Phase 2: Processing {len(to_compile)} files "
            f"(concurrency={concurrency})"
        )

        # Phase 2: compile files with bounded concurrency
        semaphore = asyncio.Semaphore(concurrency)
        results: List[FileCompileResult] = []

        async def _bounded(entry: FileEntry) -> FileCompileResult:
            async with semaphore:
                return await self._compile_single_file(entry, shallow=shallow)

        tasks = [_bounded(f) for f in to_compile]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if result.error:
                report.errors.append(f"{result.path}: {result.error}")
            else:
                if result.tree:
                    report.trees_built += 1
                # Update manifest
                manifest.files[result.path] = FileManifestEntry(
                    file_hash=get_fast_hash(result.path) or "",
                    compiled_at=datetime.now(timezone.utc).isoformat(),
                    has_tree=result.tree is not None,
                    cluster_ids=result.cluster_ids,
                    size_bytes=Path(result.path).stat().st_size if Path(result.path).exists() else 0,
                )

        # Phase 3: aggregate results into knowledge network
        await self._log.info("[Compile] Phase 3: Knowledge aggregation")
        for r in results:
            if r.error or not r.summary:
                continue
            created, merged = await self._aggregate_to_knowledge_network(r)
            report.clusters_created += created
            report.clusters_merged += merged

        # Phase 4: cross-references
        await self._log.info("[Compile] Phase 4: Building cross-references")
        report.cross_refs_built = await self._build_cross_references(results)

        # Phase 5: persist manifest
        manifest.last_compile_at = datetime.now(timezone.utc).isoformat()
        self._save_manifest(manifest)
        self._storage.force_sync()

        report.elapsed_seconds = time.monotonic() - t0
        await self._log.info(
            f"[Compile] Done in {report.elapsed_seconds:.1f}s — "
            f"trees={report.trees_built}, created={report.clusters_created}, "
            f"merged={report.clusters_merged}, errors={len(report.errors)}"
        )
        return report

    async def get_status(self, paths: List[str]) -> CompileStatus:
        """Return current compile status for the given paths."""
        manifest = self._load_manifest()
        path_set = {str(Path(p).resolve()) for p in paths}

        compiled_count = 0
        tree_count = 0
        cluster_ids: Set[str] = set()
        for fp, entry in manifest.files.items():
            for p in path_set:
                if fp.startswith(p):
                    compiled_count += 1
                    if entry.has_tree:
                        tree_count += 1
                    cluster_ids.update(entry.cluster_ids)
                    break

        return CompileStatus(
            total_compiled_files=compiled_count,
            total_clusters=len(cluster_ids),
            total_trees=tree_count,
            last_compile_at=manifest.last_compile_at,
            manifest_path=str(self._manifest_path),
        )

    # ------------------------------------------------------------------ #
    #  File discovery and change detection                                #
    # ------------------------------------------------------------------ #

    async def _discover_files(self, paths: List[str]) -> List[FileEntry]:
        """Walk paths and return all compilation-eligible files."""
        entries: List[FileEntry] = []
        seen: Set[str] = set()

        for base in paths:
            base_path = Path(base).expanduser().resolve()
            if base_path.is_file():
                candidates = [base_path]
            elif base_path.is_dir():
                candidates = sorted(base_path.rglob("*"))
            else:
                continue

            for fp in candidates:
                if not fp.is_file():
                    continue
                if fp.suffix.lower() not in self._ELIGIBLE_EXTENSIONS:
                    continue
                abs_path = str(fp.resolve())
                if abs_path in seen:
                    continue
                seen.add(abs_path)
                fh = get_fast_hash(abs_path)
                if fh is None:
                    continue
                entries.append(FileEntry(
                    path=abs_path,
                    size_bytes=fp.stat().st_size,
                    file_hash=fh,
                ))

        return entries

    def _detect_changes(
        self, discovered: List[FileEntry], manifest: CompileManifest,
    ) -> ChangeSet:
        """Compare discovered files against the manifest for incremental compile."""
        changes = ChangeSet()
        current_paths = {f.path for f in discovered}

        for entry in discovered:
            prev = manifest.files.get(entry.path)
            if prev is None:
                changes.added.append(entry)
            elif prev.file_hash != entry.file_hash:
                changes.modified.append(entry)
            else:
                changes.unchanged.append(entry.path)

        for old_path in manifest.files:
            if old_path not in current_paths:
                changes.deleted.append(old_path)

        return changes

    # ------------------------------------------------------------------ #
    #  Single-file compilation                                            #
    # ------------------------------------------------------------------ #

    async def _compile_single_file(
        self,
        entry: FileEntry,
        *,
        shallow: bool = False,
    ) -> FileCompileResult:
        """Unified compile pipeline: tree-if-eligible -> summary -> topics -> evidence.

        When *shallow* is True (or file is ineligible for tree indexing),
        the pipeline skips tree building and summarises via a direct LLM call.
        """
        result = FileCompileResult(path=entry.path)
        try:
            await self._log.info(f"[Compile] Processing: {Path(entry.path).name}")

            extraction = await fast_extract(file_path=entry.path)
            content = extraction.content
            if not content or len(content.strip()) < 100:
                result.error = "Insufficient text content"
                return result

            use_tree = (
                not shallow
                and DocumentTreeIndexer.should_build_tree(entry.path, len(content))
            )

            if use_tree:
                result.tree = await self._tree_indexer.build_tree(
                    entry.path, content,
                )

            result.summary = await self._extract_summary(
                entry.path, content, result.tree,
            )
            result.topics = await self._extract_topics(result.summary)
            result.evidence = self._build_evidence(entry, content, result)

        except Exception as exc:
            result.error = str(exc)
            await self._log.warning(f"[Compile] Failed: {entry.path}: {exc}")

        return result

    async def _extract_summary(
        self,
        file_path: str,
        content: str,
        tree: Optional[DocumentTree] = None,
    ) -> str:
        """Generate a document-level summary.

        When a tree is available its root already contains an LLM-synthesized
        summary (produced by ``_synthesize_root_summary`` during tree build),
        so we reuse it directly — no redundant LLM call.
        """
        if tree and tree.root and tree.root.summary:
            return tree.root.summary

        preview = content[:16000] if len(content) > 16000 else content
        from sirchmunk.llm.prompts import COMPILE_DOC_SUMMARY
        prompt = COMPILE_DOC_SUMMARY.format(
            file_name=Path(file_path).name,
            document_content=preview,
        )
        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        return resp.content.strip()

    def _build_evidence(
        self,
        entry: FileEntry,
        content: str,
        result: FileCompileResult,
    ) -> EvidenceUnit:
        """Build an EvidenceUnit, populating snippets/tree_path from tree leaves."""
        from sirchmunk.schema.metadata import FileInfo

        snippets: List[str] = []
        tree_path: Optional[List[str]] = None

        if result.tree and result.tree.root:
            leaves = result.tree.root.all_leaves()
            tree_path = [leaf.node_id for leaf in leaves]
            for leaf in leaves:
                start, end = leaf.char_range
                snippet = content[start:end][:500]
                if snippet.strip():
                    snippets.append(snippet)

        return EvidenceUnit(
            doc_id=FileInfo.get_cache_key(entry.path),
            file_or_url=Path(entry.path),
            summary=result.summary,
            is_found=True,
            snippets=snippets,
            tree_path=tree_path,
            extracted_at=datetime.now(timezone.utc),
        )

    async def _extract_topics(self, summary: str) -> List[str]:
        """Extract key topics/entities from a document summary."""
        from sirchmunk.llm.prompts import COMPILE_TOPIC_EXTRACTION
        prompt = COMPILE_TOPIC_EXTRACTION.format(summary=summary)
        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        try:
            raw = resp.content.strip()
            if raw.startswith("["):
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(t) for t in parsed if t]
            return [t.strip() for t in raw.split(",") if t.strip()]
        except (json.JSONDecodeError, TypeError):
            return []

    # ------------------------------------------------------------------ #
    #  Knowledge aggregation (LLM Wiki Ingest)                            #
    # ------------------------------------------------------------------ #

    async def _aggregate_to_knowledge_network(
        self, result: FileCompileResult,
    ) -> Tuple[int, int]:
        """Aggregate a file's compile result into the knowledge network.

        Three-tier similarity strategy (per design doc):
          - similarity >= 0.80  → merge into existing cluster
          - 0.50 <= sim < 0.80  → create new cluster + weak edge to similar
          - similarity < 0.50   → create standalone cluster

        Returns:
            (clusters_created, clusters_merged)
        """
        created, merged = 0, 0
        if not result.summary:
            return created, merged

        embedding = self._encode_text(result.summary)

        # Search for similar existing clusters across a wider range
        best_match: Optional[Dict[str, Any]] = None
        if embedding is not None:
            similar = await self._storage.search_similar_clusters(
                query_embedding=embedding,
                top_k=3,
                similarity_threshold=0.50,
            )
            if similar:
                best_match = similar[0]

        if best_match and best_match["similarity"] >= 0.80:
            # Tier 1: merge into existing cluster
            cluster = await self._storage.get(best_match["id"])
            if cluster:
                await self._merge_into_cluster(cluster, result)
                # Re-compute embedding for merged content
                await self._update_cluster_embedding(cluster)
                result.cluster_ids.append(cluster.id)
                merged += 1
                return created, merged

        # Create a new cluster (Tier 2 or Tier 3)
        cluster = await self._create_cluster(result)
        if cluster:
            result.cluster_ids.append(cluster.id)
            await self._store_cluster_embedding(cluster, embedding, result.summary)
            created += 1

            # Tier 2: build weak edges to moderately similar clusters
            if best_match and best_match["similarity"] >= 0.50:
                for s in (similar or []):
                    if s["similarity"] >= 0.50:
                        target = await self._storage.get(s["id"])
                        if target:
                            self._add_edge(cluster, target.id, "embed_sim", s["similarity"])
                            self._add_edge(target, cluster.id, "embed_sim", s["similarity"])
                            await self._storage.update(target)
                await self._storage.update(cluster)

        return created, merged

    def _encode_text(self, text: str) -> Optional[Any]:
        """Encode text to embedding vector, returns None on failure."""
        if not self._embedding:
            return None
        try:
            return self._embedding.encode(text)
        except Exception:
            return None

    async def _store_cluster_embedding(
        self, cluster: KnowledgeCluster, embedding: Optional[Any], text: str,
    ) -> None:
        """Store embedding for a cluster if available."""
        if embedding is None or not self._embedding:
            return
        text_hash = hashlib.md5(text.encode()).hexdigest()
        vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        await self._storage.store_embedding(
            cluster.id, vec,
            self._embedding.model_id or "default",
            text_hash,
        )

    async def _update_cluster_embedding(self, cluster: KnowledgeCluster) -> None:
        """Re-compute and store embedding after content merge."""
        content_text = str(cluster.content)[:2000] if cluster.content else ""
        if not content_text:
            return
        embedding = self._encode_text(content_text)
        await self._store_cluster_embedding(cluster, embedding, content_text)

    async def _merge_into_cluster(
        self,
        cluster: KnowledgeCluster,
        result: FileCompileResult,
    ) -> None:
        """Merge a file compile result into an existing cluster."""
        # Append evidence
        if result.evidence:
            existing_doc_ids = {e.doc_id for e in cluster.evidences}
            if result.evidence.doc_id not in existing_doc_ids:
                cluster.evidences.append(result.evidence)

        # Enrich content via LLM merge
        from sirchmunk.llm.prompts import COMPILE_MERGE_KNOWLEDGE
        prompt = COMPILE_MERGE_KNOWLEDGE.format(
            existing_content=str(cluster.content)[:3000],
            new_summary=result.summary[:3000],
        )
        resp = await self._llm.achat([{"role": "user", "content": prompt}])
        cluster.content = resp.content.strip()

        # Update metadata
        cluster.search_results = list(set(
            (cluster.search_results or []) + [result.path]
        ))
        merge_count = getattr(cluster, "merge_count", 0) or 0
        cluster.merge_count = merge_count + 1

        # Lifecycle promotion
        if cluster.merge_count >= 3 and cluster.lifecycle == Lifecycle.EMERGING:
            cluster.lifecycle = Lifecycle.STABLE

        await self._storage.update(cluster)

    async def _create_cluster(
        self, result: FileCompileResult,
    ) -> Optional[KnowledgeCluster]:
        """Create a new KnowledgeCluster from a file compile result."""
        cluster_text = result.summary
        cluster_id = f"C{hashlib.sha256(cluster_text.encode('utf-8')).hexdigest()[:10]}"

        name = Path(result.path).stem[:60]
        if result.topics:
            name = result.topics[0][:60]

        cluster = KnowledgeCluster(
            id=cluster_id,
            name=name,
            description=[result.summary[:500]],
            content=result.summary,
            evidences=[result.evidence] if result.evidence else [],
            patterns=result.topics[:5],
            lifecycle=Lifecycle.EMERGING,
            confidence=0.5,
            abstraction_level=AbstractionLevel.TECHNIQUE,
            hotness=0.3,
            search_results=[result.path],
        )

        ok = await self._storage.insert(cluster)
        return cluster if ok else None

    # ------------------------------------------------------------------ #
    #  Cross-references                                                   #
    # ------------------------------------------------------------------ #

    async def _build_cross_references(
        self, results: List[FileCompileResult],
    ) -> int:
        """Build co-occurrence edges between clusters that share source files.

        Two clusters are co-occurring when the same source file contributed
        evidence to both (e.g., different sections compiled into different
        clusters).  Includes historical data from the manifest.
        """
        # Build a complete map: cluster_id -> set of source file paths
        cluster_to_files: Dict[str, Set[str]] = {}

        # From current compile results
        for r in results:
            for cid in r.cluster_ids:
                cluster_to_files.setdefault(cid, set()).add(r.path)

        # From manifest (historical data)
        manifest = self._load_manifest()
        for fp, entry in manifest.files.items():
            for cid in entry.cluster_ids:
                cluster_to_files.setdefault(cid, set()).add(fp)

        # Find cluster pairs that share at least one source file
        cluster_ids = list(cluster_to_files.keys())
        edges_created = 0
        pairs_seen: Set[Tuple[str, str]] = set()

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cid_a, cid_b = cluster_ids[i], cluster_ids[j]
                shared = cluster_to_files[cid_a] & cluster_to_files[cid_b]
                if not shared:
                    continue

                pair_key = (min(cid_a, cid_b), max(cid_a, cid_b))
                if pair_key in pairs_seen:
                    continue
                pairs_seen.add(pair_key)

                weight = min(len(shared) * 0.25, 1.0)
                c_a = await self._storage.get(cid_a)
                c_b = await self._storage.get(cid_b)
                if c_a and c_b:
                    self._add_edge(c_a, cid_b, "co_occur", weight)
                    self._add_edge(c_b, cid_a, "co_occur", weight)
                    await self._storage.update(c_a)
                    await self._storage.update(c_b)
                    edges_created += 1

        return edges_created

    @staticmethod
    def _add_edge(
        cluster: KnowledgeCluster, target_id: str, source: str, weight: float,
    ) -> None:
        """Add or update a WeakSemanticEdge on a cluster."""
        for edge in cluster.related_clusters:
            if edge.target_cluster_id == target_id and edge.source == source:
                edge.weight = max(edge.weight, weight)
                return
        cluster.related_clusters.append(
            WeakSemanticEdge(target_cluster_id=target_id, weight=weight, source=source)
        )

    # ------------------------------------------------------------------ #
    #  Manifest I/O                                                       #
    # ------------------------------------------------------------------ #

    def _load_manifest(self) -> CompileManifest:
        if self._manifest_path.exists():
            try:
                return CompileManifest.from_json(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass
        return CompileManifest()

    def _save_manifest(self, manifest: CompileManifest) -> None:
        self._manifest_path.write_text(manifest.to_json(), encoding="utf-8")
