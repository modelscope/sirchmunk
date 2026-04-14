# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import ast
import hashlib
import json
import logging
import math
import os
import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from sirchmunk.base import BaseSearch
from sirchmunk.learnings.knowledge_base import KnowledgeBase
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    KEYWORD_QUERY_PLACEHOLDER,
    generate_keyword_extraction_prompt,
    FAST_QUERY_ANALYSIS,
    FAST_QUERY_ANALYSIS_WITH_CATALOG,
    ROI_RESULT_SUMMARY,
    SEARCH_RESULT_SUMMARY,
    DOC_SUMMARY,
    DOC_CHUNK_SUMMARY,
    DOC_MERGE_SUMMARIES,
)
from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.schema.knowledge import (
    AbstractionLevel,
    EvidenceUnit,
    KnowledgeCluster,
    Lifecycle,
)
from sirchmunk.schema.request import ContentItem, Message, Request
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.storage.knowledge_storage import KnowledgeStorage
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH
from sirchmunk.utils.embedding_util import EmbeddingUtil
from sirchmunk.utils.deps import check_dependencies
from sirchmunk.utils import create_logger, LogCallback
from loguru import logger as _loguru_logger
from sirchmunk.utils.install_rga import install_rga
from sirchmunk.utils.utils import (
    KeywordValidation,
    extract_fields,
)

# Only for quick simple-chat intent detection to reduce unnecessary LLM calls
_CHAT_QUERY_RE = re.compile(
    r"^("
    # Greetings (ZH / EN / pinyin / JA / KO)
    r"你好|您好|嗨|哈喽|喂|早上好|下午好|晚上好|早安|午安|晚安"
    r"|hello|hi|hey|howdy|greetings|yo"
    r"|nihao|ni\s*hao"
    r"|good\s*(morning|afternoon|evening|night)"
    r"|こんにちは|こんばんは|おはよう"
    r"|안녕하세요|안녕"
    # Identity / capability
    r"|who\s+are\s+you|what\s+are\s+you|你是谁|你是什么"
    r"|介绍.*你自己|tell\s+me\s+about\s+yourself"
    r"|what\s+can\s+you\s+do|你能做什么|你会什么"
    # Small talk
    r"|how\s+are\s+you|你好吗|你怎么样|what'?s\s+up"
    # Thanks
    r"|thank\s*you|thanks|谢谢|感谢|多谢"
    # Goodbye
    r"|bye|goodbye|再见|拜拜|see\s+you"
    # Ping / test
    r"|test(ing)?|ping"
    r")[\s!！？?。.，,~～…]*$",
    re.IGNORECASE,
)

_CHAT_RESPONSE_SYSTEM = (
    "You are Sirchmunk, an intelligent document search and analysis assistant. "
    "The user sent a conversational message (greeting, identity question, etc.) "
    "rather than a search query. Respond naturally and helpfully in 1-3 sentences. "
    "Reply in the same language as the user's message."
)

_NO_RESULTS_MESSAGE = "No results found."

# Soft-similarity threshold for gradient cluster reuse (P2)
_SOFT_SIM_THRESHOLD = 0.65


@dataclass
class SoftClusterHit:
    """Signals from clusters that are related but below the hard reuse threshold.

    Carries structured hints (keywords, file paths, background context) that
    downstream retrieval phases can exploit without short-circuiting the search.
    """

    patterns: List[str]
    file_paths: List[str]
    context_summary: str
    cluster_ids: List[str]


@dataclass
class KnowledgeProbeResult:
    """Rich result from knowledge cache probing (P3).

    Replaces the flat ``List[str]`` that ``_probe_knowledge_cache`` used to return.
    """

    file_paths: List[str]
    extra_keywords: List[str]
    background_context: str


@dataclass
class CompileHints:
    """Zero-LLM hints gathered from compile manifest and tree cache (P4)."""

    file_paths: List[str]
    extra_keywords: List[str]


@dataclass
class CompileArtifacts:
    """Compile artifact availability context for adaptive activation in FAST mode.

    Created once at the start of ``_search_fast()`` via
    ``_detect_compile_artifacts()`` and threaded through all pipeline steps.
    Each step checks the relevant field and falls back gracefully when the
    artifact is absent.
    """

    catalog: Optional[List[Dict[str, str]]]
    catalog_map: Dict[str, Dict[str, str]]  # path -> catalog entry for O(1) lookup
    tree_indexer: Optional[Any]  # DocumentTreeIndexer (lazy import)
    tree_available_paths: Set[str]  # file paths that have cached tree indices


class AgenticSearch(BaseSearch):

    def __init__(
        self,
        llm: Optional[OpenAIChat] = None,
        embedding: Optional[EmbeddingUtil] = None,
        work_path: Optional[Union[str, Path]] = None,
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
        verbose: bool = True,
        log_callback: LogCallback = None,
        reuse_knowledge: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Normalise and store default search paths
        if paths is not None:
            if isinstance(paths, (str, Path)):
                self.paths: Optional[List[str]] = [str(Path(paths).expanduser().resolve())]
            else:
                self.paths = [str(Path(p).expanduser().resolve()) for p in paths]
        else:
            self.paths = None

        _env_work = os.getenv("SIRCHMUNK_WORK_PATH")
        default_wp = os.path.expanduser(_env_work) if _env_work else DEFAULT_SIRCHMUNK_WORK_PATH
        work_path = work_path or default_wp
        self.work_path: Path = Path(work_path).expanduser().resolve()

        self.llm: OpenAIChat = llm or OpenAIChat(
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL_NAME", "gpt-5.2"),
            log_callback=log_callback,
        )

        self.grep_retriever: GrepRetriever = GrepRetriever(work_path=self.work_path)

        # Create bound logger with callback - returns AsyncLogger instance
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

        # Pass log_callback to KnowledgeBase so it can also log through the same callback
        self.knowledge_base = KnowledgeBase(
            llm=self.llm,
            work_path=self.work_path,
            log_callback=log_callback
        )

        # Initialize KnowledgeManager for persistent storage
        self.knowledge_storage = KnowledgeStorage(work_path=str(self.work_path))

        # Load historical knowledge clusters from cache
        self._load_historical_knowledge()

        self.verbose: bool = verbose

        self.llm_usages: List[Dict[str, Any]] = []

        # Maximum number of queries to keep per cluster (FIFO strategy)
        self.max_queries_per_cluster: int = 5

        # Initialize embedding client for cluster reuse.
        # EmbeddingUtil.__init__ is cheap (stores config only).
        # start_loading() is called immediately so the background thread
        # can download / construct the model while the first search runs.
        # By the time the search finishes and needs to persist the cluster
        # embedding, the model is typically ready.
        self.embedding_client = None
        self.cluster_sim_threshold: float = kwargs.pop('cluster_sim_threshold', 0.85)
        self.cluster_sim_top_k: int = kwargs.pop('cluster_sim_top_k', 3)
        if reuse_knowledge:
            try:
                # Use provided embedding instance if available
                if embedding is not None:
                    self.embedding_client = embedding
                    self.embedding_client.start_loading()
                    _loguru_logger.info(
                        f"Using provided embedding client (model={self.embedding_client.model_id or 'default'}, cache_dir={self.embedding_client._cache_dir or 'default'})"
                    )
                else:
                    embedding_cache = os.getenv("EMBEDDING_CACHE_DIR")
                    cache_dir = (
                        os.path.expanduser(embedding_cache)
                        if embedding_cache
                        else str(self.work_path / ".cache" / "models")
                    )
                    embedding_model_id = os.getenv("EMBEDDING_MODEL_ID")
                    self.embedding_client = EmbeddingUtil(
                        model_id=embedding_model_id,
                        cache_dir=cache_dir
                    )
                    self.embedding_client.start_loading()
                    _loguru_logger.info(
                        f"Embedding client created (model={embedding_model_id or 'default'}, cache_dir={cache_dir}), background model loading started"
                    )
            except Exception as e:
                _loguru_logger.error(
                    f"Failed to initialize embedding client: {e}. "
                    "Knowledge cluster embeddings will NOT be stored. "
                    "Ensure sentence-transformers, torch, and modelscope are installed."
                )
                self.embedding_client = None
        else:
            _loguru_logger.info(
                "Knowledge reuse disabled (reuse_knowledge=False). "
                "Embeddings will not be computed."
            )

        if not check_dependencies():
            _loguru_logger.info("Installing rga (ripgrep-all) and rg (ripgrep)...")
            install_rga()

        # Suppress noisy pypdf warnings about malformed PDF cross-references.
        # pypdf._reader emits logging.warning() for "Ignoring wrong pointing object".
        logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

        # ---- Agentic (ReAct) components (lazy-initialised on first use) ----
        self._tool_registry = None
        self._dir_scanner = None

        # ---- Spec-path cache for per-search-path context ----
        self.spec_path: Path = self.work_path / ".cache" / "spec"
        self.spec_path.mkdir(parents=True, exist_ok=True)
        self._spec_lock = asyncio.Lock()  # guards concurrent spec writes

    def update_log_callback(self, log_callback: LogCallback = None) -> None:
        """Replace the per-request log callback on all sub-components.

        This allows a singleton ``AgenticSearch`` instance to stream logs
        through a different WebSocket / callback on every request without
        having to reconstruct heavy resources (embedding model, knowledge
        storage, etc.).
        """
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

        self.llm._logger = create_logger(log_callback=log_callback, enable_async=False)
        self.llm._logger_async = create_logger(log_callback=log_callback, enable_async=True)

        self.knowledge_base.log_callback = log_callback
        self.knowledge_base._log = create_logger(log_callback=log_callback, enable_async=True)

        # Reset per-request token accounting
        self.llm_usages = []

    def _resolve_paths(
        self,
        paths: Optional[Union[str, Path, List[str], List[Path]]],
    ) -> List[str]:
        """Resolve and normalise paths with layered fallback.

        Priority (highest → lowest):
            1. Explicit ``paths`` argument  (``search(..., paths=xxx)``)
            2. Instance default ``self.paths``  (constructor ``paths=``)
            3. ``SIRCHMUNK_SEARCH_PATHS`` environment variable (comma-separated)
            4. Current working directory

        Always returns ``List[str]`` so callers need no further coercion.
        """
        if paths is not None:
            if isinstance(paths, (str, Path)):
                return [str(paths)]
            return [str(p) for p in paths]
        if self.paths is not None:
            return list(self.paths)
        env_paths = os.getenv("SIRCHMUNK_SEARCH_PATHS", "")
        if env_paths:
            parsed = [p.strip() for p in env_paths.split(",") if p.strip()]
            if parsed:
                _loguru_logger.info(
                    f"[paths] Using SIRCHMUNK_SEARCH_PATHS: {parsed}"
                )
                return parsed
        cwd = str(Path.cwd())
        _loguru_logger.info(
            f"[paths] No paths provided; using current working directory: {cwd}"
        )
        return [cwd]

    @staticmethod
    def validate_search_paths(
        paths: List[str],
        *,
        require_exists: bool = False,
    ) -> List[str]:
        """Sanitise and validate a list of search paths or URLs.

        Performs cross-platform checks for argument-injection, null-byte
        injection, and (optionally) filesystem existence.  Invalid entries
        are silently dropped with a warning log so that one bad element
        does not abort the entire search.

        Args:
            paths: Raw path/URL strings from the caller.
            require_exists: When *True*, filesystem paths that do not
                exist on disk are also rejected.

        Returns:
            A deduplicated list of safe paths/URLs (order-preserved).
        """
        from urllib.parse import urlparse

        seen: set = set()
        clean: List[str] = []

        for raw in paths:
            p = str(raw).strip()

            if not p:
                continue

            # Null-byte injection
            if "\x00" in p:
                _loguru_logger.warning(
                    f"[validate] Rejected path containing null byte: {p!r}"
                )
                continue

            # Detect URLs and validate separately
            if p.startswith(("http://", "https://", "ftp://", "ftps://")):
                parsed = urlparse(p)
                if not parsed.hostname:
                    _loguru_logger.warning(
                        f"[validate] Rejected malformed URL (no host): {p}"
                    )
                    continue
                if p not in seen:
                    seen.add(p)
                    clean.append(p)
                continue

            # Argument-injection: paths starting with a hyphen can be
            # misinterpreted as CLI flags by rga / ripgrep.
            if p.startswith("-"):
                _loguru_logger.warning(
                    f"[validate] Rejected path starting with hyphen "
                    f"(possible argument injection): {p}"
                )
                continue

            # Resolve to an absolute, normalised path (handles `..`, `~`,
            # symlinks, and mixed separators on Windows).
            try:
                resolved = str(Path(p).expanduser().resolve())
            except (OSError, ValueError) as exc:
                _loguru_logger.warning(
                    f"[validate] Rejected unresolvable path: {p} ({exc})"
                )
                continue

            if require_exists and not os.path.exists(resolved):
                _loguru_logger.warning(
                    f"[validate] Rejected non-existent path: {resolved}"
                )
                continue

            if resolved not in seen:
                seen.add(resolved)
                clean.append(resolved)

        return clean

    def _load_historical_knowledge(self):
        """Load historical knowledge clusters from local cache."""
        try:
            stats = self.knowledge_storage.get_stats()
            cluster_count = stats.get('custom_stats', {}).get('total_clusters', 0)
            _loguru_logger.info(f"Loaded {cluster_count} historical knowledge clusters from cache")
        except Exception as e:
            _loguru_logger.warning(f"Failed to load historical knowledge: {e}")

    async def _try_reuse_cluster(self, query: str, paths: Optional[List[str]] = None) -> Optional[KnowledgeCluster]:
        """Try to reuse existing knowledge cluster based on semantic similarity.

        The method waits (non-blocking) for the embedding model to become
        ready so that reuse works reliably even on the first search call
        within a process.

        Args:
            query: The search query string.
            paths: Optional list of file paths to filter cluster search scope.

        Returns:
            KnowledgeCluster if a suitable cached cluster is found, None otherwise.
        """
        if not self.embedding_client:
            return None

        try:
            # Wait briefly for the model so reuse can work when it's already loading.
            # Use a short timeout to avoid blocking the first request (e.g. in Docker
            # the model may take 30–60s to load; we skip reuse and do full search instead).
            if not self.embedding_client.is_ready():
                self.embedding_client.start_loading()
                try:
                    await self.embedding_client._ensure_model_async(timeout=5)
                except Exception:
                    await self._logger.debug(
                        "Embedding model not ready yet, skipping cluster reuse"
                    )
                    return None

            await self._logger.info("Searching for similar knowledge clusters...")

            query_embedding = (await self.embedding_client.embed([query]))[0]

            similar_clusters = await self.knowledge_storage.search_similar_clusters(
                query_embedding=query_embedding,
                top_k=self.cluster_sim_top_k,
                similarity_threshold=self.cluster_sim_threshold,
                search_paths=paths,
            )

            if not similar_clusters:
                await self._logger.info("No similar clusters found, performing new search...")
                return None

            best_match = similar_clusters[0]
            await self._logger.success(
                f"Found similar cluster: {best_match['name']} "
                f"(similarity: {best_match['similarity']:.3f})"
            )

            existing_cluster = await self.knowledge_storage.get(best_match["id"])
            if not existing_cluster:
                await self._logger.warning("Failed to retrieve cluster, falling back to new search")
                return None

            # Validate cluster has usable content BEFORE mutating it
            content = existing_cluster.content
            if isinstance(content, list):
                content = "\n".join(content)
            if not content:
                await self._logger.warning(
                    f"Cluster {existing_cluster.id} has empty content, falling back to full search"
                )
                return None

            # Mutate only after validation passes
            self._add_query_to_cluster(existing_cluster, query)
            existing_cluster.hotness = min(1.0, (existing_cluster.hotness or 0.5) + 0.1)
            existing_cluster.last_modified = datetime.now(timezone.utc)

            # Recompute embedding with updated queries list
            try:
                from sirchmunk.utils.embedding_util import compute_text_hash

                combined_text = self.knowledge_storage.combine_cluster_fields(
                    existing_cluster.queries
                )
                text_hash = compute_text_hash(combined_text)
                embedding_vector = (await self.embedding_client.embed([combined_text]))[0]

                await self.knowledge_storage.store_embedding(
                    cluster_id=existing_cluster.id,
                    embedding_vector=embedding_vector,
                    embedding_model=self.embedding_client.model_id,
                    embedding_text_hash=text_hash,
                )
            except Exception as emb_error:
                await self._logger.warning(f"Failed to update embedding: {emb_error}")

            await self.knowledge_storage.update(existing_cluster)

            # Flush to parquet so the updated cluster is visible to future searches
            try:
                self.knowledge_storage.force_sync()
            except Exception as sync_err:
                await self._logger.warning(f"Parquet force_sync failed: {sync_err}")

            await self._logger.success("Reused existing knowledge cluster")
            return existing_cluster

        except Exception as e:
            await self._logger.warning(
                f"Failed to search similar clusters: {e}. Falling back to full search."
            )
            return None

    async def _try_soft_reuse(
        self, query: str, paths: Optional[List[str]] = None,
    ) -> Optional[SoftClusterHit]:
        """Gradient reuse: extract structured hints from moderately similar clusters.

        Called when ``_try_reuse_cluster`` misses (similarity < hard threshold).
        Uses a softer threshold to find clusters that are *related* but not
        close enough for full reuse.  Returns patterns, file paths, and a
        background context summary that downstream phases can exploit.
        """
        if not self.embedding_client or not self.embedding_client.is_ready():
            return None

        try:
            query_embedding = (await self.embedding_client.embed([query]))[0]
            similar = await self.knowledge_storage.search_similar_clusters(
                query_embedding=query_embedding,
                top_k=5,
                similarity_threshold=_SOFT_SIM_THRESHOLD,
                search_paths=paths,
            )
            if not similar:
                return None

            patterns: List[str] = []
            file_paths: List[str] = []
            context_parts: List[str] = []
            cluster_ids: List[str] = []
            seen_paths: set = set()

            for match in similar:
                cid = match["id"]
                cluster_ids.append(cid)
                c = await self.knowledge_storage.get(cid)
                if not c:
                    continue
                for p in getattr(c, "patterns", []) or []:
                    if p and p not in patterns:
                        patterns.append(p)
                for ev in getattr(c, "evidences", []):
                    fp = str(getattr(ev, "file_or_url", ""))
                    if fp and fp not in seen_paths and Path(fp).exists():
                        seen_paths.add(fp)
                        file_paths.append(fp)
                content = c.content
                if isinstance(content, list):
                    content = "\n".join(content)
                if content:
                    context_parts.append(str(content)[:500])

            if not patterns and not file_paths:
                return None

            await self._logger.info(
                f"[SoftReuse] {len(similar)} soft hits: "
                f"{len(patterns)} patterns, {len(file_paths)} files"
            )
            return SoftClusterHit(
                patterns=patterns[:10],
                file_paths=file_paths[:10],
                context_summary="\n\n".join(context_parts[:3]),
                cluster_ids=cluster_ids,
            )
        except Exception:
            return None

    def _add_query_to_cluster(self, cluster: KnowledgeCluster, query: str) -> None:
        """
        Add query to cluster's queries list with FIFO strategy.
        Keeps only the most recent N queries (where N = max_queries_per_cluster).

        Args:
            cluster: KnowledgeCluster to update
            query: New query to add
        """
        # Add query if not already present
        if query not in cluster.queries:
            cluster.queries.append(query)

        # Apply FIFO strategy: keep only the most recent N queries
        if len(cluster.queries) > self.max_queries_per_cluster:
            # Remove oldest queries (from the beginning)
            cluster.queries = cluster.queries[-self.max_queries_per_cluster:]

    @staticmethod
    def _enrich_reused_content(cluster: KnowledgeCluster) -> str:
        """Build the answer text from a reused cluster.

        When the cluster carries compiled evidence with non-empty snippets
        (populated during ``sirchmunk compile``), appends them as supporting
        excerpts so the user sees both the summary and the underlying source
        material.
        """
        content = cluster.content
        if isinstance(content, list):
            content = "\n".join(content)
        content = str(content or "")

        evidence_parts: List[str] = []
        for ev in getattr(cluster, "evidences", []):
            snippets = getattr(ev, "snippets", None)
            if not snippets:
                continue
            source = str(getattr(ev, "file_or_url", "unknown"))
            for snip in snippets:
                text = snip if isinstance(snip, str) else snip.get("snippet", "")
                if text and text.strip():
                    evidence_parts.append(f"[{Path(source).name}] {text.strip()}")

        if evidence_parts:
            content += "\n\n---\nSupporting evidence:\n" + "\n\n".join(evidence_parts[:5])

        return content

    async def _save_cluster_with_embedding(self, cluster: KnowledgeCluster) -> None:
        """Save knowledge cluster to persistent storage, compute embedding, and flush to parquet.

        The final ``force_sync()`` ensures the embedding vector is written to
        the parquet file immediately so that subsequent searches (even across
        process restarts) can find it via ``search_similar_clusters``.

        Args:
            cluster: KnowledgeCluster to save
        """
        # Save knowledge cluster to persistent storage.
        # insert() returns False (without raising) when the cluster already
        # exists, so we explicitly fall back to update() in that case.
        try:
            inserted = await self.knowledge_storage.insert(cluster)
            if inserted:
                await self._logger.info(f"Saved knowledge cluster {cluster.id} to cache")
            else:
                await self.knowledge_storage.update(cluster)
                await self._logger.info(f"Updated knowledge cluster {cluster.id} in cache")
        except Exception as e:
            try:
                await self.knowledge_storage.update(cluster)
                await self._logger.info(f"Updated knowledge cluster {cluster.id} in cache")
            except Exception as update_error:
                await self._logger.warning(f"Failed to save knowledge cluster: {update_error}")
                return

        # Compute and store embedding for the cluster when the model is ready.
        # Use a short wait to avoid blocking the response if the model is still
        # loading (e.g. first request in Docker). If not ready, skip embedding
        # so the cluster is still saved and can be reused after the next load.
        if self.embedding_client:
            try:
                if not self.embedding_client.is_ready():
                    try:
                        await self.embedding_client._ensure_model_async(timeout=3)
                    except Exception:
                        pass
                if self.embedding_client.is_ready():
                    from sirchmunk.utils.embedding_util import compute_text_hash

                    combined_text = self.knowledge_storage.combine_cluster_fields(
                        cluster.queries
                    )
                    text_hash = compute_text_hash(combined_text)

                    embedding_vector = (await self.embedding_client.embed([combined_text]))[0]

                    await self.knowledge_storage.store_embedding(
                        cluster_id=cluster.id,
                        embedding_vector=embedding_vector,
                        embedding_model=self.embedding_client.model_id,
                        embedding_text_hash=text_hash,
                    )

                    await self._logger.info(
                        f"Stored embedding for cluster {cluster.id} "
                        f"(dim={len(embedding_vector)}, model={self.embedding_client.model_id})"
                    )
                else:
                    await self._logger.debug(
                        f"Embedding model not ready — skipping embedding for cluster {cluster.id}"
                    )

            except Exception as e:
                await self._logger.warning(f"Failed to compute embedding for cluster {cluster.id}: {e}")
        else:
            await self._logger.debug(
                f"Embedding client not configured — skipping embedding for cluster {cluster.id}"
            )

        # Flush DuckDB → parquet immediately so embedding data is persisted.
        # Without this, the daemon sync (60 s interval) or atexit hook might
        # run before the embedding is written, leaving NULL in the parquet.
        try:
            self.knowledge_storage.force_sync()
        except Exception as e:
            await self._logger.warning(f"Parquet force_sync failed: {e}")

    @staticmethod
    def _make_answer_cluster(
        query: str,
        answer: str,
        prefix: str = "FS",
        file_paths: Optional[List[str]] = None,
    ) -> KnowledgeCluster:
        """Create a fallback KnowledgeCluster wrapping an answer string.

        Used when the full evidence pipeline didn't produce a cluster
        (e.g. FAST early-termination or ReAct fallback).  Populates all
        key attributes so callers never receive a half-empty cluster.
        """
        _digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        resources = [
            {"type": "file", "value": fp} for fp in (file_paths or [])
        ]
        # Build evidences from file_paths so return_context=True yields non-empty evidences
        # Use answer content as snippets since we don't have raw evidence in this fallback path
        answer_snippet = answer if answer else ""
        evidences: List[EvidenceUnit] = []
        for i, fp in enumerate(file_paths or []):
            doc_id = hashlib.sha256(fp.encode("utf-8")).hexdigest()[:12]
            evidences.append(
                EvidenceUnit(
                    doc_id=doc_id,
                    file_or_url=fp,
                    summary=answer if answer else f"Source file for: {query[:500]}",
                    is_found=True,
                    # First evidence gets the answer snippet; others get empty to avoid duplication
                    snippets=[answer_snippet] if i == 0 and answer_snippet else [],
                    extracted_at=datetime.now(timezone.utc),
                )
            )
        return KnowledgeCluster(
            id=f"{prefix}{_digest}",
            name=query[:60],
            description=[f"Search result for: {query}"],
            content=answer,
            queries=[query],
            evidences=evidences if evidences else None,
            search_results=list(file_paths or []),
            resources=resources or None,
            confidence=0.5,
            abstraction_level=AbstractionLevel.TECHNIQUE,
            hotness=0.5,
            lifecycle=Lifecycle.EMERGING,
        )

    @staticmethod
    def _build_fast_cluster(
        query: str,
        answer: str,
        file_path: str,
        evidence: str,
        keywords: List[str],
    ) -> KnowledgeCluster:
        """Build a KnowledgeCluster from FAST-mode grep evidence.

        Richer than ``_make_answer_cluster``: contains a real EvidenceUnit
        sourced from the file that was actually retrieved.
        """
        _digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
        doc_id = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:12]

        evidence_unit = EvidenceUnit(
            doc_id=doc_id,
            file_or_url=file_path,
            summary=evidence[:500] if evidence else "",
            is_found=True,
            snippets=[evidence[:2000]] if evidence else [],
            extracted_at=datetime.now(timezone.utc),
        )

        return KnowledgeCluster(
            id=f"FS{_digest}",
            name=query[:60],
            description=[f"FAST search result for: {query}"],
            content=answer,
            evidences=[evidence_unit],
            patterns=keywords[:3],
            confidence=0.7,
            abstraction_level=AbstractionLevel.TECHNIQUE,
            landmark_potential=0.3,
            hotness=0.5,
            lifecycle=Lifecycle.EMERGING,
            queries=[query],
            search_results=[file_path],
            resources=[{"type": "file", "value": file_path}],
        )

    async def _search_by_filename(
        self,
        query: str,
        paths: Union[str, Path, List[str], List[Path]],
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        grep_timeout: Optional[float] = 60.0,
        top_k: Optional[int] = 10,
    ) -> List[Dict[str, Any]]:
        """
        Perform filename-only search without LLM keyword extraction.

        Args:
            query: Search query (used as filename pattern)
            paths: Paths to search in
            max_depth: Maximum directory depth
            include: File patterns to include
            exclude: File patterns to exclude
            grep_timeout: Timeout for grep operations
            top_k: Maximum number of results to return

        Returns:
            List of file matches with metadata
        """
        await self._logger.info("Performing filename-only search...")

        # Extract potential filename patterns from query
        patterns = []

        # Check if query looks like a file pattern (contains file extensions or wildcards)
        if any(char in query for char in ['*', '?', '[', ']']):
            # Treat as direct glob/regex pattern
            patterns = [query]
            await self._logger.info(f"Using direct pattern: {query}")
        else:
            # Split into words and create flexible patterns
            words = [w.strip() for w in query.strip().split() if w.strip()]

            if not words:
                await self._logger.warning("No valid words in query")
                return []

            # Strategy: Create patterns for each word that match anywhere in filename
            # Use non-greedy matching and case-insensitive by default
            for word in words:
                # Escape special regex characters in the word
                escaped_word = re.escape(word)
                # Match word anywhere in filename (case-insensitive handled in retrieve_by_filename)
                pattern = f".*{escaped_word}.*"
                patterns.append(pattern)
                await self._logger.debug(f"Created pattern for word '{word}': {pattern}")

        if not patterns:
            await self._logger.warning("No valid filename patterns extracted from query")
            return []

        await self._logger.info(f"Searching with {len(patterns)} pattern(s): {patterns}")

        try:
            # Use GrepRetriever's filename search
            await self._logger.debug(f"Calling retrieve_by_filename with {len(patterns)} patterns")
            results = await self.grep_retriever.retrieve_by_filename(
                patterns=patterns,
                path=paths,
                case_sensitive=False,
                max_depth=max_depth,
                include=include,
                exclude=exclude or ["*.pyc", "*.log"],
                timeout=grep_timeout,
            )

            if results:
                results = results[:top_k]
                await self._logger.success(f"Found {len(results)} matching files")
            else:
                await self._logger.warning("No files matched the patterns")

            return results

        except Exception as e:
            await self._logger.error(f"Filename search failed: {e}")
            await self._logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    @staticmethod
    def _parse_summary_response(llm_response: str) -> Tuple[str, bool, bool]:
        """
        Parse LLM response to extract summary and quality decisions.

        Args:
            llm_response: Raw LLM response containing SUMMARY, SHOULD_ANSWER and SHOULD_SAVE tags

        Returns:
            Tuple of (summary_text, should_save_flag, should_answer_flag)
        """
        summary_fields = extract_fields(
            content=llm_response,
            tags=["SUMMARY", "SHOULD_ANSWER", "SHOULD_SAVE"],
        )

        summary = str(summary_fields.get("summary") or "").strip()
        should_answer_str = str(summary_fields.get("should_answer") or "false").strip().lower()
        should_save_str = str(summary_fields.get("should_save") or "false").strip().lower()

        should_answer = should_answer_str in ["true", "yes", "1"]
        should_save = should_save_str in ["true", "yes", "1"]

        # If extraction failed, use entire response as summary and default to conservative:
        # not answerable and not saveable.
        if not summary:
            summary = llm_response.strip()
            should_answer = False
            should_save = False

        return summary, should_save, should_answer

    @staticmethod
    def _extract_and_validate_multi_level_keywords(
        llm_resp: str,
        num_levels: int = 3
    ) -> List[Dict[str, float]]:
        """
        Extract and validate multiple sets of keywords from LLM response.

        Args:
            llm_resp: LLM response containing keyword sets
            num_levels: Number of keyword granularity levels to extract

        Returns:
            List of keyword dicts, one for each level: [level1_keywords, level2_keywords, ...]
        """
        keyword_sets: List[Dict[str, float]] = []

        # Generate tags dynamically based on num_levels
        tags = [f"KEYWORDS_LEVEL_{i + 1}" for i in range(num_levels)]

        # Extract all fields at once
        extracted_fields = extract_fields(content=llm_resp, tags=tags)

        for level_idx, tag in enumerate(tags, start=1):
            keywords_dict: Dict[str, float] = {}
            keywords_json: Optional[str] = extracted_fields.get(tag.lower(), None)

            if not keywords_json:
                keyword_sets.append({})
                continue

            # Try to parse as dict format
            try:
                keywords_dict = json.loads(keywords_json)
            except json.JSONDecodeError:
                try:
                    keywords_dict = ast.literal_eval(keywords_json)
                except Exception:
                    keyword_sets.append({})
                    continue

            # Validate using Pydantic model
            try:
                validated = KeywordValidation(root=keywords_dict).model_dump()
                keyword_sets.append(validated)
            except Exception:
                keyword_sets.append({})

        return keyword_sets

    @staticmethod
    def _extract_alt_keywords(llm_resp: str) -> Dict[str, float]:
        """Extract cross-lingual keywords from ``<KEYWORDS_ALT>`` block."""
        fields = extract_fields(content=llm_resp, tags=["KEYWORDS_ALT"])
        raw = fields.get("keywords_alt")
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k: float(v) for k, v in parsed.items() if isinstance(k, str)}
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return {k: float(v) for k, v in parsed.items() if isinstance(k, str)}
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # Agentic (ReAct) infrastructure — lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_tool_registry(
        self,
        paths: List[str],
        enable_dir_scan: bool = False,
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> "ToolRegistry":
        """Build (or rebuild) the tool registry for the given search paths.

        The registry is cached on ``self._tool_registry`` and re-created
        only when ``paths`` change (detected via sorted hash).

        Args:
            paths: Normalised list of path strings.
            enable_dir_scan: Whether to include the directory-scan tool.
            max_depth: Maximum directory depth for keyword search.
            include: File patterns to include (glob).
            exclude: File patterns to exclude (glob).

        Returns:
            Ready-to-use ToolRegistry.
        """
        from sirchmunk.agentic.tools import (
            FileReadTool,
            KeywordSearchTool,
            KnowledgeQueryTool,
            ToolRegistry,
        )

        # Cache key: paths + filter params (all affect tool behaviour)
        cache_key = (
            tuple(sorted(paths)),
            max_depth,
            tuple(include) if include else None,
            tuple(exclude) if exclude else None,
        )
        if (
                self._tool_registry is not None
                and getattr(self, "_tool_registry_key", None) == cache_key
        ):
            return self._tool_registry

        registry = ToolRegistry()

        # Tool 1: Knowledge cache (zero cost)
        registry.register(KnowledgeQueryTool(self.knowledge_storage))

        # Tool 2: Keyword search (low cost)
        registry.register(
            KeywordSearchTool(
                retriever=self.grep_retriever,
                paths=paths,
                max_depth=max_depth if max_depth is not None else 5,
                max_results=10,
                include=include,
                exclude=exclude,
            )
        )

        # Tool 3: File read (medium cost)
        registry.register(FileReadTool(max_chars_per_file=30000))

        # Tool 4: Directory scan (optional, medium cost)
        if enable_dir_scan:
            from sirchmunk.agentic.dir_scan_tool import DirScanTool
            from sirchmunk.scan.dir_scanner import DirectoryScanner

            if self._dir_scanner is None:
                self._dir_scanner = DirectoryScanner(llm=self.llm, max_files=500)
            registry.register(DirScanTool(
                scanner=self._dir_scanner,
                paths=paths,
            ))

        self._tool_registry = registry
        self._tool_registry_key = cache_key
        return registry

    # ------------------------------------------------------------------
    # Knowledge compile entry point
    # ------------------------------------------------------------------

    async def compile(
        self,
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
        *,
        incremental: bool = True,
        shallow: bool = False,
        max_files: Optional[int] = None,
        concurrency: int = 3,
    ) -> Dict[str, Any]:
        """Compile document collections into structured knowledge indices.

        Optional offline pre-processing step that builds tree indices and
        knowledge clusters.  Products are automatically leveraged by
        subsequent search() calls.

        Args:
            paths: Directories or files to compile. Falls back to self.paths.
            incremental: Skip unchanged files (default True).
            shallow: Skip tree building — use direct LLM summarisation only.
            max_files: Cap on files — triggers importance sampling for large sets.
            concurrency: Max parallel file compilations.

        Returns:
            CompileReport as a dict.
        """
        from sirchmunk.learnings.compiler import KnowledgeCompiler
        from sirchmunk.learnings.tree_indexer import DocumentTreeIndexer

        resolved = self._resolve_paths(paths)
        await self._logger.info(
            f"[Compile] Starting compile for {len(resolved)} path(s)"
        )

        tree_cache = self.work_path / ".cache" / "compile" / "trees"
        _cb = getattr(self._logger, 'log_callback', None)
        tree_indexer = DocumentTreeIndexer(
            llm=self.llm,
            cache_dir=tree_cache,
            log_callback=_cb,
        )

        compiler = KnowledgeCompiler(
            llm=self.llm,
            embedding_client=self.embedding_client,
            knowledge_storage=self.knowledge_storage,
            tree_indexer=tree_indexer,
            work_path=self.work_path,
            log_callback=_cb,
        )

        report = await compiler.compile(
            paths=resolved,
            incremental=incremental,
            shallow=shallow,
            max_files=max_files,
            concurrency=concurrency,
        )

        return report.to_dict()

    async def compile_status(
        self,
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
    ) -> Dict[str, Any]:
        """Return current compile status for the given paths."""
        from sirchmunk.learnings.compiler import KnowledgeCompiler
        from sirchmunk.learnings.tree_indexer import DocumentTreeIndexer

        resolved = self._resolve_paths(paths)

        tree_cache = self.work_path / ".cache" / "compile" / "trees"
        tree_indexer = DocumentTreeIndexer(
            llm=self.llm, cache_dir=tree_cache,
        )

        compiler = KnowledgeCompiler(
            llm=self.llm,
            embedding_client=self.embedding_client,
            knowledge_storage=self.knowledge_storage,
            tree_indexer=tree_indexer,
            work_path=self.work_path,
        )

        status = await compiler.get_status(resolved)
        return {
            "total_compiled_files": status.total_compiled_files,
            "total_clusters": status.total_clusters,
            "total_trees": status.total_trees,
            "last_compile_at": status.last_compile_at,
            "manifest_path": status.manifest_path,
        }

    async def compile_lint(
        self,
        *,
        auto_fix: bool = False,
    ) -> Dict[str, Any]:
        """Run knowledge health checks and optionally auto-fix issues."""
        from sirchmunk.learnings.lint import KnowledgeLint

        linter = KnowledgeLint(
            knowledge_storage=self.knowledge_storage,
            work_path=self.work_path,
            log_callback=getattr(self._logger, 'log_callback', None),
        )

        report = await linter.run(auto_fix=auto_fix)
        return report.to_dict()

    # ------------------------------------------------------------------
    # Unified search entry point
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
        *,
        mode: Literal["DEEP", "FAST", "FILENAME_ONLY"] = "FAST",
        max_loops: int = 10,
        max_token_budget: int = 128000,
        max_depth: Optional[int] = 5,
        top_k_files: int = 5,
        enable_dir_scan: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        return_context: bool = False,
        spec_stale_hours: float = 72.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        llm_fallback: bool = False,
    ) -> Union[str, SearchContext, List[Dict[str, Any]]]:
        """Perform intelligent search with multi-mode support.

        Modes:
            +--------------+-------------------+-------------------------------------------+
            | Mode         | Speed / LLM Calls | Description                               |
            +--------------+-------------------+-------------------------------------------+
            | FILENAME_ONLY| Very Fast / 0     | Pattern-based file discovery, no LLM.     |
            | FAST         | 1-5s / 0-2        | Greedy: cluster reuse or keyword search    |
            |              |                   | → best file → answer. Early termination.  |
            | DEEP         | 5-30s / 4-6       | Parallel multi-path retrieval + ReAct     |
            |              |                   | refinement with Monte-Carlo evidence.     |
            +--------------+-------------------+-------------------------------------------+

        FAST architecture (greedy early-termination):

        ┌──────────────────────────────────────────────────────────┐
        │ Step 0  Cluster reuse check (instant short-circuit)       │
        ├──────────────────────────────────────────────────────────┤
        │ Step 1  LLM query analysis → keywords + file hints       │
        │         (single call, stream=False)                      │
        ├──────────────────────────────────────────────────────────┤
        │ Step 2  rga keyword search → ranked file hits + snippets │
        │         (no LLM, greedy: take first good results)        │
        ├──────────────────────────────────────────────────────────┤
        │ Step 3  Read top file(s) content                         │
        │         (no LLM, early termination at top_k_files)       │
        ├──────────────────────────────────────────────────────────┤
        │ Step 4  LLM answer synthesis from evidence               │
        └──────────────────────────────────────────────────────────┘

        DEEP architecture (phases execute as parallel as possible):

        ┌──────────────────────────────────────────────────────────┐
        │ Phase 0a Direct document analysis (intent-gated,         │
        │          short-circuit if query is doc-level operation)   │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 0  Cluster reuse check (instant, short-circuit)    │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 1  Parallel probing (all concurrent):              │
        │  ├─ LLM keyword extraction                               │
        │  ├─ DirectoryScanner.scan() (filesystem only, fast)      │
        │  ├─ Knowledge cache similarity search                    │
        │  └─ Spec-path cache load                                 │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 2  Parallel retrieval (depends on Phase 1):        │
        │  ├─ keyword_search per extracted keyword (concurrent rga)│
        │  └─ DirectoryScanner.rank() (LLM ranks candidates)      │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 3  Merge + evidence assembly:                      │
        │  └─ knowledge_base.build() (parallel per-file Monte      │
        │     Carlo evidence sampling)                             │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 4  Summary / ReAct refinement:                     │
        │  └─ If evidence sufficient → LLM summary                 │
        │     Else → ReAct loop for adaptive follow-up             │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 5  Persistence (concurrent, awaited):                │
        │  ├─ Save cluster + embeddings                            │
        │  └─ Save spec-path cache                                 │
        └──────────────────────────────────────────────────────────┘

        Args:
            query: User's search query.
            paths: Directories / files to search.  Falls back to
                ``self.paths`` or the current working directory.
            mode: Search mode — ``"DEEP"``, ``"FAST"``, or ``"FILENAME_ONLY"``.
            max_loops: Maximum ReAct iterations (DEEP mode, default: 10).
            max_token_budget: LLM token budget (DEEP mode, default: 128000).
            max_depth: Maximum directory depth for file search (default: 5).
                Used in both FILENAME_ONLY and DEEP modes.
            top_k_files: Max files for evidence extraction (default: 5).
            enable_dir_scan: Enable directory scanning (FAST and DEEP modes).
            include: File glob patterns to include (e.g. ``["*.py", "*.md"]``).
                Used in both FILENAME_ONLY and DEEP modes.
            exclude: File glob patterns to exclude (e.g. ``["*.log"]``).
                Used in both FILENAME_ONLY and DEEP modes.
            return_context: If True, return a ``SearchContext`` object
                that carries ``answer``, ``cluster`` (KnowledgeCluster),
                and full pipeline telemetry (LLM usage, files read, etc.).
            spec_stale_hours: Hours before spec cache is stale (default: 72).
            chat_history: Optional list of chat messages for context (DEEP mode).
            llm_fallback: When True, if no relevant documents are found,
                the LLM will attempt to answer the query from its own
                knowledge. Default False.

        Returns:
            - ``str``: Answer summary (default).
            - ``SearchContext``: If *return_context* — contains ``answer``,
              ``cluster``, and telemetry in a single object.
            - ``List[Dict]``: File matches in FILENAME_ONLY mode.
        """
        paths = self.validate_search_paths(
            self._resolve_paths(paths),
        )
        if not paths:
            msg = "No valid search paths remain after validation."
            _loguru_logger.warning(msg)
            if return_context:
                ctx = SearchContext()
                ctx.answer = msg
                return ctx
            return msg

        # ---- Chat intent short-circuit (rule-based, no LLM cost) ----
        if mode != "FILENAME_ONLY" and self._is_chat_query(query):
            answer, cluster, ctx = await self._respond_chat(query, chat_history=chat_history)
            if return_context:
                ctx.answer = answer
                return ctx
            return answer

        # ---- FILENAME_ONLY: pattern-based file discovery, no LLM ----
        if mode == "FILENAME_ONLY":
            results = await self._search_by_filename(
                query=query, paths=paths, max_depth=max_depth,
                include=include, exclude=exclude, top_k=top_k_files,
            )
            if not results:
                msg = f"No files found matching query: '{query}'"
                await self._logger.warning(msg)
                return msg
            await self._logger.success(f"Retrieved {len(results)} matching files")
            return results

        # ---- FAST / DEEP → both produce (answer, cluster, context) ----
        if mode == "FAST":
            answer, cluster, context = await self._search_fast(
                query=query, paths=paths, max_depth=max_depth,
                top_k_files=top_k_files, enable_dir_scan=enable_dir_scan,
                include=include, exclude=exclude,
                llm_fallback=llm_fallback,
            )
        else:
            answer, cluster, context = await self._search_deep(
                query=query, paths=paths,
                max_loops=max_loops, max_token_budget=max_token_budget,
                max_depth=max_depth, top_k_files=top_k_files,
                enable_dir_scan=enable_dir_scan,
                include=include, exclude=exclude,
                spec_stale_hours=spec_stale_hours,
                llm_fallback=llm_fallback,
            )

        # ---- Unified return wrapping ----
        if return_context:
            prefix = "FS" if mode == "FAST" else "DS"
            context.answer = answer
            if (answer or "").strip().lower() == _NO_RESULTS_MESSAGE.lower():
                context.cluster = cluster
                return context
            # Use read_file_ids from context if available, otherwise empty
            fallback_files = list(context.read_file_ids) if context.read_file_ids else None
            context.cluster = cluster or self._make_answer_cluster(
                query, answer, prefix, file_paths=fallback_files,
            )
            return context
        return answer

    # ------------------------------------------------------------------
    # DEEP mode — parallel multi-path retrieval with ReAct fallback
    # ------------------------------------------------------------------

    async def _search_deep(
        self,
        query: str,
        paths: List[str],
        *,
        max_loops: int = 10,
        max_token_budget: int = 128000,
        max_depth: Optional[int] = 5,
        top_k_files: int = 5,
        enable_dir_scan: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        spec_stale_hours: float = 72.0,
        llm_fallback: bool = False,
    ) -> Tuple[str, Optional[KnowledgeCluster], SearchContext]:
        """Parallel multi-path retrieval pipeline (Phases 0a–5).

        Returns:
            ``(answer, cluster, context)`` tuple.
        """
        context = SearchContext(
            max_token_budget=max_token_budget,
            max_loops=max_loops,
        )
        _llm_usage_start = len(self.llm_usages)

        # ==============================================================
        # Phase 0a: Direct document analysis (intent-gated short-circuit)
        # ==============================================================
        direct = await self._try_direct_doc_analysis(query, paths)
        if direct is not None:
            return direct, self._make_answer_cluster(query, direct, "DQ", file_paths=paths), context

        # ==============================================================
        # Phase 0: Cluster reuse (instant short-circuit)
        # When reuse_knowledge=True and a similar cluster is found, we
        # return here — Phase 5 (Persistence) is not executed for that path.
        # ==============================================================
        reused = await self._try_reuse_cluster(query, paths)
        if reused is not None:
            return self._enrich_reused_content(reused), reused, context

        # P2: gradient reuse — extract hints from moderately similar clusters
        soft_hit = await self._try_soft_reuse(query, paths)

        await self._logger.info(f"[search] Starting multi-path retrieval for: '{query[:80]}'")

        # ==============================================================
        # Phase 1: Parallel probing — five paths fire concurrently
        # ==============================================================
        await self._logger.info("[Phase 1] Parallel probing: keywords + dir_scan + knowledge + spec_cache + tree_index")
        context.increment_loop()

        phase1_results = await asyncio.gather(
            self._probe_keywords(query),
            self._probe_dir_scan(paths, enable_dir_scan),
            self._probe_knowledge_cache(query),
            self._load_spec_context(paths, stale_hours=spec_stale_hours),
            self._probe_tree_index(query),
            return_exceptions=True,
        )

        kw_result = phase1_results[0] if not isinstance(phase1_results[0], Exception) else ({}, [])
        scan_result = phase1_results[1] if not isinstance(phase1_results[1], Exception) else None
        knowledge_probe = phase1_results[2] if not isinstance(phase1_results[2], Exception) else KnowledgeProbeResult([], [], "")
        spec_context = phase1_results[3] if not isinstance(phase1_results[3], Exception) else ""
        tree_hits = phase1_results[4] if not isinstance(phase1_results[4], Exception) else []

        for i, label in enumerate(["keywords", "dir_scan", "knowledge", "spec_cache", "tree_index"]):
            if isinstance(phase1_results[i], Exception):
                await self._logger.warning(f"[Phase 1] {label} probe failed: {phase1_results[i]}")

        # Backwards compat: knowledge_probe may be a plain list from old code paths
        if isinstance(knowledge_probe, list):
            knowledge_probe = KnowledgeProbeResult(file_paths=knowledge_probe, extra_keywords=[], background_context="")

        query_keywords, initial_keywords = kw_result if isinstance(kw_result, tuple) else ({}, [])

        # P2: inject soft-hit patterns into keywords
        if soft_hit:
            for p in soft_hit.patterns:
                if p not in initial_keywords:
                    initial_keywords.append(p)
                if p not in query_keywords:
                    query_keywords[p] = 0.6

        # P3: inject extra keywords from structured knowledge probe
        for kw in knowledge_probe.extra_keywords:
            if kw not in initial_keywords:
                initial_keywords.append(kw)
            if kw not in query_keywords:
                query_keywords[kw] = 0.5

        # P2 + P3: append background context for Phase 4 LLM prompt
        if soft_hit and soft_hit.context_summary:
            spec_context = f"{spec_context}\n\n{soft_hit.context_summary}" if spec_context else soft_hit.context_summary
        if knowledge_probe.background_context:
            spec_context = f"{spec_context}\n\n{knowledge_probe.background_context}" if spec_context else knowledge_probe.background_context

        await self._logger.info(
            f"[Phase 1] Results: keywords={len(initial_keywords)}, "
            f"dir_scan={'OK' if scan_result else 'N/A'}, "
            f"knowledge_files={len(knowledge_probe.file_paths)}, "
            f"tree_hits={len(tree_hits)}, "
            f"soft_hit={'YES' if soft_hit else 'NO'}, "
            f"spec_cache={'YES' if spec_context else 'NO'}"
        )

        # ==============================================================
        # Phase 2: Parallel retrieval — keyword search + dir_scan rank
        # ==============================================================
        await self._logger.info("[Phase 2] Parallel retrieval: rga keyword search + dir_scan LLM rank")
        context.increment_loop()

        phase2_tasks = []

        if initial_keywords:
            phase2_tasks.append(
                self._retrieve_by_keywords(
                    initial_keywords, paths,
                    max_depth=max_depth, include=include, exclude=exclude,
                )
            )
        else:
            phase2_tasks.append(self._async_noop([]))

        if scan_result is not None and enable_dir_scan:
            phase2_tasks.append(
                self._rank_dir_scan_candidates(query, scan_result)
            )
        else:
            phase2_tasks.append(self._async_noop([]))

        phase2_results = await asyncio.gather(*phase2_tasks, return_exceptions=True)

        keyword_files = phase2_results[0] if not isinstance(phase2_results[0], Exception) else []
        dir_scan_files = phase2_results[1] if not isinstance(phase2_results[1], Exception) else []

        for i, label in enumerate(["keyword_search", "dir_scan_rank"]):
            if isinstance(phase2_results[i], Exception):
                await self._logger.warning(f"[Phase 2] {label} failed: {phase2_results[i]}")

        await self._logger.info(
            f"[Phase 2] Results: keyword_files={len(keyword_files)}, "
            f"dir_scan_files={len(dir_scan_files)}"
        )

        # ==============================================================
        # Phase 3: Merge file paths + build KnowledgeCluster
        # P1 tree hits get highest priority; P2 soft-hit files next
        # ==============================================================
        context.increment_loop()
        extra_knowledge_files = knowledge_probe.file_paths
        if soft_hit:
            extra_knowledge_files = soft_hit.file_paths + extra_knowledge_files
        merged_files = self._merge_file_paths(
            keyword_files=list(tree_hits) + keyword_files,
            dir_scan_files=dir_scan_files,
            knowledge_hits=extra_knowledge_files,
        )
        await self._logger.info(f"[Phase 3] Merged {len(merged_files)} unique candidate files")

        cluster: Optional[KnowledgeCluster] = None
        if merged_files:
            cluster = await self._build_cluster(
                query=query, file_paths=merged_files,
                query_keywords=query_keywords, top_k_files=top_k_files,
            )

        # ==============================================================
        # Phase 3.5: Graph context enrichment (P5)
        # Append related knowledge from graph neighbours to cluster content
        # so the answer-generation LLM has richer context.
        # ==============================================================
        graph_ctx = ""
        if cluster:
            graph_ctx = await self._gather_graph_context(cluster)
            if graph_ctx and cluster.content:
                if isinstance(cluster.content, list):
                    cluster.content = "\n".join(cluster.content)
                cluster.content = f"{cluster.content}\n\n{graph_ctx}"

        # ==============================================================
        # Phase 4: Generate answer — cluster summary or ReAct refinement
        # ==============================================================
        context.increment_loop()
        answer: str = ""
        should_save: bool = True

        if cluster and cluster.content:
            await self._logger.info("[Phase 4] Evidence sufficient, generating summary")
            answer, should_save, should_answer = await self._summarise_cluster(query, cluster)
            if not should_answer:
                if llm_fallback:
                    await self._logger.info(
                        "[Phase 4] Summary gate rejected evidence, llm_fallback=True → LLM fallback"
                    )
                    answer, should_save = await self._summarise_cluster_fallback(query)
                else:
                    await self._logger.warning(
                        "[Phase 4] Summary gate rejected evidence and llm_fallback=False "
                        "→ returning no results"
                    )
                    return _NO_RESULTS_MESSAGE, None, context
            if not cluster.search_results:
                cluster.search_results = list(merged_files)
        elif llm_fallback:
            await self._logger.info(
                "[Phase 4] Evidence insufficient, llm_fallback=True \u2192 LLM summary"
            )
            answer, should_save = await self._summarise_cluster_fallback(query)
        else:
            await self._logger.info("[Phase 4] Evidence insufficient, launching ReAct refinement")
            # P5: enrich ReAct context with graph knowledge
            react_spec = f"{spec_context}\n\n{graph_ctx}" if graph_ctx else spec_context
            react_answer, context = await self._react_refinement(
                query=query, paths=paths,
                initial_keywords=initial_keywords, spec_context=react_spec,
                enable_dir_scan=enable_dir_scan,
                max_loops=max_loops, max_token_budget=max_token_budget,
                max_depth=max_depth, include=include, exclude=exclude,
            )

            if not cluster:
                cluster = await self._build_cluster_from_context(
                    query=query, answer=react_answer, context=context,
                    query_keywords=query_keywords, top_k_files=top_k_files,
                )
            elif react_answer and not cluster.content:
                cluster.content = react_answer

            if not cluster:
                await self._logger.warning(
                    "[Phase 4] ReAct found no buildable evidence and llm_fallback=False "
                    "→ returning no results"
                )
                return _NO_RESULTS_MESSAGE, None, context

            # Final DEEP decision is always made in the summary call.
            answer, should_save, should_answer = await self._summarise_cluster(query, cluster)
            if not should_answer:
                if llm_fallback:
                    await self._logger.info(
                        "[Phase 4] Final summary gate rejected evidence, llm_fallback=True → LLM fallback"
                    )
                    answer, should_save = await self._summarise_cluster_fallback(query)
                else:
                    await self._logger.warning(
                        "[Phase 4] Final summary gate rejected evidence and llm_fallback=False "
                        "→ returning no results"
                    )
                    return _NO_RESULTS_MESSAGE, None, context

        # Sync LLM token accounting into context
        new_usages = self.llm_usages[_llm_usage_start:]
        for usage in new_usages:
            if usage and isinstance(usage, dict):
                total_tok = usage.get("total_tokens", 0)
                if total_tok == 0:
                    total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                context.add_llm_tokens(total_tok, usage=usage)

        # ==============================================================
        # Phase 5: Persistence (quality-gated)
        # Skipped when Phase 4 quality check says the answer is low-quality
        # or when Phase 0 reused a cluster (early-returned above).
        # ==============================================================
        phase5_tasks = []
        if cluster and should_save:
            self._add_query_to_cluster(cluster, query)
            phase5_tasks.append(self._save_cluster_with_embedding(cluster))
        elif not should_save:
            await self._logger.info("[Phase 5] Quality gate: low-quality answer, skipping cluster save")
            cluster = None
        phase5_tasks.append(self._save_spec_context(paths, context, scan_result=scan_result))
        results = await asyncio.gather(*phase5_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                _loguru_logger.warning(f"[Phase 5] Persistence task failed: {r}")

        await self._logger.success(f"[search] Complete: {context.summary()}")
        return answer, cluster, context

    # ------------------------------------------------------------------
    # Phase 0a: Direct document analysis (intent-gated)
    # ------------------------------------------------------------------

    async def _try_direct_doc_analysis(
        self,
        query: str,
        paths: List[str],
    ) -> Optional[str]:
        """Short-circuit for document-level queries (e.g. "请总结这篇文档").

        Uses the LLM to classify query intent (language-agnostic).  When
        a whole-document operation is detected **and** suitable files exist
        in *paths*, their content is fed directly to the LLM — bypassing
        the heavyweight keyword / dir-scan / evidence pipeline.

        Returns:
            LLM answer string, or None if the short-circuit does not apply.
        """
        from sirchmunk.doc_qa import (
            detect_doc_intent,
            collect_doc_files,
            analyse_documents,
        )

        # Step 1: file gate — skip early if paths contain no loadable docs
        doc_files = collect_doc_files(paths)
        if not doc_files:
            return None

        # Step 2: LLM intent classification (cheap, stream=False)
        operation = await detect_doc_intent(query, self.llm, self.llm_usages)
        if operation is None:
            return None

        filenames = ", ".join(Path(d.path).name for d in doc_files)
        await self._logger.info(
            f"[DocQA] Intent '{operation}' detected — "
            f"loading {len(doc_files)} file(s) for direct analysis: {filenames}"
        )

        # Step 3: for summary operations, use the chunked summarizer
        # with optional smart dir scanning; for other operations, use the
        # general analyser.
        if operation in ("summarize", "summary", "extract"):
            scan_result = None
            if self._has_directory_paths(paths):
                scan_result = await self._probe_dir_scan(paths, max_files=300)
            answer = await self._summarize_documents(
                query, paths, scan_result=scan_result,
            )
        else:
            answer = await analyse_documents(
                query=query,
                doc_files=doc_files,
                llm=self.llm,
                llm_usages=self.llm_usages,
            )

        if answer:
            await self._logger.success("[DocQA] Direct document analysis complete")
        return answer

    # ------------------------------------------------------------------
    # Chat intent detection — short-circuit for non-search queries
    # ------------------------------------------------------------------

    @staticmethod
    def _is_chat_query(query: str) -> bool:
        """Return True for obvious conversational queries (rule-based, no LLM)."""
        return bool(_CHAT_QUERY_RE.match(query.strip()))

    async def _respond_chat(
        self,
        query: str,
        context: Optional[SearchContext] = None,
        *,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, Optional[KnowledgeCluster], SearchContext]:
        """Generate a direct conversational response (single LLM call, no retrieval)."""
        await self._logger.info(
            f"[search] Chat intent detected — responding directly: '{query[:60]}'"
        )
        ctx = context or SearchContext()
        messages = [
            {"role": "system", "content": _CHAT_RESPONSE_SYSTEM},
            *(chat_history or []),
            {"role": "user", "content": query},
        ]
        resp = await self.llm.achat(messages=messages, stream=False)
        self.llm_usages.append(resp.usage)
        if resp.usage and isinstance(resp.usage, dict):
            ctx.add_llm_tokens(
                resp.usage.get("total_tokens", 0), usage=resp.usage,
            )
        return resp.content or "", None, ctx

    # ------------------------------------------------------------------
    # Document summarization — shared by FAST & DEEP summary intent
    # ------------------------------------------------------------------

    _SUMMARY_MAX_CONTEXT_CHARS = 100_000
    _SUMMARY_CHUNK_CHARS = 50_000
    _SUMMARY_MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB — sampling handles large files

    async def _summarize_documents(
        self,
        query: str,
        paths: List[str],
        *,
        top_k_files: int = 5,
        scan_result=None,
    ) -> Optional[str]:
        """Summarize documents from *paths* with smart content sampling.

        When *scan_result* (from a prior directory scan) is provided, the
        LLM ranks candidates first so only the most relevant files are
        summarized.  Otherwise falls back to ``collect_doc_files``.

        Small files are loaded in full; large files are sampled (head + mid +
        tail).  When the total content exceeds the LLM context budget, the
        documents are processed in chunks — each chunk is summarized
        independently, then the partial summaries are merged in a final pass.

        Returns:
            Summary string, or ``None`` if no documents could be loaded.
        """
        from sirchmunk.doc_qa import collect_doc_files, _extract_text, _sample_text

        summary_paths: Optional[List[str]] = None

        # When a scan result is available, use LLM ranking to pick candidates
        if scan_result is not None:
            ranked = await self._rank_dir_scan_candidates(
                query, scan_result,
                top_k=top_k_files * 2,
                include_medium=True,
            )
            if ranked:
                summary_paths = ranked[:top_k_files]
                await self._logger.info(
                    f"[Summary] Dir scan selected {len(summary_paths)} relevant file(s)"
                )

        doc_files = collect_doc_files(
            summary_paths or paths,
            max_files=top_k_files,
            max_file_size=self._SUMMARY_MAX_FILE_SIZE,
        )
        if not doc_files:
            await self._logger.warning(
                f"[Summary] No loadable documents found in paths: {paths}"
            )
            return None

        doc_texts: List[Tuple[str, str]] = []
        total_chars = 0
        for df in doc_files:
            text = await _extract_text(df)
            if text:
                fname = Path(df.path).name
                doc_texts.append((fname, text))
                total_chars += len(text)
            else:
                await self._logger.warning(
                    f"[Summary] Text extraction failed for: {Path(df.path).name}"
                )

        if not doc_texts:
            await self._logger.warning("[Summary] No text could be extracted from collected documents")
            return None

        await self._logger.info(
            f"[Summary] Loaded {len(doc_texts)} doc(s), "
            f"total {total_chars} chars"
        )

        needs_sampling = total_chars > self._SUMMARY_MAX_CONTEXT_CHARS
        per_file_budget = (
            self._SUMMARY_MAX_CONTEXT_CHARS // len(doc_texts)
            if needs_sampling else 0
        )

        parts: List[str] = []
        for fname, text in doc_texts:
            content = _sample_text(text, per_file_budget) if needs_sampling else text
            parts.append(f"#### File: {fname}\n```\n{content}\n```")

        combined = "\n\n".join(parts)

        if len(combined) <= self._SUMMARY_CHUNK_CHARS:
            return await self._llm_summarize_docs(combined, query)

        return await self._llm_chunked_summarize(combined, query)

    async def _llm_summarize_docs(self, documents: str, query: str) -> str:
        """Single-pass LLM summarization."""
        prompt = DOC_SUMMARY.format(documents=documents, user_input=query)
        resp = await self.llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        self.llm_usages.append(resp.usage)
        return resp.content or ""

    async def _llm_chunked_summarize(self, combined: str, query: str) -> str:
        """Multi-pass chunked summarization for large content."""
        chunk_size = self._SUMMARY_CHUNK_CHARS
        chunks = [
            combined[i:i + chunk_size]
            for i in range(0, len(combined), chunk_size)
        ]
        await self._logger.info(
            f"[Summary] Content exceeds single-pass limit — "
            f"splitting into {len(chunks)} chunk(s)"
        )

        partial_summaries: List[str] = []
        for idx, chunk in enumerate(chunks, 1):
            await self._logger.info(f"[Summary] Summarizing chunk {idx}/{len(chunks)}")
            prompt = DOC_CHUNK_SUMMARY.format(chunk=chunk, user_input=query)
            resp = await self.llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            self.llm_usages.append(resp.usage)
            if resp.content:
                partial_summaries.append(resp.content)

        if not partial_summaries:
            return ""
        if len(partial_summaries) == 1:
            return partial_summaries[0]

        merged_input = "\n\n---\n\n".join(
            f"**Part {i}**\n{s}" for i, s in enumerate(partial_summaries, 1)
        )
        prompt = DOC_MERGE_SUMMARIES.format(summaries=merged_input, user_input=query)
        resp = await self.llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        self.llm_usages.append(resp.usage)
        return resp.content or ""

    # ------------------------------------------------------------------
    # FAST mode — greedy search with early termination
    # ------------------------------------------------------------------

    _FAST_TEXT_EXTENSIONS = {
        ".txt", ".md", ".rst", ".csv", ".log", ".tsv",
        ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".xml",
        ".html", ".htm", ".sh", ".toml", ".cfg", ".ini", ".conf",
        ".css", ".bash", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    }
    _FAST_CONTEXT_WINDOW = 30  # ± lines around each grep hit
    _FAST_MAX_EVIDENCE_CHARS = 15_000
    _FAST_SMALL_FILE_THRESHOLD = 100_000  # 100K chars - read full file instead of grep sampling

    # --- Wiki-enhanced ranking constants ---
    _WIKI_BLEND_ALPHA = 0.7
    """TF-IDF weight in the hybrid score; Wiki weight = 1 - alpha."""
    _WIKI_MAX_SCORE = 10.0
    """Upper bound for the wiki relevance score."""
    _WIKI_CATALOG_KEYWORD_OVERLAP_MAX = 5.0
    """Maximum sub-score for catalog summary keyword overlap."""
    _WIKI_TREE_AVAILABILITY_BONUS = 2.0
    """Bonus for files that have a compiled tree index."""
    _WIKI_CATALOG_PRESENCE_FULL = 3.0
    """Catalog presence bonus for summaries > 100 chars."""
    _WIKI_CATALOG_PRESENCE_MEDIUM = 2.0
    """Catalog presence bonus for summaries > 30 chars."""
    _WIKI_CATALOG_PRESENCE_MINIMAL = 1.0
    """Catalog presence bonus for summaries > 0 chars."""
    _TREE_CACHE_SCAN_LIMIT = 200
    """Max tree JSON files to parse during artifact detection."""
    _CATALOG_LISTING_MAX_ENTRIES = 20
    """Max catalog entries in the enriched listing for Step 1."""
    _CATALOG_KEYWORD_MIN_LEN = 2
    """Minimum character length for a catalog keyword token."""
    _CATALOG_KEYWORD_MAX_LEN = 20
    """Maximum character length for a catalog keyword token."""
    _CATALOG_SUMMARY_TRUNCATE = 200
    """Max chars of catalog summary shown in the listing."""

    _LLM_FALLBACK_EVIDENCE = (
        "[No relevant documents found]\n\n"
        "The search did not find relevant content in the available documents. "
        "Please answer the user's question based on your own knowledge. "
        "Clearly indicate that this answer is from LLM knowledge, "
        "not from retrieved documents."
    )

    async def _search_fast(
        self,
        query: str,
        paths: List[str],
        *,
        max_depth: Optional[int] = 5,
        top_k_files: int = 3,
        enable_dir_scan: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        llm_fallback: bool = False,
    ) -> Tuple[str, Optional[KnowledgeCluster], SearchContext]:
        """Greedy search: 2-3 LLM calls, single best file, focused evidence.

        Two-level keyword cascade extracted in one LLM call:
        primary (compound phrase) is tried first; if it misses, fallback
        (atomic terms) is tried.  When ``enable_dir_scan`` is True and
        paths contain directories, a directory scan runs concurrently with
        keyword extraction and acts as a fallback retrieval path.

        Returns:
            ``(answer, cluster, context)`` — same triple as ``_search_deep``
            so the caller can handle both modes uniformly.
        """
        context = SearchContext()
        await self._logger.info(f"[FAST] Starting greedy search for: '{query[:80]}'")

        # --- Adaptive compile artifact detection (one-shot, zero LLM) ---
        artifacts = self._detect_compile_artifacts()
        if artifacts.catalog or artifacts.tree_available_paths:
            await self._logger.info(
                f"[FAST:Artifacts] catalog={'yes' if artifacts.catalog else 'no'} "
                f"({len(artifacts.catalog) if artifacts.catalog else 0} docs), "
                f"trees={len(artifacts.tree_available_paths)}"
            )

        # ==============================================================
        # Step 0: Cluster reuse — instant short-circuit (no LLM cost)
        # When reuse succeeds we return here; no persistence step runs.
        # ==============================================================
        reused = await self._try_reuse_cluster(query, paths)
        if reused is not None:
            await self._logger.success("[FAST] Reused cached knowledge cluster")
            return self._enrich_reused_content(reused), reused, context

        # P2: gradient reuse — structured hints from moderately similar clusters
        soft_hit = await self._try_soft_reuse(query, paths)

        # ==============================================================
        # Step 1: Fused LLM query analysis + document routing
        # When a compiled document catalog exists, the LLM sees all
        # document summaries and selects the most relevant ones in the
        # same call that extracts keywords (zero extra LLM cost).
        # ==============================================================
        catalog = artifacts.catalog
        catalog_routed_files: List[str] = []
        catalog_confidence: str = "low"

        if catalog:
            listing = self._build_enriched_catalog_listing(catalog)
            prompt = FAST_QUERY_ANALYSIS_WITH_CATALOG.format(
                user_input=query, document_listing=listing,
            )
        else:
            prompt = FAST_QUERY_ANALYSIS.format(user_input=query)

        resp = await self.llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        self.llm_usages.append(resp.usage)
        if resp.usage and isinstance(resp.usage, dict):
            context.add_llm_tokens(
                resp.usage.get("total_tokens", 0), usage=resp.usage,
            )

        analysis = self._parse_fast_json(resp.content)
        query_type = analysis.get("type", "search")
        file_hints = analysis.get("file_hints", [])

        # Extract catalog-routed files from the fused response
        if catalog:
            selected_indices = analysis.get("selected_docs", [])
            catalog_confidence = analysis.get("doc_confidence", "low")
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < len(catalog):
                    fp = catalog[idx]["path"]
                    if Path(fp).exists():
                        catalog_routed_files.append(fp)
            if catalog_routed_files:
                await self._logger.info(
                    f"[FAST:Step1] Catalog routing ({catalog_confidence}): "
                    f"{[Path(p).name for p in catalog_routed_files]}"
                )

        if query_type == "chat":
            chat_reply = analysis.get("response", "")
            if chat_reply:
                await self._logger.info("[FAST:Step1] LLM classified as chat intent")
                return chat_reply, None, context
            return (await self._respond_chat(query, context))

        if query_type == "summary":
            await self._logger.info("[FAST:Step1] Summary intent detected — delegating to doc analysis")
            # When user names a specific file, resolve it and skip dir scan + rank
            summary_paths: Optional[List[str]] = None
            if file_hints:
                summary_paths = self._resolve_file_hints(paths, file_hints)
                if summary_paths:
                    await self._logger.info(
                        f"[FAST:Summary] Resolved file hint(s) → {[Path(p).name for p in summary_paths]}"
                    )
            if summary_paths:
                answer = await self._summarize_documents(
                    query, summary_paths,
                    top_k_files=len(summary_paths),
                    scan_result=None,
                )
                if answer:
                    return answer, self._make_answer_cluster(query, answer, "FS", file_paths=summary_paths), context
            # No hint or resolve failed: run dir scan (if enabled) then rank + summarize
            scan_result = await self._probe_dir_scan(paths, enable=enable_dir_scan,
                                                     max_files=300) if enable_dir_scan else None
            answer = await self._summarize_documents(
                query, paths,
                top_k_files=top_k_files,
                scan_result=scan_result,
            )
            if answer:
                return answer, self._make_answer_cluster(query, answer, "FS", file_paths=paths), context
            await self._logger.info("[FAST:Step1] Summary fallback — no documents, continuing search")

        primary = analysis.get("primary", [])[:2]
        fallback = analysis.get("fallback", [])[:3]
        primary_alt = analysis.get("primary_alt", [])[:2]
        fallback_alt = analysis.get("fallback_alt", [])[:3]

        if primary_alt:
            primary = primary + primary_alt
        if fallback_alt:
            fallback = fallback + fallback_alt

        # --- IDF weights from LLM ---
        keyword_idfs: Dict[str, float] = analysis.get("idf", {})
        if not keyword_idfs:
            all_kws = (primary or []) + (fallback or [])
            keyword_idfs = {kw: max(0.5, min(1.0, len(kw) / 5.0)) for kw in all_kws}

        if not primary and not fallback:
            await self._logger.warning("[FAST] No keywords extracted")
            msg = f"Could not extract search terms from query: '{query}'"
            return msg, None, context

        # ==============================================================
        # Step 1.5: Compile-aware enrichment (P2 + P4, zero LLM calls)
        # Catalog-routed files from the fused Step 1 are merged here.
        # ==============================================================
        all_kw_set = set(primary + fallback)

        # P2: inject soft-hit patterns as fallback keywords
        if soft_hit:
            for p in soft_hit.patterns:
                if p not in all_kw_set:
                    fallback.append(p)
                    all_kw_set.add(p)
                    keyword_idfs.setdefault(p, 0.6)

        # P4: compile hints from manifest + tree cache
        compile_hints = await self._probe_compile_hints(primary + fallback)
        for kw in compile_hints.extra_keywords:
            if kw not in all_kw_set:
                fallback.append(kw)
                all_kw_set.add(kw)
                keyword_idfs.setdefault(kw, 0.5)

        compile_hint_files: List[str] = []
        # Catalog-routed files get highest priority
        seen_hint_paths: set = set()
        for fp in catalog_routed_files:
            if fp not in seen_hint_paths:
                seen_hint_paths.add(fp)
                compile_hint_files.append(fp)
        if soft_hit:
            for fp in soft_hit.file_paths:
                if fp not in seen_hint_paths:
                    seen_hint_paths.add(fp)
                    compile_hint_files.append(fp)
        for fp in compile_hints.file_paths:
            if fp not in seen_hint_paths:
                seen_hint_paths.add(fp)
                compile_hint_files.append(fp)

        if compile_hint_files:
            await self._logger.info(
                f"[FAST:Step1.5] Compile hints: {len(compile_hint_files)} files "
                f"(catalog={len(catalog_routed_files)}, soft={len(soft_hit.file_paths) if soft_hit else 0}), "
                f"{len(compile_hints.extra_keywords)} extra keywords"
            )

        await self._logger.info(
            f"[FAST:Step1] Primary: {primary}, Fallback: {fallback}"
        )

        # ==============================================================
        # Step 2: rga cascade — primary first, fallback only if needed
        # When catalog routing has high confidence, catalog-routed files
        # are used directly (skipping rga) to avoid noise from unrelated
        # files.  Otherwise rga runs first and catalog acts as fallback.
        # ==============================================================
        context.add_search(query)
        include_patterns = list(include or [])
        for hint in file_hints:
            if "*" in hint or "." in hint:
                include_patterns.append(hint)

        rga_kwargs = dict(
            paths=paths, max_depth=max_depth,
            include=include_patterns or None, exclude=exclude,
        )

        best_files: Optional[List[Dict[str, Any]]] = None
        used_level = "primary"
        evidence = ""
        file_path: Optional[str] = None  # set when best_files found

        # High-confidence catalog routing: skip rga, use catalog directly
        if catalog_routed_files and catalog_confidence == "high":
            used_level = "catalog_route"
            await self._logger.info(
                f"[FAST:Step2] High-confidence catalog routing → "
                f"{[Path(p).name for p in catalog_routed_files[:top_k_files]]}"
            )
            best_files = [
                {"path": p, "matches": [], "total_matches": 0, "weighted_score": 0.0}
                for p in catalog_routed_files[:top_k_files]
            ]

        if not best_files and primary:
            best_files = await self._fast_find_best_file(
                primary, top_k=top_k_files, keyword_idfs=keyword_idfs,
                query=query, artifacts=artifacts,
                **rga_kwargs,
            )

        if not best_files and fallback:
            used_level = "fallback"
            await self._logger.info(
                "[FAST:Step2] Primary miss, trying fine-grained fallback"
            )
            best_files = await self._fast_find_best_file(
                fallback, top_k=top_k_files, keyword_idfs=keyword_idfs,
                query=query, artifacts=artifacts,
                **rga_kwargs,
            )

        # --- Fallback: compile-hint files when rga misses (catalog + P2 + P4) ---
        if not best_files and compile_hint_files:
            used_level = "compile_hint"
            await self._logger.info(
                f"[FAST:Step2] rga miss — using {len(compile_hint_files)} compile-hint files"
            )
            best_files = [
                {"path": p, "matches": [], "total_matches": 0, "weighted_score": 0.0}
                for p in compile_hint_files[:top_k_files]
            ]

        # --- Fallback: use dir_scan only when rga misses and dir scan is enabled ---
        if not best_files and enable_dir_scan:
            scan_result = await self._probe_dir_scan(paths, enable=True, max_files=300)
            if scan_result is not None:
                await self._logger.info("[FAST:Step2] rga miss — falling back to dir_scan ranking")
                ranked_paths = await self._rank_dir_scan_candidates(
                    query, scan_result, top_k=10, include_medium=True,
                )
                if ranked_paths:
                    used_level = "dir_scan"
                    best_files = [{"path": p, "matches": [], "total_matches": 0, "weighted_score": 0.0} for p in ranked_paths[:top_k_files]]

        if not best_files:
            if llm_fallback:
                await self._logger.info(
                    "[FAST:Step2] No files found, llm_fallback=True \u2192 skip to LLM summary"
                )
                evidence = self._LLM_FALLBACK_EVIDENCE
            else:
                await self._logger.warning(
                    f"[FAST:Step2] No matching files found in paths: {paths}. "
                )
                return _NO_RESULTS_MESSAGE, None, context

        if best_files:
            file_path = best_files[0]["path"]
            match_objects = best_files[0].get("matches", [])
            wiki_info = ""
            if best_files[0].get("wiki_relevance") is not None:
                wiki_info = f", wiki={best_files[0]['wiki_relevance']:.1f}"
            await self._logger.info(
                f"[FAST:Step2] Best file ({used_level}): {Path(file_path).name} "
                f"({best_files[0].get('total_matches', 0)} hits, "
                f"score={best_files[0].get('weighted_score', 0):.2f}{wiki_info})"
            )

            # ==============================================================
            # Step 2.5 + Step 3: Tree navigation (1 LLM call) runs in
            # parallel with rga evidence sampling (0 LLM).  The merged
            # result is higher quality than either alone.
            # ==============================================================

            async def _rga_evidence() -> str:
                """Collect rga-based evidence from best_files (zero LLM)."""
                parts: List[str] = []
                chars = 0
                for bf in best_files:
                    if chars >= self._FAST_MAX_EVIDENCE_CHARS:
                        break
                    fp = bf["path"]
                    fn = Path(fp).name
                    ext = Path(fp).suffix.lower()
                    ev = None
                    if ext in self._FAST_TEXT_EXTENSIONS:
                        try:
                            sz = Path(fp).stat().st_size
                            if sz < self._FAST_SMALL_FILE_THRESHOLD:
                                full = Path(fp).read_text(errors="replace")
                                if len(full) < self._FAST_SMALL_FILE_THRESHOLD:
                                    ev = f"[{fn}]\n{full}"
                        except Exception:
                            pass
                    if ev is None:
                        ev = await self._fast_sample_evidence(fp, bf.get("matches", []))
                    if ev:
                        remaining = self._FAST_MAX_EVIDENCE_CHARS - chars
                        parts.append(ev[:remaining])
                        chars += len(parts[-1])
                        context.mark_file_read(fp)
                return "\n\n---\n\n".join(parts)

            # Launch tree navigation for the primary file alongside rga
            tree_nav_target = best_files[0]["path"]
            rga_task = _rga_evidence()
            tree_task = self._navigate_tree_for_evidence(tree_nav_target, query)

            rga_ev, tree_ev = await asyncio.gather(rga_task, tree_task)

            # Merge: tree evidence first (highest quality), then rga
            evidence_parts_final: List[str] = []
            if tree_ev:
                evidence_parts_final.append(tree_ev)
            if rga_ev:
                evidence_parts_final.append(rga_ev)
            evidence = "\n\n---\n\n".join(evidence_parts_final)

            if not evidence or len(evidence.strip()) < 20:
                if llm_fallback:
                    await self._logger.info(
                        "[FAST:Step3] No usable evidence, llm_fallback=True → LLM summary"
                    )
                    evidence = self._LLM_FALLBACK_EVIDENCE
                else:
                    await self._logger.warning("[FAST:Step3] No usable evidence extracted")
                    return _NO_RESULTS_MESSAGE, None, context

            tree_available = file_path in artifacts.tree_available_paths if artifacts else False
            await self._logger.info(
                f"[FAST:Step3] Evidence: {len(evidence)} chars "
                f"(tree={'yes' if tree_ev else 'no'}, rga={'yes' if rga_ev else 'no'}, "
                f"tree_indexed={'yes' if tree_available else 'no'})"
            )

        keywords_used = primary if used_level == "primary" else fallback

        # ==============================================================
        # Step 4: LLM answer from focused evidence (single call)
        # Wiki-enhanced: inject document context when catalog available.
        # ==============================================================
        doc_context = self._build_answer_context(file_path, artifacts) if best_files else None
        if doc_context:
            from sirchmunk.llm.prompts import ROI_RESULT_SUMMARY_WITH_CONTEXT
            answer_prompt = ROI_RESULT_SUMMARY_WITH_CONTEXT.format(
                user_input=query,
                text_content=evidence,
                document_context=doc_context,
            )
            await self._logger.info(
                f"[FAST:Step4] Wiki-enhanced answer generation with catalog context"
            )
        else:
            answer_prompt = ROI_RESULT_SUMMARY.format(
                user_input=query,
                text_content=evidence,
            )
        answer_resp = await self.llm.achat(
            messages=[{"role": "user", "content": answer_prompt}],
            stream=True,
        )
        self.llm_usages.append(answer_resp.usage)
        if answer_resp.usage and isinstance(answer_resp.usage, dict):
            context.add_llm_tokens(
                answer_resp.usage.get("total_tokens", 0), usage=answer_resp.usage,
            )

        answer, should_save, should_answer = self._parse_summary_response(
            answer_resp.content or ""
        )

        # ==============================================================
        # Step 5: Self-correction retry (conditional, ≤1 extra LLM call)
        # When the answer gate rejects the first attempt, try alternative
        # evidence sources before giving up.
        # ==============================================================
        if not should_answer:
            retry_evidence = await self._fast_self_correct(
                query, best_files, catalog_routed_files, context,
            )
            if retry_evidence:
                await self._logger.info(
                    f"[FAST:Step5] Retrying with {len(retry_evidence)} chars of alternative evidence"
                )
                retry_prompt = ROI_RESULT_SUMMARY.format(
                    user_input=query, text_content=retry_evidence,
                )
                retry_resp = await self.llm.achat(
                    messages=[{"role": "user", "content": retry_prompt}],
                    stream=True,
                )
                self.llm_usages.append(retry_resp.usage)
                if retry_resp.usage and isinstance(retry_resp.usage, dict):
                    context.add_llm_tokens(
                        retry_resp.usage.get("total_tokens", 0), usage=retry_resp.usage,
                    )
                answer, should_save, should_answer = self._parse_summary_response(
                    retry_resp.content or ""
                )

        if not should_answer:
            if llm_fallback:
                await self._logger.info(
                    "[FAST:Step5] Retry also rejected, llm_fallback=True → LLM fallback"
                )
                answer, should_save = await self._summarise_fast_fallback(query, context)
            else:
                await self._logger.warning(
                    "[FAST:Step5] Evidence rejected after retry, llm_fallback=False "
                    "→ returning no results"
                )
                return _NO_RESULTS_MESSAGE, None, context

        if not should_save:
            await self._logger.info("[FAST] Quality gate: low-quality answer, skipping cluster save")
            await self._logger.success("[FAST] Search complete (no persist)")
            return answer, None, context

        cluster = self._build_fast_cluster(
            query, answer, file_path or "", evidence, keywords_used,
        )
        self._add_query_to_cluster(cluster, query)
        try:
            await self._save_cluster_with_embedding(cluster)
        except Exception as exc:
            _loguru_logger.warning(
                f"[FAST] Failed to save cluster with embedding: {exc}"
            )

        await self._logger.success("[FAST] Search complete")
        return answer, cluster, context

    # ---- FAST helpers ----

    @staticmethod
    def _count_keyword_tf_per_file(raw_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count matches per file from rga JSON output."""
        counts: Dict[str, int] = {}
        current_path: Optional[str] = None
        for item in raw_results:
            item_type = item.get("type")
            if item_type == "begin":
                current_path = item.get("data", {}).get("path", {}).get("text")
            elif item_type == "match" and current_path is not None:
                counts[current_path] = counts.get(current_path, 0) + 1
            elif item_type == "end":
                current_path = None
        return counts

    @staticmethod
    def _dedup_merged_files(
        merged: List[Dict[str, Any]],
        per_file_kw_tf: Dict[str, Dict[str, int]],
        match_limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Deduplicate merged file entries by path, combining matches from
        multiple keyword searches into a single entry per file.

        When the same file appears in multiple rga begin/end groups (one per
        keyword search), this merges them so downstream scoring and evidence
        extraction operate on a single, complete representation.

        Args:
            merged: File entries from GrepRetriever.merge_results(), may
                contain duplicates.
            per_file_kw_tf: Pre-computed per-file keyword TF counts (not
                modified, used only for reference).
            match_limit: Maximum matches to keep per file after merging.

        Returns:
            Deduplicated list with one entry per unique file path.
        """
        if not merged:
            return merged

        seen: Dict[str, int] = {}  # path -> index in deduped
        deduped: List[Dict[str, Any]] = []

        for entry in merged:
            fpath = entry["path"]
            if fpath in seen:
                # Merge into existing entry
                idx = seen[fpath]
                existing = deduped[idx]
                existing["matches"].extend(entry.get("matches", []))
                existing["lines"].extend(entry.get("lines", []))
                existing["total_matches"] += entry.get("total_matches", 0)
            else:
                # New file — clone to avoid mutating original
                seen[fpath] = len(deduped)
                deduped.append({
                    "path": fpath,
                    "matches": list(entry.get("matches", [])),
                    "lines": list(entry.get("lines", [])),
                    "total_matches": entry.get("total_matches", 0),
                    "total_score": entry.get("total_score", 0.0),
                })

        # Trim matches to limit per file
        for entry in deduped:
            if len(entry["matches"]) > match_limit:
                # Sort by score descending, keep top
                entry["matches"].sort(
                    key=lambda x: x.get("score", 0.0), reverse=True
                )
                entry["matches"] = entry["matches"][:match_limit]

        return deduped

    @staticmethod
    def _prune_by_score(
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        relative_ratio: float = 0.30,
        gap_ratio: float = 0.50,
        min_count: int = 1,
    ) -> List[Dict[str, Any]]:
        """Dynamically prune ranked file candidates by score distribution.

        Applies a three-stage filter to remove clearly irrelevant files:

        1. **Relative threshold**: Discard files scoring below
           ``max_score * relative_ratio`` (default 30%).
        2. **Gap detection**: Scan adjacently ranked files; when the score
           drop from one to the next exceeds ``prev_score * gap_ratio``
           (default 50%), truncate the list at that point.
        3. **Minimum guarantee**: Ensure at least ``min_count`` files
           survive (default 1).

        Finally the result is capped at ``top_k``.

        Args:
            candidates: File dicts sorted by ``weighted_score`` descending.
            top_k: Maximum number of files to return.
            relative_ratio: Fraction of the top score used as a floor.
            gap_ratio: Maximum tolerated relative drop between adjacent
                candidates.
            min_count: Minimum number of candidates to keep regardless of
                score.

        Returns:
            Pruned list of candidates (length in [min_count, top_k]).
        """
        if not candidates:
            return []

        max_score = candidates[0].get("weighted_score", 0.0)

        # Step 1: Relative threshold filter
        threshold = max_score * relative_ratio
        filtered = [f for f in candidates if f.get("weighted_score", 0.0) >= threshold]
        if not filtered:
            filtered = candidates[:min_count]

        # Step 2: Gap detection truncation
        result = [filtered[0]]
        for i in range(1, len(filtered)):
            prev_score = filtered[i - 1].get("weighted_score", 0.0)
            curr_score = filtered[i].get("weighted_score", 0.0)
            if prev_score > 0 and (prev_score - curr_score) > prev_score * gap_ratio:
                break
            result.append(filtered[i])

        # Step 3: Minimum guarantee
        if len(result) < min_count and len(filtered) >= min_count:
            result = filtered[:min_count]

        # Cap at top_k
        return result[:top_k]

    @staticmethod
    def _compute_wiki_relevance(
        file_path: str,
        query: str,
        keywords: List[str],
        catalog_map: Dict[str, Dict[str, str]],
        tree_available_paths: Set[str],
    ) -> float:
        """Compute wiki-based relevance score for a candidate file (0-10 scale).

        Uses three sub-scores derived from compile artifacts:

        1. **Catalog summary overlap** (0-``_WIKI_CATALOG_KEYWORD_OVERLAP_MAX``):
           proportion of query keywords that appear in the catalog entry's
           summary.  When *keywords* is empty, falls back to whole-query
           substring matching against the summary to avoid returning 0 for
           valid queries.
        2. **Tree availability bonus** (0-``_WIKI_TREE_AVAILABILITY_BONUS``):
           a file with a compiled tree index likely has rich structure.
        3. **Catalog presence bonus** (0-``_WIKI_CATALOG_PRESENCE_FULL``):
           files important enough to be in the catalog get a baseline boost.

        All scoring is pure text matching — no LLM, no embedding.

        Args:
            file_path: Absolute path of the candidate file.
            query: Original user query.
            keywords: Extracted search keywords from FAST Step 1.
            catalog_map: ``{path: catalog_entry}`` from CompileArtifacts.
            tree_available_paths: Set of file paths with cached tree indices.

        Returns:
            Float in [0, 10] representing wiki-derived relevance.
        """
        cls = AgenticSearch  # access class constants from static method
        score = 0.0

        entry = catalog_map.get(file_path)

        # Sub-score 1: Catalog summary keyword overlap
        if entry:
            summary_lower = (entry.get("summary", "") + " " + entry.get("name", "")).lower()
            query_lower = query.lower()
            matches = 0
            total = 0
            for kw in keywords:
                if kw:
                    total += 1
                    if kw.lower() in summary_lower:
                        matches += 1
            # Also check whole query as a substring
            if len(query_lower) >= 2 and query_lower in summary_lower:
                matches += 1
                total += 1
            # When keywords list is empty but query is non-empty, fall back to
            # character-level overlap so the sub-score is not silently 0.
            if total == 0 and query_lower:
                # Simple overlap: count how many query chars appear in summary
                overlap = sum(1 for ch in query_lower if ch in summary_lower)
                ratio = overlap / max(len(query_lower), 1)
                score += ratio * cls._WIKI_CATALOG_KEYWORD_OVERLAP_MAX
            elif total > 0:
                score += (matches / total) * cls._WIKI_CATALOG_KEYWORD_OVERLAP_MAX

        # Sub-score 2: Tree availability bonus
        if file_path in tree_available_paths:
            score += cls._WIKI_TREE_AVAILABILITY_BONUS

        # Sub-score 3: Catalog presence bonus
        if entry:
            summary_len = len(entry.get("summary", ""))
            if summary_len > 100:
                score += cls._WIKI_CATALOG_PRESENCE_FULL
            elif summary_len > 30:
                score += cls._WIKI_CATALOG_PRESENCE_MEDIUM
            elif summary_len > 0:
                score += cls._WIKI_CATALOG_PRESENCE_MINIMAL

        return min(score, cls._WIKI_MAX_SCORE)

    async def _fast_find_best_file(
        self,
        keywords: List[str],
        paths: List[str],
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        top_k: int = 1,
        keyword_idfs: Optional[Dict[str, float]] = None,
        query: str = "",
        artifacts: Optional["CompileArtifacts"] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Search per keyword via rga and return the top-k best-matching files
        ranked by IDF-weighted log-TF scoring, optionally enhanced with
        wiki-derived relevance from compile artifacts.

        Args:
            keywords: Search keywords from FAST Step 1.
            paths: Search paths.
            max_depth: Maximum directory depth for rga.
            include: Glob patterns to include.
            exclude: Glob patterns to exclude.
            top_k: Number of top files to return.
            keyword_idfs: Pre-computed IDF values for keywords.
            query: Original user query (used for wiki relevance scoring).
            artifacts: Compile artifacts for adaptive wiki-enhanced ranking.

        Returns:
            List of merged file dicts (path, matches, lines, total_matches, weighted_score) or None.
        """
        all_raw: List[Dict[str, Any]] = []
        per_file_kw_tf: Dict[str, Dict[str, int]] = {}  # {file_path: {keyword: count}}

        for kw in keywords:
            try:
                results = await self.grep_retriever.retrieve(
                    terms=kw, path=paths, literal=True, regex=False,
                    max_depth=max_depth, include=include, exclude=exclude,
                    timeout=30.0,
                )
                if results:
                    all_raw.extend(results)
                    # Track per-file TF for this keyword
                    kw_counts = self._count_keyword_tf_per_file(results)
                    for fpath, count in kw_counts.items():
                        per_file_kw_tf.setdefault(fpath, {})[kw] = count
            except Exception as exc:
                await self._logger.warning(
                    f"[FAST] rga literal search failed for '{kw}': {exc}"
                )

        # Fallback: escaped-regex OR (handles adapters that only work in regex mode)
        if not all_raw and keywords:
            try:
                escaped = [re.escape(kw) for kw in keywords]
                pattern = "|".join(escaped)
                results = await self.grep_retriever.retrieve(
                    terms=pattern, path=paths, literal=False, regex=True,
                    max_depth=max_depth, include=include, exclude=exclude,
                    timeout=30.0,
                )
                if results:
                    all_raw.extend(results)
                    # For regex OR fallback, attribute matches to individual keywords
                    # by checking which keywords appear in each match line
                    # (simplified: count total matches per file, distribute proportionally)
                    regex_counts = self._count_keyword_tf_per_file(results)
                    for fpath, count in regex_counts.items():
                        # Attribute to all keywords equally (approximation for OR regex)
                        per_kw_share = max(1, count // len(keywords)) if keywords else count
                        for kw in keywords:
                            existing = per_file_kw_tf.get(fpath, {}).get(kw, 0)
                            if existing == 0:  # Only fill if not already set by literal search
                                per_file_kw_tf.setdefault(fpath, {})[kw] = per_kw_share
            except Exception as exc:
                await self._logger.warning(
                    f"[FAST] rga regex search failed: {exc}"
                )

        # Fallback: filename search
        if not all_raw:
            try:
                fn_results = await self.grep_retriever.retrieve_by_filename(
                    patterns=[f".*{re.escape(kw)}.*" for kw in keywords],
                    path=paths, case_sensitive=False, max_depth=max_depth,
                    timeout=30.0,
                )
                if fn_results:
                    return [{"path": fn_results[0]["path"], "matches": [], "lines": [], "total_matches": 0, "weighted_score": 0.0}]
            except Exception as exc:
                await self._logger.warning(
                    f"[FAST] filename search failed: {exc}"
                )
            return None

        merged = GrepRetriever.merge_results(all_raw, limit=20)
        if not merged:
            return None

        # Deduplicate file entries from multi-keyword searches
        merged = self._dedup_merged_files(merged, per_file_kw_tf)

        # --- IDF × (1 + log TF) weighted scoring ---
        _idfs = keyword_idfs or {}
        for f in merged:
            fpath = f["path"]
            kw_tf = per_file_kw_tf.get(fpath, {})
            score = 0.0
            for kw in keywords:
                tf = kw_tf.get(kw, 0)
                if tf > 0:
                    idf = _idfs.get(kw, max(0.5, min(1.0, len(kw) / 5.0)))
                    score += idf * (1.0 + math.log(tf))
            f["weighted_score"] = score

        # --- Wiki-enhanced hybrid scoring (adaptive: only when artifacts exist) ---
        if artifacts and artifacts.catalog_map:
            # Normalize TF-IDF scores to [0, 10] to align with Wiki score range
            max_tf_idf = max((f["weighted_score"] for f in merged), default=1.0)
            if max_tf_idf <= 0:
                max_tf_idf = 1.0
            for f in merged:
                wiki_score = self._compute_wiki_relevance(
                    f["path"], query, keywords,
                    artifacts.catalog_map, artifacts.tree_available_paths,
                )
                f["wiki_relevance"] = wiki_score
                # Normalize TF-IDF to [0, 10] before blending
                tf_idf_norm = (f["weighted_score"] / max_tf_idf) * self._WIKI_MAX_SCORE
                f["weighted_score"] = (
                    self._WIKI_BLEND_ALPHA * tf_idf_norm
                    + (1 - self._WIKI_BLEND_ALPHA) * wiki_score
                )

        merged.sort(key=lambda f: f["weighted_score"], reverse=True)
        pruned = self._prune_by_score(merged, top_k=top_k)

        return pruned if pruned else None

    async def _fast_sample_evidence(
        self,
        file_path: str,
        match_objects: List[Dict[str, Any]],
    ) -> str:
        """Build focused evidence from grep hits: context windows for text
        files, raw match snippets for binary formats.

        Args:
            file_path: Absolute path to the best file.
            match_objects: Match event dicts from ``merge_results``.

        Returns:
            Formatted evidence string.
        """
        fname = Path(file_path).name
        ext = Path(file_path).suffix.lower()

        # Extract match line numbers
        hit_lines: List[int] = []
        for m in match_objects:
            ln = m.get("data", {}).get("line_number")
            if isinstance(ln, int):
                hit_lines.append(ln)

        # Diagnostic logging when falling back to snippet mode
        if not hit_lines and match_objects:
            await self._logger.warning(
                f"[FAST] No line_number in {len(match_objects)} match(es) for {fname}, "
                f"falling back to snippet mode"
            )

        # --- Text files: read context windows around hits ---
        if ext in self._FAST_TEXT_EXTENSIONS and hit_lines:
            # Expand context window for sparse hits
            window = self._FAST_CONTEXT_WINDOW
            if len(hit_lines) <= 2:
                window = max(window, 100)  # ±100 lines for 1-2 hits
            evidence = self._read_context_windows(
                file_path, hit_lines,
                window=window,
                max_chars=self._FAST_MAX_EVIDENCE_CHARS,
            )
            if evidence:
                full_evidence = f"[{fname}]\n{evidence}"
                if len(full_evidence) < 100:
                    await self._logger.info(
                        f"[FAST] Context window evidence too thin ({len(full_evidence)} chars) for {fname}, "
                        f"attempting file head extraction"
                    )
                    head_evidence = await self._fast_read_file_head(file_path)
                    if head_evidence and len(head_evidence) > len(full_evidence):
                        return head_evidence
                return full_evidence

        # --- Non-text files or no line numbers: use grep snippets ---
        snippets: List[str] = []
        total = 0
        for m in match_objects:
            line_text = m.get("data", {}).get("lines", {}).get("text", "").rstrip()
            if not line_text:
                continue
            snippets.append(line_text)
            total += len(line_text)
            if total >= self._FAST_MAX_EVIDENCE_CHARS:
                break

        if snippets:
            snippet_evidence = f"[{fname}]\n" + "\n".join(snippets)
            # If snippet evidence is too thin, try file head for richer context
            if len(snippet_evidence) < 100:
                await self._logger.info(
                    f"[FAST] Evidence too thin ({len(snippet_evidence)} chars) for {fname}, "
                    f"attempting file head extraction"
                )
                head_evidence = await self._fast_read_file_head(file_path)
                if head_evidence and len(head_evidence) > len(snippet_evidence):
                    return head_evidence
            return snippet_evidence

        # Last resort: try reading file head
        return await self._fast_read_file_head(file_path)

    @staticmethod
    def _read_context_windows(
        file_path: str,
        hit_lines: List[int],
        window: int = 30,
        max_chars: int = 15_000,
    ) -> Optional[str]:
        """Read context windows around *hit_lines* from a text file.

        Merges overlapping windows to avoid duplication.  Stops when
        *max_chars* is reached.
        """
        # Merge overlapping intervals
        intervals = sorted(set(
            (max(1, ln - window), ln + window) for ln in hit_lines
        ))
        merged: List[tuple] = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        # Read file and extract windows
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except Exception:
            return None

        parts: List[str] = []
        total = 0
        for start, end in merged:
            s = max(0, start - 1)  # 0-indexed
            e = min(len(all_lines), end)
            chunk = "".join(all_lines[s:e])
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    chunk = chunk[:remaining] + "\n[...truncated...]"
                    parts.append(chunk)
                break
            parts.append(chunk)
            total += len(chunk)

        if not parts:
            return None

        # Join windows with separator when there are gaps
        return "\n[...]\n".join(parts)

    @classmethod
    async def _fast_read_file_head(
        cls, file_path: str, max_chars: int = 8_000,
    ) -> str:
        """Read the head of a file as last-resort evidence."""
        try:
            p = Path(file_path)
            if p.suffix.lower() in cls._FAST_TEXT_EXTENSIONS:
                text = p.read_text(encoding="utf-8", errors="replace")
            else:
                from sirchmunk.utils.file_utils import fast_extract
                result = await fast_extract(file_path)
                text = result.content if result and result.content else ""
            if text:
                return f"[{p.name}]\n{text[:max_chars]}"
        except Exception:
            pass
        return ""

    def _load_document_catalog(self) -> Optional[List[Dict[str, str]]]:
        """Load the compiled document catalog for fused query+route prompt.

        Returns None when compile has not been run or catalog is missing.
        """
        catalog_path = self.work_path / ".cache" / "compile" / "document_catalog.json"
        if not catalog_path.exists():
            return None
        try:
            entries = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(entries, list) and entries:
                return entries
        except Exception:
            pass
        return None

    def _detect_compile_artifacts(self) -> CompileArtifacts:
        """One-shot probe of all compile artifacts for adaptive FAST activation.

        Reads the document catalog and scans the tree cache directory to
        determine which compile products are available.  Called once at the
        start of ``_search_fast()``; the result is passed to downstream
        helpers so they can enable enhanced logic only when artifacts exist.

        Cost: one JSON read (catalog) + one directory listing (tree cache).
        Tree path results are cached in ``_tree_paths_cache`` so subsequent
        calls within the same instance avoid re-parsing every JSON file.
        Returns a ``CompileArtifacts`` with ``None``/empty fields when
        compile has not been run.
        """
        catalog = self._load_document_catalog()
        catalog_map: Dict[str, Dict[str, str]] = {}
        if catalog:
            for entry in catalog:
                p = entry.get("path", "")
                if p:
                    catalog_map[p] = entry

        indexer = self._get_tree_indexer()
        # Use cached tree paths when available to avoid re-parsing all JSONs
        tree_paths: Set[str] = getattr(self, "_tree_paths_cache", None) or set()
        if indexer is not None and not tree_paths:
            tree_cache = self.work_path / ".cache" / "compile" / "trees"
            if tree_cache.exists():
                try:
                    from sirchmunk.learnings.tree_indexer import DocumentTree
                    for tf in sorted(tree_cache.glob("*.json"))[:self._TREE_CACHE_SCAN_LIMIT]:
                        try:
                            tree = DocumentTree.from_json(
                                tf.read_text(encoding="utf-8")
                            )
                            if tree.file_path:
                                tree_paths.add(tree.file_path)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Cache for future calls within this instance
            self._tree_paths_cache = tree_paths

        return CompileArtifacts(
            catalog=catalog,
            catalog_map=catalog_map,
            tree_indexer=indexer,
            tree_available_paths=tree_paths,
        )

    @staticmethod
    def _extract_catalog_keywords(summary: str, max_kw: int = 3) -> List[str]:
        """Extract salient keywords from a catalog summary via simple heuristics.

        Uses word-length filtering, Chinese character detection, and CJK n-gram
        extraction to pick the most informative tokens.  For CJK-heavy text
        (which does not use whitespace word boundaries), consecutive CJK
        character runs are extracted as additional candidate tokens.

        No LLM or embedding involved.

        Args:
            summary: Document summary text from the compiled catalog.
            max_kw: Maximum number of keywords to return.

        Returns:
            List of up to *max_kw* keywords.
        """
        cls = AgenticSearch
        if max_kw <= 0:
            return []
        summary_text = str(summary or "").strip()
        if not summary_text:
            return []
        import re as _re

        # Split on whitespace and common punctuation (incl. CJK punctuation)
        tokens = _re.split(
            r'[\s,;\uff0c\uff1b\u3001\u3002\uff1a:!?\uff01\uff1f()\[\]{}\u201c\u201d\u2018\u2019\u0022\u0027/\\|`~@#$%^&*=+<>]+',
            summary_text,
        )

        # For CJK text, also extract consecutive CJK character runs (2-6 chars)
        # so that e.g. "停车位申请条件" yields ["停车位申请条件", "停车位", "申请条件", ...]
        cjk_runs = _re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]{2,}', summary_text)
        # Generate sub-phrases from long CJK runs (bigrams/trigrams/4-grams)
        cjk_ngrams: List[str] = []
        max_ngram_per_run = 40
        for run in cjk_runs:
            cjk_ngrams.append(run)
            if len(run) > 4:
                # Extract 2-4 char sub-phrases from each run
                added = 0
                for n in (4, 3, 2):
                    for i in range(len(run) - n + 1):
                        cjk_ngrams.append(run[i:i + n])
                        added += 1
                        if added >= max_ngram_per_run:
                            break
                    if added >= max_ngram_per_run:
                        break

        tokens = tokens + cjk_ngrams

        # Filter: keep tokens with appropriate length and not purely numeric
        candidates = [
            t for t in tokens
            if t
            and len(t) >= cls._CATALOG_KEYWORD_MIN_LEN
            and not t.isdigit()
            and len(t) <= cls._CATALOG_KEYWORD_MAX_LEN
            and not _re.fullmatch(r"[_\-.]+", t)
        ]
        # Prefer longer tokens (more specific)
        candidates.sort(key=len, reverse=True)
        # Deduplicate case-insensitively
        seen: Set[str] = set()
        chosen_norms: List[str] = []
        result: List[str] = []
        for c in candidates:
            lower = c.lower()
            if lower not in seen:
                # Avoid noisy micro-fragments when a longer token already exists.
                if len(lower) <= 4 and any(lower in kept for kept in chosen_norms):
                    continue
                seen.add(lower)
                chosen_norms.append(lower)
                result.append(c)
            if len(result) >= max_kw:
                break
        return result

    def _build_enriched_catalog_listing(
        self,
        catalog: List[Dict[str, str]],
        max_entries: Optional[int] = None,
    ) -> str:
        """Build an enriched catalog listing with keywords for FAST Step 1.

        Compared to the plain ``[i] name: summary[:200]`` format, this adds
        extracted keywords to help the LLM make more informed document
        selections.

        Args:
            catalog: Entries from ``document_catalog.json``.
            max_entries: Cap to prevent prompt overflow.

        Returns:
            Formatted listing string for injection into the FAST query
            analysis prompt.
        """
        if not isinstance(catalog, list) or not catalog:
            return ""
        lines: List[str] = []
        _max = max_entries if max_entries is not None else self._CATALOG_LISTING_MAX_ENTRIES
        if _max <= 0:
            return ""
        _trunc = self._CATALOG_SUMMARY_TRUNCATE
        for i, entry in enumerate(catalog[:_max]):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("path") or "")
            summary = str(entry.get("summary") or "")
            # Keep one-line prompt entries to avoid accidental prompt pollution.
            name = " ".join(name.split())
            summary = " ".join(summary.split())
            if not name:
                name = f"doc_{i}"
            kws = AgenticSearch._extract_catalog_keywords(summary)
            kw_str = ", ".join(kws) if kws else ""
            shown_summary = summary[:_trunc]
            if len(summary) > _trunc:
                shown_summary += "..."
            if kw_str:
                lines.append(f"[{i}] {name}: {shown_summary}  [Keywords: {kw_str}]")
            else:
                lines.append(f"[{i}] {name}: {shown_summary}")
        return "\n".join(lines)

    def _build_answer_context(
        self,
        best_file_path: str,
        artifacts: CompileArtifacts,
    ) -> Optional[str]:
        """Build document context from catalog for wiki-enhanced answer generation.

        Returns a short context string describing the source document, or
        None when no catalog entry exists for *best_file_path*.

        Args:
            best_file_path: Path of the top-ranked file from Step 2.
            artifacts: Compile artifact availability context.

        Returns:
            Context string or None.
        """
        if not artifacts.catalog_map:
            return None
        entry = artifacts.catalog_map.get(best_file_path)
        if not entry:
            return None
        name = entry.get("name", Path(best_file_path).name)
        summary = entry.get("summary", "")
        if not summary:
            return None
        return f"Source Document: {name}\nDocument Overview: {summary}"

    async def _navigate_tree_for_evidence(
        self, file_path: str, query: str,
    ) -> Optional[str]:
        """LLM-driven tree navigation: select relevant sections and read leaf content.

        Uses 1 LLM call to drill into the compiled tree index for
        *file_path*, returning concatenated leaf content as evidence.
        Returns None when no tree cache is available.
        """
        indexer = self._get_tree_indexer()
        if indexer is None:
            return None
        tree = indexer.load_tree(file_path)
        if tree is None or tree.root is None:
            return None

        try:
            leaves = await indexer.navigate(tree, query, max_results=3)
        except Exception:
            return None

        if not leaves:
            return None

        fname = Path(file_path).name
        # Read leaf content from the original document via char_range
        parts: List[str] = []
        try:
            from sirchmunk.utils.file_utils import fast_extract
            extraction = await fast_extract(file_path=file_path)
            full_text = extraction.content or ""
        except Exception:
            full_text = ""

        for leaf in leaves:
            start, end = leaf.char_range
            if full_text and end > start:
                segment = full_text[start:end]
            else:
                segment = leaf.summary or ""
            if segment.strip():
                header = f"[{fname} → {leaf.title}]"
                parts.append(f"{header}\n{segment[:3000]}")

        if not parts:
            return None

        evidence = "\n\n".join(parts)
        await self._logger.info(
            f"[FAST:TreeNav] Extracted {len(parts)} sections, "
            f"{len(evidence)} chars from {fname}"
        )
        return evidence

    async def _fast_self_correct(
        self,
        query: str,
        best_files: Optional[List[Dict[str, Any]]],
        catalog_routed_files: List[str],
        context: SearchContext,
    ) -> Optional[str]:
        """Attempt to gather alternative evidence when the first answer is rejected.

        Three strategies tried in order:
        A) Tree-navigate a 2nd catalog-routed file not yet tried.
        B) Retrieve the most semantically similar compiled cluster's content.
        C) Tree-navigate the 2nd-best rga file if available.

        Returns alternative evidence string, or None if all strategies fail.
        """
        first_file = best_files[0]["path"] if best_files else ""

        # Strategy A: 2nd catalog-routed file via tree navigation
        for fp in catalog_routed_files:
            if fp == first_file:
                continue
            tree_ev = await self._navigate_tree_for_evidence(fp, query)
            if tree_ev and len(tree_ev.strip()) > 50:
                context.mark_file_read(fp)
                return tree_ev

        # Strategy B: cluster content from knowledge storage
        if self.embedding_client and self.knowledge_storage:
            try:
                qe = self.embedding_client.encode(query)
                if qe is not None:
                    vec = qe.tolist() if hasattr(qe, "tolist") else list(qe)
                    hits = await self.knowledge_storage.search_similar_clusters(
                        query_embedding=vec, top_k=2, similarity_threshold=0.50,
                    )
                    if hits:
                        parts: List[str] = []
                        for h in hits[:2]:
                            c = await self.knowledge_storage.get(h["id"])
                            if c and c.content:
                                parts.append(str(c.content)[:3000])
                                for ev in (c.evidences or [])[:3]:
                                    for s in (ev.snippets or [])[:2]:
                                        parts.append(s[:500])
                        if parts:
                            return "\n\n---\n\n".join(parts)
            except Exception:
                pass

        # Strategy C: 2nd rga file via tree navigation
        if best_files and len(best_files) > 1:
            fp2 = best_files[1]["path"]
            tree_ev = await self._navigate_tree_for_evidence(fp2, query)
            if tree_ev and len(tree_ev.strip()) > 50:
                context.mark_file_read(fp2)
                return tree_ev

        return None

    @staticmethod
    def _parse_fast_json(text: str) -> Dict[str, Any]:
        """Extract JSON from the FAST query analysis LLM response."""
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    # ------------------------------------------------------------------
    # Phase 1 probes (each designed to run concurrently)
    # ------------------------------------------------------------------

    async def _probe_keywords(
        self, query: str,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Extract multi-level keywords from the query via LLM.

        Also extracts cross-lingual alternative keywords from the
        ``<KEYWORDS_ALT>`` block and merges them into the result list.

        Returns:
            Tuple of (keyword_idf_dict, keyword_list).
        """
        await self._logger.info("[Probe:Keywords] Extracting keywords...")
        dynamic_prompt = generate_keyword_extraction_prompt(num_levels=2)
        keyword_prompt = dynamic_prompt.replace(KEYWORD_QUERY_PLACEHOLDER, query)
        kw_response = await self.llm.achat(
            messages=[{"role": "user", "content": keyword_prompt}],
            stream=False,
        )
        self.llm_usages.append(kw_response.usage)

        keyword_sets = self._extract_and_validate_multi_level_keywords(
            kw_response.content, num_levels=2,
        )

        alt_keywords = self._extract_alt_keywords(kw_response.content)
        if alt_keywords:
            await self._logger.info(f"[Probe:Keywords] Cross-lingual alt: {list(alt_keywords.keys())}")

        for kw_set in keyword_sets:
            if kw_set:
                merged = {**kw_set, **alt_keywords}
                kw_list = list(merged.keys())
                await self._logger.info(f"[Probe:Keywords] Extracted: {kw_list}")
                return merged, kw_list

        if alt_keywords:
            return alt_keywords, list(alt_keywords.keys())

        return {}, []

    @staticmethod
    def _has_directory_paths(paths: List[str]) -> bool:
        """Return True if any element in *paths* is a directory."""
        return any(Path(p).is_dir() for p in paths)

    @staticmethod
    def _resolve_file_hints(
        paths: List[str],
        file_hints: List[str],
        max_depth: int = 8,
    ) -> List[str]:
        """Resolve file_hints (filenames) to absolute paths under *paths*.

        Lightweight name-only search: no metadata extraction. Used when the
        user clearly asks for a specific document (e.g. "总结《foo.pdf》")
        so we can skip full dir scan + LLM rank.

        Returns:
            List of absolute path strings that match any hint (deduplicated,
            order preserved). Empty if no matches.
        """
        if not file_hints:
            return []

        hints = [h.strip() for h in file_hints if (h and isinstance(h, str))]
        if not hints:
            return []

        def _name_matches(name: str, hint: str) -> bool:
            name_n = name.strip()
            hint_n = hint.strip()
            if not hint_n:
                return False
            if name_n == hint_n:
                return True
            if hint_n.lower() in name_n.lower():
                return True
            if Path(name_n).stem == Path(hint_n).stem:
                return True
            return False

        seen: set = set()
        out: List[str] = []

        def walk_dir(d: Path, depth: int) -> None:
            if depth > max_depth or len(out) >= 20:
                return
            try:
                for entry in sorted(d.iterdir(), key=lambda p: p.name):
                    if len(out) >= 20:
                        return
                    if entry.name.startswith("."):
                        continue
                    if entry.is_file():
                        for hint in hints:
                            if _name_matches(entry.name, hint):
                                resolved = str(entry.resolve())
                                if resolved not in seen:
                                    seen.add(resolved)
                                    out.append(resolved)
                                break
                    elif entry.is_dir():
                        walk_dir(entry, depth + 1)
            except PermissionError:
                pass

        for p_str in paths:
            p = Path(p_str).resolve()
            if p.is_file():
                for hint in hints:
                    if _name_matches(p.name, hint):
                        resolved = str(p)
                        if resolved not in seen:
                            seen.add(resolved)
                            out.append(resolved)
                        break
            elif p.is_dir():
                walk_dir(p, 0)

        return out

    async def _probe_dir_scan(
        self,
        paths: List[str],
        enable: bool = True,
        max_files: int = 500,
    ):
        """Scan directories for file metadata (filesystem only, no LLM).

        Automatically skips scanning when all *paths* are single files.

        Args:
            paths: Normalised list of path strings to scan.
            enable: Whether directory scanning is enabled.
            max_files: Cap on number of files to scan (lower = faster).

        Returns:
            ScanResult or None if disabled / all paths are files.
        """
        if not enable or not self._has_directory_paths(paths):
            return None

        from sirchmunk.scan.dir_scanner import DirectoryScanner

        if self._dir_scanner is None or self._dir_scanner.max_files != max_files:
            self._dir_scanner = DirectoryScanner(llm=self.llm, max_files=max_files)

        await self._logger.info("[Probe:DirScan] Scanning directories...")
        scan_result = await self._dir_scanner.scan(paths)
        await self._logger.info(
            f"[Probe:DirScan] Found {scan_result.total_files} files "
            f"in {scan_result.total_dirs} dirs ({scan_result.scan_duration_ms:.0f}ms)"
        )
        return scan_result

    async def _probe_knowledge_cache(
        self, query: str,
    ) -> KnowledgeProbeResult:
        """Structured knowledge probe: embedding search with graph expansion.

        Uses embedding similarity (threshold 0.50) when available, falling back
        to SQL LIKE.  Extracts file paths, topic keywords, and background
        context from matched clusters and their graph neighbours.
        """
        empty = KnowledgeProbeResult([], [], "")
        try:
            clusters: List[KnowledgeCluster] = []

            # Prefer embedding search for semantic quality
            if self.embedding_client and self.embedding_client.is_ready():
                try:
                    qe = (await self.embedding_client.embed([query]))[0]
                    similar = await self.knowledge_storage.search_similar_clusters(
                        query_embedding=qe, top_k=5, similarity_threshold=0.50,
                    )
                    for m in (similar or []):
                        c = await self.knowledge_storage.get(m["id"])
                        if c:
                            clusters.append(c)
                except Exception:
                    pass

            # Fallback to SQL LIKE when embedding unavailable or empty
            if not clusters:
                clusters = await self.knowledge_storage.find(query, limit=3)

            if not clusters:
                return empty

            seen_paths: set = set()
            file_paths: List[str] = []
            extra_keywords: List[str] = []
            context_parts: List[str] = []
            seen_kw: set = set()

            def _collect_cluster(c: KnowledgeCluster) -> None:
                for ev in getattr(c, "evidences", []):
                    fp = str(getattr(ev, "file_or_url", ""))
                    if fp and fp not in seen_paths and Path(fp).exists():
                        seen_paths.add(fp)
                        file_paths.append(fp)
                for p in getattr(c, "patterns", []) or []:
                    if p and p.lower() not in seen_kw:
                        seen_kw.add(p.lower())
                        extra_keywords.append(p)
                content = c.content
                if isinstance(content, list):
                    content = "\n".join(content)
                if content:
                    context_parts.append(str(content)[:500])

            for c in clusters:
                _collect_cluster(c)

            # One-hop graph expansion via WeakSemanticEdge
            neighbour_ids: set = set()
            for c in clusters:
                for edge in getattr(c, "related_clusters", []):
                    tid = getattr(edge, "target_cluster_id", None)
                    if tid and tid not in neighbour_ids:
                        neighbour_ids.add(tid)

            for nid in list(neighbour_ids)[:6]:
                try:
                    neighbour = await self.knowledge_storage.get(nid)
                    if neighbour:
                        _collect_cluster(neighbour)
                except Exception:
                    pass

            if file_paths:
                await self._logger.info(
                    f"[Probe:Knowledge] {len(file_paths)} files, "
                    f"{len(extra_keywords)} keywords from "
                    f"{len(clusters)} clusters + {len(neighbour_ids)} neighbours"
                )

            return KnowledgeProbeResult(
                file_paths=file_paths,
                extra_keywords=extra_keywords[:15],
                background_context="\n\n".join(context_parts[:3]),
            )
        except Exception:
            return empty

    async def _probe_tree_index(self, query: str) -> List[str]:
        """LLM-driven file discovery via compiled tree root summaries (PageIndex).

        Loads all cached document trees, presents their root summaries to the
        LLM, and asks it to select the most relevant 1-3 documents.  For
        selected trees, optionally drills one level deeper into children.

        Returns file paths of the most relevant documents.
        """
        tree_cache = self.work_path / ".cache" / "compile" / "trees"
        if not tree_cache.exists():
            return []

        try:
            from sirchmunk.learnings.tree_indexer import DocumentTree

            trees: List[DocumentTree] = []
            for tree_file in sorted(tree_cache.glob("*.json"))[:50]:
                try:
                    t = DocumentTree.from_json(
                        tree_file.read_text(encoding="utf-8")
                    )
                    if t.root and t.file_path:
                        trees.append(t)
                except Exception:
                    continue

            if not trees:
                return []

            # If few trees, return all without LLM
            if len(trees) <= 2:
                return [t.file_path for t in trees if Path(t.file_path).exists()]

            # LLM-driven selection among tree roots
            listing = "\n".join(
                f"[{i}] {Path(t.file_path).name}: {(t.root.summary or '')[:200]}"
                for i, t in enumerate(trees)
            )
            prompt = (
                f'Given the query: "{query}"\n\n'
                f"Select the 1-3 most relevant documents (by index number):\n{listing}\n\n"
                f"Return ONLY a JSON array of index numbers, e.g. [0, 2]"
            )
            resp = await self.llm.achat([{"role": "user", "content": prompt}])
            self.llm_usages.append(resp.usage)

            selected_indices: List[int] = []
            try:
                raw = resp.content.strip()
                m = re.search(r"\[[\d\s,]+\]", raw)
                if m:
                    selected_indices = [
                        idx for idx in json.loads(m.group())
                        if isinstance(idx, int) and 0 <= idx < len(trees)
                    ]
            except (json.JSONDecodeError, TypeError):
                pass

            if not selected_indices:
                selected_indices = list(range(min(2, len(trees))))

            result_paths: List[str] = []
            for idx in selected_indices:
                fp = trees[idx].file_path
                if Path(fp).exists():
                    result_paths.append(fp)

            if result_paths:
                await self._logger.info(
                    f"[Probe:TreeIndex] LLM selected {len(result_paths)} documents "
                    f"from {len(trees)} tree indices"
                )
            return result_paths
        except Exception:
            return []

    async def _probe_compile_hints(self, keywords: List[str]) -> CompileHints:
        """Zero-LLM enrichment from compile manifest and tree cache.

        Scans the compile manifest for clusters whose patterns overlap with
        the query keywords, and scans cached tree root summaries for keyword
        matches.  No LLM calls — only local JSON reads and in-memory DB lookups.
        """
        empty = CompileHints([], [])
        if not keywords:
            return empty

        kw_lower = {k.lower() for k in keywords}
        file_paths: List[str] = []
        extra_keywords: List[str] = []
        seen_paths: set = set()
        seen_kw: set = set(kw_lower)

        # --- Cluster pattern matching via manifest ---
        manifest_path = self.work_path / ".cache" / "compile" / "manifest.json"
        if manifest_path.exists():
            try:
                from sirchmunk.learnings.compiler import CompileManifest
                manifest = CompileManifest.from_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                cluster_ids: set = set()
                for entry in manifest.files.values():
                    cluster_ids.update(entry.cluster_ids)

                for cid in list(cluster_ids)[:50]:
                    try:
                        c = await self.knowledge_storage.get(cid)
                    except Exception:
                        continue
                    if not c:
                        continue
                    cluster_patterns = [
                        p.lower() for p in (getattr(c, "patterns", []) or []) if p
                    ]
                    if kw_lower & set(cluster_patterns):
                        for ev in getattr(c, "evidences", []):
                            fp = str(getattr(ev, "file_or_url", ""))
                            if fp and fp not in seen_paths and Path(fp).exists():
                                seen_paths.add(fp)
                                file_paths.append(fp)
                        for p in cluster_patterns:
                            if p not in seen_kw:
                                seen_kw.add(p)
                                extra_keywords.append(p)
            except Exception:
                pass

        # --- Tree root summary scanning (keyword substring match) ---
        tree_cache = self.work_path / ".cache" / "compile" / "trees"
        if tree_cache.exists():
            try:
                from sirchmunk.learnings.tree_indexer import DocumentTree
                for tree_file in sorted(tree_cache.glob("*.json"))[:100]:
                    try:
                        tree = DocumentTree.from_json(
                            tree_file.read_text(encoding="utf-8")
                        )
                    except Exception:
                        continue
                    if not tree.root or not tree.file_path:
                        continue
                    summary_lower = (tree.root.summary or "").lower()
                    if any(kw in summary_lower for kw in kw_lower):
                        fp = tree.file_path
                        if fp not in seen_paths and Path(fp).exists():
                            seen_paths.add(fp)
                            file_paths.append(fp)
            except Exception:
                pass

        return CompileHints(
            file_paths=file_paths[:15],
            extra_keywords=extra_keywords[:10],
        )

    @staticmethod
    async def _async_noop(default=None):
        """No-op coroutine used as placeholder in gather()."""
        return default

    # ------------------------------------------------------------------
    # Phase 2 retrievers
    # ------------------------------------------------------------------

    async def _retrieve_by_keywords(
        self,
        keywords: List[str],
        paths: List[str],
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        """Run keyword search via rga and return discovered file paths.

        Each keyword is searched concurrently (literal per-term strategy).
        """
        from sirchmunk.agentic.tools import KeywordSearchTool

        tool = KeywordSearchTool(
            retriever=self.grep_retriever,
            paths=paths,
            max_depth=max_depth if max_depth is not None else 5,
            max_results=20,
            include=include,
            exclude=exclude,
        )
        ctx = SearchContext()  # lightweight context for this probe
        result_text, meta = await tool.execute(context=ctx, keywords=keywords)

        # Extract discovered file paths from the tool's context logs
        discovered: List[str] = []
        for log_entry in ctx.retrieval_logs:
            discovered.extend(log_entry.metadata.get("files_discovered", []))

        await self._logger.info(
            f"[Retrieve:Keywords] {len(discovered)} files from rga search"
        )
        return discovered

    async def _rank_dir_scan_candidates(
        self,
        query: str,
        scan_result,
        *,
        top_k: int = 20,
        include_medium: bool = False,
    ) -> List[str]:
        """Run LLM ranking on dir_scan candidates and return relevant paths.

        Args:
            include_medium: When True, include both high and medium relevance.
        """
        if self._dir_scanner is None:
            return []

        ranked = await self._dir_scanner.rank(query, scan_result, top_k=top_k)
        accept = {"high", "medium"} if include_medium else {"high"}
        paths = [
            c.path for c in ranked.ranked_candidates
            if c.relevance in accept
        ]
        await self._logger.info(
            f"[Retrieve:DirScan] {len(paths)} relevant files "
            f"(accept={accept})"
        )
        return paths

    async def _scan_and_rank_paths(
        self,
        query: str,
        paths: List[str],
        *,
        max_files: int = 300,
        top_k: int = 20,
        include_medium: bool = True,
    ) -> List[str]:
        """Scan directories and return LLM-ranked relevant file paths.

        Combines :meth:`_probe_dir_scan` (filesystem walk) and
        :meth:`_rank_dir_scan_candidates` (LLM ranking) in one call.
        Automatically skips scanning when all *paths* are single files.

        Returns:
            Ranked file paths (high + optionally medium relevance),
            or empty list when scanning is not applicable.
        """
        scan_result = await self._probe_dir_scan(
            paths, enable=True, max_files=max_files,
        )
        if scan_result is None:
            return []

        return await self._rank_dir_scan_candidates(
            query, scan_result,
            top_k=top_k, include_medium=include_medium,
        )

    # ------------------------------------------------------------------
    # Phase 3: Merge + cluster build
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_file_paths(
        keyword_files: List[str],
        dir_scan_files: List[str],
        knowledge_hits: List[str],
    ) -> List[str]:
        """Merge file paths from all retrieval paths, dedup, preserve priority.

        Priority: keyword_search > knowledge_cache > dir_scan.
        """
        seen: set = set()
        merged: List[str] = []

        for fp in keyword_files + knowledge_hits + dir_scan_files:
            if fp and fp not in seen:
                seen.add(fp)
                merged.append(fp)

        return merged

    def _get_tree_indexer(self):
        """Lazily construct a DocumentTreeIndexer for search-time tree navigation."""
        from sirchmunk.learnings.tree_indexer import DocumentTreeIndexer

        tree_cache = self.work_path / ".cache" / "compile" / "trees"
        if not tree_cache.exists():
            return None
        _cb = getattr(self._logger, 'log_callback', None)
        return DocumentTreeIndexer(
            llm=self.llm,
            cache_dir=tree_cache,
            log_callback=_cb,
        )

    async def _build_cluster(
        self,
        query: str,
        file_paths: List[str],
        query_keywords: Dict[str, float],
        top_k_files: int = 5,
        top_k_snippets: int = 5,
    ) -> Optional[KnowledgeCluster]:
        """Build a KnowledgeCluster via knowledge_base.build().

        Constructs the Request wrapper and delegates to the knowledge
        base for parallel Monte Carlo evidence sampling.  When compiled
        tree indices exist, passes a ``tree_indexer`` so that evidence
        extraction can navigate to relevant sections before sampling.
        """
        try:
            request = Request(
                messages=[
                    Message(
                        role="user",
                        content=[ContentItem(type="text", text=query)],
                    ),
                ],
            )
            retrieved_infos = [{"path": fp} for fp in file_paths]

            cluster = await self.knowledge_base.build(
                request=request,
                retrieved_infos=retrieved_infos,
                keywords=query_keywords,
                top_k_files=top_k_files,
                top_k_snippets=top_k_snippets,
                verbose=self.verbose,
                tree_indexer=self._get_tree_indexer(),
            )
            self.llm_usages.extend(self.knowledge_base.llm_usages)
            self.knowledge_base.llm_usages.clear()

            if cluster:
                await self._logger.success(
                    f"[Phase 3] KnowledgeCluster built: {cluster.name} "
                    f"({len(cluster.evidences)} evidence units)"
                )
            return cluster
        except Exception as exc:
            await self._logger.warning(f"[Phase 3] knowledge_base.build() failed: {exc}")
            return None

    async def _gather_graph_context(self, cluster: KnowledgeCluster) -> str:
        """Enrich answer context with knowledge from graph neighbours.

        Traverses the cluster's ``related_clusters`` edges (sorted by weight),
        fetches the top neighbours, and returns a joined summary string that
        can be appended to the cluster content before answer generation.
        """
        edges = sorted(
            getattr(cluster, "related_clusters", []) or [],
            key=lambda e: getattr(e, "weight", 0),
            reverse=True,
        )
        if not edges:
            return ""

        parts: List[str] = []
        for edge in edges[:3]:
            tid = getattr(edge, "target_cluster_id", None)
            if not tid:
                continue
            try:
                neighbour = await self.knowledge_storage.get(tid)
            except Exception:
                continue
            if not neighbour:
                continue
            content = neighbour.content
            if isinstance(content, list):
                content = "\n".join(content)
            name = getattr(neighbour, "name", "") or ""
            snippet = str(content or "")[:300]
            if snippet:
                parts.append(f"- {name}: {snippet}")

        if not parts:
            return ""
        await self._logger.info(
            f"[Phase 3.5] Graph context: {len(parts)} neighbour summaries"
        )
        return "Related knowledge:\n" + "\n".join(parts)

    # ------------------------------------------------------------------
    # Phase 4: Answer generation
    # ------------------------------------------------------------------

    async def _summarise_cluster(
        self, query: str, cluster: KnowledgeCluster,
    ) -> Tuple[str, bool, bool]:
        """Generate a final answer summary from a KnowledgeCluster.

        Returns:
            ``(summary_text, should_save, should_answer)`` where:
            - should_save: quality verdict for persistence
            - should_answer: evidence sufficiency verdict for answering
        """
        sep = "\n"
        cluster_text_content = (
            f"{cluster.name}\n\n"
            f"{sep.join(cluster.description)}\n\n"
            f"{cluster.content if isinstance(cluster.content, str) else sep.join(cluster.content)}"
        )

        result_sum_prompt = SEARCH_RESULT_SUMMARY.format(
            user_input=query,
            text_content=cluster_text_content,
        )

        await self._logger.info("[Phase 4] Generating search result summary...")
        response = await self.llm.achat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
        )
        self.llm_usages.append(response.usage)

        summary, should_save, should_answer = self._parse_summary_response(response.content)
        return summary, should_save, should_answer

    async def _summarise_cluster_fallback(self, query: str) -> Tuple[str, bool]:
        """Generate an answer using the DEEP summary prompt with fallback evidence.

        Reuses the existing ``SEARCH_RESULT_SUMMARY`` prompt, feeding it the
        standard fallback text so that the LLM answers from its own knowledge
        without adding an extra LLM call to the pipeline.
        """
        result_sum_prompt = SEARCH_RESULT_SUMMARY.format(
            user_input=query,
            text_content=self._LLM_FALLBACK_EVIDENCE,
        )
        await self._logger.info("[Phase 4] Generating fallback summary from LLM knowledge...")
        response = await self.llm.achat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
        )
        self.llm_usages.append(response.usage)
        summary, _, _ = self._parse_summary_response(response.content)
        return summary, False  # Never save fallback answers

    async def _summarise_fast_fallback(
        self, query: str, context: "SearchContext",
    ) -> Tuple[str, bool]:
        """Generate an answer using the FAST summary prompt with fallback evidence.

        Reuses the existing ``ROI_RESULT_SUMMARY`` prompt, feeding it the
        standard fallback text so that the LLM answers from its own knowledge.
        """
        answer_prompt = ROI_RESULT_SUMMARY.format(
            user_input=query,
            text_content=self._LLM_FALLBACK_EVIDENCE,
        )
        answer_resp = await self.llm.achat(
            messages=[{"role": "user", "content": answer_prompt}],
            stream=True,
        )
        self.llm_usages.append(answer_resp.usage)
        if answer_resp.usage and isinstance(answer_resp.usage, dict):
            context.add_llm_tokens(
                answer_resp.usage.get("total_tokens", 0), usage=answer_resp.usage,
            )
        answer, _, _ = self._parse_summary_response(answer_resp.content or "")
        return answer, False  # Never save fallback answers

    async def _react_refinement(
        self,
        query: str,
        paths: List[str],
        initial_keywords: List[str],
        spec_context: str,
        enable_dir_scan: bool,
        max_loops: int,
        max_token_budget: int,
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Tuple[str, SearchContext]:
        """Fall back to ReAct loop when parallel probing yields insufficient evidence.

        The ReAct agent receives pre-extracted keywords and cached
        directory context so it doesn't waste turns re-discovering them.
        """
        from sirchmunk.agentic.react_agent import ReActSearchAgent

        registry = self._ensure_tool_registry(
            paths, enable_dir_scan,
            max_depth=max_depth,
            include=include,
            exclude=exclude,
        )
        agent = ReActSearchAgent(
            llm=self.llm,
            tool_registry=registry,
            max_loops=max_loops,
            max_token_budget=max_token_budget,
        )

        augmented_query = query
        if spec_context:
            augmented_query = (
                f"{query}\n\n"
                f"[System hint — cached directory context]\n{spec_context}"
            )

        answer, context = await agent.run(
            query=augmented_query,
            initial_keywords=initial_keywords or None,
        )
        return answer, context

    async def _build_cluster_from_context(
        self,
        query: str,
        answer: str,
        context: SearchContext,
        query_keywords: Dict[str, float],
        top_k_files: int = 5,
    ) -> Optional[KnowledgeCluster]:
        """Build a KnowledgeCluster from files discovered during a ReAct session.

        Collects file paths from ``context.read_file_ids`` and retrieval
        logs, then delegates to ``_build_cluster()``.  Falls back to a
        lightweight answer-only cluster when no files were discovered.
        """
        if not answer or len(answer) < 50:
            return None

        # Collect all discovered file paths
        discovered: List[str] = list(context.read_file_ids)
        for log_entry in context.retrieval_logs:
            if log_entry.tool_name == "keyword_search":
                for p in log_entry.metadata.get("files_discovered", []):
                    if p not in discovered:
                        discovered.append(p)

        if discovered:
            cluster = await self._build_cluster(
                query=query,
                file_paths=discovered,
                query_keywords=query_keywords,
                top_k_files=top_k_files,
            )
            if cluster:
                if not cluster.search_results:
                    cluster.search_results = list(discovered)
                return cluster

        # Fallback: lightweight cluster from answer text
        try:
            return self._make_answer_cluster(
                query, answer, prefix="R", file_paths=discovered,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Spec-path caching  (Task 4)
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_hash(path_str: str) -> str:
        """Deterministic hash of a search path string for cache filename."""
        return hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:16]

    def _spec_file(self, path_str: str) -> Path:
        """Return the spec-cache file path for a given search path."""
        return self.spec_path / f"{self._spec_hash(path_str)}.json"

    async def _load_spec_context(
        self,
        paths: List[str],
        *,
        stale_hours: float = 72.0,
    ) -> str:
        """Load cached spec context for each search path and merge.

        Returns a condensed text block summarising previously-cached
        directory metadata that the ReAct agent can use as a hint.
        Stale files (older than ``stale_hours``) are silently ignored.

        Args:
            paths: Normalised list of path strings.
            stale_hours: Maximum age of the cache in hours before it is
                considered stale and skipped (default: 72).

        Returns:
            Merged context string, or empty string if nothing cached.
        """
        parts: List[str] = []
        now = datetime.now(timezone.utc)
        stale_seconds = stale_hours * 3600

        for sp in paths:
            spec_file = self._spec_file(sp)
            if not spec_file.exists():
                continue
            try:
                raw = spec_file.read_text(encoding="utf-8")
                data = json.loads(raw)

                # Skip if stale (handle both naive and aware timestamps)
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
                if cached_at.tzinfo is None:
                    cached_at = cached_at.replace(tzinfo=timezone.utc)
                if (now - cached_at).total_seconds() > stale_seconds:
                    await self._logger.debug(f"[SpecCache] Stale cache for {sp} (>{stale_hours}h), skipping")
                    continue

                summary = data.get("summary", "")
                # Append file metadata (title + preview) for richer context
                file_meta = data.get("file_metadata", [])
                meta_lines: List[str] = []
                for fm in file_meta:
                    title = fm.get("title", "")
                    preview = fm.get("preview", "")
                    kw = fm.get("keywords", [])
                    line = f"  - {fm.get('filename', '?')}"
                    if title:
                        line += f"  [title: {title}]"
                    if kw:
                        line += f"  [keywords: {', '.join(kw[:5])}]"
                    if preview:
                        line += f"\n    preview: {preview[:200]}"
                    meta_lines.append(line)

                combined = summary or ""
                if meta_lines:
                    combined += "\nKnown files:\n" + "\n".join(meta_lines)
                if combined:
                    parts.append(f"[{sp}]\n{combined}")
            except Exception as exc:
                await self._logger.debug(f"[SpecCache] Failed to load {spec_file}: {exc}")

        return "\n\n".join(parts)

    async def _save_spec_context(
        self,
        paths: List[str],
        context: SearchContext,
        scan_result=None,
    ) -> None:
        """Persist spec-path context for each search path.

        Saves a JSON file per search-path containing: directory stats,
        files discovered, dir_scan file metadata (title, preview, keywords),
        searches performed, and a short summary.
        Uses ``self._spec_lock`` to prevent concurrent-write corruption.

        Args:
            paths: Normalised list of path strings.
            context: Completed SearchContext from a ReAct session.
            scan_result: Optional ScanResult from DirectoryScanner.scan().
        """
        # Build a path→FileCandidate lookup from scan_result
        scan_candidates: Dict[str, Any] = {}
        if scan_result is not None:
            for c in getattr(scan_result, "candidates", []):
                scan_candidates[c.path] = c

        async with self._spec_lock:
            for sp in paths:
                spec_file = self._spec_file(sp)
                try:
                    # Collect relevant info for this specific path
                    files_in_path = [
                        f for f in context.read_file_ids if f.startswith(sp)
                    ]
                    searches = context.search_history

                    # Build a brief summary
                    summary_lines = [
                        f"Total files read: {len(files_in_path)}",
                        f"Searches: {', '.join(searches[:10])}",
                    ]
                    if files_in_path:
                        summary_lines.append("Files read:")
                        for fp in files_in_path[:20]:
                            summary_lines.append(f"  - {fp}")

                    # Collect dir_scan metadata for files under this search path
                    file_metadata: List[Dict[str, Any]] = []
                    for cpath, cand in scan_candidates.items():
                        if cpath.startswith(sp):
                            entry: Dict[str, Any] = {
                                "path": cand.path,
                                "filename": cand.filename,
                                "extension": cand.extension,
                                "size_bytes": cand.size_bytes,
                                "mime_type": cand.mime_type,
                            }
                            if cand.title:
                                entry["title"] = cand.title
                            if cand.author:
                                entry["author"] = cand.author
                            if cand.page_count:
                                entry["page_count"] = cand.page_count
                            if cand.keywords:
                                entry["keywords"] = cand.keywords
                            if cand.preview:
                                entry["preview"] = cand.preview[:500]
                            if cand.encoding:
                                entry["encoding"] = cand.encoding
                            if cand.line_count:
                                entry["line_count"] = cand.line_count
                            if cand.relevance:
                                entry["relevance"] = cand.relevance
                            if cand.reason:
                                entry["reason"] = cand.reason
                            file_metadata.append(entry)

                    data = {
                        "search_path": sp,
                        "cached_at": datetime.now(timezone.utc).isoformat(),
                        "total_llm_tokens": context.total_llm_tokens,
                        "loop_count": context.loop_count,
                        "files_read": files_in_path,
                        "search_history": searches,
                        "summary": "\n".join(summary_lines),
                        "file_metadata": file_metadata,
                        "retrieval_logs": [
                            log.to_dict() for log in context.retrieval_logs
                        ],
                    }

                    # Atomic write: write to temp, then rename
                    tmp_path = spec_file.with_suffix(".tmp")
                    tmp_path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    tmp_path.replace(spec_file)

                    await self._logger.debug(
                        f"[SpecCache] Saved spec for {sp} -> {spec_file.name} "
                        f"({len(file_metadata)} file entries)"
                    )

                except Exception as exc:
                    await self._logger.warning(f"[SpecCache] Failed to save spec for {sp}: {exc}")
