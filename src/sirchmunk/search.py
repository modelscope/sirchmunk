# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import ast
import hashlib
import json
import logging
import os
import re
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from sirchmunk.base import BaseSearch
from sirchmunk.learnings.evidence_cache import EvidenceCache
from sirchmunk.learnings.knowledge_base import KnowledgeBase
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    generate_keyword_extraction_prompt,
    FAST_QUERY_ANALYSIS,
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
from sirchmunk.schema.metadata import FileInfo
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
from sirchmunk.utils.chat_utils import CHAT_QUERY_RE, CHAT_RESPONSE_SYSTEM


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
        enable_memory: bool = False,
        rga_max_count: Optional[int] = None,
        ugrep_corpus_path: Optional[Union[str, Path]] = None,
        highfreq_file_threshold: int = 0,
        rga_max_parse_lines: int = 0,
        merge_max_files: int = 0,
        title_lookup_fn=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store rga max_count setting
        self._rga_max_count = rga_max_count
        self._title_lookup_fn = title_lookup_fn

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

        self.grep_retriever: GrepRetriever = GrepRetriever(
            work_path=self.work_path,
            ugrep_corpus_path=ugrep_corpus_path,
            highfreq_file_threshold=highfreq_file_threshold,
            rga_max_parse_lines=rga_max_parse_lines,
            merge_max_files=merge_max_files,
        )

        # Create bound logger with callback - returns AsyncLogger instance
        self._logger = create_logger(log_callback=log_callback, enable_async=True)

        # Shared evidence cache across all searches in this session
        self._evidence_cache = EvidenceCache()

        # Pass log_callback to KnowledgeBase so it can also log through the same callback
        self.knowledge_base = KnowledgeBase(
            llm=self.llm,
            work_path=self.work_path,
            log_callback=log_callback,
            evidence_cache=self._evidence_cache,
        )

        self.verbose: bool = verbose
        self.llm_usages: List[Dict[str, Any]] = []
        self.max_queries_per_cluster: int = 5
        self.cluster_sim_threshold: float = kwargs.pop('cluster_sim_threshold', 0.85)
        self.cluster_sim_top_k: int = kwargs.pop('cluster_sim_top_k', 3)

        logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

        # ---- Agentic (ReAct) components (lazy-initialised on first use) ----
        self._tool_registry = None
        self._dir_scanner = None

        # ---- Spec-path cache for per-search-path context ----
        self.spec_path: Path = self.work_path / ".cache" / "spec"
        self.spec_path.mkdir(parents=True, exist_ok=True)
        self._spec_lock = asyncio.Lock()

        # ---- Heavy components (populated by background warm-up thread) ----
        self.knowledge_storage: Optional[KnowledgeStorage] = None
        self.embedding_client = None
        self._memory = None
        self._pending_feedback: List[asyncio.Task] = []

        # ---- Background warm-up: non-blocking ----
        self._warmup_event = threading.Event()
        _warmup_thread = threading.Thread(
            target=self._warmup_bg,
            kwargs={
                "reuse_knowledge": reuse_knowledge,
                "enable_memory": enable_memory,
                "embedding": embedding,
            },
            daemon=True,
            name="sirchmunk-warmup",
        )
        _warmup_thread.start()

    # ------------------------------------------------------------------
    # Background warm-up
    # ------------------------------------------------------------------

    def _warmup_bg(self, *, reuse_knowledge: bool, enable_memory: bool = False, embedding: EmbeddingUtil = None) -> None:
        """Background thread: initialise heavy components without blocking __init__."""
        try:
            # 1. KnowledgeStorage — DuckDB + Parquet load
            try:
                self.knowledge_storage = KnowledgeStorage(work_path=str(self.work_path))
                self._load_historical_knowledge()
            except Exception as exc:
                _loguru_logger.warning(f"[warmup] KnowledgeStorage init failed: {exc}")

            # 2. Embedding model — triggers its own daemon thread for download/load
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
                            embedding_cache
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

            # 3. System dependencies (rga / rg)
            try:
                if not check_dependencies():
                    _loguru_logger.info("[warmup] Installing rga (ripgrep-all) and rg (ripgrep)...")
                    install_rga()
            except Exception as exc:
                _loguru_logger.warning(f"[warmup] Dependency check failed: {exc}")

            # 4. Self-evolving retrieval memory (opt-in)
            if enable_memory:
                try:
                    from sirchmunk.memory import RetrievalMemory
                    self._memory = RetrievalMemory(
                        work_path=str(self.work_path),
                        embedding_util=self.embedding_client,
                    )
                    self.grep_retriever.set_memory(self._memory)
                    _loguru_logger.info("[warmup] Retrieval memory enabled and initialised")
                except Exception as exc:
                    _loguru_logger.info(f"[warmup] Retrieval memory not available: {exc}")
            else:
                _loguru_logger.info("[warmup] Retrieval memory disabled")

        except Exception as exc:
            _loguru_logger.error(f"[warmup] Unexpected error: {exc}")
        finally:
            self._warmup_event.set()
            _loguru_logger.info("[warmup] Background warm-up complete")

    async def _ensure_warmup(self, timeout: float = 30.0) -> None:
        """Wait for background warm-up to finish (non-blocking for the event loop).

        Called once at the top of ``search()``.  If warm-up is already done
        this is a no-op (~0 ms).  Otherwise it awaits in a thread-pool so
        the asyncio loop isn't blocked.
        """
        if self._warmup_event.is_set():
            return
        loop = asyncio.get_running_loop()
        ready = await loop.run_in_executor(
            None, self._warmup_event.wait, timeout,
        )
        if not ready:
            _loguru_logger.warning(
                "[warmup] Timed out — proceeding with available components"
            )

    def inject_evaluation(
        self,
        query: str,
        em_score: float,
        f1_score: float,
        llm_judge_verdict: Optional[str] = None,
    ) -> None:
        """Back-fill evaluation metrics into memory for the given query.

        Called by benchmark harnesses after ground-truth evaluation is
        available.  This closes the learning loop by providing gradient
        confidence to all memory layers.
        """
        if self._memory:
            self._memory.inject_evaluation(
                query=query,
                em_score=em_score,
                f1_score=f1_score,
                llm_judge_verdict=llm_judge_verdict,
            )

    async def flush_memory(self) -> None:
        """Await all pending feedback tasks and flush memory to disk.

        Must be called before process exit to guarantee feedback persistence.
        """
        if self._pending_feedback:
            done = [t for t in self._pending_feedback if not t.done()]
            if done:
                await asyncio.gather(*done, return_exceptions=True)
            self._pending_feedback.clear()
        if self._memory:
            try:
                self._memory.close()
            except Exception:
                pass

    async def trigger_distillation_sweep(self) -> int:
        """Attempt distillation across all observed query types.

        Called after evaluation injection to ensure distillation triggers
        even when individual per-query checks missed the batch threshold.
        Returns the number of successful distillations.
        """
        if not self._memory:
            return 0
        return await self._memory.trigger_distillation_sweep()

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

    async def _try_reuse_cluster(self, query: str) -> Optional[KnowledgeCluster]:
        """Try to reuse existing knowledge cluster based on semantic similarity.

        The method waits (non-blocking) for the embedding model to become
        ready so that reuse works reliably even on the first search call
        within a process.

        Returns:
            KnowledgeCluster if a suitable cached cluster is found, None otherwise.
        """
        if not self.embedding_client or not self.knowledge_storage:
            return None

        try:
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

    async def _save_cluster_with_embedding(self, cluster: KnowledgeCluster) -> None:
        """Save knowledge cluster to persistent storage, compute embedding, and flush to parquet.

        The final ``force_sync()`` ensures the embedding vector is written to
        the parquet file immediately so that subsequent searches (even across
        process restarts) can find it via ``search_similar_clusters``.

        Args:
            cluster: KnowledgeCluster to save
        """
        if not self.knowledge_storage:
            return

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
            evidences=evidences,
            search_results=list(file_paths or []),
            resources=resources or [],
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
    def _parse_summary_response(llm_response: str) -> Tuple[str, bool]:
        """
        Parse LLM response to extract summary and save decision.

        Args:
            llm_response: Raw LLM response containing SUMMARY and SHOULD_SAVE tags

        Returns:
            Tuple of (summary_text, should_save_flag)
        """
        # Extract SUMMARY content
        summary_fields = extract_fields(content=llm_response, tags=["SUMMARY", "SHOULD_SAVE"])

        summary = summary_fields.get("summary", "").strip()
        should_save_str = summary_fields.get("should_save", "true").strip().lower()

        # Parse should_save flag
        should_save = should_save_str in ["true", "yes", "1"]

        # If extraction failed, use entire response as summary and assume should save
        if not summary:
            summary = llm_response.strip()
            should_save = True

        return summary, should_save

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

    _TEMPLATE_PLACEHOLDERS = frozenset({
        "keyword1", "keyword2", "keyword3",
        "translated_keyword1", "translated_keyword2", "translated_keyword3",
        "idf_value",
    })

    @classmethod
    def _is_placeholder(cls, term: str) -> bool:
        """True if *term* is a prompt-template placeholder, not a real keyword."""
        return term.lower().strip('"') in cls._TEMPLATE_PLACEHOLDERS

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
                return {
                    k: float(v) for k, v in parsed.items()
                    if isinstance(k, str) and not AgenticSearch._is_placeholder(k)
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return {
                        k: float(v) for k, v in parsed.items()
                        if isinstance(k, str) and not AgenticSearch._is_placeholder(k)
                    }
            except Exception:
                pass
        return {}

    @staticmethod
    def _fallback_query_keywords(query: str) -> Dict[str, float]:
        """Heuristic keyword extraction when LLM extraction fails.

        Extracts capitalized named entities and non-stop content words
        directly from the query.  Avoids the expensive ReAct fallback
        for every query when the LLM prompt format is incompatible.
        """
        import re as _re
        _STOP = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "may", "might", "shall", "should", "must",
            "of", "in", "to", "for", "with", "on", "at", "from", "by",
            "about", "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "not", "so", "yet", "both", "also",
            "what", "which", "who", "where", "when", "how", "that",
            "this", "it", "its", "than", "between", "out",
        })
        kw: Dict[str, float] = {}

        named = _re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        for name in named:
            if name.lower() not in _STOP:
                kw[name] = 7.0

        words = query.split()
        for w in words:
            clean = _re.sub(r"[^\w]", "", w)
            if not clean or len(clean) < 3 or clean.lower() in _STOP:
                continue
            if clean not in kw:
                kw[clean] = 5.0

        return kw

    # ------------------------------------------------------------------
    # Agentic (ReAct) infrastructure — lazy initialisation
    # ------------------------------------------------------------------

    def _get_bm25_scorer(self):
        """Lazily create a shared BM25Scorer with the best available tokenizer."""
        if getattr(self, "_shared_bm25_scorer", None) is not None:
            return self._shared_bm25_scorer
        try:
            from sirchmunk.utils.tokenizer_util import TokenizerUtil
            tokenizer = TokenizerUtil()
        except Exception:
            tokenizer = None
        from sirchmunk.utils.bm25_util import BM25Scorer
        self._shared_bm25_scorer = BM25Scorer(tokenizer=tokenizer)
        return self._shared_bm25_scorer

    @staticmethod
    def _detect_text_only_corpus(paths: List[str], sample_limit: int = 20) -> bool:
        """Check if search paths contain only plain-text files (no binary adapters needed).

        Samples up to *sample_limit* files and returns True when none of the
        sampled files have a binary-adapter extension (.pdf, .docx, .epub, etc.).
        For text-only corpora rga adapter caching is useless and the
        ``--rga-no-cache`` flag should be passed to avoid overhead.
        """
        _ADAPTER_EXTS = {".pdf", ".docx", ".odt", ".epub", ".fb2", ".ipynb", ".xlsx"}
        sampled = 0
        for p in paths:
            pp = Path(p)
            if not pp.is_dir():
                continue
            for child in pp.iterdir():
                if child.is_file():
                    if child.suffix.lower() in _ADAPTER_EXTS:
                        return False
                    sampled += 1
                    if sampled >= sample_limit:
                        break
            if sampled >= sample_limit:
                break
        return sampled > 0

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
            TitleLookupTool,
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
        bm25_scorer = self._get_bm25_scorer()

        # Tool 1: Knowledge cache (zero cost)
        registry.register(KnowledgeQueryTool(self.knowledge_storage))

        # Detect text-only corpus (extensionless files) to skip rga adapter cache
        _rga_no_cache = self._detect_text_only_corpus(paths)

        # Tool 2: Keyword search (low cost, BM25-reranked)
        registry.register(
            KeywordSearchTool(
                retriever=self.grep_retriever,
                paths=paths,
                max_depth=max_depth if max_depth is not None else 5,
                max_results=10,
                include=include,
                exclude=exclude,
                bm25_scorer=bm25_scorer,
                max_count=self._rga_max_count,
                rga_no_cache=_rga_no_cache,
            )
        )

        # Tool 3: File read (medium cost, keyword-focused extraction)
        # When BA-ReAct belief tracking is active, the deep_extract_fn
        # enables lazy MCES activation for high-value large files.
        registry.register(FileReadTool(
            max_chars_per_file=30000,
            bm25_scorer=bm25_scorer,
            base_paths=paths,
            deep_extract_fn=self._make_deep_extract_fn(),
        ))

        # Tool 4: Title lookup (zero cost — direct index lookup)
        if self._title_lookup_fn is not None:
            registry.register(TitleLookupTool(
                lookup_fn=self._title_lookup_fn,
            ))

        # Tool 5: Directory scan (optional, medium cost)
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
    # BA-ReAct: Lazy MCES deep extraction callback
    # ------------------------------------------------------------------

    def _make_deep_extract_fn(self):
        """Create a closure that runs MCES deep evidence extraction.

        The returned async callable is injected into ``FileReadTool`` and
        invoked transparently when the belief state triggers deep extraction
        for a high-value large file.  Results are cached on the
        ``SearchContext.mces_cache`` for Phase 3 reuse.
        """
        from sirchmunk.learnings.evidence_processor import EvidenceSampler

        llm = self.llm
        evidence_cache = self._evidence_cache
        log_callback = getattr(self, "log_callback", None)
        embedding_util = getattr(self, "embedding_client", None)

        async def _deep_extract(
            doc_content: str,
            keywords: list,
            context,
            file_path: str,
        ):
            query = getattr(context, "query", "") or ""
            if not query:
                return None

            sampler = EvidenceSampler(
                llm=llm,
                doc_content=doc_content,
                verbose=False,
                log_callback=log_callback,
                embedding_util=embedding_util,
                cache=evidence_cache,
            )
            kw_dict = {k: 1.0 for k in keywords} if keywords else {}
            roi = await sampler.get_roi(
                query=query,
                keywords=kw_dict,
                confidence_threshold=7.0,
            )

            # Track LLM tokens consumed by MCES
            for u in sampler.llm_usages:
                tok = u.get("total_tokens", 0)
                if tok == 0:
                    tok = u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
                context.add_llm_tokens(tok, usage=u)

            # Cache RoiResult for Phase 3 cluster construction
            context.mces_cache[file_path] = {
                "summary": roi.summary,
                "is_found": roi.is_found,
                "snippets": roi.snippets,
            }

            # Update belief state with MCES scores
            belief = getattr(context, "belief_state", None)
            if belief:
                best_score = max(
                    (s.get("score", 0) for s in roi.snippets),
                    default=0,
                )
                belief.update_from_mces(file_path, best_score, roi.is_found)

            if not roi.is_found:
                return None

            parts = [roi.summary]
            for s in roi.snippets[:3]:
                score = s.get("score", 0)
                snippet = s.get("snippet", "")[:500]
                if snippet:
                    parts.append(f"\n--- Evidence (score={score:.1f}) ---\n{snippet}")
            return "\n".join(parts)

        return _deep_extract

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
        max_depth: Optional[int] = 8,
        top_k_files: int = 5,
        enable_dir_scan: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        return_context: bool = False,
        spec_stale_hours: float = 72.0,
        chat_history: Optional[List[Dict[str, str]]] = None,
        enable_thinking: bool = False,
        enable_cross_lingual: bool = True,
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

        DEEP architecture (ReAct-first iterative retrieval):

        ┌──────────────────────────────────────────────────────────┐
        │ Phase 0a Direct document analysis (intent-gated,         │
        │          short-circuit if query is doc-level operation)   │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 0  Cluster reuse check (instant, short-circuit)    │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 1  Parallel warm-start probing (all concurrent):   │
        │  ├─ LLM keyword extraction                               │
        │  ├─ DirectoryScanner.scan() (filesystem only, fast)      │
        │  ├─ Knowledge cache similarity search                    │
        │  └─ Spec-path cache load                                 │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 2  ReAct-driven iterative retrieval:               │
        │  └─ Agent loop with warm-start keywords + tools:         │
        │     keyword_search (BM25-reranked), file_read (focused   │
        │     extraction), knowledge_query, dir_scan (optional)    │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 3  Cluster construction from ReAct discoveries     │
        ├──────────────────────────────────────────────────────────┤
        │ Phase 4  Persistence (concurrent, quality-gated):        │
        │  ├─ Save cluster + embeddings                            │
        │  └─ Save spec-path cache                                 │
        └──────────────────────────────────────────────────────────┘

        Args:
            query: User's search query.
            paths: Directories / files to search.  Falls back to
                ``self.paths`` or the current working directory.
            mode: Search mode — ``"DEEP"``, ``"FAST"``, or ``"FILENAME_ONLY"``.
            max_loops: Maximum ReAct iterations (DEEP mode, default: 5).
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
            chat_history: Optional list of chat messages for intent detection (DEEP mode).
            enable_thinking: If True, enable model reasoning/thinking for
                complex processing steps (final answer synthesis, ReAct
                reasoning). Simple extraction/classification steps always
                run with thinking disabled. Default: False.
            enable_cross_lingual: If True, enable cross-lingual keyword extraction and matching.

        Returns:
            - ``str``: Answer summary (default).
            - ``SearchContext``: If *return_context* — contains ``answer``,
              ``cluster``, and telemetry in a single object.
            - ``List[Dict]``: File matches in FILENAME_ONLY mode.
        """
        # ---- Ensure background warm-up is complete ----
        await self._ensure_warmup()

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

        # ---- Memory: strategy hint (zero-LLM) ----
        _STRATEGY_CONFIDENCE_MIN = 0.7
        if self._memory and mode != "FILENAME_ONLY":
            try:
                hint = self._memory.suggest_strategy(query)
                if hint and hint.confidence >= _STRATEGY_CONFIDENCE_MIN:
                    _loguru_logger.info(
                        "[memory] Strategy hint applied: conf={:.2f} mode={}",
                        hint.confidence, hint.mode,
                    )
                    if hint.mode:
                        mode = hint.mode
                    if hint.top_k_files is not None:
                        top_k_files = hint.top_k_files
                    if hint.max_loops is not None:
                        if hint.resolution_level is not None and hint.resolution_level <= 1:
                            max_loops = max(max_loops, hint.max_loops)
                        else:
                            max_loops = hint.max_loops
                    if hint.enable_dir_scan is not None:
                        enable_dir_scan = hint.enable_dir_scan
                    if hint.token_budget is not None and hint.token_budget > 0:
                        max_token_budget = max(hint.token_budget, 20000)
                    _loguru_logger.debug(
                        "[memory] strategy hint applied: mode={}, resolution_level={}, token_budget={}",
                        hint.mode, hint.resolution_level, hint.token_budget,
                    )
            except Exception:
                pass

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
                enable_thinking=enable_thinking,
                enable_cross_lingual=enable_cross_lingual,
            )
        else:
            answer, cluster, context = await self._search_deep(
                query=query, paths=paths,
                max_loops=max_loops, max_token_budget=max_token_budget,
                max_depth=max_depth, top_k_files=top_k_files,
                enable_dir_scan=enable_dir_scan,
                include=include, exclude=exclude,
                spec_stale_hours=spec_stale_hours,
                enable_thinking=enable_thinking,
                enable_cross_lingual=enable_cross_lingual,
            )

        # ---- Unified return wrapping ----
        if return_context:
            prefix = "FS" if mode == "FAST" else "DS"
            context.answer = answer
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
        max_loops: int = 5,
        max_token_budget: int = 128000,
        max_depth: Optional[int] = 5,
        top_k_files: int = 5,
        enable_dir_scan: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        spec_stale_hours: float = 72.0,
        enable_thinking: bool = False,
        enable_cross_lingual: bool = True,
    ) -> Tuple[str, Optional[KnowledgeCluster], SearchContext]:
        """ReAct-first iterative retrieval pipeline (Phases 0a–4).

        Phase 0a/0 provide short-circuit paths for cached knowledge.
        Phase 1 extracts warm-start data (keywords, dir_scan, knowledge).
        Phase 2 drives a ReAct agent loop with enhanced tools.
        Phase 3 constructs a knowledge cluster from ReAct discoveries.
        Phase 4 persists quality-gated results.

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
        # return here — Phase 4 (Persistence) is not executed for that path.
        # ==============================================================
        reused = await self._try_reuse_cluster(query)
        if reused is not None:
            content = reused.content
            if isinstance(content, list):
                content = "\n".join(content)
            return str(content), reused, context

        await self._logger.info(f"[search] Starting multi-path retrieval for: '{query[:80]}'")

        # ==============================================================
        # Phase 1: Parallel probing — all four paths fire concurrently
        # ==============================================================
        await self._logger.info("[Phase 1] Parallel probing: keywords + dir_scan + knowledge + spec_cache")
        context.increment_loop()

        phase1_results = await asyncio.gather(
            self._probe_keywords(query, enable_cross_lingual=enable_cross_lingual),
            self._probe_dir_scan(paths, enable_dir_scan),
            self._probe_knowledge_cache(query),
            self._load_spec_context(paths, stale_hours=spec_stale_hours),
            return_exceptions=True,
        )

        kw_result = phase1_results[0] if not isinstance(phase1_results[0], Exception) else ({}, [])
        scan_result = phase1_results[1] if not isinstance(phase1_results[1], Exception) else None
        knowledge_hits = phase1_results[2] if not isinstance(phase1_results[2], Exception) else []
        spec_context = phase1_results[3] if not isinstance(phase1_results[3], Exception) else ""

        for i, label in enumerate(["keywords", "dir_scan", "knowledge", "spec_cache"]):
            if isinstance(phase1_results[i], Exception):
                await self._logger.warning(f"[Phase 1] {label} probe failed: {phase1_results[i]}")

        query_keywords, initial_keywords = kw_result if isinstance(kw_result, tuple) else ({}, [])

        # Memory: expand keywords with semantic bridge + filter noise
        if self._memory and query_keywords:
            try:
                query_keywords = self._memory.expand_keywords(query_keywords)
            except Exception:
                pass
            try:
                clean_kws = self._memory.filter_noise_keywords(
                    list(query_keywords.keys())
                )
                if clean_kws:
                    query_keywords = {
                        k: v for k, v in query_keywords.items()
                        if k in clean_kws
                    }
                    initial_keywords = [
                        k for k in initial_keywords if k in clean_kws
                    ]
            except Exception:
                pass
        await self._logger.info(
            f"[Phase 1] Results: keywords={len(initial_keywords)}, "
            f"dir_scan={'OK' if scan_result else 'N/A'}, "
            f"knowledge_hits={len(knowledge_hits)}, "
            f"spec_cache={'YES' if spec_context else 'NO'}"
        )

        # Memory → BeliefState: extract cross-session priors for warm-start
        _memory_prior = None
        _memory_bridge = None
        if self._memory:
            try:
                from sirchmunk.memory.bridge import MemoryBridge
                _memory_bridge = MemoryBridge(
                    self._memory,
                    title_lookup_fn=self._title_lookup_fn,
                )
                _memory_prior = _memory_bridge.extract_priors(
                    query=query,
                    extra_keywords=(
                        list(query_keywords.keys()) if query_keywords else None
                    ),
                )
                if _memory_prior and not _memory_prior.is_empty:
                    _loguru_logger.info(
                        "[memory] Prior: entity_paths={}, dead={}, "
                        "chain={}, strategy_rules={}",
                        len(_memory_prior.entity_paths),
                        len(_memory_prior.dead_paths),
                        "yes" if _memory_prior.chain_hint else "no",
                        len(_memory_prior.strategy_rules),
                    )
            except Exception:
                pass

        # ==============================================================
        # Phase 1.5: Memory-Augmented Planning (MAP)
        # Generate a search plan from distilled meta-knowledge.
        # High-confidence plans drive guided execution; lower ones
        # serve as warm-start hints for the ReAct agent.
        # ==============================================================
        _search_plan = None
        if self._memory:
            try:
                _search_plan = await self._memory.plan_search(query)
                if _search_plan:
                    _loguru_logger.info(
                        "[MAP] Plan: conf={:.2f}, steps={}, strategy={}",
                        _search_plan.confidence,
                        len(_search_plan.plan_steps),
                        _search_plan.keyword_strategy,
                    )
                    # Inject plan into MemoryPrior for BeliefState access
                    if _memory_prior is not None:
                        _memory_prior.search_plan = _search_plan
                    # Adaptive loop budget from meta-knowledge
                    learned_budget = self._memory.get_optimal_loop_budget(query)
                    if learned_budget and learned_budget < max_loops:
                        max_loops = max(learned_budget, 3)
                        _loguru_logger.debug(
                            "[MAP] Adaptive loop budget: %d", max_loops,
                        )
            except Exception:
                pass

        # ==============================================================
        # Phase 2: ReAct-driven iterative retrieval
        # The agent receives Phase 1 warm-start data (keywords, spec
        # context) and uses enhanced tools (BM25-reranked keyword search,
        # keyword-focused file read) for multi-hop reasoning.
        # ==============================================================
        await self._logger.info("[Phase 2] Launching ReAct agent for iterative retrieval")

        answer, context = await self._react_refinement(
            query=query, paths=paths,
            initial_keywords=initial_keywords, spec_context=spec_context,
            enable_dir_scan=enable_dir_scan,
            max_loops=max_loops, max_token_budget=max_token_budget,
            max_depth=max_depth, include=include, exclude=exclude,
            enable_thinking=enable_thinking,
            memory_prior=_memory_prior,
            search_plan=_search_plan,
        )

        # ==============================================================
        # Phase 3: Cluster construction from ReAct discoveries
        # ==============================================================
        cluster: Optional[KnowledgeCluster] = None
        should_save: bool = False

        if answer and self._is_evidence_meaningful(answer):
            cluster = await self._build_cluster_from_context(
                query=query, answer=answer, context=context,
                query_keywords=query_keywords, top_k_files=top_k_files,
            )
            should_save = cluster is not None

        # Sync LLM token accounting into context
        new_usages = self.llm_usages[_llm_usage_start:]
        for usage in new_usages:
            if usage and isinstance(usage, dict):
                total_tok = usage.get("total_tokens", 0)
                if total_tok == 0:
                    total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                context.add_llm_tokens(total_tok, usage=usage)

        # ==============================================================
        # Phase 4: Persistence (quality-gated)
        # ==============================================================
        phase4_tasks = []
        if cluster and should_save:
            self._add_query_to_cluster(cluster, query)
            phase4_tasks.append(self._save_cluster_with_embedding(cluster))
        else:
            await self._logger.info("[Phase 4] Quality gate: skipping cluster save")
            cluster = None
        phase4_tasks.append(self._save_spec_context(paths, context, scan_result=scan_result))
        results = await asyncio.gather(*phase4_tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                _loguru_logger.warning(f"[Phase 4] Persistence task failed: {r}")

        # Memory: record feedback signal enriched with belief data
        if self._memory:
            try:
                from sirchmunk.memory.schemas import FeedbackSignal
                _now = datetime.now(timezone.utc)
                _start = (
                    context.start_time
                    if context.start_time.tzinfo
                    else context.start_time.replace(tzinfo=timezone.utc)
                )
                elapsed = max(0.0, (_now - _start).total_seconds())
                _discovered = list(context.read_file_ids)
                for log_entry in context.retrieval_logs:
                    if log_entry.tool_name == "keyword_search":
                        for p in log_entry.metadata.get("files_discovered", []):
                            if p not in _discovered:
                                _discovered.append(p)

                belief_state = getattr(context, "belief_state", None)
                _high_value = []
                if belief_state:
                    try:
                        _high_value = belief_state.to_feedback_dict().get(
                            "high_value_files", [],
                        )
                    except Exception:
                        pass
                _useful_count = (
                    len(_high_value) if _high_value
                    else (len(context.read_file_ids) if (answer and answer.strip()) else 0)
                )

                signal = FeedbackSignal(
                    query=query,
                    mode="DEEP",
                    answer_found=bool(answer and answer.strip()),
                    answer_text=(answer or "")[:2000],
                    cluster_confidence=(
                        getattr(cluster, "confidence", 0.0) if cluster else 0.0
                    ),
                    react_loops=context.loop_count,
                    files_read_count=context.total_known_files,
                    files_useful_count=_useful_count,
                    total_tokens=context.total_llm_tokens,
                    latency_sec=elapsed,
                    keywords_used=list(query_keywords.keys()) if query_keywords else [],
                    paths_searched=paths,
                    files_read=list(context.read_file_ids),
                    files_discovered=_discovered,
                )

                # Enrich with BA-ReAct belief data
                if _memory_bridge and belief_state:
                    try:
                        _memory_bridge.enrich_feedback(signal, belief_state)
                    except Exception as _enrich_err:
                        _loguru_logger.debug(
                            "[memory] Belief enrichment failed: {}", _enrich_err,
                        )

                task = asyncio.ensure_future(self._memory.record_feedback(signal))
                self._pending_feedback.append(task)
            except Exception as _sig_err:
                _loguru_logger.debug(
                    "[memory] Feedback signal construction failed: {}", _sig_err,
                )

        # Meta-RL: record abstract trajectory for strategy distillation.
        # Outcome is heuristic confidence (ground-truth injected later).
        if self._memory:
            try:
                _heuristic_outcome = 0.5
                if answer and answer.strip():
                    _heuristic_outcome = 0.6
                    if cluster and getattr(cluster, "confidence", 0) > 0.5:
                        _heuristic_outcome = 0.75
                self._memory.record_trajectory(query, context, _heuristic_outcome)
            except Exception:
                pass
            # Trigger strategy distillation when enough trajectories accumulate
            try:
                if self._memory.should_distill(query):
                    asyncio.ensure_future(self._memory.trigger_distillation(query))
            except Exception:
                pass

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
        return bool(CHAT_QUERY_RE.match(query.strip()))

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
            {"role": "system", "content": CHAT_RESPONSE_SYSTEM},
            *(chat_history or []),
            {"role": "user", "content": query},
        ]
        resp = await self.llm.achat(messages=messages, stream=False, enable_thinking=False)
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
            enable_thinking=False,
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
                enable_thinking=False,
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
            enable_thinking=False,
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
        enable_thinking: bool = False,
        enable_cross_lingual: bool = True,
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

        # ==============================================================
        # Step 0: Cluster reuse — instant short-circuit (no LLM cost)
        # When reuse succeeds we return here; no persistence step runs.
        # ==============================================================
        reused = await self._try_reuse_cluster(query)
        if reused is not None:
            content = reused.content
            if isinstance(content, list):
                content = "\n".join(content)
            await self._logger.success("[FAST] Reused cached knowledge cluster")
            return str(content), reused, context

        # ==============================================================
        # Step 1: LLM query analysis only (dir scan deferred until needed)
        # ==============================================================
        prompt = FAST_QUERY_ANALYSIS.format(user_input=query)
        resp = await self.llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            enable_thinking=False,
        )
        self.llm_usages.append(resp.usage)
        if resp.usage and isinstance(resp.usage, dict):
            context.add_llm_tokens(
                resp.usage.get("total_tokens", 0), usage=resp.usage,
            )

        analysis = self._parse_fast_json(resp.content)
        query_type = analysis.get("type", "search")
        file_hints = analysis.get("file_hints", [])

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

        # Cross-lingual keywords (disabled by default for monolingual datasets)
        if enable_cross_lingual:
            primary_alt = analysis.get("primary_alt", [])[:2]
            fallback_alt = analysis.get("fallback_alt", [])[:3]
            if primary_alt:
                primary = primary + primary_alt
            if fallback_alt:
                fallback = fallback + fallback_alt

        if not primary and not fallback:
            await self._logger.warning("[FAST] No keywords extracted")
            msg = f"Could not extract search terms from query: '{query}'"
            return msg, None, context

        await self._logger.info(
            f"[FAST:Step1] Primary: {primary}, Fallback: {fallback}"
        )

        # ==============================================================
        # Step 2: rga cascade — primary first, fallback only if needed
        # Dir scan runs only when enabled, for fallback when rga misses.
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

        best_file: Optional[Dict[str, Any]] = None
        used_level = "primary"

        if primary:
            best_file = await self._fast_find_best_file(primary, **rga_kwargs)

        if not best_file and fallback:
            used_level = "fallback"
            await self._logger.info(
                "[FAST:Step2] Primary miss, trying fine-grained fallback"
            )
            best_file = await self._fast_find_best_file(fallback, **rga_kwargs)

        # --- Fallback: use dir_scan only when rga misses and dir scan is enabled ---
        if not best_file and enable_dir_scan:
            scan_result = await self._probe_dir_scan(paths, enable=True, max_files=300)
            if scan_result is not None:
                await self._logger.info("[FAST:Step2] rga miss — falling back to dir_scan ranking")
                ranked_paths = await self._rank_dir_scan_candidates(
                    query, scan_result, top_k=10, include_medium=True,
                )
                if ranked_paths:
                    used_level = "dir_scan"
                    best_file = {"path": ranked_paths[0], "matches": [], "total_matches": 0}

        if not best_file:
            await self._logger.warning(
                f"[FAST:Step2] No matching files found in paths: {paths}. "
                "If files are PDFs/DOCX, ensure poppler-utils and pandoc are installed."
            )
            msg = f"No relevant content found for query: '{query}'"
            return msg, None, context

        file_path = best_file["path"]
        match_objects = best_file["matches"]
        await self._logger.info(
            f"[FAST:Step2] Best file ({used_level}): {Path(file_path).name} "
            f"({best_file['total_matches']} hits)"
        )

        # ==============================================================
        # Step 3: Context sampling around grep hits (no LLM)
        # ==============================================================
        evidence = await self._fast_sample_evidence(file_path, match_objects)
        context.mark_file_read(file_path)

        if not evidence or len(evidence.strip()) < 20:
            await self._logger.warning("[FAST:Step3] No usable evidence extracted")
            msg = f"Found file but could not extract content for query: '{query}'"
            return msg, None, context

        await self._logger.info(
            f"[FAST:Step3] Evidence: {len(evidence)} chars from {Path(file_path).name}"
        )

        # ==============================================================
        # Step 4: LLM answer from focused evidence (single call)
        # ==============================================================
        answer_prompt = ROI_RESULT_SUMMARY.format(
            user_input=query,
            text_content=evidence,
        )
        import time as _time
        _t0 = _time.time()
        answer_resp = await self.llm.achat(
            messages=[{"role": "user", "content": answer_prompt}],
            stream=True,
            enable_thinking=enable_thinking,
        )
        await self._logger.info(f"[Timing] FAST answer synthesis: {_time.time()-_t0:.2f}s")
        self.llm_usages.append(answer_resp.usage)
        if answer_resp.usage and isinstance(answer_resp.usage, dict):
            context.add_llm_tokens(
                answer_resp.usage.get("total_tokens", 0), usage=answer_resp.usage,
            )

        answer, should_save = self._parse_summary_response(answer_resp.content or "")
        keywords_used = primary if used_level == "primary" else fallback

        if not should_save:
            await self._logger.info("[FAST] Quality gate: low-quality answer, skipping cluster save")
            await self._logger.success("[FAST] Search complete (2 LLM calls, no persist)")
            return answer, None, context

        cluster = self._build_fast_cluster(
            query, answer, file_path, evidence, keywords_used,
        )
        self._add_query_to_cluster(cluster, query)
        try:
            await self._save_cluster_with_embedding(cluster)
        except Exception as exc:
            _loguru_logger.warning(
                f"[FAST] Failed to save cluster with embedding: {exc}"
            )

        # Memory: record feedback signal (fire-and-forget)
        if self._memory:
            try:
                from sirchmunk.memory.schemas import FeedbackSignal
                _now = datetime.now(timezone.utc)
                _start = (
                    context.start_time
                    if context.start_time.tzinfo
                    else context.start_time.replace(tzinfo=timezone.utc)
                )
                elapsed = max(0.0, (_now - _start).total_seconds())
                signal = FeedbackSignal(
                    query=query,
                    mode="FAST",
                    answer_found=bool(answer and answer.strip()),
                    answer_text=(answer or "")[:2000],
                    files_read_count=len(context.read_file_ids),
                    files_useful_count=len(
                        context.read_file_ids
                    ) if (answer and answer.strip()) else 0,
                    total_tokens=context.total_llm_tokens,
                    latency_sec=elapsed,
                    keywords_used=keywords_used,
                    paths_searched=paths,
                    files_read=list(context.read_file_ids),
                    files_discovered=[file_path] if file_path else [],
                )
                task = asyncio.ensure_future(self._memory.record_feedback(signal))
                self._pending_feedback.append(task)
            except Exception as _sig_err:
                _loguru_logger.debug(
                    "[memory] FAST feedback signal failed: {}", _sig_err,
                )

        await self._logger.success("[FAST] Search complete (2 LLM calls)")
        return answer, cluster, context

    # ---- FAST helpers ----

    async def _fast_find_best_file(
        self,
        keywords: List[str],
        paths: List[str],
        max_depth: Optional[int] = 5,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Search per keyword via rga and return the single best-matching file.

        Returns:
            Merged file dict (path, matches, lines, total_matches) or None.
        """
        all_raw: List[Dict[str, Any]] = []

        for kw in keywords:
            try:
                results = await self.grep_retriever.retrieve(
                    terms=kw, path=paths, literal=True, regex=False,
                    max_depth=max_depth, include=include, exclude=exclude,
                    timeout=30.0,
                )
                if results:
                    all_raw.extend(results)
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
                    return {
                        "path": fn_results[0]["path"],
                        "matches": [], "lines": [], "total_matches": 0,
                    }
            except Exception as exc:
                await self._logger.warning(
                    f"[FAST] filename search failed: {exc}"
                )
            return None

        merged = GrepRetriever.merge_results(
            all_raw, limit=20, max_files=self.grep_retriever._merge_max_files,
        )
        if not merged:
            return None

        # Greedy: pick the file with the most matches
        merged.sort(key=lambda f: f["total_matches"], reverse=True)
        return merged[0]

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

        # --- Text files: read context windows around hits ---
        if ext in self._FAST_TEXT_EXTENSIONS and hit_lines:
            evidence = self._read_context_windows(
                file_path, hit_lines,
                window=self._FAST_CONTEXT_WINDOW,
                max_chars=self._FAST_MAX_EVIDENCE_CHARS,
            )
            if evidence:
                return f"[{fname}]\n{evidence}"

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
            return f"[{fname}]\n" + "\n".join(snippets)

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
        self, query: str, *, enable_cross_lingual: bool = True,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Extract multi-level keywords from the query via LLM.

        Also extracts cross-lingual alternative keywords from the
        ``<KEYWORDS_ALT>`` block and merges them into the result list
        when ``enable_cross_lingual`` is True.
        Falls back to heuristic extraction if LLM output is unparseable.

        Args:
            query: User query to extract keywords from.
            enable_cross_lingual: Whether to include translated keywords.
                Disable for monolingual datasets like HotpotQA.

        Returns:
            Tuple of (keyword_idf_dict, keyword_list).
        """
        await self._logger.info("[Probe:Keywords] Extracting keywords...")

        try:
            dynamic_prompt = generate_keyword_extraction_prompt(num_levels=2)
            keyword_prompt = dynamic_prompt.format(user_input=query)
            import time as _time
            _t0 = _time.time()
            kw_response = await self.llm.achat(
                messages=[{"role": "user", "content": keyword_prompt}],
                stream=False,
                enable_thinking=False,
            )
            await self._logger.info(f"[Timing] Keyword extraction: {_time.time()-_t0:.2f}s")
            self.llm_usages.append(kw_response.usage)

            keyword_sets = self._extract_and_validate_multi_level_keywords(
                kw_response.content, num_levels=2,
            )

            keyword_sets = [
                {k: v for k, v in ks.items() if not self._is_placeholder(k)}
                for ks in keyword_sets
            ]

            # Cross-lingual keywords (disabled for monolingual datasets)
            alt_keywords: Dict[str, float] = {}
            if enable_cross_lingual:
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
        except Exception as exc:
            await self._logger.info(f"[Phase 1] keywords probe failed: {exc}")

        fallback = self._fallback_query_keywords(query)
        if fallback:
            kw_list = list(fallback.keys())
            await self._logger.info(f"[Probe:Keywords] Heuristic fallback: {kw_list}")
            return fallback, kw_list

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
    ) -> List[str]:
        """Search knowledge cache for related clusters, return known file paths.

        Returns:
            List of file paths from previously cached clusters.
        """
        if not self.knowledge_storage:
            return []
        try:
            clusters = await self.knowledge_storage.find(query, limit=3)
            if not clusters:
                return []

            file_paths: List[str] = []
            for c in clusters:
                for ev in getattr(c, "evidences", []):
                    fp = str(getattr(ev, "file_or_url", ""))
                    if fp and Path(fp).exists():
                        file_paths.append(fp)

            if file_paths:
                await self._logger.info(
                    f"[Probe:Knowledge] Found {len(file_paths)} files from cached clusters"
                )
            return file_paths
        except Exception:
            return []

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
        """Run keyword search via rga and return discovered file paths."""
        from sirchmunk.agentic.tools import KeywordSearchTool

        tool = KeywordSearchTool(
            retriever=self.grep_retriever,
            paths=paths,
            max_depth=max_depth if max_depth is not None else 5,
            max_results=20,
            include=include,
            exclude=exclude,
            max_count=self._rga_max_count,
        )
        ctx = SearchContext()
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

    @staticmethod
    def _bm25_rerank_files(
        query: str,
        file_paths: List[str],
        top_k: int = 10,
    ) -> Optional[List[str]]:
        """Use BM25 to rerank candidate files by query relevance.

        Reads first 2000 chars of each file as document representation,
        scores them against the query using ``BM25Scorer``, and returns
        the top-k most relevant file paths.  Falls back to None on error
        (caller should keep the original list).
        """
        from sirchmunk.utils.bm25_util import BM25Scorer

        docs: List[str] = []
        valid_paths: List[str] = []
        for fp in file_paths:
            try:
                text = Path(fp).read_text(encoding="utf-8", errors="ignore")[:2000]
                if text.strip():
                    docs.append(text)
                    valid_paths.append(fp)
            except Exception:
                continue

        if len(docs) < 2:
            return None

        try:
            scorer = BM25Scorer()
            k = min(top_k, len(docs))
            indices = scorer.rerank(query, docs, k)
            if indices is None:
                return None
            reranked = [valid_paths[i] for i in indices if 0 <= i < len(valid_paths)]
            return reranked if reranked else None
        except Exception:
            return None

    # Large-file threshold (chars): files above this size trigger
    # keyword-guided snippet extraction before Monte Carlo sampling.
    _SNIPPET_EXTRACTION_THRESHOLD = 30_000

    @staticmethod
    def _extract_keyword_snippets(
        file_path: str,
        keywords: Dict[str, float],
        max_snippet_chars: int = 20_000,
    ) -> Optional[str]:
        """Extract keyword-relevant lines from a large file.

        For wiki-style files (one JSON article per line), only keeps lines
        that contain at least one keyword.  Returns None when the file is
        small enough to process whole or when no keyword matches.
        """
        try:
            raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            if len(raw) <= AgenticSearch._SNIPPET_EXTRACTION_THRESHOLD:
                return None

            kw_lower = [k.lower() for k in keywords]
            matched_lines = []
            total_chars = 0
            for line in raw.splitlines():
                ll = line.lower()
                if any(k in ll for k in kw_lower):
                    matched_lines.append(line)
                    total_chars += len(line)
                    if total_chars >= max_snippet_chars:
                        break

            if not matched_lines:
                return None
            return "\n".join(matched_lines)
        except Exception:
            return None

    async def _build_cluster(
        self,
        query: str,
        file_paths: List[str],
        query_keywords: Dict[str, float],
        top_k_files: int = 5,
        top_k_snippets: int = 5,
        mces_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[KnowledgeCluster]:
        """Build a KnowledgeCluster via knowledge_base.build().

        For files already processed by Lazy MCES during Phase 2 (cached in
        ``mces_cache``), the cached evidence is reused directly — skipping
        redundant re-extraction.  Remaining files go through the standard
        keyword-snippet pre-filter + MCES pipeline.
        """
        import tempfile

        mces_cache = mces_cache or {}

        try:
            # Partition files into cached (from Phase 2 MCES) and uncached
            cached_evidences: List[EvidenceUnit] = []
            uncached_paths: List[str] = []

            for fp in file_paths[:top_k_files]:
                cached = mces_cache.get(fp)
                if cached:
                    cached_evidences.append(EvidenceUnit(
                        doc_id=FileInfo.get_cache_key(fp),
                        file_or_url=Path(fp),
                        summary=cached.get("summary", ""),
                        is_found=cached.get("is_found", False),
                        snippets=cached.get("snippets", []),
                        extracted_at=datetime.now(tz=timezone.utc),
                        conflict_group=[],
                    ))
                else:
                    uncached_paths.append(fp)

            if cached_evidences:
                await self._logger.info(
                    f"[Phase 3] Reusing {len(cached_evidences)} cached MCES results, "
                    f"{len(uncached_paths)} files need fresh extraction"
                )

            # Process uncached files through the standard pipeline
            mces_evidences: List[EvidenceUnit] = []
            if uncached_paths:
                request = Request(
                    messages=[
                        Message(
                            role="user",
                            content=[ContentItem(type="text", text=query)],
                        ),
                    ],
                )

                effective_paths: List[str] = []
                temp_files: List[str] = []
                for fp in uncached_paths:
                    snippet_text = self._extract_keyword_snippets(fp, query_keywords)
                    if snippet_text:
                        tf = tempfile.NamedTemporaryFile(
                            mode="w", suffix=".txt", delete=False, encoding="utf-8",
                        )
                        tf.write(snippet_text)
                        tf.close()
                        effective_paths.append(tf.name)
                        temp_files.append(tf.name)
                    else:
                        effective_paths.append(fp)

                retrieved_infos = [{"path": fp} for fp in effective_paths]

                cluster = await self.knowledge_base.build(
                    request=request,
                    retrieved_infos=retrieved_infos,
                    keywords=query_keywords,
                    top_k_files=len(uncached_paths),
                    top_k_snippets=top_k_snippets,
                    verbose=self.verbose,
                )
                self.llm_usages.extend(self.knowledge_base.llm_usages)
                self.knowledge_base.llm_usages.clear()

                for tf_path in temp_files:
                    try:
                        os.remove(tf_path)
                    except OSError:
                        pass

                if cluster:
                    # Restore original paths in evidence units
                    for ev in (getattr(cluster, "evidences", None) or []):
                        for orig_fp, eff_fp in zip(uncached_paths, effective_paths):
                            if str(getattr(ev, "file_or_url", "")) == eff_fp and eff_fp != orig_fp:
                                ev.file_or_url = Path(orig_fp)
                                break
                    mces_evidences = list(cluster.evidences or [])

            # Merge cached + freshly extracted evidence
            all_evidences = cached_evidences + mces_evidences

            if not all_evidences:
                return None

            # Build cluster from the merged evidence set
            all_summaries = [ev.summary for ev in all_evidences if ev.summary]
            combined_summary = "\n\n".join(all_summaries) if all_summaries else query

            cluster_text = combined_summary or query
            cluster_id = f"C{hashlib.sha256(cluster_text.encode('utf-8')).hexdigest()[:10]}"

            cluster = KnowledgeCluster(
                id=cluster_id,
                name=query[:60],
                description=[f"Search result for: {query}"],
                content=combined_summary,
                scripts=[],
                resources=[],
                patterns=[],
                constraints=[],
                evidences=all_evidences,
                confidence=0.5,
                abstraction_level=AbstractionLevel.TECHNIQUE,
                landmark_potential=0.5,
                hotness=0.5,
                lifecycle=Lifecycle.EMERGING,
                create_time=datetime.now(tz=timezone.utc),
                last_modified=datetime.now(tz=timezone.utc),
                version=1,
                related_clusters=[],
            )

            n_ev = len(all_evidences)
            n_cached = len(cached_evidences)
            await self._logger.success(
                f"[Phase 3] KnowledgeCluster built: {cluster.name} "
                f"({n_ev} evidence units, {n_cached} from cache)"
            )
            return cluster
        except Exception as exc:
            await self._logger.warning(f"[Phase 3] knowledge_base.build() failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Phase 4: Answer generation
    # ------------------------------------------------------------------

    _EMPTY_EVIDENCE_PATTERNS = re.compile(
        r"(?i)"
        r"(?:no\s+(?:data|information|evidence|results?|relevant)\s+found)"
        r"|(?:no\s+(?:information\s+is\s+)?available)"
        r"|(?:could\s+not\s+(?:find|determine|identify))"
        r"|(?:nothing\s+(?:found|relevant))"
        r"|(?:search\s+returned?\s+no)"
        r"|(?:no\s+matching)"
        r"|(?:insufficient\s+(?:data|evidence|information))"
        # Chinese
        r"|(?:资料缺失)"
        r"|(?:未找到)"
        r"|(?:没有找到)"
        # French
        r"|(?:aucun[e]?\s+(?:donn[ée]e|r[ée]ponse|information|r[ée]sultat)\s+trouv[ée])"
        r"|(?:donn[ée]es?\s+insuffisantes?)"
        # Spanish
        r"|(?:datos?\s+insuficientes?)"
        r"|(?:no\s+se\s+(?:encontr[oó]|hall[oó]))"
        # German
        r"|(?:keine\s+(?:daten|ergebnisse|informationen)\s+gefunden)"
    )

    @classmethod
    def _is_evidence_meaningful(cls, content: str) -> bool:
        """Return True only if content has real substance, not just placeholders."""
        if not content:
            return False
        stripped = content.strip()
        if len(stripped) < 80:
            return False
        if cls._EMPTY_EVIDENCE_PATTERNS.search(stripped[:300]):
            return False
        return True

    @staticmethod
    def _assemble_evidence_snippets(cluster: Any) -> str:
        """Recover usable content from individual evidence summaries/snippets.

        When the overall cluster content assembly fails (e.g. due to JSON
        parsing errors in LLM scoring), individual evidence units may still
        contain useful summaries or raw text snippets.  This method gathers
        those fragments so a summary can be generated without a full
        ReAct fallback, saving significant latency.

        Returns an empty string if no meaningful content can be recovered.
        """
        if not cluster:
            return ""
        parts: List[str] = []
        for ev in (getattr(cluster, "evidences", None) or []):
            summary = getattr(ev, "summary", "")
            if summary and len(summary.strip()) > 30:
                parts.append(summary.strip())
                continue
            for snip in (getattr(ev, "snippets", None) or []):
                text = snip.get("snippet", "") if isinstance(snip, dict) else str(snip)
                if text and len(text.strip()) > 50:
                    parts.append(text.strip())
        if not parts:
            return ""
        combined = "\n\n".join(parts)
        return combined if len(combined) > 100 else ""

    async def _summarise_cluster(
        self,
        query: str,
        cluster: KnowledgeCluster,
        *,
        enable_thinking: bool = False,
    ) -> Tuple[str, bool]:
        """Generate a final answer summary from a KnowledgeCluster.

        Returns:
            ``(summary_text, should_save)`` — *should_save* is the LLM's
            quality verdict on whether the result is worth persisting.
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
        import time as _time
        _t0 = _time.time()
        response = await self.llm.achat(
            messages=[{"role": "user", "content": result_sum_prompt}],
            stream=True,
            enable_thinking=enable_thinking,
        )
        await self._logger.info(f"[Timing] Result summary: {_time.time()-_t0:.2f}s")
        self.llm_usages.append(response.usage)

        summary, should_save = self._parse_summary_response(response.content)
        return summary, should_save

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
        enable_thinking: bool = False,
        memory_prior: Any = None,
        search_plan: Any = None,
    ) -> Tuple[str, SearchContext]:
        """Fall back to ReAct loop when parallel probing yields insufficient evidence.

        The ReAct agent receives pre-extracted keywords and cached
        directory context so it doesn't waste turns re-discovering them.
        When *memory_prior* is provided, the agent's BeliefState is
        warm-started with cross-session priors from RetrievalMemory.
        When *search_plan* is provided, the agent may use guided
        execution (high confidence) or inject the plan as a hint.
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
            enable_thinking=enable_thinking,
            enable_decomposition=True,
            memory_prior=memory_prior,
            search_plan=search_plan,
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
            mces_cache = getattr(context, "mces_cache", {})
            cluster = await self._build_cluster(
                query=query,
                file_paths=discovered,
                query_keywords=query_keywords,
                top_k_files=top_k_files,
                mces_cache=mces_cache,
            )
            if cluster:
                if not cluster.search_results:
                    cluster.search_results = list(discovered)
                # When the evidence pipeline failed to find relevant content
                # (e.g. due to JSON parse errors in scoring) but ReAct already
                # produced a high-quality answer by reading the same files,
                # use the ReAct answer as the authoritative cluster content.
                any_found = any(
                    getattr(ev, "is_found", False)
                    for ev in (cluster.evidences or [])
                )
                if not any_found and answer and len(answer) > len(cluster.content or ""):
                    cluster.content = answer
                    cluster.name = query[:60]
                    cluster.description = [f"Search result for: {query}"]
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
