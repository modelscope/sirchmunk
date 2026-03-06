# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import ast
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from sirchmunk.base import BaseSearch
from sirchmunk.learnings.knowledge_base import KnowledgeBase
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.llm.prompts import (
    generate_keyword_extraction_prompt,
    FAST_QUERY_ANALYSIS,
    ROI_RESULT_SUMMARY,
    SEARCH_RESULT_SUMMARY,
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
from sirchmunk.utils.deps import check_dependencies
from sirchmunk.utils import create_logger, LogCallback
from loguru import logger as _loguru_logger
from sirchmunk.utils.install_rga import install_rga
from sirchmunk.utils.utils import (
    KeywordValidation,
    extract_fields,
    parse_llm_json,
)


class AgenticSearch(BaseSearch):

    def __init__(
        self,
        llm: Optional[OpenAIChat] = None,
        work_path: Optional[Union[str, Path]] = None,
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
        verbose: bool = True,
        log_callback: LogCallback = None,
        reuse_knowledge: bool = True,
        vlm: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._vlm = vlm

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
                from sirchmunk.utils.embedding_util import EmbeddingUtil
                
                self.embedding_client = EmbeddingUtil(
                    cache_dir=str(self.work_path / ".cache" / "models")
                )
                self.embedding_client.start_loading()
                _loguru_logger.info(
                    "Embedding client created, background model loading started"
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
            print("Installing rga (ripgrep-all) and rg (ripgrep)...", flush=True)
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

        # ---- Agent memory (shared with vision pipeline) ----
        self._agent_memory = None
        try:
            from sirchmunk.memory.agent_memory import AgentMemory
            mem_path = str(self.work_path / ".cache" / "agent_memory.duckdb")
            self._agent_memory = AgentMemory(db_path=mem_path)
        except Exception:
            pass

        # ---- Vision search (lazy-initialised on first VISION-mode call) ----
        self._vision_search = None

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
        """Resolve and normalise paths: arg > self.paths > cwd.

        Always returns ``List[str]`` so callers need no further coercion.
        """
        if paths is not None:
            if isinstance(paths, (str, Path)):
                return [str(paths)]
            return [str(p) for p in paths]
        if self.paths is not None:
            return list(self.paths)
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
        """Load historical knowledge clusters from local cache"""
        try:
            stats = self.knowledge_storage.get_stats()
            cluster_count = stats.get('custom_stats', {}).get('total_clusters', 0)
            # Use sync logger for initialization
            print(f"Loaded {cluster_count} historical knowledge clusters from cache")
        except Exception as e:
            print(f"[WARNING] Failed to load historical knowledge: {e}")
    
    async def _try_reuse_cluster(self, query: str) -> Optional[KnowledgeCluster]:
        """Try to reuse existing knowledge cluster based on semantic similarity.

        Returns:
            KnowledgeCluster if a suitable cached cluster is found, None otherwise.
        """
        if not self.embedding_client:
            return None

        # Skip cluster reuse while the embedding model is still loading in
        # its background thread; kick off loading so it's ready next time.
        if not self.embedding_client.is_ready():
            self.embedding_client.start_loading()
            return None

        try:
            await self._logger.info("Searching for similar knowledge clusters...")
            
            # Compute query embedding
            query_embedding = (await self.embedding_client.embed([query]))[0]
            
            # Search for similar clusters
            similar_clusters = await self.knowledge_storage.search_similar_clusters(
                query_embedding=query_embedding,
                top_k=self.cluster_sim_top_k,
                similarity_threshold=self.cluster_sim_threshold,
            )
            
            if not similar_clusters:
                await self._logger.info("No similar clusters found, performing new search...")
                return None
            
            # Found similar cluster - process reuse
            best_match = similar_clusters[0]
            await self._logger.success(
                f"♻️ Found similar cluster: {best_match['name']} "
                f"(similarity: {best_match['similarity']:.3f})"
            )
            
            # Retrieve full cluster object
            existing_cluster = await self.knowledge_storage.get(best_match["id"])
            
            if not existing_cluster:
                await self._logger.warning("Failed to retrieve cluster, falling back to new search")
                return None
            
            # Add current query to queries list with FIFO strategy
            self._add_query_to_cluster(existing_cluster, query)
            
            # Update hotness and timestamp for reused cluster
            existing_cluster.hotness = min(1.0, (existing_cluster.hotness or 0.5) + 0.1)
            existing_cluster.last_modified = datetime.now()
            
            # Recompute embedding with new query (before update to avoid double save)
            if self.embedding_client and self.embedding_client.is_ready():
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
                    await self._logger.debug(f"Updated embedding for cluster {existing_cluster.id}")
                except Exception as emb_error:
                    await self._logger.warning(f"Failed to update embedding: {emb_error}")
            
            # Single update call - saves cluster data and embedding together
            await self.knowledge_storage.update(existing_cluster)
            
            # Validate cluster has usable content
            content = existing_cluster.content
            if isinstance(content, list):
                content = "\n".join(content)
            if not content:
                await self._logger.warning(
                    f"Cluster {existing_cluster.id} has empty content, falling back to full search"
                )
                return None

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
        # Save knowledge cluster to persistent storage
        try:
            await self.knowledge_storage.insert(cluster)
            await self._logger.info(f"Saved knowledge cluster {cluster.id} to cache")
        except Exception as e:
            # If cluster exists, update it instead
            try:
                await self.knowledge_storage.update(cluster)
                await self._logger.info(f"Updated knowledge cluster {cluster.id} in cache")
            except Exception as update_error:
                await self._logger.warning(f"Failed to save knowledge cluster: {update_error}")
                return

        # Compute and store embedding for the cluster.
        # embed() internally awaits model readiness via _ensure_model_async(),
        # so even if the background loading thread hasn't finished yet, we
        # block (non-blocking async) until the model is ready rather than
        # silently skipping the embedding — which would make the cluster
        # invisible to future similarity searches.
        if self.embedding_client:
            try:
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
    
    def _record_search_episode(
        self,
        query: str,
        query_type: str,
        paths: List[str],
        hit_paths: Optional[List[str]] = None,
        timings: Optional[Dict[str, float]] = None,
        phase_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a search episode to the shared agent memory.

        Silently no-ops when ``AgentMemory`` is unavailable (e.g. vision
        extras not installed).
        """
        if self._agent_memory is None:
            return
        try:
            import uuid
            from sirchmunk.memory.agent_memory import SearchEpisode
            ep = SearchEpisode(
                episode_id=str(uuid.uuid4())[:12],
                query=query,
                query_type=query_type,
                paths=paths,
                top_k=0,
                result_count=len(hit_paths or []),
                hit_paths=hit_paths or [],
                timings=timings or {},
                phase_stats=phase_stats or {},
            )
            self._agent_memory.record_episode(ep)
        except Exception:
            pass

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
        return KnowledgeCluster(
            id=f"{prefix}{_digest}",
            name=query[:60],
            description=[f"Search result for: {query}"],
            content=answer,
            queries=[query],
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
                await self._logger.success(f" ✓ Found {len(results)} matching files", flush=True)
            else:
                await self._logger.warning("No files matched the patterns")
            
            return results
        
        except Exception as e:
            await self._logger.error(f"Filename search failed: {e}")
            import traceback
            await self._logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    @staticmethod
    def _parse_summary_response(llm_response: str) -> tuple[str, bool]:
        """
        Parse LLM response to extract summary and save decision.
        
        Args:
            llm_response: Raw LLM response containing SUMMARY and SHOULD_SAVE tags
        
        Returns:
            Tuple of (summary_text, should_save_flag)
        """
        # Extract SUMMARY content
        summary_fields = extract_fields(content=llm_response, tags=["SUMMARY", "SHOULD_SAVE"])
        
        summary = (summary_fields.get("summary") or "").strip()
        should_save_str = (summary_fields.get("should_save") or "true").strip().lower()
        
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
        tags = [f"KEYWORDS_LEVEL_{i+1}" for i in range(num_levels)]

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
                except Exception as e:
                    keyword_sets.append({})
                    continue

            # Validate using Pydantic model
            try:
                validated = KeywordValidation(root=keywords_dict).model_dump()
                keyword_sets.append(validated)
            except Exception as e:
                keyword_sets.append({})

        return keyword_sets

    # ------------------------------------------------------------------
    # Agentic (ReAct) infrastructure — lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_tool_registry(
        self,
        paths: List[str],
        enable_dir_scan: bool = True,
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
    # Vision search (lazy initialisation)
    # ------------------------------------------------------------------

    def _ensure_vision_search(self):
        """Build the VisionSearch orchestrator on first use.

        Heavy dependencies (torch, transformers, SigLIP2) are imported and
        loaded only when VISION mode is actually requested, keeping FAST /
        DEEP searches lightweight.

        Returns:
            Ready-to-use ``VisionSearch`` instance.
        """
        if self._vision_search is not None:
            return self._vision_search

        from sirchmunk.vision.vision_search import VisionSearch
        from sirchmunk.llm.vlm_chat import VLMClient

        vlm = self._vlm
        if vlm is None:
            base_url = os.getenv("VLM_BASE_URL", "")
            api_key = os.getenv("VLM_API_KEY", os.getenv("LLM_API_KEY", ""))
            if self.llm is not None:
                if not base_url:
                    try:
                        base_url = str(self.llm._client.base_url)
                    except AttributeError:
                        base_url = ""
                if not api_key:
                    try:
                        api_key = self.llm._client.api_key
                    except AttributeError:
                        pass
            vlm = VLMClient(
                base_url=base_url,
                api_key=api_key,
                model=os.getenv("VLM_MODEL", "qwen3.5-plus"),
            )

        vision_work = str(self.work_path / ".cache" / "vision")
        self._vision_search = VisionSearch(
            vlm=vlm,
            work_path=vision_work,
            memory=self._agent_memory,
        )
        _loguru_logger.info(
            f"VisionSearch initialised (work_path={vision_work})"
        )
        return self._vision_search

    async def _try_vision_dispatch(
        self,
        query: str,
        paths: list,
        mode: str = "FAST",
    ) -> Optional[list]:
        """Check whether *query* has vision intent and dispatch if so.

        Uses the same LLM call as FAST keyword extraction (the
        ``modality`` field).  Returns ``None`` when the query is
        text-oriented so the caller can continue with the text pipeline.
        Returns a list of ``VisionSearchResult`` when vision intent is
        detected and the pipeline succeeds.  Falls back to ``None`` on
        any vision pipeline error.
        """
        prompt = FAST_QUERY_ANALYSIS.format(user_input=query)
        resp = await self.llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        self.llm_usages.append(resp.usage)
        analysis = self._parse_fast_json(resp.content)
        modality = analysis.get("modality", "text").lower()

        if modality != "vision":
            return None

        await self._logger.info(
            f"[search] Vision intent detected — routing to vision pipeline ({mode})"
        )
        try:
            vs = self._ensure_vision_search()
            return await vs.search(
                query=query,
                paths=[str(p) for p in paths],
                top_k=10,
                mode=mode,
            )
        except Exception as e:
            await self._logger.warning(
                f"[search] Vision pipeline failed ({e}), continuing with text"
            )
            return None

    # ------------------------------------------------------------------
    # Unified search entry point
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str = "",
        paths: Optional[Union[str, Path, List[str], List[Path]]] = None,
        *,
        mode: Literal["DEEP", "FAST", "FILENAME_ONLY"] = "FAST",
        query_images: Optional[List[Any]] = None,
        max_loops: int = 10,
        max_token_budget: int = 64000,
        max_depth: Optional[int] = 8,
        top_k_files: int = 3,
        top_k: int = 10,
        enable_dir_scan: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        return_context: bool = False,
        spec_stale_hours: float = 72.0,
    ) -> Union[str, SearchContext, List[Dict[str, Any]]]:
        """Perform intelligent search with multi-mode support.

        Vision intent is **auto-detected** during query analysis — if the
        LLM determines the user is looking for images, the query is
        transparently routed to the vision pipeline.  When ``query_images``
        is provided, vision mode is forced.  The vision pipeline inherits
        the current ``mode`` (FAST or DEEP):

        - **FAST vision**: Visual Grep → SigLIP2 → return (no VLM call).
        - **DEEP vision**: Visual Grep → SigLIP2 → VLM verification →
          feedback loop.

        Modes:
            +--------------+-------------------+-------------------------------------------+
            | Mode         | Speed / LLM Calls | Description                               |
            +--------------+-------------------+-------------------------------------------+
            | FILENAME_ONLY| Very Fast / 0     | Pattern-based file discovery, no LLM.     |
            | FAST         | 1-5s / 0-2        | Greedy: cluster reuse or keyword search    |
            |              |                   | → best file → answer. Early termination.  |
            |              |                   | Auto-detects vision intent.               |
            | DEEP         | 5-30s / 4-6       | Parallel multi-path retrieval + ReAct     |
            |              |                   | refinement with Monte-Carlo evidence.     |
            |              |                   | Auto-detects vision intent.               |
            +--------------+-------------------+-------------------------------------------+

        Args:
            query: User's search query.
            paths: Directories / files to search.  Falls back to
                ``self.paths`` or the current working directory.
            mode: Search mode — ``"DEEP"``, ``"FAST"``, or
                ``"FILENAME_ONLY"``.
            max_loops: Maximum ReAct iterations (DEEP mode, default: 10).
            max_token_budget: LLM token budget (DEEP mode, default: 64000).
            max_depth: Maximum directory depth for file search (default: 5).
            top_k_files: Max files for evidence extraction (default: 3).
            top_k: Max results for vision search (default: 10).
            enable_dir_scan: Enable directory scanning tool (DEEP mode).
            include: File glob patterns to include.
            exclude: File glob patterns to exclude.
            return_context: If True, return ``(answer, SearchContext)`` tuple.
            return_cluster: If True, return the full KnowledgeCluster.
            include: File glob patterns to include (e.g. ``["*.py", "*.md"]``).
                Used in both FILENAME_ONLY and DEEP modes.
            exclude: File glob patterns to exclude (e.g. ``["*.log"]``).
                Used in both FILENAME_ONLY and DEEP modes.
            return_context: If True, return a ``SearchContext`` object
                that carries ``answer``, ``cluster`` (KnowledgeCluster),
                and full pipeline telemetry (LLM usage, files read, etc.).
            spec_stale_hours: Hours before spec cache is stale (default: 72).
            query_images: Reference images for vision similarity or
                hybrid search.  Each element can be a file path ``str``,
                base64 data URI ``str``, raw ``bytes``, or a
                ``PIL.Image.Image`` object.  When provided, forces vision
                mode.

        Returns:
            - ``str``: Answer summary (default).
            - ``SearchContext``: If *return_context* — contains ``answer``,
              ``cluster``, and telemetry in a single object.
            - ``List[Dict]``: File matches in FILENAME_ONLY mode.
            - ``str``: Answer summary (default for FAST/DEEP text search).
            - ``(str, SearchContext)``: If *return_context* (text search).
            - ``KnowledgeCluster``: If *return_cluster* (text search).
            - ``List[Dict]``: File matches (FILENAME_ONLY).
            - ``List[VisionSearchResult]``: Image results (vision).
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
        # Force vision mode when query_images are provided
        if query_images:
            paths = self._resolve_paths(paths)
            vs = self._ensure_vision_search()
            return await vs.search(
                query=query or "",
                paths=[str(p) for p in paths],
                top_k=top_k,
                query_images=query_images,
                mode=mode,
            )

        paths = self._resolve_paths(paths)

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
                top_k_files=top_k_files, top_k=top_k,
                include=include, exclude=exclude,
            )
        else:
            answer, cluster, context = await self._search_deep(
                query=query, paths=paths,
                max_loops=max_loops, max_token_budget=max_token_budget,
                max_depth=max_depth, top_k_files=top_k_files,
                enable_dir_scan=enable_dir_scan,
                include=include, exclude=exclude,
                spec_stale_hours=spec_stale_hours,
            )

        # ---- Unified return wrapping ----
        if return_context:
            prefix = "FS" if mode == "FAST" else "DS"
            context.answer = answer
            context.cluster = cluster or self._make_answer_cluster(
                query, answer, prefix,
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
        max_token_budget: int = 64000,
        max_depth: Optional[int] = 5,
        top_k_files: int = 3,
        enable_dir_scan: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        spec_stale_hours: float = 72.0,
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
            return direct, self._make_answer_cluster(query, direct, "DQ"), context

        # ==============================================================
        # Phase 0: Cluster reuse (instant short-circuit)
        # ==============================================================
        reused = await self._try_reuse_cluster(query)
        if reused is not None:
            content = reused.content
            if isinstance(content, list):
                content = "\n".join(content)
            return str(content), reused, context

        await self._logger.info(f"[search] Starting multi-path retrieval for: '{query[:80]}'")

        # ==============================================================
        # Phase 0c: Vision intent quick-check (single LLM call, no-op
        #           if the query doesn't look image-related)
        # ==============================================================
        vision_result = await self._try_vision_dispatch(query, paths, mode="DEEP")
        if vision_result is not None:
            summary = (
                f"Found {len(vision_result)} image(s) matching your query."
                if vision_result else "No matching images found."
            )
            return summary, self._make_answer_cluster(query, summary, "VS"), context

        # ==============================================================
        # Phase 1: Parallel probing — all four paths fire concurrently
        # ==============================================================
        await self._logger.info("[Phase 1] Parallel probing: keywords + dir_scan + knowledge + spec_cache")
        context.increment_loop()

        phase1_results = await asyncio.gather(
            self._probe_keywords(query),
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

        await self._logger.info(
            f"[Phase 1] Results: keywords={len(initial_keywords)}, "
            f"dir_scan={'OK' if scan_result else 'N/A'}, "
            f"knowledge_hits={len(knowledge_hits)}, "
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
        # ==============================================================
        context.increment_loop()
        merged_files = self._merge_file_paths(
            keyword_files=keyword_files,
            dir_scan_files=dir_scan_files,
            knowledge_hits=knowledge_hits,
        )
        await self._logger.info(f"[Phase 3] Merged {len(merged_files)} unique candidate files")

        cluster: Optional[KnowledgeCluster] = None
        if merged_files:
            cluster = await self._build_cluster(
                query=query, file_paths=merged_files,
                query_keywords=query_keywords, top_k_files=top_k_files,
            )

        # ==============================================================
        # Phase 4: Generate answer — cluster summary or ReAct refinement
        # ==============================================================
        context.increment_loop()
        answer: str = ""

        if cluster and cluster.content:
            await self._logger.info("[Phase 4] Evidence sufficient, generating summary")
            answer = await self._summarise_cluster(query, cluster)
            if not cluster.search_results:
                cluster.search_results = list(merged_files)
        else:
            await self._logger.info("[Phase 4] Evidence insufficient, launching ReAct refinement")
            answer, context = await self._react_refinement(
                query=query, paths=paths,
                initial_keywords=initial_keywords, spec_context=spec_context,
                enable_dir_scan=enable_dir_scan,
                max_loops=max_loops, max_token_budget=max_token_budget,
                max_depth=max_depth, include=include, exclude=exclude,
            )

            if not cluster:
                cluster = await self._build_cluster_from_context(
                    query=query, answer=answer, context=context,
                    query_keywords=query_keywords, top_k_files=top_k_files,
                )
            elif answer and not cluster.content:
                cluster.content = answer

        # Sync LLM token accounting into context
        new_usages = self.llm_usages[_llm_usage_start:]
        for usage in new_usages:
            if usage and isinstance(usage, dict):
                total_tok = usage.get("total_tokens", 0)
                if total_tok == 0:
                    total_tok = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                context.add_llm_tokens(total_tok, usage=usage)

        # ==============================================================
        # Phase 5: Persistence
        # ==============================================================
        phase5_tasks = []
        if cluster:
            self._add_query_to_cluster(cluster, query)
            phase5_tasks.append(self._save_cluster_with_embedding(cluster))
        phase5_tasks.append(self._save_spec_context(paths, context, scan_result=scan_result))
        await asyncio.gather(*phase5_tasks, return_exceptions=True)

        await self._logger.success(f"[search] Complete: {context.summary()}")

        self._record_search_episode(
            query=query, query_type="text_deep", paths=paths,
            hit_paths=list(context.read_file_ids) if context else [],
        )
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

        # Step 3: extract, (optionally sample), and analyse
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
    # FAST mode — greedy search with early termination
    # ------------------------------------------------------------------

    _FAST_TEXT_EXTENSIONS = {
        ".txt", ".md", ".rst", ".csv", ".log", ".tsv",
        ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".xml",
        ".html", ".htm", ".sh", ".toml", ".cfg", ".ini", ".conf",
        ".css", ".bash", ".java", ".c", ".cpp", ".h", ".go", ".rs",
    }
    _FAST_CONTEXT_WINDOW = 30       # ± lines around each grep hit
    _FAST_MAX_EVIDENCE_CHARS = 15_000

    async def _search_fast(
        self,
        query: str,
        paths: List[str],
        *,
        max_depth: Optional[int] = 5,
        top_k_files: int = 2,
        top_k: int = 10,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[KnowledgeCluster], SearchContext]:
        """Greedy search: 2 LLM calls, single best file, focused evidence.

        Two-level keyword cascade extracted in one LLM call:
        primary (compound phrase) is tried first; if it misses, fallback
        (atomic terms) is tried.  Greedy early-termination at every step.

        Returns:
            ``(answer, cluster, context)`` — same triple as ``_search_deep``
            so the caller can handle both modes uniformly.
        """
        context = SearchContext()
        await self._logger.info(f"[FAST] Starting greedy search for: '{query[:80]}'")

        # ==============================================================
        # Step 0: Cluster reuse — instant short-circuit (no LLM cost)
        # ==============================================================
        reused = await self._try_reuse_cluster(query)
        if reused is not None:
            content = reused.content
            if isinstance(content, list):
                content = "\n".join(content)
            await self._logger.success("[FAST] Reused cached knowledge cluster")
            return str(content), reused, context

        # ==============================================================
        # Step 1: LLM → 2-level keywords in one call (stream=False)
        # ==============================================================
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
        primary = analysis.get("primary", [])[:2]
        fallback = analysis.get("fallback", [])[:3]
        file_hints = analysis.get("file_hints", [])
        modality = analysis.get("modality", "text").lower()

        # ==============================================================
        # Step 1b: Vision intent auto-detection
        # ==============================================================
        if modality == "vision":
            await self._logger.info(
                f"[FAST:Step1] Vision intent detected — routing to vision pipeline"
            )
            try:
                vs = self._ensure_vision_search()
                return await vs.search(
                    query=query,
                    paths=[str(p) for p in paths],
                    top_k=top_k,
                    mode="FAST",
                )
            except Exception as e:
                await self._logger.warning(
                    f"[FAST] Vision search failed ({e}), falling back to text"
                )

        if not primary and not fallback:
            await self._logger.warning("[FAST] No keywords extracted")
            msg = f"Could not extract search terms from query: '{query}'"
            return msg, None, context

        await self._logger.info(
            f"[FAST:Step1] Primary: {primary}, Fallback: {fallback}"
        )

        # ==============================================================
        # Step 2: rga cascade — primary first, fallback only if needed
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
        answer_resp = await self.llm.achat(
            messages=[{"role": "user", "content": answer_prompt}],
            stream=True,
        )
        self.llm_usages.append(answer_resp.usage)
        if answer_resp.usage and isinstance(answer_resp.usage, dict):
            context.add_llm_tokens(
                answer_resp.usage.get("total_tokens", 0), usage=answer_resp.usage,
            )

        answer = answer_resp.content or ""
        keywords_used = primary if used_level == "primary" else fallback
        cluster = self._build_fast_cluster(
            query, answer, file_path, evidence, keywords_used,
        )

        # Persist the FAST cluster so it can be reused by future queries
        self._add_query_to_cluster(cluster, query)
        try:
            await self._save_cluster_with_embedding(cluster)
        except Exception as exc:
            _loguru_logger.warning(
                f"[FAST] Failed to save cluster with embedding: {exc}"
            )

        await self._logger.success("[FAST] Search complete (2 LLM calls)")

        self._record_search_episode(
            query=query, query_type="text_fast", paths=paths,
            hit_paths=[file_path],
        )
        return answer_resp.content or ""
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
                    timeout=15.0,
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
                    timeout=15.0,
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
                    timeout=15.0,
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

        merged = GrepRetriever.merge_results(all_raw, limit=20)
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

    @staticmethod
    async def _fast_read_file_head(
        file_path: str, max_chars: int = 8_000,
    ) -> str:
        """Read the head of a file as last-resort evidence."""
        try:
            p = Path(file_path)
            if p.suffix.lower() in AgenticSearch._FAST_TEXT_EXTENSIONS:
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
        return parse_llm_json(text)

    # ------------------------------------------------------------------
    # Phase 1 probes (each designed to run concurrently)
    # ------------------------------------------------------------------

    async def _probe_keywords(
        self, query: str,
    ) -> Tuple[Dict[str, float], List[str]]:
        """Extract multi-level keywords from the query via LLM.

        Returns:
            Tuple of (keyword_idf_dict, keyword_list).
        """
        await self._logger.info("[Probe:Keywords] Extracting keywords...")
        dynamic_prompt = generate_keyword_extraction_prompt(num_levels=2)
        keyword_prompt = dynamic_prompt.format(user_input=query)
        kw_response = await self.llm.achat(
            messages=[{"role": "user", "content": keyword_prompt}],
            stream=False,
        )
        self.llm_usages.append(kw_response.usage)

        keyword_sets = self._extract_and_validate_multi_level_keywords(
            kw_response.content, num_levels=2,
        )
        for kw_set in keyword_sets:
            if kw_set:
                kw_list = list(kw_set.keys())
                await self._logger.info(f"[Probe:Keywords] Extracted: {kw_list}")
                return kw_set, kw_list

        return {}, []

    async def _probe_dir_scan(
        self,
        paths: List[str],
        enable: bool = True,
    ):
        """Scan directories for file metadata (filesystem only, no LLM).

        Args:
            paths: Normalised list of path strings to scan.
            enable: Whether directory scanning is enabled.

        Returns:
            ScanResult or None if disabled / no scanner.
        """
        if not enable:
            return None

        from sirchmunk.scan.dir_scanner import DirectoryScanner

        if self._dir_scanner is None:
            self._dir_scanner = DirectoryScanner(llm=self.llm, max_files=500)

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
        self, query: str, scan_result,
    ) -> List[str]:
        """Run LLM ranking on dir_scan candidates and return high-relevance paths only."""
        if self._dir_scanner is None:
            return []

        ranked = await self._dir_scanner.rank(query, scan_result, top_k=20)
        paths = [
            c.path for c in ranked.ranked_candidates
            if c.relevance == "high"
        ]
        await self._logger.info(
            f"[Retrieve:DirScan] {len(paths)} high-relevance files"
        )
        return paths

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

    async def _build_cluster(
        self,
        query: str,
        file_paths: List[str],
        query_keywords: Dict[str, float],
        top_k_files: int = 3,
        top_k_snippets: int = 5,
    ) -> Optional[KnowledgeCluster]:
        """Build a KnowledgeCluster via knowledge_base.build().

        Constructs the Request wrapper and delegates to the knowledge
        base for parallel Monte Carlo evidence sampling.
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

    # ------------------------------------------------------------------
    # Phase 4: Answer generation
    # ------------------------------------------------------------------

    async def _summarise_cluster(
        self, query: str, cluster: KnowledgeCluster,
    ) -> str:
        """Generate a final answer summary from a KnowledgeCluster.

        Same pipeline as ``search()``'s final summarisation step.
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

        summary, should_save = self._parse_summary_response(response.content)
        return summary

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
        top_k_files: int = 3,
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
        now = datetime.now()
        stale_seconds = stale_hours * 3600

        for sp in paths:
            spec_file = self._spec_file(sp)
            if not spec_file.exists():
                continue
            try:
                raw = spec_file.read_text(encoding="utf-8")
                data = json.loads(raw)

                # Skip if stale
                cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
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
                        "cached_at": datetime.now().isoformat(),
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
