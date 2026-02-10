# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Sirchmunk Service Wrapper for MCP Server.

Provides a high-level interface to Sirchmunk's AgenticSearch functionality,
managing initialization, configuration, and session state.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .config import Config

if TYPE_CHECKING:
    from sirchmunk.schema.knowledge import KnowledgeCluster


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output.
    
    Used during initialization to prevent third-party libraries
    (ModelScope, transformers, etc.) from printing to stdout,
    which would break MCP stdio protocol.
    """
    # Check if we're in stdio MCP mode (stdout should be protected)
    if os.environ.get("MCP_TRANSPORT") == "stdio":
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            sys.stdout = old_stdout
    else:
        yield


class SirchmunkService:
    """Service wrapper for AgenticSearch with lifecycle management.
    
    This class manages the AgenticSearch instance and provides a clean interface
    for MCP tool implementations.
    
    Attributes:
        config: Configuration object
        search: AgenticSearch instance
        initialized: Whether the service is initialized
    """
    
    def __init__(self, config: Config):
        """Initialize Sirchmunk service.
        
        Args:
            config: Configuration object
        
        Raises:
            RuntimeError: If initialization fails
        """
        self.config = config
        self.searcher: Optional[AgenticSearch] = None
        self.initialized = False
        
        logger.info(f"Initializing Sirchmunk service with config: {config.sirchmunk.work_path}")
        
        try:
            self._initialize_search()
            self.initialized = True
            logger.info("Sirchmunk service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Sirchmunk service: {e}")
            raise RuntimeError(f"Sirchmunk service initialization failed: {e}") from e
    
    def _initialize_search(self) -> None:
        """Initialize AgenticSearch instance with configuration.
        
        Raises:
            Exception: If AgenticSearch initialization fails
        """
        # Import sirchmunk modules inside function to allow stdout suppression
        # These imports may trigger model downloads that print to stdout
        with suppress_stdout():
            from sirchmunk.search import AgenticSearch
            from sirchmunk.llm.openai_chat import OpenAIChat
        
        # Create LLM client
        llm = OpenAIChat(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            model=self.config.llm.model_name,
        )
        
        # Create AgenticSearch instance with stdout suppression
        # AgenticSearch may load embedding models which print progress
        with suppress_stdout():
            self.searcher = AgenticSearch(
                llm=llm,
                work_path=self.config.sirchmunk.work_path,
                paths=self.config.sirchmunk.paths,
                verbose=False,  # Disable verbose in stdio mode to prevent stdout pollution
                reuse_knowledge=self.config.sirchmunk.enable_cluster_reuse,
                cluster_sim_threshold=self.config.sirchmunk.cluster_similarity.threshold,
                cluster_sim_top_k=self.config.sirchmunk.cluster_similarity.top_k,
            )
        
        logger.info("AgenticSearch instance created")
    
    async def search(
        self,
        query: str,
        paths: Optional[Union[str, List[str]]] = None,
        mode: str = "DEEP",
        max_depth: Optional[int] = None,
        top_k_files: Optional[int] = None,
        max_loops: Optional[int] = None,
        max_token_budget: Optional[int] = None,
        enable_dir_scan: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        return_cluster: bool = False,
    ) -> Union[str, List[Dict[str, Any]], KnowledgeCluster, None]:
        """Search and retrieve various types of raw documents using AgenticSearch.
        
        Supports DEEP mode (parallel multi-path + ReAct refinement) and
        FILENAME_ONLY mode (fast filename pattern matching, no LLM).
        
        Args:
            query: Search query or question to find relevant documents
            paths: Paths to search in (files or directories).
                Optional — falls back to configured default or cwd.
            mode: Search mode (DEEP, FILENAME_ONLY)
            max_depth: Maximum directory depth to search
            top_k_files: Number of top files to return
            max_loops: Maximum ReAct iterations (DEEP mode)
            max_token_budget: LLM token budget (DEEP mode)
            enable_dir_scan: Enable directory scanning (DEEP mode)
            include: File patterns to include (glob)
            exclude: File patterns to exclude (glob)
            return_cluster: Whether to return full KnowledgeCluster object
        
        Returns:
            Search results: str (summary), List[Dict] (FILENAME_ONLY), 
            KnowledgeCluster (if return_cluster=True), or None (if no results)
        
        Raises:
            RuntimeError: If service is not initialized
            ValueError: If parameters are invalid
        """
        if not self.initialized or self.searcher is None:
            raise RuntimeError("Sirchmunk service is not initialized")
        
        # Validate mode
        if mode not in ("DEEP", "FILENAME_ONLY"):
            raise ValueError(f"Invalid mode: {mode}. Must be DEEP or FILENAME_ONLY")
        
        # Normalize paths
        if isinstance(paths, str):
            paths = [paths]
        
        # Validate search paths if provided
        if paths:
            for p in paths:
                path_obj = Path(p)
                if not path_obj.exists():
                    logger.warning(f"Search path does not exist: {p}")
        
        # Apply defaults from configuration
        max_depth = max_depth or self.config.sirchmunk.search_defaults.max_depth
        top_k_files = top_k_files or self.config.sirchmunk.search_defaults.top_k_files
        
        logger.info(
            f"Starting search: mode={mode}, query='{query[:50]}...', "
            f"paths={len(paths) if paths else 'default'}, max_depth={max_depth}"
        )
        
        try:
            # Build kwargs (only pass params that are set)
            kwargs: Dict[str, Any] = {
                "query": query,
                "paths": paths,
                "mode": mode,
                "max_depth": max_depth,
                "top_k_files": top_k_files,
                "enable_dir_scan": enable_dir_scan,
                "include": include,
                "exclude": exclude,
                "return_cluster": return_cluster,
            }
            if max_loops is not None:
                kwargs["max_loops"] = max_loops
            if max_token_budget is not None:
                kwargs["max_token_budget"] = max_token_budget

            result = await self.searcher.search(**kwargs)
            
            logger.info(f"Search completed: mode={mode}, result_type={type(result).__name__}")
            return result
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    async def get_cluster(self, cluster_id: str) -> Optional[KnowledgeCluster]:
        """Retrieve a knowledge cluster by ID.
        
        Args:
            cluster_id: Cluster ID (e.g., 'C1007')
        
        Returns:
            KnowledgeCluster if found, None otherwise
        
        Raises:
            RuntimeError: If service is not initialized
        """
        if not self.initialized or self.searcher is None:
            raise RuntimeError("Sirchmunk service is not initialized")
        
        try:
            cluster = await self.searcher.knowledge_storage.get(cluster_id)
            if cluster:
                logger.info(f"Retrieved cluster: {cluster_id}")
            else:
                logger.warning(f"Cluster not found: {cluster_id}")
            return cluster
        except Exception as e:
            logger.error(f"Failed to get cluster {cluster_id}: {e}")
            raise
    
    async def list_clusters(
        self,
        limit: int = 10,
        sort_by: str = "last_modified",
    ) -> List[Dict[str, Any]]:
        """List saved knowledge clusters with optional filtering.
        
        Args:
            limit: Maximum number of clusters to return
            sort_by: Sort field (hotness, confidence, last_modified)
        
        Returns:
            List of cluster metadata dictionaries
        
        Raises:
            RuntimeError: If service is not initialized
        """
        if not self.initialized or self.searcher is None:
            raise RuntimeError("Sirchmunk service is not initialized")
        
        try:
            # Get all cluster IDs
            all_clusters = await self.searcher.knowledge_storage.list_all()
            
            # Sort clusters
            if sort_by == "hotness":
                all_clusters.sort(key=lambda c: c.hotness or 0.0, reverse=True)
            elif sort_by == "confidence":
                all_clusters.sort(key=lambda c: c.confidence or 0.0, reverse=True)
            else:  # last_modified
                all_clusters.sort(key=lambda c: c.last_modified, reverse=True)
            
            # Limit results
            result_clusters = all_clusters[:limit]
            
            # Convert to dictionaries
            results = []
            for cluster in result_clusters:
                results.append({
                    "id": cluster.id,
                    "name": cluster.name,
                    "confidence": cluster.confidence,
                    "hotness": cluster.hotness,
                    "lifecycle": cluster.lifecycle.value,
                    "version": cluster.version,
                    "last_modified": cluster.last_modified.isoformat() if cluster.last_modified else None,
                    "queries": cluster.queries,
                    "evidences_count": len(cluster.evidences),
                })
            
            logger.info(f"Listed {len(results)} clusters (limit={limit}, sort_by={sort_by})")
            return results
        
        except Exception as e:
            logger.error(f"Failed to list clusters: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics.
        
        Returns:
            Dictionary with service statistics
        
        Raises:
            RuntimeError: If service is not initialized
        """
        if not self.initialized or self.searcher is None:
            raise RuntimeError("Sirchmunk service is not initialized")
        
        try:
            # Get knowledge manager stats
            stats = self.searcher.knowledge_storage.get_stats()
            
            # Add service-level stats
            stats["service"] = {
                "initialized": self.initialized,
                "work_path": str(self.config.sirchmunk.work_path),
                "cluster_reuse_enabled": self.config.sirchmunk.enable_cluster_reuse,
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the service.
        
        Performs cleanup operations like closing connections and saving state.
        """
        logger.info("Shutting down Sirchmunk service")
        
        try:
            # Currently no cleanup needed, but this provides extension point
            self.initialized = False
            logger.info("Sirchmunk service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
