# Copyright (c) ModelScope Contributors. All rights reserved.
"""
MCP Server implementation for Sirchmunk using FastMCP.

Provides the main MCP server that exposes Sirchmunk functionality
as MCP tools following the Model Context Protocol specification.
"""

import asyncio
import logging
import sys
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .config import Config
from .service import SirchmunkService


logger = logging.getLogger(__name__)

# Global service instance (initialized when server starts)
_service: Optional[SirchmunkService] = None


def create_server(config: Config) -> FastMCP:
    """Create and configure FastMCP server instance.
    
    Args:
        config: Configuration object
    
    Returns:
        Configured FastMCP server instance
    """
    global _service
    
    # Initialize service
    _service = SirchmunkService(config)
    
    # Create FastMCP server
    mcp = FastMCP(
        name=config.mcp.server_name,
    )
    
    logger.info(
        f"Creating MCP server: {config.mcp.server_name}"
    )
    
    # Register tools using decorators
    @mcp.tool()
    async def sirchmunk_search(
        query: str,
        paths: Optional[List[str]] = None,
        mode: str = "DEEP",
        max_depth: int = 5,
        top_k_files: int = 3,
        max_loops: int = 10,
        max_token_budget: int = 64000,
        enable_dir_scan: bool = True,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        return_cluster: bool = False,
    ) -> str:
        """Search local files, documents, and raw data on disk. Supports 100+ file formats
        including PDF, Word, Excel, PowerPoint, CSV, JSON, YAML, Markdown, HTML, source code,
        images (OCR), archives (zip/tar/gz), emails (eml/msg), eBooks (epub), Jupyter notebooks,
        and more — no pre-indexing or embedding required.

        USE THIS TOOL WHEN YOU NEED TO:
        - Search through local files and directories on disk
        - Find information inside documents (PDF, DOCX, XLSX, PPTX, etc.)
        - Search raw data files that other tools cannot parse
        - Answer questions about content in local files or codebases
        - Locate specific files by name or content pattern

        Modes:
        - DEEP: Comprehensive search with LLM-powered analysis. Reads file contents,
          extracts evidence via Monte Carlo sampling, and synthesizes an answer. (10-30s)
        - FILENAME_ONLY: Fast filename pattern matching across directories. (<1s)

        Args:
            query: Natural language question or search keywords.
                Examples: 'How does authentication work?', 'database schema migration',
                'find all config files related to logging'
            paths: Local filesystem paths to search (files or directories).
                Examples: ['/home/user/projects'], ['./src', './docs'], ['/data/reports']
                Optional — falls back to configured SIRCHMUNK_SEARCH_PATHS or cwd.
            mode: Search mode - DEEP (comprehensive content analysis) or FILENAME_ONLY (fast file discovery)
            max_depth: Maximum directory depth to search (1-20, default: 5)
            top_k_files: Number of top files to analyze (1-20, default: 3)
            max_loops: Maximum ReAct agent iterations for adaptive retrieval (1-20, default: 10)
            max_token_budget: Token budget for DEEP mode (default: 64000)
            enable_dir_scan: Enable directory scanning for file discovery (DEEP mode, default: True)
            include: File patterns to include (glob, e.g., ['*.py', '*.md', '*.pdf'])
            exclude: File patterns to exclude (glob, e.g., ['*.pyc', '*.log', 'node_modules'])
            return_cluster: Return full KnowledgeCluster object with evidences (DEEP mode only)

        Returns:
            Search results as formatted text with source references
        """
        if _service is None:
            return "Error: Service not initialized"

        logger.info(f"sirchmunk_search: mode={mode}, query='{query[:50]}...'")

        try:
            result = await _service.searcher.search(
                query=query,
                paths=paths,
                mode=mode,
                max_depth=max_depth,
                top_k_files=top_k_files,
                max_loops=max_loops,
                max_token_budget=max_token_budget,
                enable_dir_scan=enable_dir_scan,
                include=include,
                exclude=exclude,
                return_cluster=return_cluster,
            )

            if result is None:
                return f"No results found for query: {query}"

            if isinstance(result, str):
                return result

            if isinstance(result, list):
                # FILENAME_ONLY mode returns list of file matches
                return _format_filename_results(result, query)

            if hasattr(result, "__str__"):
                return str(result)

            return str(result)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return f"Search failed: {str(e)}"

    @mcp.tool()
    async def sirchmunk_scan_dir(
        query: str,
        paths: List[str],
        max_depth: int = 8,
        max_files: int = 500,
        top_k: int = 20,
    ) -> str:
        """Scan local directories to discover and rank files by relevance to a query.

        Recursively scans local filesystem paths to collect file metadata (title, size,
        type, author, page count, keywords, content preview), then uses an LLM to rank
        the most promising file candidates. Useful for exploring unfamiliar directories
        or finding relevant documents before performing a deep content search.

        USE THIS TOOL WHEN YOU NEED TO:
        - Explore what files exist in a local directory
        - Find relevant documents in a large folder structure
        - Get a ranked overview of file candidates before deep search

        Args:
            query: Natural language question or topic to rank files by relevance
            paths: Local filesystem directory paths to scan (e.g., ['/home/user/docs'])
            max_depth: Maximum recursion depth (1-20, default: 8)
            max_files: Maximum number of files to scan (default: 500)
            top_k: Number of top candidates for LLM ranking (default: 20)

        Returns:
            Ranked list of local file candidates with relevance scores and metadata
        """
        if _service is None:
            return "Error: Service not initialized"

        logger.info(f"sirchmunk_scan_dir: query='{query[:50]}...'")

        try:
            from sirchmunk.scan.dir_scanner import DirectoryScanner

            scanner = DirectoryScanner(
                llm=_service.searcher.llm,
                max_depth=max_depth,
                max_files=max_files,
            )
            result = await scanner.scan_and_rank(
                query=query,
                paths=paths,
                top_k=top_k,
            )

            return _format_scan_results(result, query)
        except Exception as e:
            logger.error(f"Dir scan failed: {e}", exc_info=True)
            return f"Directory scan failed: {str(e)}"

    @mcp.tool()
    async def sirchmunk_get_cluster(cluster_id: str) -> str:
        """Retrieve a cached knowledge cluster from a previous local file search.

        Knowledge clusters are automatically created and saved when sirchmunk_search
        runs in DEEP mode. Each cluster contains structured evidence extracted from
        local files — including source-linked snippets, synthesized analysis, design
        patterns, and confidence scores. Use this to recall previous search results
        without re-searching.

        Args:
            cluster_id: Knowledge cluster ID (e.g., 'C1a2b3c4d5')

        Returns:
            Full cluster information with evidences, patterns, and analysis
        """
        if _service is None:
            return "Error: Service not initialized"
        
        logger.info(f"sirchmunk_get_cluster: cluster_id={cluster_id}")
        
        try:
            cluster = await _service.get_cluster(cluster_id)
            
            if cluster is None:
                return f"Cluster not found: {cluster_id}"
            
            return str(cluster)
        
        except Exception as e:
            logger.error(f"Get cluster failed: {e}", exc_info=True)
            return f"Failed to retrieve cluster: {str(e)}"
    
    @mcp.tool()
    async def sirchmunk_list_clusters(
        limit: int = 10,
        sort_by: str = "last_modified",
    ) -> str:
        """List cached knowledge clusters from previous local file searches.

        Shows all knowledge clusters that were automatically created by sirchmunk_search.
        Each cluster represents a past search result with its query history, confidence
        score, and evidence count. Use this to discover what has already been searched
        and avoid redundant searches.

        Args:
            limit: Maximum number of clusters to return (1-100, default: 10)
            sort_by: Sort field - hotness (query frequency), confidence (quality), or last_modified (default)

        Returns:
            List of cluster metadata with IDs, names, queries, and scores
        """
        if _service is None:
            return "Error: Service not initialized"
        
        logger.info(f"sirchmunk_list_clusters: limit={limit}, sort_by={sort_by}")
        
        try:
            clusters = await _service.list_clusters(limit=limit, sort_by=sort_by)
            
            if not clusters:
                return "No knowledge clusters found."
            
            return _format_cluster_list(clusters, sort_by)
        
        except Exception as e:
            logger.error(f"List clusters failed: {e}", exc_info=True)
            return f"Failed to list clusters: {str(e)}"
    
    return mcp


def _format_scan_results(result, query: str) -> str:
    """Format DirectoryScanner results for MCP output.

    Args:
        result: ScanResult from DirectoryScanner
        query: Original query

    Returns:
        Formatted markdown string
    """
    lines = [
        "# Directory Scan Results",
        "",
        f"**Query**: `{query}`",
        f"**Files scanned**: {result.total_files}",
        f"**Directories traversed**: {result.total_dirs}",
        f"**Scan time**: {result.scan_duration_ms:.0f}ms",
        f"**Rank time**: {result.rank_duration_ms:.0f}ms",
        "",
    ]

    for i, c in enumerate(result.ranked_candidates, 1):
        tag = f"[{c.relevance}]" if c.relevance else "[?]"
        lines.append(f"## {i}. {tag} {c.filename}")
        lines.append(f"- **Path**: `{c.path}`")
        lines.append(f"- **Type**: {c.extension} | **Size**: {c._human_size()}")
        if c.title:
            lines.append(f"- **Title**: {c.title}")
        if c.reason:
            lines.append(f"- **Reason**: {c.reason}")
        if c.keywords:
            lines.append(f"- **Keywords**: {', '.join(c.keywords[:5])}")
        lines.append("")

    return "\n".join(lines)


def _format_filename_results(results: List[Dict[str, Any]], query: str) -> str:
    """Format FILENAME_ONLY mode results.
    
    Args:
        results: List of filename match dictionaries
        query: Original query
    
    Returns:
        Formatted string representation
    """
    lines = [
        f"# Filename Search Results",
        f"",
        f"**Query**: `{query}`",
        f"**Found**: {len(results)} matching file(s)",
        f"",
    ]
    
    for i, result in enumerate(results, 1):
        lines.append(f"## {i}. {result.get('filename', 'unknown')}")
        lines.append(f"- **Path**: `{result.get('path', 'unknown')}`")
        if 'match_score' in result:
            lines.append(f"- **Relevance**: {result['match_score']:.2f}")
        if "matched_pattern" in result:
            lines.append(f"- **Pattern**: `{result['matched_pattern']}`")
        lines.append("")
    
    return "\n".join(lines)


def _format_cluster_list(clusters: List[Dict[str, Any]], sort_by: str) -> str:
    """Format cluster list.
    
    Args:
        clusters: List of cluster metadata dictionaries
        sort_by: Sort field used
    
    Returns:
        Formatted string representation
    """
    lines = [
        f"# Knowledge Clusters",
        f"",
        f"**Total**: {len(clusters)} cluster(s)",
        f"**Sorted by**: {sort_by}",
        f"",
    ]
    
    for i, cluster in enumerate(clusters, 1):
        lines.append(f"## {i}. {cluster.get('name', 'Unnamed')}")
        lines.append(f"- **ID**: `{cluster.get('id', 'unknown')}`")
        lines.append(f"- **Lifecycle**: {cluster.get('lifecycle', 'unknown')}")
        lines.append(f"- **Version**: {cluster.get('version', 0)}")
        
        if cluster.get('confidence') is not None:
            lines.append(f"- **Confidence**: {cluster['confidence']:.2f}")
        
        if cluster.get('hotness') is not None:
            lines.append(f"- **Hotness**: {cluster['hotness']:.2f}")
        
        if cluster.get('last_modified'):
            lines.append(f"- **Last Modified**: {cluster['last_modified']}")
        
        if cluster.get('queries'):
            queries_preview = ", ".join(f'"{q}"' for q in cluster['queries'][:3])
            if len(cluster['queries']) > 3:
                queries_preview += f" (+{len(cluster['queries']) - 3} more)"
            lines.append(f"- **Related Queries**: {queries_preview}")
        
        lines.append(f"- **Evidences**: {cluster.get('evidences_count', 0)}")
        lines.append("")
    
    return "\n".join(lines)


async def run_stdio_server(config: Config) -> None:
    """Run MCP server with stdio transport.
    
    This is the default transport mode for Claude Desktop and other
    MCP clients that communicate via standard input/output.
    
    Args:
        config: Configuration object
    
    Note:
        This mode should be launched by an MCP client, not run directly
        in an interactive terminal. Manual terminal input will cause
        JSON parsing errors.
    """
    logger.info("Starting MCP server with stdio transport")
    
    # Create server
    mcp = create_server(config)
    
    # Run with stdio transport
    logger.info("MCP server listening on stdio")
    logger.info("Waiting for MCP client connection...")
    
    await mcp.run_stdio_async()


async def run_http_server(config: Config) -> None:
    """Run MCP server with Streamable HTTP transport.
    
    This transport mode runs an HTTP server that communicates via
    HTTP with streaming support, suitable for web-based clients.
    
    Args:
        config: Configuration object
    
    Note:
        HTTP transport requires uvicorn to be installed.
    """
    logger.info(
        f"Starting MCP server with HTTP transport on {config.mcp.host}:{config.mcp.port}"
    )
    
    # Create server
    mcp = create_server(config)
    
    # Run with HTTP transport using uvicorn
    try:
        import uvicorn
        uvicorn.run(
            mcp.sse_app(),
            host=config.mcp.host,
            port=config.mcp.port,
            log_level="info",
        )
    except ImportError:
        raise RuntimeError(
            "HTTP transport requires uvicorn. Install with: pip install uvicorn"
        )


async def main() -> None:
    """Main entry point for MCP server.
    
    Loads configuration and starts the appropriate transport server.
    """
    # Configure logging — explicit stderr to keep stdout clean for MCP JSON-RPC.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )
    
    try:
        # Load configuration from environment
        config = Config.from_env()
        
        # Set log level from config
        logging.getLogger().setLevel(config.mcp.log_level)
        
        logger.info(f"Loaded configuration: transport={config.mcp.transport}")
        
        # Start appropriate transport server
        if config.mcp.transport == "stdio":
            await run_stdio_server(config)
        elif config.mcp.transport == "http":
            await run_http_server(config)
        else:
            raise ValueError(f"Unknown transport: {config.mcp.transport}")
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
    
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
