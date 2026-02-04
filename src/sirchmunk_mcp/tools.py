# Copyright (c) ModelScope Contributors. All rights reserved.
"""
MCP Tools definitions for Sirchmunk.

Defines the MCP tools that expose Sirchmunk functionality to MCP clients.
"""

import json
import logging
from typing import Any, Dict, List

from mcp.types import Tool, TextContent

from .service import SirchmunkService


logger = logging.getLogger(__name__)


# Tool definitions following MCP specification
SIRCHMUNK_SEARCH_TOOL = Tool(
    name="sirchmunk_search",
    description=(
        "Intelligent code and document search with multi-mode support. "
        "DEEP mode provides comprehensive knowledge extraction with full context analysis. "
        "FAST mode offers quick content search without deep LLM processing. "
        "FILENAME_ONLY mode performs fast filename pattern matching without content search."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query or question (e.g., 'How does authentication work?')",
            },
            "search_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Paths to search in (files or directories)",
            },
            "mode": {
                "type": "string",
                "enum": ["DEEP", "FAST", "FILENAME_ONLY"],
                "default": "DEEP",
                "description": (
                    "Search mode: DEEP (comprehensive analysis, 10-30s), "
                    "FAST (quick search, 3-8s), FILENAME_ONLY (file discovery, <1s)"
                ),
            },
            "max_depth": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum directory depth to search",
            },
            "top_k_files": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 20,
                "description": "Number of top files to return",
            },
            "keyword_levels": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 5,
                "description": "Keyword granularity levels (DEEP mode only, more levels = more fallback options)",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File patterns to include (glob, e.g., ['*.py', '*.md'])",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "File patterns to exclude (glob, e.g., ['*.pyc', '*.log'])",
            },
            "return_cluster": {
                "type": "boolean",
                "default": False,
                "description": "Return full KnowledgeCluster object (DEEP/FAST modes only)",
            },
        },
        "required": ["query", "search_paths"],
    },
)


SIRCHMUNK_GET_CLUSTER_TOOL = Tool(
    name="sirchmunk_get_cluster",
    description=(
        "Retrieve a previously saved knowledge cluster by its ID. "
        "Knowledge clusters are automatically saved during DEEP mode searches "
        "and contain rich information including evidences, patterns, and constraints."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "cluster_id": {
                "type": "string",
                "description": "Knowledge cluster ID (e.g., 'C1007')",
            },
        },
        "required": ["cluster_id"],
    },
)


SIRCHMUNK_LIST_CLUSTERS_TOOL = Tool(
    name="sirchmunk_list_clusters",
    description=(
        "List all saved knowledge clusters with optional filtering and sorting. "
        "Useful for discovering previously searched topics and reusing knowledge."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum number of clusters to return",
            },
            "sort_by": {
                "type": "string",
                "enum": ["hotness", "confidence", "last_modified"],
                "default": "last_modified",
                "description": (
                    "Sort field: hotness (query frequency), "
                    "confidence (quality score), last_modified (most recent)"
                ),
            },
        },
    },
)


# Tool registry
TOOLS = [
    SIRCHMUNK_SEARCH_TOOL,
    SIRCHMUNK_GET_CLUSTER_TOOL,
    SIRCHMUNK_LIST_CLUSTERS_TOOL,
]


async def handle_sirchmunk_search(
    service: SirchmunkService,
    arguments: Dict[str, Any],
) -> List[TextContent]:
    """Handle sirchmunk_search tool invocation.
    
    Args:
        service: SirchmunkService instance
        arguments: Tool arguments from MCP client
    
    Returns:
        List of TextContent with search results
    
    Raises:
        ValueError: If required arguments are missing or invalid
    """
    # Extract required arguments
    query = arguments.get("query")
    search_paths = arguments.get("search_paths")
    
    if not query:
        raise ValueError("Missing required argument: query")
    if not search_paths:
        raise ValueError("Missing required argument: search_paths")
    
    # Extract optional arguments with defaults
    mode = arguments.get("mode", "DEEP")
    max_depth = arguments.get("max_depth")
    top_k_files = arguments.get("top_k_files")
    keyword_levels = arguments.get("keyword_levels")
    include = arguments.get("include")
    exclude = arguments.get("exclude")
    return_cluster = arguments.get("return_cluster", False)
    
    logger.info(f"Handling sirchmunk_search: mode={mode}, query='{query[:50]}...'")
    
    try:
        # Perform search
        result = await service.search(
            query=query,
            search_paths=search_paths,
            mode=mode,
            max_depth=max_depth,
            top_k_files=top_k_files,
            keyword_levels=keyword_levels,
            include=include,
            exclude=exclude,
            return_cluster=return_cluster,
        )
        
        # Format response based on result type
        if result is None:
            response_text = f"No results found for query: {query}"
        
        elif isinstance(result, str):
            # DEEP or FAST mode: string summary
            response_text = result
        
        elif isinstance(result, list):
            # FILENAME_ONLY mode: list of file matches
            response_text = _format_filename_results(result, query)
        
        elif hasattr(result, "to_dict"):
            # KnowledgeCluster object
            response_text = _format_cluster(result)
        
        else:
            response_text = str(result)
        
        return [TextContent(
            type="text",
            text=response_text,
        )]
    
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        error_message = f"Search failed: {str(e)}"
        return [TextContent(
            type="text",
            text=error_message,
        )]


async def handle_sirchmunk_get_cluster(
    service: SirchmunkService,
    arguments: Dict[str, Any],
) -> List[TextContent]:
    """Handle sirchmunk_get_cluster tool invocation.
    
    Args:
        service: SirchmunkService instance
        arguments: Tool arguments from MCP client
    
    Returns:
        List of TextContent with cluster information
    
    Raises:
        ValueError: If required arguments are missing
    """
    cluster_id = arguments.get("cluster_id")
    
    if not cluster_id:
        raise ValueError("Missing required argument: cluster_id")
    
    logger.info(f"Handling sirchmunk_get_cluster: cluster_id={cluster_id}")
    
    try:
        cluster = await service.get_cluster(cluster_id)
        
        if cluster is None:
            response_text = f"Cluster not found: {cluster_id}"
        else:
            response_text = _format_cluster(cluster)
        
        return [TextContent(
            type="text",
            text=response_text,
        )]
    
    except Exception as e:
        logger.error(f"Get cluster failed: {e}", exc_info=True)
        error_message = f"Failed to retrieve cluster: {str(e)}"
        return [TextContent(
            type="text",
            text=error_message,
        )]


async def handle_sirchmunk_list_clusters(
    service: SirchmunkService,
    arguments: Dict[str, Any],
) -> List[TextContent]:
    """Handle sirchmunk_list_clusters tool invocation.
    
    Args:
        service: SirchmunkService instance
        arguments: Tool arguments from MCP client
    
    Returns:
        List of TextContent with cluster listing
    """
    limit = arguments.get("limit", 10)
    sort_by = arguments.get("sort_by", "last_modified")
    
    logger.info(f"Handling sirchmunk_list_clusters: limit={limit}, sort_by={sort_by}")
    
    try:
        clusters = await service.list_clusters(limit=limit, sort_by=sort_by)
        
        if not clusters:
            response_text = "No knowledge clusters found."
        else:
            response_text = _format_cluster_list(clusters, sort_by)
        
        return [TextContent(
            type="text",
            text=response_text,
        )]
    
    except Exception as e:
        logger.error(f"List clusters failed: {e}", exc_info=True)
        error_message = f"Failed to list clusters: {str(e)}"
        return [TextContent(
            type="text",
            text=error_message,
        )]


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
        lines.append(f"## {i}. {result['filename']}")
        lines.append(f"- **Path**: `{result['path']}`")
        lines.append(f"- **Relevance**: {result['match_score']:.2f}")
        if "matched_pattern" in result:
            lines.append(f"- **Pattern**: `{result['matched_pattern']}`")
        lines.append("")
    
    return "\n".join(lines)


def _format_cluster(cluster: Any) -> str:
    """Format KnowledgeCluster object.
    
    Args:
        cluster: KnowledgeCluster object
    
    Returns:
        Formatted string representation
    """
    # Use the cluster's __str__ method for human-readable format
    return str(cluster)


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
        lines.append(f"## {i}. {cluster['name']}")
        lines.append(f"- **ID**: `{cluster['id']}`")
        lines.append(f"- **Lifecycle**: {cluster['lifecycle']}")
        lines.append(f"- **Version**: {cluster['version']}")
        
        if cluster['confidence'] is not None:
            lines.append(f"- **Confidence**: {cluster['confidence']:.2f}")
        
        if cluster['hotness'] is not None:
            lines.append(f"- **Hotness**: {cluster['hotness']:.2f}")
        
        if cluster['last_modified']:
            lines.append(f"- **Last Modified**: {cluster['last_modified']}")
        
        if cluster['queries']:
            queries_preview = ", ".join(f'"{q}"' for q in cluster['queries'][:3])
            if len(cluster['queries']) > 3:
                queries_preview += f" (+{len(cluster['queries']) - 3} more)"
            lines.append(f"- **Related Queries**: {queries_preview}")
        
        lines.append(f"- **Evidences**: {cluster['evidences_count']}")
        lines.append("")
    
    return "\n".join(lines)


# Tool handler registry
TOOL_HANDLERS = {
    "sirchmunk_search": handle_sirchmunk_search,
    "sirchmunk_get_cluster": handle_sirchmunk_get_cluster,
    "sirchmunk_list_clusters": handle_sirchmunk_list_clusters,
}
