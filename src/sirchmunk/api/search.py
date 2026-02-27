# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Search API endpoints for CLI and programmatic access.

Provides HTTP endpoints for executing AgenticSearch queries,
designed for CLI client mode and external integrations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import logging

import os

from sirchmunk.search import AgenticSearch
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.utils.constants import DEFAULT_SIRCHMUNK_WORK_PATH


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


# === Request/Response Models ===

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query or question")
    paths: Optional[List[str]] = Field(
        default=None,
        description="Paths to search (directories or files). Falls back to configured default or cwd."
    )
    mode: Literal["DEEP", "FILENAME_ONLY"] = Field(
        default="DEEP",
        description="Search mode: DEEP (comprehensive analysis) or FILENAME_ONLY (fast file discovery)"
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum directory depth to search"
    )
    top_k_files: Optional[int] = Field(
        default=None,
        description="Number of top files to return"
    )
    max_loops: Optional[int] = Field(
        default=None,
        description="Maximum ReAct iterations (DEEP mode)"
    )
    max_token_budget: Optional[int] = Field(
        default=None,
        description="LLM token budget (DEEP mode)"
    )
    enable_dir_scan: bool = Field(
        default=True,
        description="Enable directory scanning (DEEP mode)"
    )
    include_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to include (glob)"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="File patterns to exclude (glob)"
    )
    return_cluster: bool = Field(
        default=False,
        description="Return full KnowledgeCluster object"
    )


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    success: bool
    data: dict
    error: Optional[str] = None


# === Cached Search Instance ===

_search_instance: Optional[AgenticSearch] = None
_search_config: Optional[tuple] = None


def _get_search_instance() -> AgenticSearch:
    """Get or create AgenticSearch instance.

    Caches the instance but automatically recreates it when the LLM
    configuration (detected via os.environ) changes, so that settings
    modified through the WebUI take effect without a server restart.

    Returns:
        AgenticSearch instance

    Raises:
        HTTPException: If LLM API key is not configured
    """
    global _search_instance, _search_config

    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("LLM_MODEL_NAME", "gpt-5.2")

    current_config = (api_key, base_url, model_name)

    if _search_instance is not None and current_config == _search_config:
        return _search_instance

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="LLM_API_KEY is not configured. Set it in your environment or .env file."
        )

    llm = OpenAIChat(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )

    enable_cluster_reuse = os.getenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "true").lower() == "true"
    cluster_sim_threshold = float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.85"))
    cluster_sim_top_k = int(os.getenv("CLUSTER_SIM_TOP_K", "3"))

    _search_instance = AgenticSearch(
        llm=llm,
        work_path=DEFAULT_SIRCHMUNK_WORK_PATH,
        verbose=False,
        reuse_knowledge=enable_cluster_reuse,
        cluster_sim_threshold=cluster_sim_threshold,
        cluster_sim_top_k=cluster_sim_top_k,
    )
    _search_config = current_config

    logger.info("AgenticSearch instance created for API")

    return _search_instance


# === API Endpoints ===

@router.post("/search")
async def execute_search(request: SearchRequest) -> SearchResponse:
    """Execute an AgenticSearch query.
    
    This endpoint performs a full search using AgenticSearch,
    including keyword extraction, file retrieval, content analysis,
    and summary generation.
    
    Args:
        request: Search request parameters
        
    Returns:
        SearchResponse with search results
        
    Raises:
        HTTPException: If search fails or configuration is invalid
    """
    try:
        searcher = _get_search_instance()
        
        paths = request.paths
        
        logger.info(f"Executing search: query='{request.query}', mode={request.mode}, paths={paths}")
        
        # Build search kwargs
        search_kwargs = {
            "query": request.query,
            "paths": paths,
            "mode": request.mode,
            "enable_dir_scan": request.enable_dir_scan,
            "return_cluster": request.return_cluster,
        }
        
        # Add optional parameters if provided
        if request.max_depth is not None:
            search_kwargs["max_depth"] = request.max_depth
        if request.top_k_files is not None:
            search_kwargs["top_k_files"] = request.top_k_files
        if request.max_loops is not None:
            search_kwargs["max_loops"] = request.max_loops
        if request.max_token_budget is not None:
            search_kwargs["max_token_budget"] = request.max_token_budget
        if request.include_patterns:
            search_kwargs["include"] = request.include_patterns
        if request.exclude_patterns:
            search_kwargs["exclude"] = request.exclude_patterns
        
        # Execute search
        result = await searcher.search(**search_kwargs)
        
        # Format response
        if request.return_cluster and hasattr(result, "to_dict"):
            # Return full cluster data
            return SearchResponse(
                success=True,
                data={
                    "type": "cluster",
                    "cluster": result.to_dict(),
                }
            )
        elif isinstance(result, list):
            # FILENAME_ONLY mode returns list
            return SearchResponse(
                success=True,
                data={
                    "type": "files",
                    "files": result,
                    "count": len(result),
                }
            )
        else:
            # Standard text summary
            return SearchResponse(
                success=True,
                data={
                    "type": "summary",
                    "summary": str(result) if result else "No results found.",
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/status")
async def get_search_status():
    """Get search service status.
    
    Returns:
        Service status information
    """
    try:
        api_key = os.getenv("LLM_API_KEY", "")
        model_name = os.getenv("LLM_MODEL_NAME", "gpt-5.2")
        has_api_key = bool(api_key)
        
        return {
            "success": True,
            "data": {
                "status": "ready" if has_api_key else "not_configured",
                "llm_configured": has_api_key,
                "llm_model": model_name if has_api_key else None,
                "work_path": DEFAULT_SIRCHMUNK_WORK_PATH,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
