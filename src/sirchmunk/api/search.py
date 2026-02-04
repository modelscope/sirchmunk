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

from sirchmunk.search import AgenticSearch
from sirchmunk.llm.openai_chat import OpenAIChat
from sirchmunk.utils.constants import (
    LLM_BASE_URL, 
    LLM_API_KEY, 
    LLM_MODEL_NAME,
    DEFAULT_SIRCHMUNK_WORK_PATH,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


# === Request/Response Models ===

class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., description="Search query or question")
    search_paths: List[str] = Field(
        default_factory=list,
        description="Paths to search (directories or files)"
    )
    mode: Literal["FAST", "DEEP", "FILENAME_ONLY"] = Field(
        default="DEEP",
        description="Search mode: FAST, DEEP, or FILENAME_ONLY"
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum directory depth to search"
    )
    top_k_files: Optional[int] = Field(
        default=None,
        description="Number of top files to return"
    )
    keyword_levels: Optional[int] = Field(
        default=None,
        description="Number of keyword granularity levels"
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


def _get_search_instance() -> AgenticSearch:
    """Get or create AgenticSearch instance.
    
    Uses lazy initialization and caches the instance for reuse.
    
    Returns:
        AgenticSearch instance
        
    Raises:
        HTTPException: If LLM API key is not configured
    """
    global _search_instance
    
    if _search_instance is None:
        if not LLM_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="LLM_API_KEY is not configured. Set it in your environment or .env file."
            )
        
        llm = OpenAIChat(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL_NAME,
        )
        
        _search_instance = AgenticSearch(
            llm=llm,
            work_path=DEFAULT_SIRCHMUNK_WORK_PATH,
            verbose=False,
        )
        
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
        
        # Use current directory if no paths provided
        search_paths = request.search_paths if request.search_paths else ["."]
        
        logger.info(f"Executing search: query='{request.query}', mode={request.mode}, paths={search_paths}")
        
        # Build search kwargs
        search_kwargs = {
            "input": request.query,
            "search_paths": search_paths,
            "mode": request.mode,
            "return_cluster": request.return_cluster,
        }
        
        # Add optional parameters if provided
        if request.max_depth is not None:
            search_kwargs["max_depth"] = request.max_depth
        if request.top_k_files is not None:
            search_kwargs["top_k_files"] = request.top_k_files
        if request.keyword_levels is not None:
            search_kwargs["keyword_levels"] = request.keyword_levels
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
        has_api_key = bool(LLM_API_KEY)
        
        return {
            "success": True,
            "data": {
                "status": "ready" if has_api_key else "not_configured",
                "llm_configured": has_api_key,
                "llm_model": LLM_MODEL_NAME if has_api_key else None,
                "work_path": DEFAULT_SIRCHMUNK_WORK_PATH,
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
