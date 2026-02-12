# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Configuration management for Sirchmunk MCP Server.

Handles loading and validation of configuration from environment variables,
configuration files, and default values.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class LLMConfig(BaseModel):
    """Configuration for LLM service."""
    
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="LLM API base URL"
    )
    api_key: str = Field(
        description="LLM API key (required)"
    )
    model_name: str = Field(
        default="gpt-5.2",
        description="LLM model name"
    )
    timeout: float = Field(
        default=60.0,
        description="Request timeout in seconds",
        gt=0
    )
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("LLM API key cannot be empty")
        return v


class ClusterSimilarityConfig(BaseModel):
    """Configuration for cluster similarity search."""
    
    threshold: float = Field(
        default=0.85,
        description="Similarity threshold for cluster reuse",
        ge=0.0,
        le=1.0
    )
    top_k: int = Field(
        default=3,
        description="Number of similar clusters to retrieve",
        ge=1,
        le=10
    )


class SearchDefaultsConfig(BaseModel):
    """Default configuration for search operations."""
    
    max_depth: int = Field(
        default=5,
        description="Maximum directory depth to search",
        ge=1,
        le=20
    )
    top_k_files: int = Field(
        default=3,
        description="Number of top files to return",
        ge=1,
        le=20
    )
    keyword_levels: int = Field(
        default=3,
        description="Number of keyword granularity levels",
        ge=1,
        le=5
    )
    grep_timeout: float = Field(
        default=60.0,
        description="Timeout for grep operations in seconds",
        gt=0
    )
    max_queries_per_cluster: int = Field(
        default=5,
        description="Maximum number of queries to keep per cluster (FIFO)",
        ge=1,
        le=20
    )


class SirchmunkConfig(BaseModel):
    """Configuration for Sirchmunk service."""
    
    work_path: Path = Field(
        default_factory=lambda: Path.home() / ".sirchmunk",
        description="Working directory for Sirchmunk data"
    )
    paths: Optional[list] = Field(
        default=None,
        description="Default search paths (directories or files). "
                    "Falls back to current working directory when None."
    )
    verbose: bool = Field(
        default=True,
        description="Enable verbose logging"
    )
    enable_cluster_reuse: bool = Field(
        default=True,
        description="Enable knowledge cluster reuse with embeddings"
    )
    cluster_similarity: ClusterSimilarityConfig = Field(
        default_factory=ClusterSimilarityConfig
    )
    search_defaults: SearchDefaultsConfig = Field(
        default_factory=SearchDefaultsConfig
    )
    
    @field_validator("work_path")
    @classmethod
    def validate_work_path(cls, v: Path) -> Path:
        """Ensure work path is absolute and exists."""
        v = v.expanduser().resolve()
        v.mkdir(parents=True, exist_ok=True)
        return v


class MCPServerConfig(BaseModel):
    """Configuration for MCP server."""
    
    server_name: str = Field(
        default="sirchmunk",
        description="MCP server name"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    transport: str = Field(
        default="stdio",
        description="MCP transport protocol (stdio or http)"
    )
    
    # HTTP-specific settings
    host: str = Field(
        default="localhost",
        description="Host for HTTP transport"
    )
    port: int = Field(
        default=8080,
        description="Port for HTTP transport",
        ge=1024,
        le=65535
    )
    
    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: str) -> str:
        """Validate transport protocol."""
        v = v.lower()
        if v not in ("stdio", "http"):
            raise ValueError(f"Invalid transport: {v}. Must be 'stdio' or 'http'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        v = v.upper()
        valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class Config(BaseModel):
    """Master configuration for Sirchmunk MCP Server."""
    
    llm: LLMConfig
    sirchmunk: SirchmunkConfig = Field(default_factory=SirchmunkConfig)
    mcp: MCPServerConfig = Field(default_factory=MCPServerConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables.
        
        Automatically loads .env file from work_path (~/.sirchmunk/.env by default).
        
        Environment variables:
            LLM_BASE_URL: LLM API base URL
            LLM_API_KEY: LLM API key (required)
            LLM_MODEL_NAME: LLM model name
            SIRCHMUNK_WORK_PATH: Sirchmunk working directory
            SIRCHMUNK_SEARCH_PATHS: Default search paths (comma-separated)
            SIRCHMUNK_VERBOSE: Enable verbose logging
            SIRCHMUNK_ENABLE_CLUSTER_REUSE: Enable cluster reuse
            CLUSTER_SIM_THRESHOLD: Similarity threshold
            CLUSTER_SIM_TOP_K: Top-K similar clusters
            DEFAULT_MAX_DEPTH: Default max directory depth
            DEFAULT_TOP_K_FILES: Default top-K files
            DEFAULT_KEYWORD_LEVELS: Default keyword levels
            GREP_TIMEOUT: Grep operation timeout
            MAX_QUERIES_PER_CLUSTER: Max queries per cluster
            MCP_SERVER_NAME: MCP server name
            MCP_LOG_LEVEL: Logging level
            MCP_TRANSPORT: MCP transport protocol
            MCP_HOST: MCP server host (HTTP mode)
            MCP_PORT: MCP server port (HTTP mode)
        
        Returns:
            Config: Loaded configuration
        
        Raises:
            ValueError: If required configuration is missing or invalid
        """
        # Load .env from work_path (default: ~/.sirchmunk/.env)
        work_path = Path(os.getenv("SIRCHMUNK_WORK_PATH", str(Path.home() / ".sirchmunk")))
        work_path = work_path.expanduser().resolve()
        env_file = work_path / ".env"
        
        if env_file.exists():
            load_dotenv(env_file, override=False)
        
        # LLM configuration
        llm_config = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
            model_name=os.getenv("LLM_MODEL_NAME", "gpt-5.2"),
            timeout=float(os.getenv("LLM_TIMEOUT", "60.0")),
        )
        
        # Parse SIRCHMUNK_SEARCH_PATHS (supports English comma, Chinese comma,
        # or os.pathsep as delimiters)
        raw_paths = os.getenv("SIRCHMUNK_SEARCH_PATHS", "").strip()
        parsed_paths: Optional[list] = None
        if raw_paths:
            # Split by English comma, Chinese comma (，), or os.pathsep
            _sep_pattern = r"[,，" + re.escape(os.pathsep) + r"]"
            parsed_paths = [
                p.strip() for p in re.split(_sep_pattern, raw_paths) if p.strip()
            ] or None

        # Sirchmunk configuration
        sirchmunk_config = SirchmunkConfig(
            work_path=Path(os.getenv("SIRCHMUNK_WORK_PATH", str(Path.home() / ".sirchmunk"))),
            paths=parsed_paths,
            verbose=os.getenv("SIRCHMUNK_VERBOSE", "true").lower() == "true",
            enable_cluster_reuse=os.getenv("SIRCHMUNK_ENABLE_CLUSTER_REUSE", "true").lower() == "true",
            cluster_similarity=ClusterSimilarityConfig(
                threshold=float(os.getenv("CLUSTER_SIM_THRESHOLD", "0.85")),
                top_k=int(os.getenv("CLUSTER_SIM_TOP_K", "3")),
            ),
            search_defaults=SearchDefaultsConfig(
                max_depth=int(os.getenv("DEFAULT_MAX_DEPTH", "5")),
                top_k_files=int(os.getenv("DEFAULT_TOP_K_FILES", "3")),
                keyword_levels=int(os.getenv("DEFAULT_KEYWORD_LEVELS", "3")),
                grep_timeout=float(os.getenv("GREP_TIMEOUT", "60.0")),
                max_queries_per_cluster=int(os.getenv("MAX_QUERIES_PER_CLUSTER", "5")),
            ),
        )
        
        # MCP server configuration
        mcp_config = MCPServerConfig(
            server_name=os.getenv("MCP_SERVER_NAME", "sirchmunk"),
            log_level=os.getenv("MCP_LOG_LEVEL", "INFO"),
            transport=os.getenv("MCP_TRANSPORT", "stdio"),
            host=os.getenv("MCP_HOST", "localhost"),
            port=int(os.getenv("MCP_PORT", "8080")),
        )
        
        return cls(
            llm=llm_config,
            sirchmunk=sirchmunk_config,
            mcp=mcp_config,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self.model_dump()
