# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Basic tests for Sirchmunk MCP Server.

These tests verify core functionality of the MCP server components.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from sirchmunk_mcp.config import Config, LLMConfig, SirchmunkConfig, MCPServerConfig
from sirchmunk_mcp.tools import TOOLS, _format_filename_results, _format_cluster_list


class TestConfig:
    """Test configuration management."""
    
    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Valid config
        config = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-4",
        )
        assert config.api_key == "sk-test"
        
        # Invalid: empty API key
        with pytest.raises(ValueError):
            LLMConfig(
                base_url="https://api.openai.com/v1",
                api_key="",
                model_name="gpt-4",
            )
    
    def test_config_from_env(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            "LLM_API_KEY": "sk-test-key",
            "LLM_MODEL_NAME": "gpt-4-test",
            "SIRCHMUNK_WORK_PATH": "/tmp/sirchmunk_test",
            "MCP_LOG_LEVEL": "DEBUG",
        }):
            config = Config.from_env()
            
            assert config.llm.api_key == "sk-test-key"
            assert config.llm.model_name == "gpt-4-test"
            assert config.mcp.log_level == "DEBUG"


class TestTools:
    """Test MCP tools definitions and formatting."""
    
    def test_tools_defined(self):
        """Test that all tools are properly defined."""
        assert len(TOOLS) == 3
        
        tool_names = {tool.name for tool in TOOLS}
        assert "sirchmunk_search" in tool_names
        assert "sirchmunk_get_cluster" in tool_names
        assert "sirchmunk_list_clusters" in tool_names
    
    def test_sirchmunk_search_tool_schema(self):
        """Test sirchmunk_search tool schema."""
        search_tool = next(t for t in TOOLS if t.name == "sirchmunk_search")
        
        assert "query" in search_tool.inputSchema["properties"]
        assert "search_paths" in search_tool.inputSchema["properties"]
        assert "mode" in search_tool.inputSchema["properties"]
        
        # Check required fields
        assert "query" in search_tool.inputSchema["required"]
        assert "search_paths" in search_tool.inputSchema["required"]
    
    def test_format_filename_results(self):
        """Test formatting of filename search results."""
        results = [
            {
                "filename": "test.py",
                "path": "/path/to/test.py",
                "match_score": 0.95,
                "matched_pattern": ".*test.*",
            },
            {
                "filename": "utils.py",
                "path": "/path/to/utils.py",
                "match_score": 0.85,
                "matched_pattern": ".*utils.*",
            },
        ]
        
        formatted = _format_filename_results(results, "test utils")
        
        assert "test.py" in formatted
        assert "utils.py" in formatted
        assert "0.95" in formatted
        assert "0.85" in formatted
    
    def test_format_cluster_list(self):
        """Test formatting of cluster list."""
        clusters = [
            {
                "id": "C1001",
                "name": "Test Cluster",
                "confidence": 0.9,
                "hotness": 0.8,
                "lifecycle": "stable",
                "version": 1,
                "last_modified": "2024-01-01T00:00:00",
                "queries": ["test query 1", "test query 2"],
                "evidences_count": 5,
            },
        ]
        
        formatted = _format_cluster_list(clusters, "hotness")
        
        assert "C1001" in formatted
        assert "Test Cluster" in formatted
        assert "0.90" in formatted
        assert "0.80" in formatted


@pytest.mark.asyncio
class TestService:
    """Test Sirchmunk service wrapper."""
    
    async def test_service_initialization_requires_api_key(self):
        """Test that service initialization requires valid API key."""
        # This test would require valid LLM credentials
        # Skipped in unit tests, should be tested in integration tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
