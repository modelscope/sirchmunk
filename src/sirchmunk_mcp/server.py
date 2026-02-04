# Copyright (c) ModelScope Contributors. All rights reserved.
"""
MCP Server implementation for Sirchmunk.

Provides the main MCP server that exposes Sirchmunk functionality
as MCP tools following the Model Context Protocol specification.
"""

import asyncio
import logging
from typing import Any, Dict

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
)

from .config import Config
from .service import SirchmunkService
from .tools import TOOLS, TOOL_HANDLERS


logger = logging.getLogger(__name__)


def create_server(config: Config) -> Server:
    """Create and configure MCP server instance.
    
    Args:
        config: Configuration object
    
    Returns:
        Configured MCP Server instance
    """
    # Initialize service
    service = SirchmunkService(config)
    
    # Create MCP server
    server = Server(config.mcp.server_name)
    
    logger.info(
        f"Creating MCP server: {config.mcp.server_name} v{config.mcp.server_version}"
    )
    
    # Register list_tools handler
    @server.list_tools()
    async def list_tools() -> ListToolsResult:
        """List available tools.
        
        Returns:
            ListToolsResult with all available tools
        """
        logger.debug(f"Listing {len(TOOLS)} available tools")
        return ListToolsResult(tools=TOOLS)
    
    # Register call_tool handler
    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: Dict[str, Any],
    ) -> CallToolResult:
        """Handle tool invocation.
        
        Args:
            name: Tool name
            arguments: Tool arguments
        
        Returns:
            CallToolResult with tool execution results
        
        Raises:
            ValueError: If tool is not found
        """
        logger.info(f"Tool call: {name}")
        logger.debug(f"Tool arguments: {arguments}")
        
        # Get tool handler
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            error_msg = f"Unknown tool: {name}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Execute tool handler
            content = await handler(service, arguments)
            
            logger.info(f"Tool {name} executed successfully")
            return CallToolResult(content=content)
        
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}", exc_info=True)
            # Re-raise to let MCP server handle error response
            raise
    
    # Register lifecycle hooks
    @server.initialize()
    async def initialize() -> None:
        """Initialize server on client connection."""
        logger.info("MCP server initialized")
    
    @server.shutdown()
    async def shutdown() -> None:
        """Cleanup on server shutdown."""
        logger.info("MCP server shutting down")
        await service.shutdown()
    
    return server


async def run_stdio_server(config: Config) -> None:
    """Run MCP server with stdio transport.
    
    This is the default transport mode for Claude Desktop and other
    MCP clients that communicate via standard input/output.
    
    Args:
        config: Configuration object
    """
    logger.info("Starting MCP server with stdio transport")
    
    # Create server
    server = create_server(config)
    
    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server listening on stdio")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_http_server(config: Config) -> None:
    """Run MCP server with Streamable HTTP transport.
    
    This transport mode runs an HTTP server that communicates via
    HTTP with streaming support, suitable for web-based clients.
    
    Args:
        config: Configuration object
    
    Note:
        HTTP transport implementation requires additional dependencies.
        This is a placeholder for future implementation.
    """
    logger.info(
        f"Starting MCP server with HTTP transport on {config.mcp.host}:{config.mcp.port}"
    )
    
    # TODO: Implement Streamable HTTP transport
    # This would typically involve:
    # 1. Creating an HTTP server (e.g., with aiohttp or FastAPI)
    # 2. Implementing streaming HTTP endpoints  
    # 3. Routing MCP messages through HTTP streaming
    
    raise NotImplementedError(
        "HTTP transport is not yet implemented. Please use stdio transport."
    )


async def main() -> None:
    """Main entry point for MCP server.
    
    Loads configuration and starts the appropriate transport server.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
