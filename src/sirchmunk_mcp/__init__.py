# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Sirchmunk MCP Server

A Model Context Protocol (MCP) server that exposes Sirchmunk's intelligent
code and document search capabilities as MCP tools.

Uses FastMCP for simplified MCP server implementation.
"""

from sirchmunk.version import __version__

__author__ = "ModelScope Contributors"

from .server import create_server, run_stdio_server, run_http_server
from .service import SirchmunkService
from .config import Config

__all__ = [
    "create_server",
    "run_stdio_server",
    "run_http_server",
    "SirchmunkService",
    "Config",
    "__version__",
]
