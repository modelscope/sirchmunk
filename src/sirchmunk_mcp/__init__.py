# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Sirchmunk MCP Server

A Model Context Protocol (MCP) server that exposes Sirchmunk's intelligent
code and document search capabilities as MCP tools.

Uses FastMCP for simplified MCP server implementation.
"""

from sirchmunk.version import __version__

__author__ = "ModelScope Contributors"

__all__ = [
    "create_server",
    "run_stdio_server",
    "run_http_server",
    "SirchmunkService",
    "Config",
    "__version__",
]


def __getattr__(name):
    # Reading configuration/claiming a work root must not import the server or
    # initialize its optional SDK first. Preserve the existing public exports.
    if name in ("create_server", "run_stdio_server", "run_http_server"):
        from . import server
        return getattr(server, name)
    if name == "SirchmunkService":
        from .service import SirchmunkService
        return SirchmunkService
    if name == "Config":
        from .config import Config
        return Config
    raise AttributeError(name)
