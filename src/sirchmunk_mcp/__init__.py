# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Sirchmunk MCP Server

A Model Context Protocol (MCP) server that exposes Sirchmunk's intelligent
code and document search capabilities as MCP tools.
"""

__version__ = "0.1.0"
__author__ = "ModelScope Contributors"

from .server import create_server
from .service import SirchmunkService
from .config import Config

__all__ = [
    "create_server",
    "SirchmunkService",
    "Config",
    "__version__",
]
