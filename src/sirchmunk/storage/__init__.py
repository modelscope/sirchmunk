# Copyright (c) ModelScope Contributors. All rights reserved.
"""Storage package initialization"""

from .knowledge_manager import KnowledgeManager
from .duckdb import DuckDBManager

__all__ = ["KnowledgeManager", "DuckDBManager"]
