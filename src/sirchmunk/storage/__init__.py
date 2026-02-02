# Copyright (c) ModelScope Contributors. All rights reserved.
"""Storage package initialization"""

from .knowledge_storage import KnowledgeStorage
from .duckdb import DuckDBManager

__all__ = ["KnowledgeStorage", "DuckDBManager"]
