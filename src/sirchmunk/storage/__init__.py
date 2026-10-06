# Copyright (c) ModelScope Contributors. All rights reserved.
"""Storage package initialization"""

__all__ = ["KnowledgeStorage", "DuckDBManager"]


def __getattr__(name):
    # Ownership bootstrap must run before importing data consumers/constants.
    if name == "KnowledgeStorage":
        from .knowledge_storage import KnowledgeStorage
        return KnowledgeStorage
    if name == "DuckDBManager":
        from .duckdb import DuckDBManager
        return DuckDBManager
    raise AttributeError(name)
