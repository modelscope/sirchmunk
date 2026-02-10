# Copyright (c) ModelScope Contributors. All rights reserved.
from .search import AgenticSearch
from .version import __version__

__all__ = [
    "__version__",
    "AgenticSearch",
]

# Lazy imports for agentic sub-modules
def __getattr__(name):
    if name == "ReActSearchAgent":
        from .agentic.react_agent import ReActSearchAgent
        return ReActSearchAgent
    if name == "DirectoryScanner":
        from .scan.dir_scanner import DirectoryScanner
        return DirectoryScanner
    if name == "SearchContext":
        from .schema.search_context import SearchContext
        return SearchContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
