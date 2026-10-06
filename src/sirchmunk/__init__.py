# Copyright (c) ModelScope Contributors. All rights reserved.
from .version import __version__

__all__ = [
    "__version__",
    "AgenticSearch",
]

# Lazy imports for agentic sub-modules
def __getattr__(name):
    if name == "AgenticSearch":
        # Management ownership and launch environment must be established
        # before search imports constants, storage and optional model runtimes.
        from .search import AgenticSearch
        return AgenticSearch
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
