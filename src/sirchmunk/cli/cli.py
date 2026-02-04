# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Command-line interface for Sirchmunk.

Provides commands for server management, initialization, configuration,
and search operations.

Usage:
    sirchmunk init          - Initialize Sirchmunk working directory
    sirchmunk config        - Show or generate configuration
    sirchmunk serve         - Start the API server
    sirchmunk search        - Perform a search query
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from sirchmunk.version import __version__


logger = logging.getLogger(__name__)


def _get_default_work_path() -> Path:
    """Get the default work path for Sirchmunk."""
    return Path(os.getenv("SIRCHMUNK_WORK_PATH", str(Path.home() / ".sirchmunk")))


def _setup_logging(log_level: str = "INFO"):
    """Configure logging for CLI operations."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )


def _load_env_file(env_file: Path) -> bool:
    """Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        True if file was loaded, False otherwise
    """
    if not env_file.exists():
        return False
    
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return True
    except ImportError:
        # Fallback: manual parsing if python-dotenv not installed
        try:
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            return True
        except Exception:
            return False


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize Sirchmunk working directory.
    
    Creates the work directory structure, checks dependencies, and generates
    initial configuration.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    import shutil
    
    try:
        work_path = Path(args.work_path).expanduser().resolve()
        
        print("=" * 60)
        print("  Sirchmunk Initialization")
        print("=" * 60)
        print()
        print(f"Work path: {work_path}")
        print()
        
        # Create directory structure
        print("Creating directory structure...")
        directories = [
            work_path,
            work_path / "data",
            work_path / "logs",
            work_path / ".cache",
            work_path / ".cache" / "models",
            work_path / ".cache" / "knowledge",
            work_path / ".cache" / "history",
            work_path / ".cache" / "settings",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("  ✓ Created work directory and subdirectories")
        
        # Generate default .env file if not exists
        env_file = work_path / ".env"
        if not env_file.exists():
            _generate_env_file(env_file)
            print(f"  ✓ Generated {env_file}")
        else:
            print(f"  • Skipped {env_file} (already exists)")
        
        # Check dependencies
        print()
        print("Checking dependencies...")
        
        # Check ripgrep-all
        if shutil.which("rga"):
            print("  ✓ ripgrep-all (rga) is installed")
        else:
            print("  ✗ ripgrep-all (rga) is not installed")
            print("    Installing ripgrep-all...")
            try:
                from sirchmunk.utils.install_rga import install_rga
                install_rga()
                print("  ✓ ripgrep-all installed successfully")
            except Exception as e:
                print(f"  ✗ Failed to install ripgrep-all: {e}")
                print("    Please install manually: https://github.com/phiresky/ripgrep-all")
        
        # Check ripgrep
        if shutil.which("rg"):
            print("  ✓ ripgrep (rg) is installed")
        else:
            print("  ✗ ripgrep (rg) is not installed")
            print("    Please install: https://github.com/BurntSushi/ripgrep")
        
        # Check Python packages
        try:
            import fastapi
            print(f"  ✓ FastAPI is installed")
        except ImportError:
            print("  ✗ FastAPI not found")
            print("    Install with: pip install fastapi")
        
        try:
            import uvicorn
            print(f"  ✓ uvicorn is installed")
        except ImportError:
            print("  ✗ uvicorn not found")
            print("    Install with: pip install uvicorn")
        
        # Check environment variables
        print()
        print("Checking environment variables...")
        
        # Load env file first
        _load_env_file(env_file)
        
        llm_api_key = os.getenv("LLM_API_KEY")
        if llm_api_key:
            masked_key = llm_api_key[:8] + "..." if len(llm_api_key) > 8 else "***"
            print(f"  ✓ LLM_API_KEY is set ({masked_key})")
        else:
            print("  ✗ LLM_API_KEY is not set")
            print(f"    Set it in {env_file}")
        
        llm_model = os.getenv("LLM_MODEL_NAME", "gpt-5.2")
        print(f"  • LLM_MODEL_NAME: {llm_model}")
        
        llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        print(f"  • LLM_BASE_URL: {llm_base_url}")
        
        print()
        print("=" * 60)
        print("✅ Initialization complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print(f"  1. Edit {env_file} to configure LLM_API_KEY")
        print("  2. Run 'sirchmunk serve' to start the API server")
        print("  3. Run 'sirchmunk search \"your query\"' to perform searches")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        print(f"❌ Initialization failed: {e}")
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show or generate configuration.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        work_path = _get_default_work_path().expanduser().resolve()
        
        if args.generate:
            # Generate .env file
            work_path.mkdir(parents=True, exist_ok=True)
            env_file = work_path / ".env"
            
            if env_file.exists() and not args.force:
                print(f"⚠️  {env_file} already exists.")
                print("   Use --force to overwrite.")
                return 1
            
            _generate_env_file(env_file)
            print(f"✅ Generated {env_file}")
            print()
            print("Edit this file to configure your LLM settings:")
            print(f"  {env_file}")
            return 0
        
        # Show current configuration
        print("=" * 60)
        print("Sirchmunk Configuration")
        print("=" * 60)
        print()
        
        # Load env file if exists
        env_file = work_path / ".env"
        if env_file.exists():
            _load_env_file(env_file)
            print(f"📄 Config file: {env_file}")
        else:
            print(f"📄 Config file: Not found ({env_file})")
            print("   Run 'sirchmunk config --generate' to create one.")
        
        print()
        print("Current Settings:")
        print(f"  SIRCHMUNK_WORK_PATH: {os.getenv('SIRCHMUNK_WORK_PATH', '~/.sirchmunk (default)')}")
        print(f"  LLM_BASE_URL: {os.getenv('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1 (default)')}")
        print(f"  LLM_API_KEY: {'***' + os.getenv('LLM_API_KEY', '')[-4:] if os.getenv('LLM_API_KEY') else 'Not set'}")
        print(f"  LLM_MODEL_NAME: {os.getenv('LLM_MODEL_NAME', 'qwen3-max (default)')}")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Config command failed: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the Sirchmunk API server.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load environment
        work_path = _get_default_work_path().expanduser().resolve()
        env_file = work_path / ".env"
        if env_file.exists():
            _load_env_file(env_file)
        
        # Import uvicorn here to avoid slow startup
        try:
            import uvicorn
        except ImportError:
            print("❌ uvicorn is not installed.")
            print("   Install it with: pip install uvicorn")
            return 1
        
        print("=" * 60)
        print(f"Sirchmunk API Server v{__version__}")
        print("=" * 60)
        print()
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Reload: {args.reload}")
        print()
        print(f"  API Docs: http://{args.host}:{args.port}/docs")
        print(f"  Health: http://{args.host}:{args.port}/health")
        print()
        print("Press Ctrl+C to stop the server.")
        print("=" * 60)
        print()
        
        uvicorn.run(
            "sirchmunk.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level.lower(),
        )
        
        return 0
        
    except KeyboardInterrupt:
        print("\n✅ Server stopped.")
        return 0
    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        print(f"❌ Server error: {e}")
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Perform a search query.
    
    Can operate in two modes:
    - Local mode (default): Direct search using AgenticSearch
    - Client mode (--api): Call the API server
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Load environment
        work_path = _get_default_work_path().expanduser().resolve()
        env_file = work_path / ".env"
        if env_file.exists():
            _load_env_file(env_file)
        
        query = args.query
        search_paths = args.paths or [os.getcwd()]
        
        if args.api:
            # Client mode: call API server
            return _search_via_api(
                query=query,
                search_paths=search_paths,
                api_url=args.api_url,
                mode=args.mode,
                output_format=args.output,
            )
        else:
            # Local mode: direct search
            return asyncio.run(_search_local(
                query=query,
                search_paths=search_paths,
                mode=args.mode,
                output_format=args.output,
                verbose=args.verbose,
            ))
        
    except KeyboardInterrupt:
        print("\n⚠️ Search cancelled.")
        return 130
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        print(f"❌ Search error: {e}")
        return 1


async def _search_local(
    query: str,
    search_paths: list,
    mode: str = "DEEP",
    output_format: str = "text",
    verbose: bool = False,
) -> int:
    """Execute search locally using AgenticSearch.
    
    Args:
        query: Search query
        search_paths: Paths to search
        mode: Search mode (FAST, DEEP, FILENAME_ONLY)
        output_format: Output format (text, json)
        verbose: Enable verbose output
        
    Returns:
        Exit code
    """
    from sirchmunk.search import AgenticSearch
    from sirchmunk.llm.openai_chat import OpenAIChat
    from sirchmunk.utils.constants import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME
    
    # Validate API key
    if not LLM_API_KEY:
        print("❌ LLM_API_KEY is not set.")
        print("   Configure it in ~/.sirchmunk/.env or set the environment variable.")
        return 1
    
    # Create LLM client
    llm = OpenAIChat(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
    )
    
    # Create search instance
    work_path = _get_default_work_path()
    searcher = AgenticSearch(
        llm=llm,
        work_path=str(work_path),
        verbose=verbose,
    )
    
    if not verbose:
        print(f"🔍 Searching: {query}")
        print(f"   Mode: {mode}")
        print(f"   Paths: {', '.join(search_paths)}")
        print()
    
    # Execute search
    result = await searcher.search(
        query=query,
        search_paths=search_paths,
        mode=mode,
        return_cluster=output_format == "json",
    )
    
    # Output result
    if output_format == "json":
        if hasattr(result, "to_dict"):
            output = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        else:
            output = json.dumps({"result": result}, indent=2, ensure_ascii=False)
        print(output)
    else:
        if result:
            print("=" * 60)
            print("Search Results")
            print("=" * 60)
            print()
            print(result)
        else:
            print("No results found.")
    
    return 0


def _search_via_api(
    query: str,
    search_paths: list,
    api_url: str = "http://localhost:8584",
    mode: str = "DEEP",
    output_format: str = "text",
) -> int:
    """Execute search via API server.
    
    Args:
        query: Search query
        search_paths: Paths to search
        api_url: API server URL
        mode: Search mode
        output_format: Output format
        
    Returns:
        Exit code
    """
    try:
        import requests
    except ImportError:
        print("❌ requests library is not installed.")
        print("   Install it with: pip install requests")
        return 1
    
    print(f"🔍 Searching via API: {api_url}")
    print(f"   Query: {query}")
    print(f"   Mode: {mode}")
    print()
    
    try:
        response = requests.post(
            f"{api_url}/api/v1/search",
            json={
                "query": query,
                "search_paths": search_paths,
                "mode": mode,
            },
            timeout=300,  # 5 minute timeout for long searches
        )
        response.raise_for_status()
        
        data = response.json()
        
        if output_format == "json":
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            if data.get("success"):
                print("=" * 60)
                print("Search Results")
                print("=" * 60)
                print()
                print(data.get("data", {}).get("summary", "No results found."))
            else:
                print(f"❌ Search failed: {data.get('error', 'Unknown error')}")
                return 1
        
        return 0
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API server at {api_url}")
        print("   Make sure the server is running: sirchmunk serve")
        return 1
    except requests.exceptions.Timeout:
        print("❌ Request timed out.")
        return 1
    except Exception as e:
        print(f"❌ API error: {e}")
        return 1


def _generate_env_file(env_file: Path):
    """Generate a default .env configuration file.
    
    Args:
        env_file: Path to write the .env file
    """
    content = """# ===== Sirchmunk Configuration =====
# Generated by: sirchmunk config --generate

# ===== LLM Settings =====
# LLM API base URL (OpenAI-compatible endpoint)
LLM_BASE_URL=https://api.openai.com/v1

# LLM API key (REQUIRED - get from your LLM provider)
LLM_API_KEY=

# LLM model name
LLM_MODEL_NAME=gpt-5.2

# ===== Sirchmunk Settings =====
# Working directory for data and cache
SIRCHMUNK_WORK_PATH=~/.sirchmunk

# Enable verbose logging (true/false)
SIRCHMUNK_VERBOSE=false

# ===== Search Settings =====
# Maximum directory depth to search
DEFAULT_MAX_DEPTH=5

# Number of top files to return
DEFAULT_TOP_K_FILES=3

# Number of keyword granularity levels
DEFAULT_KEYWORD_LEVELS=3

# Grep operation timeout in seconds
GREP_TIMEOUT=60.0

# ===== Cluster Settings =====
# Enable knowledge cluster reuse with embeddings
SIRCHMUNK_ENABLE_CLUSTER_REUSE=true

# Similarity threshold for cluster reuse (0.0-1.0)
CLUSTER_SIM_THRESHOLD=0.85

# Number of similar clusters to retrieve
CLUSTER_SIM_TOP_K=3

# Maximum queries per cluster (FIFO)
MAX_QUERIES_PER_CLUSTER=5
"""
    
    with open(env_file, "w") as f:
        f.write(content)


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code
    """
    print(f"sirchmunk {__version__}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="sirchmunk",
        description="Sirchmunk: Agentic Search for raw data intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sirchmunk init                    Initialize Sirchmunk
  sirchmunk config --generate       Generate configuration file
  sirchmunk serve                   Start API server
  sirchmunk serve --port 8000       Start on custom port
  sirchmunk search "find auth"      Search in current directory
  sirchmunk search "bug" ./src      Search in specific path
  sirchmunk search "api" --mode FILENAME_ONLY
                                    Quick filename search
        """,
    )
    
    parser.add_argument(
        "-V", "--version",
        action="store_true",
        help="Show version and exit",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # === init command ===
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize Sirchmunk working directory",
        description="Create directory structure and generate initial configuration.",
    )
    init_parser.add_argument(
        "--work-path",
        default=str(_get_default_work_path()),
        help="Working directory path (default: ~/.sirchmunk)",
    )
    init_parser.set_defaults(func=cmd_init)
    
    # === config command ===
    config_parser = subparsers.add_parser(
        "config",
        help="Show or generate configuration",
        description="Display current configuration or generate a new .env file.",
    )
    config_parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate .env configuration file",
    )
    config_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing configuration file",
    )
    config_parser.set_defaults(func=cmd_config)
    
    # === serve command ===
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the Sirchmunk API server",
        description="Launch the FastAPI server for API access and WebUI.",
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8584,
        help="Port to listen on (default: 8584)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    serve_parser.set_defaults(func=cmd_serve)
    
    # === search command ===
    search_parser = subparsers.add_parser(
        "search",
        help="Perform a search query",
        description="Search documents and code using AgenticSearch.",
    )
    search_parser.add_argument(
        "query",
        help="Search query or question",
    )
    search_parser.add_argument(
        "paths",
        nargs="*",
        help="Paths to search (default: current directory)",
    )
    search_parser.add_argument(
        "--mode", "-m",
        default="DEEP",
        choices=["FAST", "DEEP", "FILENAME_ONLY"],
        help="Search mode (default: DEEP)",
    )
    search_parser.add_argument(
        "--output", "-o",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    search_parser.add_argument(
        "--api",
        action="store_true",
        help="Use API server instead of local search",
    )
    search_parser.add_argument(
        "--api-url",
        default="http://localhost:8584",
        help="API server URL (default: http://localhost:8584)",
    )
    search_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    search_parser.set_defaults(func=cmd_search)
    
    # === version command ===
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    version_parser.set_defaults(func=cmd_version)
    
    return parser


def run_cmd():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --version flag
    if args.version:
        print(f"sirchmunk {__version__}")
        sys.exit(0)
    
    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Setup logging
    _setup_logging()
    
    # Execute command
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    run_cmd()
