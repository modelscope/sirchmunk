# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Command-line interface for Sirchmunk MCP Server.

Provides commands for server management, initialization, and diagnostics.
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .server import run_stdio_server, run_http_server


logger = logging.getLogger(__name__)


def get_mcp_version():
    try:
        import importlib.metadata
        return importlib.metadata.version("mcp")
    except ImportError:
        return None


def _setup_stdio_safe_environment():
    """Configure environment for safe stdio MCP communication.
    
    In MCP stdio mode, stdout is reserved exclusively for JSON-RPC messages.
    Any non-JSON output to stdout will break the protocol. This function
    sets environment variables to suppress verbose output from third-party
    libraries (ModelScope, transformers, tqdm, etc.) that might print to stdout.
    """
    # Set environment variables to suppress third-party library outputs
    # These must be set BEFORE importing the libraries
    os.environ["MODELSCOPE_LOG_LEVEL"] = "ERROR"
    os.environ["MODELSCOPE_CACHE"] = os.path.expanduser("~/.sirchmunk/.cache/models")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")


def cmd_serve(args: argparse.Namespace) -> int:
    """Run MCP server.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        # Override transport from command-line if specified
        if args.transport:
            os.environ["MCP_TRANSPORT"] = args.transport
        
        if args.host:
            os.environ["MCP_HOST"] = args.host
        
        if args.port:
            os.environ["MCP_PORT"] = str(args.port)
        
        # Determine transport mode early (before loading config triggers imports)
        transport = args.transport or os.environ.get("MCP_TRANSPORT", "stdio")
        
        # Set up safe environment BEFORE any sirchmunk imports for stdio mode
        # This prevents ModelScope/transformers from printing to stdout
        if transport == "stdio":
            os.environ["MCP_TRANSPORT"] = "stdio"  # Signal to service.py
            _setup_stdio_safe_environment()
        
        # Load configuration (may trigger sirchmunk imports)
        config = Config.from_env()
        
        # Configure logging - MUST use stderr for stdio transport
        # stdout is reserved for JSON-RPC messages in MCP stdio mode
        log_level = args.log_level or config.mcp.log_level
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr,  # Critical: logs must go to stderr, not stdout
        )
        
        logger.info(f"Sirchmunk MCP Server v{__version__}")
        logger.info(f"Transport: {config.mcp.transport}")
        
        # Run server
        if config.mcp.transport == "stdio":
            # Check if running in interactive terminal
            if sys.stdin.isatty():
                print()
                print("=" * 60)
                print("  ⚠️  Sirchmunk MCP Server - STDIO Mode")
                print("=" * 60)
                print()
                print("  STDIO mode is designed to be launched by an MCP client")
                print("  (e.g., Claude Desktop, Cursor IDE), not run directly")
                print("  in an interactive terminal.")
                print()
                print("  The server expects JSON-RPC messages from an MCP client.")
                print("  Running in a terminal will cause errors.")
                print()
                print("  Options:")
                print("  1. Configure your MCP client to launch this server")
                print("  2. Use HTTP mode: sirchmunk-mcp serve --transport http")
                print()
                print("=" * 60)
                print()
                
                # Ask user if they want to continue anyway
                try:
                    response = input("Continue anyway? (y/N): ").strip().lower()
                    if response != 'y':
                        print("Server not started. Use an MCP client to launch this server.")
                        return 0
                    print()
                    print("Starting server... Press Ctrl+C to stop.")
                    print("(Any input you type will cause errors)")
                    print()
                except (EOFError, KeyboardInterrupt):
                    print("\nServer not started.")
                    return 0
            
            asyncio.run(run_stdio_server(config))
        elif config.mcp.transport == "http":
            asyncio.run(run_http_server(config))
        else:
            logger.error(f"Unknown transport: {config.mcp.transport}")
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        return 0
    
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize Sirchmunk environment.
    
    Creates working directory, configuration files, and performs
    dependency checks.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        print(f"Initializing Sirchmunk MCP v{__version__}")
        print()
        
        # Determine work path
        if args.work_path:
            work_path = Path(args.work_path)
        else:
            work_path = Path(os.getenv("SIRCHMUNK_WORK_PATH", str(Path.home() / ".sirchmunk")))
        
        work_path = work_path.expanduser().resolve()
        
        print(f"Work path: {work_path}")
        
        # Create work directory
        work_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created work directory")
        
        # Create subdirectories
        (work_path / ".cache").mkdir(exist_ok=True)
        (work_path / ".cache" / "models").mkdir(exist_ok=True)
        (work_path / "data").mkdir(exist_ok=True)
        print(f"✓ Created subdirectories")
        
        # Check dependencies
        print()
        print("Checking dependencies...")
        
        # Check ripgrep-all
        if shutil.which("rga"):
            print("✓ ripgrep-all (rga) is installed")
        else:
            print("✗ ripgrep-all (rga) is not installed")
            print("  Installing ripgrep-all...")
            try:
                from sirchmunk.utils.install_rga import install_rga
                install_rga()
                print("✓ ripgrep-all installed successfully")
            except Exception as e:
                print(f"✗ Failed to install ripgrep-all: {e}")
                print("  Please install manually: https://github.com/phiresky/ripgrep-all")
        
        # Check Python packages
        mcp_version = get_mcp_version()
        if mcp_version:
            print(f"✓ MCP package installed (v{mcp_version})")
        else:
            print("✗ MCP package not found")
            print("  Install with: pip install mcp")
        
        try:
            import sirchmunk
            print(f"✓ sirchmunk package installed")
        except ImportError:
            print("✗ sirchmunk package not found")
            print("  Install with: pip install sirchmunk")
        
        # Check environment variables
        print()
        print("Checking environment variables...")
        
        llm_api_key = os.getenv("LLM_API_KEY")
        if llm_api_key:
            print(f"✓ LLM_API_KEY is set ({llm_api_key[:8]}...)")
        else:
            print("✗ LLM_API_KEY is not set")
            print("  Set it in your .mcp_env file or environment")
        
        print()
        print("Initialization complete!")
        print()
        print("Next steps:")
        print("1. Configure LLM_API_KEY in .mcp_env file")
        print("2. Run 'sirchmunk-mcp serve' to start the server")
        print("3. Configure your MCP client (e.g., Claude Desktop)")
        
        return 0
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Show or generate configuration.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    try:
        if args.generate:
            # Determine work path
            work_path = Path(os.getenv("SIRCHMUNK_WORK_PATH", str(Path.home() / ".sirchmunk")))
            work_path = work_path.expanduser().resolve()
            work_path.mkdir(parents=True, exist_ok=True)
            
            # Generate .mcp_env file in work_path
            env_file = work_path / ".mcp_env"
            
            env_content = """# ===== LLM Configuration =====
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL_NAME=gpt-5.2

# ===== Sirchmunk Settings =====
SIRCHMUNK_WORK_PATH=~/.sirchmunk
SIRCHMUNK_VERBOSE=false
SIRCHMUNK_ENABLE_CLUSTER_REUSE=true

# ===== Cluster Similarity Settings =====
CLUSTER_SIM_THRESHOLD=0.85
CLUSTER_SIM_TOP_K=3
MAX_QUERIES_PER_CLUSTER=5

# ===== Search Settings =====
DEFAULT_MAX_DEPTH=5
DEFAULT_TOP_K_FILES=3
DEFAULT_KEYWORD_LEVELS=3
GREP_TIMEOUT=60.0

# ===== MCP Server Settings =====
MCP_SERVER_NAME=sirchmunk
MCP_SERVER_VERSION=0.1.0
MCP_LOG_LEVEL=INFO
MCP_TRANSPORT=stdio
"""
            
            env_file.write_text(env_content)
            print(f"Generated {env_file}")
            
            # Generate MCP client config
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = Path.cwd() / "mcp_config.json"
            
            mcp_config = """{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "your-api-key",
        "LLM_MODEL_NAME": "gpt-5.2",
        "SIRCHMUNK_WORK_PATH": "~/.sirchmunk",
        "SIRCHMUNK_VERBOSE": "false"
      }
    }
  }
}
"""
            
            output_path.write_text(mcp_config)
            print(f"Generated {output_path}")
            
            print()
            print("Configuration files generated successfully!")
            print()
            print(f"⚠️  Please edit {env_file} to set your LLM_API_KEY")
            print()
            print("To use with Claude Desktop:")
            print("1. Edit mcp_config.json with your API key")
            if sys.platform == "darwin":
                config_dir = Path.home() / "Library/Application Support/Claude"
            elif sys.platform == "linux":
                config_dir = Path.home() / ".config/Claude"
            else:
                config_dir = Path(os.getenv("APPDATA", "")) / "Claude"
            
            print(f"2. Copy to: {config_dir}/claude_desktop_config.json")
            print("3. Restart Claude Desktop")
        
        else:
            # Show current configuration
            config = Config.from_env()
            print("Current Configuration:")
            print()
            print(f"LLM:")
            print(f"  Base URL: {config.llm.base_url}")
            print(f"  Model: {config.llm.model_name}")
            print(f"  API Key: {config.llm.api_key[:8]}..." if config.llm.api_key else "  API Key: (not set)")
            print()
            print(f"Sirchmunk:")
            print(f"  Work Path: {config.sirchmunk.work_path}")
            print(f"  Verbose: {config.sirchmunk.verbose}")
            print(f"  Cluster Reuse: {config.sirchmunk.enable_cluster_reuse}")
            print()
            print(f"MCP Server:")
            print(f"  Name: {config.mcp.server_name}")
            print(f"  Version: {config.mcp.server_version}")
            print(f"  Transport: {config.mcp.transport}")
            print(f"  Log Level: {config.mcp.log_level}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Config command failed: {e}", exc_info=True)
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information.
    
    Args:
        args: Command-line arguments
    
    Returns:
        Exit code
    """
    print(f"Sirchmunk MCP Server v{__version__}")
    
    mcp_version = get_mcp_version()
    if mcp_version:
        print(f"MCP package v{mcp_version}")
    else:
        print("MCP package (not installed)")

    try:
        import sirchmunk
        print(f"Sirchmunk Core (installed)")
    except ImportError:
        pass
    
    return 0


def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Sirchmunk MCP Server - Intelligent code search as an MCP service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run MCP server")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        help="Transport protocol (default: stdio)",
    )
    serve_parser.add_argument(
        "--host",
        help="Host for HTTP transport (default: localhost)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        help="Port for HTTP transport (default: 8080)",
    )
    serve_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize Sirchmunk environment")
    init_parser.add_argument(
        "--work-path",
        help="Working directory path (default: ~/.sirchmunk)",
    )
    
    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate configuration templates",
    )
    config_parser.add_argument(
        "--output",
        help="Output path for generated config",
    )
    
    # version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    # Handle --version flag
    if args.version:
        return cmd_version(args)
    
    # Handle commands
    if args.command == "serve":
        return cmd_serve(args)
    elif args.command == "init":
        return cmd_init(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "version":
        return cmd_version(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
