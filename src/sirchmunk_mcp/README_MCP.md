# Sirchmunk MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes Sirchmunk's intelligent code and document search capabilities as MCP tools.

## Features

- **🔍 Multi-Mode Search**
  - **DEEP**: Comprehensive knowledge extraction with full context analysis (~10-30s)
  - **FAST**: Quick content search without deep LLM processing (~3-8s)
  - **FILENAME_ONLY**: Fast filename pattern matching (<1s)

- **🧠 Knowledge Cluster Management**
  - Automatic knowledge extraction and storage
  - Semantic similarity-based cluster reuse
  - Version tracking and lifecycle management

- **🔌 MCP Integration**
  - Standard MCP protocol support
  - Stdio transport (Claude Desktop compatible)
  - SSE transport (coming soon)

## Installation

### From PyPI (Recommended)

```bash
pip install sirchmunk-mcp
```

### From Source

```bash
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk
pip install -e .
```

## Quick Start

### 1. Initialize

```bash
# Initialize Sirchmunk environment
sirchmunk-mcp init

# Generate configuration templates
sirchmunk-mcp config --generate
```

### 2. Configure

Edit the generated `.env` file with your LLM API credentials:

```bash
# Required
LLM_API_KEY=sk-your-api-key-here

# Optional (defaults shown)
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4-turbo-preview
SIRCHMUNK_WORK_PATH=~/.sirchmunk
```

### 3. Run Server

```bash
# Start MCP server (stdio mode)
sirchmunk-mcp serve

# Or with custom configuration
MCP_LOG_LEVEL=DEBUG sirchmunk-mcp serve
```

## Integration with Claude Desktop

### macOS

1. Edit `mcp_config.json` with your API key
2. Copy to Claude Desktop config directory:
   ```bash
   cp config/mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
3. Restart Claude Desktop

### Linux

```bash
cp config/mcp_config.json ~/.config/Claude/claude_desktop_config.json
```

### Windows

```cmd
copy config\mcp_config.json %APPDATA%\Claude\claude_desktop_config.json
```

## Usage Examples

### Example 1: Deep Code Search

```
User: "Search for transformer attention implementation in my project"

Claude: [Using sirchmunk_search tool]
{
  "query": "transformer attention implementation",
  "search_paths": ["/path/to/project"],
  "mode": "DEEP",
  "top_k_files": 3
}

Response: Comprehensive analysis with:
- 3 relevant files identified
- Key code snippets extracted
- Implementation patterns discovered
- Usage context provided
```

### Example 2: Fast Filename Search

```
User: "Find all test files in the project"

Claude: [Using sirchmunk_search tool]
{
  "query": "test",
  "search_paths": ["/path/to/project"],
  "mode": "FILENAME_ONLY",
  "top_k_files": 10
}

Response: List of matching files with paths and relevance scores
```

### Example 3: Retrieve Knowledge Cluster

```
User: "Show me the saved knowledge about authentication"

Claude: [Using sirchmunk_list_clusters tool]
{
  "limit": 5,
  "sort_by": "hotness"
}

[Then using sirchmunk_get_cluster tool]
{
  "cluster_id": "C1007"
}

Response: Detailed cluster information with evidences and patterns
```

## Available Tools

### `sirchmunk_search`

Intelligent code and document search.

**Parameters:**
- `query` (string, required): Search query or question
- `search_paths` (array, required): Paths to search in
- `mode` (string): Search mode (DEEP/FAST/FILENAME_ONLY, default: DEEP)
- `max_depth` (integer): Maximum directory depth (default: 5)
- `top_k_files` (integer): Number of top files to return (default: 3)
- `keyword_levels` (integer): Keyword granularity levels (default: 3, DEEP mode only)
- `include` (array): File patterns to include (glob)
- `exclude` (array): File patterns to exclude (glob)
- `return_cluster` (boolean): Return full KnowledgeCluster object (default: false)

**Returns:**
- DEEP/FAST mode: Formatted search result summary (string)
- FILENAME_ONLY mode: List of file matches with metadata (array)
- With `return_cluster=true`: Full KnowledgeCluster object

### `sirchmunk_get_cluster`

Retrieve a saved knowledge cluster by ID.

**Parameters:**
- `cluster_id` (string, required): Cluster ID (e.g., 'C1007')

**Returns:** Full cluster information or error message

### `sirchmunk_list_clusters`

List all saved knowledge clusters.

**Parameters:**
- `limit` (integer): Maximum number of clusters (default: 10)
- `sort_by` (string): Sort field (hotness/confidence/last_modified, default: last_modified)

**Returns:** List of cluster metadata

## Configuration

### Environment Variables

See `config/env.example` for all available environment variables.

**Key Settings:**

```bash
# LLM Configuration
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL_NAME=gpt-4-turbo-preview

# Search Behavior
DEFAULT_MAX_DEPTH=5
DEFAULT_TOP_K_FILES=3
DEFAULT_KEYWORD_LEVELS=3

# Cluster Reuse
SIRCHMUNK_ENABLE_CLUSTER_REUSE=true
CLUSTER_SIM_THRESHOLD=0.85
CLUSTER_SIM_TOP_K=3

# Logging
MCP_LOG_LEVEL=INFO
SIRCHMUNK_VERBOSE=false
```

### Programmatic Configuration

```python
from sirchmunk_mcp import Config, create_server

# Load from environment
config = Config.from_env()

# Or create custom config
config = Config(
    llm=LLMConfig(
        base_url="https://api.openai.com/v1",
        api_key="sk-xxx",
        model_name="gpt-4-turbo-preview",
    ),
    sirchmunk=SirchmunkConfig(
        work_path=Path.home() / ".sirchmunk",
        enable_cluster_reuse=True,
    )
)

# Create and run server
server = create_server(config)
await run_stdio_server(config)
```

## CLI Reference

### `sirchmunk-mcp serve`

Run the MCP server.

```bash
sirchmunk-mcp serve [OPTIONS]

Options:
  --transport {stdio,sse}  Transport protocol (default: stdio)
  --host TEXT              Host for SSE transport (default: localhost)
  --port INTEGER           Port for SSE transport (default: 8080)
  --log-level {DEBUG,INFO,WARNING,ERROR}  Logging level
```

### `sirchmunk-mcp init`

Initialize Sirchmunk environment.

```bash
sirchmunk-mcp init [OPTIONS]

Options:
  --work-path PATH  Working directory path (default: ~/.sirchmunk)
```

### `sirchmunk-mcp config`

Manage configuration.

```bash
sirchmunk-mcp config [OPTIONS]

Options:
  --generate    Generate configuration templates
  --output PATH Output path for generated config
```

### `sirchmunk-mcp version`

Show version information.

```bash
sirchmunk-mcp version
```

## Architecture

```
┌─────────────────────────────────┐
│   MCP Client (Claude Desktop)   │
└────────────┬────────────────────┘
             │ MCP Protocol (stdio)
             ↓
┌─────────────────────────────────┐
│    Sirchmunk MCP Server         │
│  ┌──────────────────────────┐   │
│  │  MCP Service Layer       │   │
│  │  - Tool Registration     │   │
│  │  - Request Handling      │   │
│  └──────────┬───────────────┘   │
│             ↓                    │
│  ┌──────────────────────────┐   │
│  │  Sirchmunk Service       │   │
│  │  - AgenticSearch Init    │   │
│  │  - Config Management     │   │
│  └──────────┬───────────────┘   │
│             ↓                    │
│  ┌──────────────────────────┐   │
│  │  AgenticSearch Core      │   │
│  │  - DEEP/FAST/FILENAME    │   │
│  │  - KnowledgeBase         │   │
│  │  - GrepRetriever         │   │
│  │  - EmbeddingUtil         │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

## Performance

| Mode | LLM Calls | Speed | Use Case |
|------|-----------|-------|----------|
| DEEP | 4-6 | 10-30s | Comprehensive analysis |
| FAST | 0-2 | 3-8s | Quick content search |
| FILENAME_ONLY | 0 | <1s | File discovery |

## Troubleshooting

### Server won't start

1. Check API key is set:
   ```bash
   echo $LLM_API_KEY
   ```

2. Verify dependencies:
   ```bash
   sirchmunk-mcp init
   ```

3. Check logs:
   ```bash
   MCP_LOG_LEVEL=DEBUG sirchmunk-mcp serve
   ```

### ripgrep-all not found

Install ripgrep-all manually:

```bash
# macOS
brew install rga

# Linux
# Download from https://github.com/phiresky/ripgrep-all/releases
```

### Claude Desktop not detecting server

1. Verify config file location
2. Check JSON syntax in `claude_desktop_config.json`
3. Restart Claude Desktop completely
4. Check Claude Desktop logs

## Development

### Setup

```bash
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint
ruff check src/

# Type check
mypy src/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Links

- [Sirchmunk GitHub](https://github.com/modelscope/sirchmunk)
- [MCP Documentation](https://modelcontextprotocol.io)
- [Claude Desktop](https://claude.ai/desktop)

## Support

- GitHub Issues: https://github.com/modelscope/sirchmunk/issues
- Documentation: https://github.com/modelscope/sirchmunk#readme
