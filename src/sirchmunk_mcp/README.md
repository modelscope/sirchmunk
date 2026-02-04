# Sirchmunk MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes Sirchmunk's intelligent code and document search capabilities as MCP tools.

## Features

- **🔍 Multi-Mode Search**
  - **DEEP**: Comprehensive knowledge extraction with full context analysis (~10-30s)
  - **FILENAME_ONLY**: Fast filename pattern matching (<1s)

- **🧠 Knowledge Cluster Management**
  - Automatic knowledge extraction and storage
  - Semantic similarity-based cluster reuse
  - Version tracking and lifecycle management

- **🔌 MCP Integration**
  - Standard MCP protocol support
  - Stdio transport (Claude Desktop / Claude Code compatible)
  - Streamable HTTP transport (for web-based clients)

---

## Quick Start (5 Minutes)

### Step 1: Install

```bash
pip install sirchmunk-mcp
```

### Step 2: Initialize

```bash
sirchmunk-mcp init
sirchmunk-mcp config --generate
```

### Step 3: Configure

Edit `.mcp_env` with your API key:

```bash
# Required
LLM_API_KEY=your-api-key
LLM_MODEL_NAME=gpt-5.2
LLM_BASE_URL=https://api.openai.com/v1
```

### Step 4: Test

Anthropic provides a dedicated debugging tool called MCP Inspector (runnable via npx). It simulates a Client's behavior and provides a web-based interface for interaction.

```bash
MCP_LOG_LEVEL=INFO npx @modelcontextprotocol/inspector sirchmunk-mcp serve
```

You should see:
```
Starting MCP inspector...
⚙️ Proxy server listening on localhost:6277
🔑 Session token: a2057c4...
   Use this token to authenticate requests or set DANGEROUSLY_OMIT_AUTH=true to disable auth

🚀 MCP Inspector is up and running at:
   http://localhost:6274/?MCP_PROXY_AUTH_TOKEN=a2057c4...

🌐 Opening browser...

```

Press `Ctrl+C` to stop.

**How to use**:
- Connect -> Tools -> List Tools -> `sirchmunk_search` -> Input parameters (`query` and `search_paths`) -> Run Tool
- Check the response for search results.

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Memory**: At least 2GB RAM recommended
- **LLM API Key**: OpenAI or compatible endpoint

### Method 1: From PyPI (Recommended)

```bash
# Create a virtual environment (optional but recommended)
conda create -n sirchmunk_mcp python=3.13 -y
conda activate sirchmunk_mcp

# Basic installation
pip install sirchmunk-mcp
```

### Method 2: From Source

```bash
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk/src/sirchmunk_mcp
pip install -e .
```

### Installing ripgrep-all (Optional)

Sirchmunk uses `ripgrep-all` for document search. 
<br/>
It will be installed automatically during initialization, but you can install it manually, see https://github.com/phiresky/ripgrep-all

---

## Integration with Claude Code / Claude Desktop

### Cursor IDE

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "LLM_MODEL_NAME": "gpt-5.2"
      }
    }
  }
}
```

### Claude Desktop

Edit the configuration file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "LLM_MODEL_NAME": "gpt-5.2",
        "SIRCHMUNK_WORK_PATH": "~/.sirchmunk",
        "MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Note:** If using virtual environment, use the full path:

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "/path/to/sirchmunk-env/bin/sirchmunk-mcp",
      "args": ["serve"],
      "env": { ... }
    }
  }
}
```

### Restart Client

After configuration, completely quit and restart your Claude client.

---

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

Response: Comprehensive analysis with code snippets and patterns
```

### Example 2: Fast Filename Search

```
User: "Find all test files in the project"

Claude: [Using sirchmunk_search tool]
{
  "query": "test",
  "search_paths": ["/path/to/project"],
  "mode": "FILENAME_ONLY"
}

Response: List of matching files with paths
```

### Example 3: Knowledge Cluster Management

```
User: "Show saved knowledge clusters"

Claude: [Using sirchmunk_list_clusters tool]

User: "Show details of cluster C1007"

Claude: [Using sirchmunk_get_cluster tool]
{
  "cluster_id": "C1007"
}
```

---

## Available Tools

### `sirchmunk_search`

Intelligent code and document search.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query or question |
| `search_paths` | array | ✅ | - | Paths to search in |
| `mode` | string | | "DEEP" | DEEP/FILENAME_ONLY |
| `max_depth` | integer | | 5 | Directory search depth |
| `top_k_files` | integer | | 3 | Files to return |
| `keyword_levels` | integer | | 3 | Keyword granularity (DEEP only) |
| `include` | array | | - | Glob patterns to include |
| `exclude` | array | | - | Glob patterns to exclude |
| `return_cluster` | boolean | | false | Return full KnowledgeCluster |

### `sirchmunk_get_cluster`

Retrieve a saved knowledge cluster by ID.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cluster_id` | string | ✅ | Cluster ID (e.g., 'C1007') |

### `sirchmunk_list_clusters`

List all saved knowledge clusters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 10 | Maximum clusters to return |
| `sort_by` | string | "last_modified" | hotness/confidence/last_modified |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Your LLM API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `LLM_MODEL_NAME` | `gpt-5.2` | Model to use |
| `SIRCHMUNK_WORK_PATH` | `~/.sirchmunk` | Working directory |
| `SIRCHMUNK_ENABLE_CLUSTER_REUSE` | `true` | Enable knowledge reuse |
| `CLUSTER_SIM_THRESHOLD` | `0.85` | Similarity threshold |
| `DEFAULT_MAX_DEPTH` | `5` | Default search depth |
| `DEFAULT_TOP_K_FILES` | `3` | Default files count |
| `MCP_LOG_LEVEL` | `INFO` | Logging level |

### Using Custom LLM Providers

**Local LLM (Ollama, LM Studio):**
```json
{
  "env": {
    "LLM_BASE_URL": "http://localhost:11434/v1",
    "LLM_API_KEY": "ollama",
    "LLM_MODEL_NAME": "llama3"
  }
}
```

**Azure OpenAI:**
```json
{
  "env": {
    "LLM_BASE_URL": "https://your-resource.openai.azure.com/",
    "LLM_API_KEY": "your-azure-key",
    "LLM_MODEL_NAME": "gpt-5.2"
  }
}
```

### Programmatic Configuration

```python
from sirchmunk_mcp import Config, create_server

# Load from environment
config = Config.from_env()

# Create and run server
server = create_server(config)
```

---

## CLI Reference

### `sirchmunk-mcp serve`

Run the MCP server.

```bash
sirchmunk-mcp serve [OPTIONS]

Options:
  --transport {stdio,http}  Transport protocol (default: stdio)
  --host TEXT               Host for HTTP transport (default: localhost)
  --port INTEGER            Port for HTTP transport (default: 8080)
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

---

## Architecture

```
┌─────────────────────────────────┐
│   MCP Client (Claude/Cursor)    │
└────────────┬────────────────────┘
             │ MCP Protocol (stdio/http)
             ↓
┌─────────────────────────────────┐
│    Sirchmunk MCP Server         │
│  ┌──────────────────────────┐   │
│  │  FastMCP Layer           │   │
│  │  - @mcp.tool() decorators│   │
│  │  - Auto tool discovery   │   │
│  └──────────┬───────────────┘   │
│             ↓                   │
│  ┌──────────────────────────┐   │
│  │  Sirchmunk Service       │   │
│  │  - AgenticSearch Init    │   │
│  │  - Config Management     │   │
│  └──────────┬───────────────┘   │
│             ↓                   │
│  ┌──────────────────────────┐   │
│  │  AgenticSearch Core      │   │
│  │  - DEEP/FILENAME         │   │
│  │  - KnowledgeBase         │   │
│  │  - GrepRetriever         │   │
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

---

## Performance

| Mode | LLM Calls | Speed | Use Case |
|------|-----------|-------|----------|
| DEEP | 4-6 | 10-30s | Comprehensive analysis |
| FILENAME_ONLY | 0 | <1s | File discovery |

### Performance Tips

1. **Use specific search paths** - Narrow paths = faster searches
2. **Choose appropriate mode** - Match mode to task complexity
3. **Leverage cluster reuse** - Similar queries reuse cached knowledge
4. **Adjust depth** - Lower `max_depth` for faster results

---

## Troubleshooting

### "Command not found: sirchmunk-mcp"

```bash
# Verify installation
pip show sirchmunk-mcp

# Reinstall
pip install --force-reinstall sirchmunk-mcp
```

### "LLM API key cannot be empty"

```bash
# Check environment variable
echo $LLM_API_KEY

# Set it
export LLM_API_KEY="your-api-key"
```

### "ripgrep-all not found"

```bash
# Try auto-install
sirchmunk-mcp init

# Or install manually (see Installation section)
```

### Claude Desktop/Cursor not showing tools

1. Verify config file location and JSON syntax:
   ```bash
   python -m json.tool ~/.cursor/mcp.json
   ```
2. Check Claude client logs for MCP errors
3. Completely restart the client (quit, not just close)

### Slow initialization

First-time startup downloads embedding models. To skip:
```bash
export SIRCHMUNK_ENABLE_CLUSTER_REUSE=false
```

### Debug Mode

```bash
MCP_LOG_LEVEL=DEBUG SIRCHMUNK_VERBOSE=true sirchmunk-mcp serve
```

Check logs in `~/.sirchmunk/logs/`.

---

## Updating

```bash
# From PyPI
pip install --upgrade sirchmunk-mcp

# Re-initialize
sirchmunk-mcp init
```

---

## Uninstallation

```bash
# Remove package
pip uninstall sirchmunk-mcp

# Remove data (optional)
rm -rf ~/.sirchmunk
```

---

## Development

```bash
# Clone and install
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk/src/sirchmunk_mcp
pip install -e ".[dev]"
```

---

## Security Best Practices

1. **Never commit API keys** - Use environment variables
2. **Review search paths** - Only include trusted directories
3. **Monitor API usage** - DEEP mode uses more tokens
4. **Update regularly** - Keep dependencies current

---

## Links

- [Sirchmunk GitHub](https://github.com/modelscope/sirchmunk)
- [MCP Documentation](https://modelcontextprotocol.io)
- [MCP Specification](https://github.com/modelcontextprotocol/modelcontextprotocol)
- [Claude Desktop](https://claude.ai)

## License

Apache License 2.0 - see [LICENSE](LICENSE)

## Support

- **Issues**: https://github.com/modelscope/sirchmunk/issues
- **Logs**: `~/.sirchmunk/logs/`
