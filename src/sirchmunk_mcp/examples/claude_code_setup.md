# Sirchmunk MCP - Claude Code Integration Guide

This guide demonstrates how to install and use Sirchmunk MCP with Claude Code (Cursor, VS Code with Claude extension, etc.).

## Prerequisites

- Python 3.9+
- Claude Code client (Cursor IDE, VS Code + Claude extension, etc.)
- LLM API key (OpenAI or compatible)

## Installation

### Option 1: Global Installation

```bash
# Install sirchmunk-mcp globally
pip install sirchmunk-mcp

# Verify installation
sirchmunk-mcp version
```

### Option 2: Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python -m venv ~/.sirchmunk-env
source ~/.sirchmunk-env/bin/activate  # macOS/Linux
# or: ~/.sirchmunk-env\Scripts\activate  # Windows

# Install
pip install sirchmunk-mcp

# Note the executable path for configuration
which sirchmunk-mcp
# Example output: /Users/you/.sirchmunk-env/bin/sirchmunk-mcp
```

## Configuration for Claude Code

### Cursor IDE

Edit `~/.cursor/mcp.json` (create if not exists):

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_MODEL_NAME": "gpt-5.2",
        "SIRCHMUNK_WORK_PATH": "~/.sirchmunk",
        "MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**If using virtual environment:**

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "/Users/you/.sirchmunk-env/bin/sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "your-api-key",
        "LLM_MODEL_NAME": "gpt-5.2"
      }
    }
  }
}
```

### VS Code + Claude Extension

Edit VS Code settings or the MCP configuration file as specified by your Claude extension.

The configuration format is typically similar to Cursor.

### Claude Desktop App

Edit the Claude Desktop configuration file:

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
        "LLM_MODEL_NAME": "gpt-5.2"
      }
    }
  }
}
```

## Restart Claude Client

After configuration:

1. **Cursor**: Restart the IDE or reload window (Cmd/Ctrl + Shift + P → "Reload Window")
2. **VS Code**: Restart extension or reload window
3. **Claude Desktop**: Quit and restart the application completely

## Verify Integration

In your Claude Code client, try one of these prompts:

### Test 1: Basic Search

```
Use sirchmunk to search for "main function" in /path/to/your/project
```

### Test 2: Filename Search

```
Find all Python files in my current project using sirchmunk
```

### Test 3: Deep Analysis

```
Explain how authentication works in this codebase using sirchmunk DEEP mode
```

## Usage Examples

### Example 1: Code Understanding

**Prompt:**
```
Use sirchmunk to find and explain the database connection handling in my project
```

**Claude will:**
1. Invoke `sirchmunk_search` with mode="DEEP"
2. Return comprehensive analysis with code snippets
3. Explain the implementation patterns

### Example 2: File Discovery

**Prompt:**
```
Search for test files in /my/project using sirchmunk FILENAME_ONLY mode
```

**Claude will:**
1. Invoke `sirchmunk_search` with mode="FILENAME_ONLY"
2. Return list of matching files with paths

### Example 3: Quick Content Search

**Prompt:**
```
Find all references to "UserService" in FAST mode
```

**Claude will:**
1. Invoke `sirchmunk_search` with mode="FAST"
2. Return relevant code snippets quickly

### Example 4: Knowledge Cluster Management

**Prompt:**
```
List all knowledge clusters saved by sirchmunk
```

**Claude will:**
1. Invoke `sirchmunk_list_clusters`
2. Display saved knowledge clusters with metadata

**Prompt:**
```
Show me details of cluster C1007
```

**Claude will:**
1. Invoke `sirchmunk_get_cluster` with cluster_id="C1007"
2. Display full cluster information

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | (required) | Your LLM API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `LLM_MODEL_NAME` | `gpt-5.2` | Model to use |
| `SIRCHMUNK_WORK_PATH` | `~/.sirchmunk` | Working directory |
| `SIRCHMUNK_ENABLE_CLUSTER_REUSE` | `true` | Enable knowledge reuse |
| `MCP_LOG_LEVEL` | `INFO` | Logging level |
| `DEFAULT_MAX_DEPTH` | `5` | Directory search depth |
| `DEFAULT_TOP_K_FILES` | `3` | Files to return |

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
    "LLM_MODEL_NAME": "gpt-4"
  }
}
```

## Troubleshooting

### "Command not found: sirchmunk-mcp"

```bash
# Check if installed
pip show sirchmunk-mcp

# Use full path in config
which sirchmunk-mcp
```

### "LLM API key cannot be empty"

Ensure `LLM_API_KEY` is set in the `env` section of your MCP config.

### Claude doesn't show Sirchmunk tools

1. Verify JSON syntax: `python -m json.tool mcp.json`
2. Check file location is correct for your client
3. Restart the client completely
4. Check client logs for MCP errors

### Slow initialization

First-time startup may download embedding models. Subsequent starts are faster.

To skip embedding (disable cluster reuse):
```json
{
  "env": {
    "SIRCHMUNK_ENABLE_CLUSTER_REUSE": "false"
  }
}
```

### Debug Mode

Enable debug logging:
```json
{
  "env": {
    "MCP_LOG_LEVEL": "DEBUG",
    "SIRCHMUNK_VERBOSE": "true"
  }
}
```

## Performance Tips

1. **Use specific search paths** - Narrow paths = faster searches
2. **Choose appropriate mode**:
   - `FILENAME_ONLY`: <1s - for file discovery
   - `FAST`: 3-8s - for quick content search
   - `DEEP`: 10-30s - for comprehensive analysis
3. **Leverage cluster reuse** - Similar queries reuse cached knowledge
4. **Adjust depth** - Lower `max_depth` for faster results

## Security Notes

1. **Never commit API keys** - Use environment variables or secure config
2. **Review search paths** - Only include directories you trust
3. **Monitor API usage** - DEEP mode consumes more tokens

## Getting Help

- **Logs**: Check `~/.sirchmunk/logs/` for detailed errors
- **Issues**: https://github.com/modelscope/sirchmunk/issues
- **Documentation**: See [README.md](../README.md)

---

Happy coding with Sirchmunk MCP! 🚀
