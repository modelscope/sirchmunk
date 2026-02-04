# Sirchmunk MCP Examples

This directory contains configuration examples for integrating Sirchmunk MCP with various Claude Code clients.

## Files

| File | Description |
|------|-------------|
| `claude_code_setup.md` | Complete setup guide for Claude Code integration |
| `cursor_mcp_config.json` | Basic configuration for Cursor IDE |
| `cursor_mcp_config_venv.json` | Configuration for virtual environment installation |
| `cursor_mcp_config_local_llm.json` | Configuration for local LLM (Ollama, etc.) |

## Quick Start

### For Cursor IDE

1. Copy the appropriate config to `~/.cursor/mcp.json`:

   ```bash
   # Basic setup
   cp cursor_mcp_config.json ~/.cursor/mcp.json
   
   # Or for virtual environment
   cp cursor_mcp_config_venv.json ~/.cursor/mcp.json
   ```

2. Edit the file to add your API key

3. Restart Cursor

### For Claude Desktop

1. Copy config to Claude Desktop config directory:

   ```bash
   # macOS
   cp cursor_mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   
   # Linux
   cp cursor_mcp_config.json ~/.config/Claude/claude_desktop_config.json
   ```

2. Edit the file to add your API key

3. Restart Claude Desktop

## Configuration Notes

### Required Settings

- `LLM_API_KEY`: Your OpenAI API key (or compatible LLM provider)

### Optional Settings

- `LLM_BASE_URL`: API endpoint (default: OpenAI)
- `LLM_MODEL_NAME`: Model to use (default: gpt-5.2)
- `SIRCHMUNK_WORK_PATH`: Working directory (default: ~/.sirchmunk)
- `MCP_LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)

### Using Virtual Environment

If you installed sirchmunk-mcp in a virtual environment, use the full path to the executable:

```bash
# Find your executable path
which sirchmunk-mcp
# Example: /Users/you/.sirchmunk-env/bin/sirchmunk-mcp
```

Then update the `command` field in your config accordingly.

## More Information

See the complete [Claude Code Setup Guide](claude_code_setup.md) for detailed instructions.
