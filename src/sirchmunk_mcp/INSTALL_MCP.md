# Sirchmunk MCP Server - Installation and Deployment Guide

This guide provides detailed instructions for installing and deploying the Sirchmunk MCP Server.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Integration with Claude Desktop](#integration-with-claude-desktop)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Updating](#updating)
8. [Uninstallation](#uninstallation)

---

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: At least 2GB RAM recommended
- **Disk Space**: ~500MB for full installation with embedding models

### Required Software

1. **Python and pip**
   ```bash
   python --version  # Should be 3.9+
   pip --version
   ```

2. **ripgrep-all (rga)** - Will be installed automatically, or manually:
   
   **macOS:**
   ```bash
   brew install rga
   ```
   
   **Linux:**
   ```bash
   # Download from GitHub releases
   curl -L https://github.com/phiresky/ripgrep-all/releases/download/v0.10.6/ripgrep_all-v0.10.6-x86_64-unknown-linux-musl.tar.gz | tar xz
   sudo mv ripgrep_all-*/rga /usr/local/bin/
   ```
   
   **Windows:**
   Download from [ripgrep-all releases](https://github.com/phiresky/ripgrep-all/releases)

3. **LLM API Key** (required)
   - OpenAI API key, or
   - Compatible API endpoint (e.g., Azure OpenAI, local LLM server)

---

## Installation Methods

### Method 1: Install from PyPI (Recommended)

```bash
# Basic installation
pip install sirchmunk-mcp

# Or install with all optional dependencies (embedding support)
pip install sirchmunk-mcp[full]
```

### Method 2: Install from Source

```bash
# Clone repository
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk

# Install in development mode
pip install -e .

# Or with all dependencies
pip install -e ".[full,dev]"
```

### Method 3: Install in Virtual Environment (Recommended for isolation)

```bash
# Create virtual environment
python -m venv sirchmunk-env

# Activate (macOS/Linux)
source sirchmunk-env/bin/activate

# Activate (Windows)
sirchmunk-env\Scripts\activate

# Install
pip install sirchmunk-mcp[full]
```

---

## Configuration

### Step 1: Initialize Sirchmunk

```bash
sirchmunk-mcp init
```

This will:
- Create working directory (`~/.sirchmunk`)
- Check dependencies
- Verify installation

### Step 2: Generate Configuration Templates

```bash
sirchmunk-mcp config --generate
```

This creates:
- `.env.example` - Environment variable template
- `mcp_config.json` - MCP client configuration template

### Step 3: Configure Environment Variables

Copy and edit the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Required
LLM_API_KEY=sk-your-actual-api-key-here

# Optional (customize as needed)
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4-turbo-preview
SIRCHMUNK_WORK_PATH=~/.sirchmunk
MCP_LOG_LEVEL=INFO
```

**Important:** Never commit `.env` with real API keys to version control!

### Step 4: Load Environment (if using standalone)

```bash
# Load environment variables
source .env  # macOS/Linux
# or
set -a; . .env; set +a  # Alternative

# Verify
echo $LLM_API_KEY
```

---

## Integration with Claude Desktop

### Option 1: Automatic Installation

Edit the generated `mcp_config.json` with your API key, then:

```bash
# macOS
cp mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
cp mcp_config.json ~/.config/Claude/claude_desktop_config.json

# Windows (PowerShell)
Copy-Item mcp_config.json "$env:APPDATA\Claude\claude_desktop_config.json"
```

### Option 2: Manual Configuration

Create or edit Claude Desktop config file at:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add Sirchmunk server configuration:

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk-mcp",
      "args": ["serve"],
      "env": {
        "LLM_API_KEY": "sk-your-api-key",
        "LLM_MODEL_NAME": "gpt-4-turbo-preview",
        "SIRCHMUNK_WORK_PATH": "~/.sirchmunk",
        "MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Note:** If you're using a virtual environment, use the full path to the executable:

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

### Step 3: Restart Claude Desktop

Completely quit and restart Claude Desktop for changes to take effect.

---

## Verification

### Check Server Configuration

```bash
sirchmunk-mcp config
```

Expected output:
```
Current Configuration:
LLM:
  Base URL: https://api.openai.com/v1
  Model: gpt-4-turbo-preview
  API Key: sk-xxx...

Sirchmunk:
  Work Path: /Users/you/.sirchmunk
  Verbose: false
  Cluster Reuse: true
...
```

### Test Standalone Server

```bash
# Run server in test mode
MCP_LOG_LEVEL=DEBUG sirchmunk-mcp serve
```

You should see:
```
INFO - Sirchmunk MCP Server v0.1.0
INFO - Transport: stdio
INFO - MCP server listening on stdio
```

Press `Ctrl+C` to stop.

### Test in Claude Desktop

1. Open Claude Desktop
2. Start a new conversation
3. Try a search command:
   ```
   Search for "authentication" in /path/to/your/project using sirchmunk
   ```

4. Claude should invoke the MCP tool and return results

---

## Troubleshooting

### Problem: "Command not found: sirchmunk-mcp"

**Solution:**
```bash
# Verify installation
pip show sirchmunk-mcp

# Check if bin directory is in PATH
echo $PATH

# Reinstall
pip install --force-reinstall sirchmunk-mcp
```

### Problem: "LLM API key cannot be empty"

**Solution:**
- Verify `LLM_API_KEY` is set in environment or config
- Check for typos in environment variable name
- Ensure API key is valid

```bash
# Test API key
export LLM_API_KEY="sk-your-key"
sirchmunk-mcp config
```

### Problem: "ripgrep-all not found"

**Solution:**
```bash
# Check if rga is installed
which rga

# If not, install:
sirchmunk-mcp init  # Will attempt to install

# Or install manually (see Prerequisites)
```

### Problem: Claude Desktop doesn't show Sirchmunk tools

**Solutions:**

1. **Verify config file location:**
   ```bash
   # macOS
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
   
   # Linux
   cat ~/.config/Claude/claude_desktop_config.json
   ```

2. **Check JSON syntax:**
   ```bash
   # Use a JSON validator
   python -m json.tool claude_desktop_config.json
   ```

3. **Check Claude Desktop logs:**
   - macOS: `~/Library/Logs/Claude/`
   - Linux: `~/.config/Claude/logs/`
   - Windows: `%APPDATA%\Claude\logs\`

4. **Restart Claude Desktop completely:**
   - Quit application (not just close window)
   - Verify process is not running
   - Restart

### Problem: "ModuleNotFoundError: No module named 'mcp'"

**Solution:**
```bash
pip install mcp>=0.9.0
```

### Problem: Slow initialization or embedding errors

**Solution:**

If embedding models take too long to download:

```bash
# Disable cluster reuse temporarily
export SIRCHMUNK_ENABLE_CLUSTER_REUSE=false

# Or install embedding dependencies separately
pip install sentence-transformers modelscope
```

---

## Updating

### Update from PyPI

```bash
pip install --upgrade sirchmunk-mcp
```

### Update from Source

```bash
cd sirchmunk
git pull
pip install -e . --upgrade
```

### After Update

```bash
# Re-initialize to check dependencies
sirchmunk-mcp init

# Restart Claude Desktop if integrated
```

---

## Uninstallation

### Remove Package

```bash
pip uninstall sirchmunk-mcp
```

### Remove Data (Optional)

```bash
# Remove work directory (contains cached data and knowledge clusters)
rm -rf ~/.sirchmunk

# Remove Claude Desktop config (if you want)
# macOS
rm ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Linux
rm ~/.config/Claude/claude_desktop_config.json
```

---

## Advanced Configuration

### Using Custom LLM Endpoint

```bash
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="local-key"
export LLM_MODEL_NAME="your-model"
```

### Using Azure OpenAI

```bash
export LLM_BASE_URL="https://your-resource.openai.azure.com/"
export LLM_API_KEY="your-azure-key"
export LLM_MODEL_NAME="gpt-4"
```

### Customizing Search Behavior

```bash
# Adjust search defaults
export DEFAULT_MAX_DEPTH=10
export DEFAULT_TOP_K_FILES=5
export DEFAULT_KEYWORD_LEVELS=4

# Adjust cluster reuse behavior
export CLUSTER_SIM_THRESHOLD=0.90
export CLUSTER_SIM_TOP_K=5
```

### Multiple Configurations

Create different environment files:

```bash
# Development
cp config/env.example .env.dev

# Production
cp config/env.example .env.prod

# Load specific config
source .env.dev
sirchmunk-mcp serve
```

---

## Security Best Practices

1. **Never commit API keys**
   - Add `.env` to `.gitignore`
   - Use environment variables or secret managers

2. **Restrict file access**
   - Configure search paths carefully
   - Use include/exclude patterns

3. **Monitor usage**
   - Check LLM API usage regularly
   - Set spending limits in your LLM provider dashboard

4. **Update regularly**
   - Keep sirchmunk-mcp updated
   - Update dependencies periodically

---

## Getting Help

- **Documentation**: [README_MCP.md](README_MCP.md)
- **GitHub Issues**: https://github.com/modelscope/sirchmunk/issues
- **Logs**: Check `~/.sirchmunk/logs/` for detailed error messages
- **Verbose Mode**: Run with `SIRCHMUNK_VERBOSE=true` for debugging

---

## Next Steps

After successful installation:

1. Read the [README_MCP.md](README_MCP.md) for usage examples
2. Explore available tools and their parameters
3. Try different search modes (DEEP/FAST/FILENAME_ONLY)
4. Experiment with knowledge cluster management
5. Customize configuration for your workflow

Happy searching! 🔍
