# Sirchmunk MCP - Quick Start Guide

Get started with Sirchmunk MCP in 5 minutes!

## 🚀 Fast Track Installation

### Step 1: Install (1 minute)

```bash
pip install sirchmunk-mcp[full]
```

### Step 2: Initialize (30 seconds)

```bash
sirchmunk-mcp init
sirchmunk-mcp config --generate
```

### Step 3: Configure (2 minutes)

Edit `.env.example` and set your API key:

```bash
# Copy template
cp .env.example .env

# Edit with your favorite editor
vim .env  # or nano, code, etc.
```

**Minimum required:**
```bash
LLM_API_KEY=sk-your-actual-api-key
```

### Step 4: Test (30 seconds)

```bash
# Quick test
MCP_LOG_LEVEL=INFO LLM_API_KEY=sk-your-key sirchmunk-mcp serve
```

You should see:
```
INFO - Sirchmunk MCP Server v0.1.0
INFO - Transport: stdio
INFO - MCP server listening on stdio
```

Press Ctrl+C to stop.

---

## 🖥️ Claude Desktop Integration (2 minutes)

### Step 1: Edit Config

```bash
vim mcp_config.json
```

Change `sk-your-api-key-here` to your actual API key.

### Step 2: Install Config

**macOS:**
```bash
cp mcp_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Linux:**
```bash
mkdir -p ~/.config/Claude
cp mcp_config.json ~/.config/Claude/claude_desktop_config.json
```

**Windows (PowerShell):**
```powershell
Copy-Item mcp_config.json "$env:APPDATA\Claude\claude_desktop_config.json"
```

### Step 3: Restart Claude Desktop

Completely quit and restart Claude Desktop.

---

## 🎯 First Search

In Claude Desktop, try:

```
Search for "main function" in /path/to/your/project using sirchmunk in DEEP mode
```

Claude will use the Sirchmunk MCP tool to perform an intelligent search!

---

## 📚 What's Next?

- [Full Documentation](README) - Complete feature guide
- [Installation Guide](INSTALL) - Detailed setup instructions
- [Configuration Options](config/env.example) - All available settings

---

## 🆘 Quick Troubleshooting

### "Command not found: sirchmunk-mcp"

```bash
pip install --force-reinstall sirchmunk-mcp
```

### "LLM API key cannot be empty"

Check your environment variable:
```bash
echo $LLM_API_KEY
```

Set it if empty:
```bash
export LLM_API_KEY="sk-your-key"
```

### Claude Desktop not showing tools

1. Verify config file exists and has correct API key
2. Check JSON syntax: `python -m json.tool claude_desktop_config.json`
3. Completely restart Claude Desktop (quit, not just close)
4. Check Claude Desktop logs in `~/Library/Logs/Claude/`

### Need more help?

- Enable debug logging: `MCP_LOG_LEVEL=DEBUG`
- Check logs in `~/.sirchmunk/logs/`
- Open issue: https://github.com/modelscope/sirchmunk/issues

---

## 🎓 Example Usage

### Filename Search (fastest)

```
Find all test files in /my/project
```

Claude invokes:
```json
{
  "query": "test",
  "search_paths": ["/my/project"],
  "mode": "FILENAME_ONLY"
}
```

### Quick Content Search

```
Search for authentication code in /my/project using FAST mode
```

### Deep Analysis

```
Explain how the authentication system works in /my/project
```

(Uses DEEP mode by default)

---

## 💡 Pro Tips

1. **Use specific paths** - More specific search paths = faster results
2. **Try different modes** - FILENAME_ONLY for files, FAST for quick content, DEEP for analysis
3. **Leverage cluster reuse** - Similar queries will reuse previous results
4. **Check stats** - `sirchmunk-mcp config` shows current settings
5. **Monitor costs** - DEEP mode uses more LLM tokens

---

## 🔧 Configuration Presets

### Minimal (fastest, lowest cost)
```bash
SIRCHMUNK_ENABLE_CLUSTER_REUSE=false
DEFAULT_KEYWORD_LEVELS=1
DEFAULT_TOP_K_FILES=1
```

### Balanced (recommended)
```bash
SIRCHMUNK_ENABLE_CLUSTER_REUSE=true
DEFAULT_KEYWORD_LEVELS=3
DEFAULT_TOP_K_FILES=3
```

### Maximum (best results, higher cost)
```bash
SIRCHMUNK_ENABLE_CLUSTER_REUSE=true
DEFAULT_KEYWORD_LEVELS=5
DEFAULT_TOP_K_FILES=5
CLUSTER_SIM_THRESHOLD=0.90
```

---

That's it! You're ready to use Sirchmunk MCP 🎉
