#!/bin/bash
set -e

WORK_PATH="${SIRCHMUNK_WORK_PATH:-/data/sirchmunk}"

# Ensure directory structure (same as `sirchmunk init`)
mkdir -p "$WORK_PATH"/{data,logs,.cache/models,.cache/knowledge,.cache/history,.cache/settings,.cache/web_static}

# Copy pre-built frontend assets if not already present
if [ -d /app/web_static ] && [ ! -f "$WORK_PATH/.cache/web_static/index.html" ]; then
    echo "[entrypoint] Copying pre-built WebUI assets..."
    cp -r /app/web_static/* "$WORK_PATH/.cache/web_static/"
fi

# Generate default .env if not present
if [ ! -f "$WORK_PATH/.env" ]; then
    echo "[entrypoint] Generating default .env from template..."
    cp /app/config/env.example "$WORK_PATH/.env"
fi

# Generate MCP client config if not present (same as `sirchmunk init`)
if [ ! -f "$WORK_PATH/mcp_config.json" ]; then
    cat > "$WORK_PATH/mcp_config.json" <<'MCPEOF'
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk",
      "args": ["mcp", "serve"],
      "env": {
        "SIRCHMUNK_SEARCH_PATHS": ""
      }
    }
  }
}
MCPEOF
    echo "[entrypoint] Generated MCP client config: $WORK_PATH/mcp_config.json"
fi

# Apply environment variable overrides into .env
# Uses Python for safe replacement — avoids sed delimiter issues with
# special characters (/, &, |, \) that may appear in URLs or API keys.
for var in LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME LLM_TIMEOUT \
           UI_THEME UI_LANGUAGE SIRCHMUNK_VERBOSE; do
    val="${!var}"
    if [ -n "$val" ]; then
        python3 -c "
import sys, pathlib
key, val, p = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
lines = p.read_text().splitlines(True) if p.exists() else []
found = False
out = []
for line in lines:
    if line.startswith(key + '='):
        out.append(key + '=' + val + '\n')
        found = True
    else:
        out.append(line)
if not found:
    out.append(key + '=' + val + '\n')
p.write_text(''.join(out))
" "$var" "$val" "$WORK_PATH/.env"
    fi
done

echo "[entrypoint] Starting: $*"
exec "$@"
