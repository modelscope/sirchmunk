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

# Copy pre-downloaded embedding model if not already present
if [ -d /app/models ] && [ -z "$(ls -A "$WORK_PATH/.cache/models/" 2>/dev/null)" ]; then
    echo "[entrypoint] Copying pre-downloaded embedding model..."
    cp -r /app/models/* "$WORK_PATH/.cache/models/"
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
for var in LLM_BASE_URL LLM_API_KEY LLM_MODEL_NAME LLM_TIMEOUT \
           UI_THEME UI_LANGUAGE SIRCHMUNK_VERBOSE; do
    val="${!var}"
    if [ -n "$val" ]; then
        if grep -q "^${var}=" "$WORK_PATH/.env" 2>/dev/null; then
            sed -i "s|^${var}=.*|${var}=${val}|" "$WORK_PATH/.env"
        else
            echo "${var}=${val}" >> "$WORK_PATH/.env"
        fi
    fi
done

echo "[entrypoint] Starting: $*"
exec "$@"
