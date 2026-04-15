# Sirchmunk Docker Images

## Available Images

| Region | Image                                                                                              |
|---|----------------------------------------------------------------------------------------------------|
| US West | `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.7`  |
| China Beijing | `modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.7` |

Image tag format: `ubuntu{ubuntu_version}-py{python_version}-{sirchmunk_version}`

## Quick Start

### 1. Pull the image

Choose the registry closest to your location:

```bash
# US West
docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.7

# China Beijing
docker pull modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.7
```

### 2. Start the service

```bash
docker run -d \
  --name sirchmunk \
  -p 8584:8584 \
  -e LLM_API_KEY="your-api-key-here" \
  -e LLM_BASE_URL="https://api.openai.com/v1" \
  -e LLM_MODEL_NAME="gpt-5.2" \
  -e LLM_TIMEOUT=60.0 \
  -e UI_THEME=light \
  -e UI_LANGUAGE=en \
  -e SIRCHMUNK_VERBOSE=false \
  -e SIRCHMUNK_SEARCH_PATHS=/mnt/docs \
  -v /path/to/your_work_path:/data/sirchmunk \
  -v /path/to/your/docs:/mnt/docs:ro \
  modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.7
```

**Parameters:**

| Parameter | Required | Default | Description |
|---|---|---|---|
| `-e LLM_API_KEY` | **Yes** | | API key from your LLM provider |
| `-e LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API endpoint (e.g., `https://api.minimax.io/v1` for MiniMax) |
| `-e LLM_MODEL_NAME` | No | `gpt-5.2` | LLM model name (e.g., `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`) |
| `-e LLM_TIMEOUT` | No | `60.0` | LLM request timeout in seconds |
| `-e UI_THEME` | No | `light` | WebUI theme (`light` / `dark`) |
| `-e UI_LANGUAGE` | No | `en` | WebUI language (`en` / `zh`) |
| `-e SIRCHMUNK_VERBOSE` | No | `false` | Enable verbose logging (`true` / `false`) |
| `-e SIRCHMUNK_SEARCH_PATHS` | No | (empty) | Comma-separated default search paths (e.g. `/mnt/docs`) |
| `-e CHAT_HISTORY_MAX_TURNS` | No | `5` | Max chat history turns for multi-turn context |
| `-e CHAT_HISTORY_MAX_TOKENS` | No | `32000` | Max tokens for chat history context |
| `-p 8584:8584` | Yes | | Expose WebUI and API port |
| `-v /data/sirchmunk:/data/sirchmunk` | Recommended | | Persist data (models, history, knowledge) across restarts |


**Mount local files for search:**

Use `-v` to mount host directories into the container, then search them via the API or WebUI.


### 3. Use the service

**WebUI** — Open http://localhost:8584 in your browser.

**API — Search via curl:**

```bash
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search question here",
    "paths": ["/mnt/docs"],
    "mode": "FAST"
  }'
```

**API — Search via Python:**

```python
import requests

response = requests.post(
    "http://localhost:8584/api/v1/search",
    json={
        "query": "your search question here",
        "paths": ["/mnt/docs"],
        "mode": "FAST",           # "FAST" (default), "DEEP", or "FILENAME_ONLY"
        "enable_dir_scan": True,  # enable directory scanning (DEEP/FAST)
        "max_depth": 5,           # optional: max directory depth
        "top_k_files": 3,         # optional: number of top files to return
        "max_token_budget": 8192, # optional: LLM token budget (DEEP)
    },
)

result = response.json()
if result["success"]:
    print(result["data"])
else:
    print(f"Error: {result['error']}")
```

**Search API fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Search query or question |
| `paths` | list | No | Directories or files to search (e.g., `["/mnt/docs"]`); falls back to `SIRCHMUNK_SEARCH_PATHS` if unset |
| `mode` | string | No | `"FAST"` (default, greedy search 2-5s), `"DEEP"` (comprehensive analysis 10-30s), or `"FILENAME_ONLY"` (file discovery <1s) |
| `enable_dir_scan` | bool | No | Enable directory scanning for file discovery in DEEP/FAST (default: true) |
| `max_depth` | int | No | Maximum directory depth to search |
| `top_k_files` | int | No | Number of top files to return |
| `max_token_budget` | int | No | LLM token budget (DEEP mode) |
| `return_context` | bool | No | Return full SearchContext with KnowledgeCluster and telemetry (default: false) |

### 4. Manage the container

```bash
# View logs
docker logs -f sirchmunk

# Stop
docker stop sirchmunk

# Restart
docker start sirchmunk

# Remove container (data is preserved in the volume)
docker rm sirchmunk

# Remove data volume (caution: deletes all persisted data)
rm -rf /path/to/your_work_path
```
