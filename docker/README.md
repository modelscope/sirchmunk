# Sirchmunk Docker Images

## Available Images

| Region | Image |
|---|---|
| US West | `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2` |
| China Beijing | `modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2` |
| China Hangzhou | `modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2` |

Image tag format: `ubuntu{ubuntu_version}-py{python_version}-{sirchmunk_version}`

## Quick Start

### 1. Pull the image

Choose the registry closest to your location:

```bash
# US West
docker pull modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2

# China Beijing
docker pull modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2

# China Hangzhou
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
```

### 2. Start the service

```bash
docker run -d \
  --name sirchmunk \
  -p 8584:8584 \
  -e LLM_API_KEY="your-api-key-here" \
  -e LLM_BASE_URL="https://api.openai.com/v1" \
  -e LLM_MODEL_NAME="gpt-4o" \
  -v sirchmunk_data:/data/sirchmunk \
  modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
```

**Required parameters:**

| Parameter | Description |
|---|---|
| `-e LLM_API_KEY` | **(Required)** API key from your LLM provider |
| `-e LLM_BASE_URL` | OpenAI-compatible API endpoint (default: `https://api.openai.com/v1`) |
| `-e LLM_MODEL_NAME` | Model name (default: `gpt-5.2`) |
| `-p 8584:8584` | Expose WebUI and API port |
| `-v sirchmunk_data:/data/sirchmunk` | Persist data across container restarts |

**Optional — mount local files for search:**

```bash
docker run -d \
  --name sirchmunk \
  -p 8584:8584 \
  -e LLM_API_KEY="your-api-key-here" \
  -e LLM_BASE_URL="https://api.openai.com/v1" \
  -e LLM_MODEL_NAME="gpt-4o" \
  -v sirchmunk_data:/data/sirchmunk \
  -v /path/to/your/docs:/mnt/docs:ro \
  modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
```

### 3. Use the service

**WebUI** — Open http://localhost:8584 in your browser.

**API — Search via curl:**

```bash
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search question here",
    "paths": ["/mnt/docs"],
    "mode": "DEEP"
  }'
```

| Field | Type | Description |
|---|---|---|
| `query` | string | Search query or question |
| `paths` | list | Directories or files to search (e.g., `["/mnt/docs"]` for mounted volumes) |
| `mode` | string | `"DEEP"` for comprehensive analysis, `"FILENAME_ONLY"` for fast file discovery |

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

# Remove data volume
docker volume rm sirchmunk_data
```
