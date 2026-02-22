# Sirchmunk Docker Deployment

## Architecture

```
docker/
├── Dockerfile.ubuntu    # Dockerfile template with {placeholder} tokens
├── build_image.py       # Build script (renders template → builds image → multi-registry push)
├── entrypoint.sh        # Container entrypoint (dir setup, .env init, model copy)
├── docker-compose.yml   # Compose file for one-command deployment
├── .env.example         # Template for Compose environment variables
└── README.md
```

The build follows the [modelscope docker pattern](https://github.com/modelscope/modelscope/tree/master/docker):

1. `Dockerfile.ubuntu` is a **template** with placeholders like `{python_image}`, `{rg_version}`, etc.
2. `build_image.py` reads the template, substitutes all placeholders, writes a final `Dockerfile` at the project root, builds the image, and pushes to Alibaba Cloud ACR registries.
3. The GitHub Actions workflow (`.github/workflows/docker-image.yaml`) runs on a **self-hosted** runner (US-West Alibaba Cloud ECS) and calls `build_image.py --push`.

## Image Tag Convention

Follows the ModelScope naming pattern:

```
ubuntu{ubuntu_version}-py{python_tag}-{sirchmunk_version}
```

Example (with version `0.0.2`, Python 3.12):
```
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
```

The `sirchmunk_version` defaults to the value in `src/sirchmunk/version.py` and can be overridden via `--sirchmunk_version`.

## Quick Start

```bash
# From the project root directory:

# 1. Create your .env file
cp config/env.example docker/.env
# Edit docker/.env and set LLM_API_KEY

# 2. Build the image (local only, no push)
python docker/build_image.py

# 3. Start with docker compose
docker compose -f docker/docker-compose.yml up -d

# 4. Open WebUI
open http://localhost:8584
```

## Build Script Usage

```bash
# Build CPU image locally (no push)
python docker/build_image.py

# Build and push to all ACR registries
python docker/build_image.py --push

# Dry-run: only generate Dockerfile, skip docker build
python docker/build_image.py --dry_run 1

# Override version (instead of auto-detecting from version.py)
python docker/build_image.py --push --sirchmunk_version 0.0.4

# Custom Python / Node / Ubuntu versions
python docker/build_image.py --python_version 3.13 --node_version 22 --ubuntu_version 24.04

# Push to custom registries
python docker/build_image.py --push --registries "my-registry.example.com/ns/sirchmunk"

# Full options
python docker/build_image.py --help
```

### Default Push Registries

When `--push` is used without `--registries`, images are pushed to:

| Region | Registry |
|---|---|
| cn-beijing | `modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk` |
| cn-hangzhou | `modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk` |
| us-west-1 | `modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk` |

### China Mainland Mirror Acceleration (中国大陆镜像加速)

Docker Hub、GitHub Releases、PyPI、npm 在中国大陆可能无法直接访问。使用 `--mirror cn` 自动切换为国内镜像源：

```bash
python docker/build_image.py --mirror cn
```

该选项会自动配置：

| 服务 | 镜像源 |
|---|---|
| Docker 基础镜像 (python/node) | `docker.m.daocloud.io` (DaoCloud) |
| PyPI (pip install) | `mirrors.aliyun.com` (阿里云) |
| npm (node packages) | `registry.npmmirror.com` (淘宝) |
| GitHub Releases (rg/rga) | `ghfast.top` (GitHub 加速) |

> **注意**: 镜像源由第三方维护，可能存在不稳定的情况。如果某个源不可用，可以修改 `build_image.py` 中的 `MIRROR_PROFILES["cn"]` 字典替换为其他可用源。

### Builder Classes

| Class | `--image_type` | Description |
|---|---|---|
| `CPUImageBuilder` | `cpu` | CPU-only image with Python + Node.js frontend (default) |

## Configuration

### Environment Variables

Set these in `docker/.env` or pass them directly:

| Variable | Required | Default | Description |
|---|---|---|---|
| `LLM_API_KEY` | Yes | | API key for your LLM provider |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | OpenAI-compatible API endpoint |
| `LLM_MODEL_NAME` | No | `gpt-5.2` | Model name |

### Mounting Local Files for Search

To search local files/directories, mount them as read-only volumes in `docker-compose.yml`:

```yaml
services:
  sirchmunk:
    volumes:
      - sirchmunk_data:/data/sirchmunk
      - /path/to/your/docs:/mnt/docs:ro
      - /path/to/your/code:/mnt/code:ro
```

Then use `/mnt/docs` or `/mnt/code` as the search path in the WebUI.

### Data Persistence

All data is stored in the `sirchmunk_data` named volume:

```
/data/sirchmunk/
├── .env                  # Configuration
├── mcp_config.json       # MCP client config
├── .cache/
│   ├── models/           # Embedding models (pre-downloaded in image)
│   ├── knowledge/        # Knowledge clusters
│   ├── history/          # Chat history
│   ├── settings/         # Settings cache
│   └── web_static/       # WebUI assets
├── data/
└── logs/
```

### What's included in the Docker image

The Docker build reproduces all steps of `sirchmunk web init`:

- **All Python dependencies**: `pip install ".[all]"` (core, web, mcp, docs, tests)
- **Pre-built WebUI frontend**: Node.js multi-stage build
- **Pre-downloaded embedding model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via ModelScope, stored at `/app/models` and copied to the work volume on first start
- **System tools**: `ripgrep` (rg) and `ripgrep-all` (rga)
- **MCP client config**: Generated at first container start

## Commands

```bash
# Start
docker compose -f docker/docker-compose.yml up -d

# View logs
docker compose -f docker/docker-compose.yml logs -f

# Stop
docker compose -f docker/docker-compose.yml down

# Rebuild after code changes
python docker/build_image.py
docker compose -f docker/docker-compose.yml up -d

# Reset all data
docker compose -f docker/docker-compose.yml down -v
```

## CI / GitHub Actions

The workflow at `.github/workflows/docker-image.yaml` runs on a **self-hosted** runner (Alibaba Cloud ECS, us-west-1) and can be triggered manually via **Actions → Build Docker Image → Run workflow**.

### Required Secrets

| Secret | Description |
|---|---|
| `ACR_USERNAME` | Alibaba Cloud ACR username |
| `ACR_PASSWORD` | Alibaba Cloud ACR password |

### Workflow Inputs

| Input | Default | Description |
|---|---|---|
| `sirchmunk_branch` | `main` | Branch to build from |
| `image_type` | `cpu` | Image type (`cpu`) |
| `sirchmunk_version` | *(auto from version.py)* | Version label for the image tag |
| `ubuntu_version` | `22.04` | Ubuntu version label |
| `python_version` | `3.12` | Python base image version |
| `node_version` | `20` | Node.js base image version |
| `mirror` | *(empty)* | Mirror profile (`cn` for China mainland) |

### Example Output

Triggering the workflow with default inputs and `version.py` containing `0.0.2` will produce:

```
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/sirchmunk:ubuntu22.04-py312-0.0.2
```

Each image is also tagged with a timestamp for traceability (e.g., `ubuntu22.04-py312-0.0.2-20260212153045`).
