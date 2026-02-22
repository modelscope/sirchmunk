# Sirchmunk Docker Deployment

## Architecture

```
docker/
├── Dockerfile.ubuntu    # Dockerfile template with {placeholder} tokens
├── build_image.py       # Build script (renders template → builds image)
├── entrypoint.sh        # Container entrypoint (dir setup, .env init)
├── docker-compose.yml   # Compose file for one-command deployment
├── .env.example         # Template for Compose environment variables
└── README.md
```

The build follows the [modelscope docker pattern](https://github.com/modelscope/modelscope/tree/master/docker):

1. `Dockerfile.ubuntu` is a **template** with placeholders like `{python_image}`, `{rg_version}`, etc.
2. `build_image.py` reads the template, substitutes all placeholders with concrete values, writes a final `Dockerfile` at the project root, and runs `docker build`.
3. The GitHub Actions workflow (`.github/workflows/docker-image.yaml`) calls `build_image.py` for CI builds.

## Quick Start

```bash
# From the project root directory:

# 1. Create your .env file
cp config/.env.example docker/.env
# Edit docker/.env and set LLM_API_KEY

# 2. Build the image
python docker/build_image.py

# 3. Start with docker compose
docker compose -f docker/docker-compose.yml up -d

# 4. Open WebUI
open http://localhost:8584
```

## Build Script Usage

```bash
# Build CPU image (default)
python docker/build_image.py --image_type cpu

# Dry-run: only generate Dockerfile, skip docker build
python docker/build_image.py --dry_run 1

# Custom Python / Node versions
python docker/build_image.py --python_version 3.13 --node_version 22

# Build and push to a registry
DOCKER_REGISTRY=ghcr.io/modelscope/sirchmunk \
    python docker/build_image.py --sirchmunk_version 0.0.2

# Full options
python docker/build_image.py --help
```

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

The workflow at `.github/workflows/docker-image.yaml` can be triggered manually
via **Actions → Build Docker Image → Run workflow**:

| Input | Default | Description |
|---|---|---|
| `sirchmunk_branch` | `main` | Branch to build from |
| `image_type` | `cpu` | Image type (`cpu`) |
| `sirchmunk_version` | `latest` | Version label for the image tag |
| `python_version` | `3.12` | Python base image version |
| `node_version` | `20` | Node.js base image version |

The built image is pushed to `ghcr.io/<owner>/sirchmunk`.
