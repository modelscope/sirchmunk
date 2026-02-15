<div align="center">

<img src="web/public/logo-v2.png" alt="Sirchmunk Logo" width="250" style="border-radius: 15px;">

# Sirchmunk: Raw data to self-evolving intelligence, real-time. 

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![ripgrep-all](https://img.shields.io/badge/ripgrep--all-Search-E67E22?style=flat-square&logo=rust&logoColor=white)](https://github.com/phiresky/ripgrep-all)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white)](https://github.com/openai/openai-python)
[![Kreuzberg](https://img.shields.io/badge/Kreuzberg-Text_Extraction-4CAF50?style=flat-square)](https://github.com/kreuzberg-dev/kreuzberg)
[![MCP](https://img.shields.io/badge/MCP-Python_SDK-8B5CF6?style=flat-square&logo=python&logoColor=white)](https://github.com/modelcontextprotocol/python-sdk)
[![Documentation](https://img.shields.io/badge/Documentation-Official-4285F4?style=flat-square&logo=readthedocs&logoColor=white)](https://modelscope.github.io/sirchmunk-web/)


[**Quick Start**](#-quick-start) · [**Key Features**](#-key-features) · [**MCP Server**](#-mcp-server) · [**Web UI**](#️-web-ui) · [**How it Works**](#️-how-it-works) · [**FAQ**](#-faq)

</div>

<div align="center">

🔍 **Agentic Search** &nbsp;•&nbsp; 🧠 **Knowledge Clustering** &nbsp;•&nbsp; 📊 **Monte Carlo Evidence Sampling**<br>
⚡ **Indexless Retrieval** &nbsp;•&nbsp; 🔄 **Self-Evolving Knowledge Base** &nbsp;•&nbsp; 💬 **Real-time Chat**

</div>

<br>

[English](README.md) | [中文](README_zh.md)


---

## 🌰 Why “Sirchmunk”？

Intelligence pipelines built upon vector-based retrieval can be _rigid and brittle_. They rely on static vector embeddings that are **expensive to compute, blind to real-time changes, and detached from the raw context**. We introduce **Sirchmunk** to usher in a more agile paradigm, where data is no longer treated as a snapshot, and insights can evolve together with the data.

---

## ✨ Key Features

### 1. Embedding-Free: Data in its Purest Form

**Sirchmunk** works directly with **raw data** -- bypassing the heavy overhead of squeezing your rich files into fixed-dimensional vectors.

* **Instant Search:** Eliminating complex pre-processing pipelines in hours long indexing; just drop your files and search immediately.
* **Full Fidelity:** Zero information loss —- stay true to your data without vector approximation.

### 2. Self-Evolving: A Living Index

Data is a stream, not a snapshot.  **Sirchmunk** is **dynamic by design**, while vector DB can become obsolete the moment your data changes.

* **Context-Aware:** Evolves in real-time with your data context.
* **LLM-Powered Autonomy:** Designed for Agents that perceive data as it lives, utilizing **token-efficient** reasoning that triggers LLM inference only when necessary to maximize intelligence while minimizing cost.

### 3. Intelligence at Scale: Real-Time & Massive

**Sirchmunk** bridges massive local repositories and the web with **high-scale throughput** and **real-time awareness**. <br/>
It serves as a unified intelligent hub for AI agents, delivering deep insights across vast datasets at the speed of thought.

---

### Traditional RAG vs. Sirchmunk

<div style="display: flex; justify-content: center; width: 100%;">
  <table style="width: 100%; max-width: 900px; border-collapse: separate; border-spacing: 0; overflow: hidden; border-radius: 12px; font-family: sans-serif; border: 1px solid rgba(128, 128, 128, 0.2); margin: 0 auto;">
    <colgroup>
      <col style="width: 25%;">
      <col style="width: 30%;">
      <col style="width: 45%;">
    </colgroup>
    <thead>
      <tr style="background-color: rgba(128, 128, 128, 0.05);">
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(128, 128, 128, 0.2); font-size: 1.3em;">Dimension</th>
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(128, 128, 128, 0.2); font-size: 1.3em; opacity: 0.7;">Traditional RAG</th>
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(58, 134, 255, 0.5); color: #3a86ff; font-weight: 800; font-size: 1.3em;">✨Sirchmunk</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">💰 Setup Cost</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">High Overhead <br/> (VectorDB, GraphDB, Complex Document Parser...)</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ Zero Infrastructure <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">Direct-to-data retrieval without vector silos</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">🕒 Data Freshness</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">Stale (Batch Re-indexing)</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ Instant &amp; Dynamic <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">Self-evolving index that reflects live changes</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">📈 Scalability</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">Linear Cost Growth</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ Extremely low RAM/CPU consumption <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">Native Elastic Support, efficiently handles large-scale datasets</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">🎯 Accuracy</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">Approximate Vector Matches</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ Deterministic &amp; Contextual <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">Hybrid logic ensuring semantic precision</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600;">⚙️ Workflow</td>
        <td style="padding: 16px; opacity: 0.6;">Complex ETL Pipelines</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef;">
          ✅ Drop-and-Search <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">Zero-config integration for rapid deployment</small>
        </td>
      </tr>
    </tbody>
  </table>
</div>

---


## Demonstration


<div align="center">
  <img src="assets/gif/Sirchmunk_Web.gif" alt="Sirchmunk WebUI" width="100%">
  <p style="font-size: 1.1em; font-weight: 600; margin-top: 8px; color: #00bcd4;">
    Access files directly to start chatting
  </p>
</div>

---


|  WeChat Group
|:-------------------------:
|  <img src="assets/pic/wechat.jpg" width="200" height="200">

---


## 🎉 News

* 🚀 **Feb 12, 2026**: Sirchmunk v0.0.3: Upgraded MCP Integration & Core Search Algorithms
  - **MCP Boost**: Enhanced Model Context Protocol support with updated setup guides.
  - **Granular Search**: Added glob pattern (include/exclude) support; auto-filters temp/cache files.
  - **New Docs**: Deep dives into "Monte Carlo Evidence Sampling" and "Self-Evolving Knowledge Clusters."
  - **System Stability**: Refactored search pipeline and implemented SHA256 deterministic IDs for Knowledge Clusters.


* 🚀 **Feb 5, 2026**: Release **v0.0.2** — MCP Support, CLI Commands & Knowledge Persistence!
  - **MCP Integration**: Full [Model Context Protocol](https://modelcontextprotocol.io) support, works seamlessly with Claude Desktop and Cursor IDE.
  - **CLI Commands**: New `sirchmunk` CLI with `init`, `serve`, `search`, `web`, and `mcp` commands.
  - **KnowledgeCluster Persistence**: DuckDB-powered storage with Parquet export for efficient knowledge management.
  - **Knowledge Reuse**: Semantic similarity-based cluster retrieval for faster searches via embedding vectors.

* 🎉🎉 Jan 22, 2026: Introducing **Sirchmunk**: Initial Release v0.0.1 Now Available!


---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **LLM API Key** (OpenAI-compatible endpoint, local or remote)
- **Node.js** 18+ (Optional, for web interface)

### Installation

```bash
# Create virtual environment (recommended)
conda create -n sirchmunk python=3.13 -y && conda activate sirchmunk 

pip install sirchmunk

# Or via UV:
uv pip install sirchmunk

# Alternatively, install from source:
git clone https://github.com/modelscope/sirchmunk.git && cd sirchmunk
pip install -e .
```

### Python SDK Usage

```python
import asyncio

from sirchmunk import AgenticSearch
from sirchmunk.llm import OpenAIChat

llm = OpenAIChat(
        api_key="your-api-key",
        base_url="your-base-url",   # e.g., https://api.openai.com/v1
        model="your-model-name"     # e.g., gpt-4o
    )

async def main():
    
    searcher = AgenticSearch(llm=llm)
    
    result: str = await searcher.search(
        query="How does transformer attention work?",
        paths=["/path/to/documents"],
    )
    
    print(result)

asyncio.run(main())
```

**⚠️ Notes:**
- Upon initialization, `AgenticSearch` automatically checks if `ripgrep-all` and `ripgrep` are installed. If they are missing, it will attempt to install them automatically. If the automatic installation fails, please install them manually.
  - References: https://github.com/BurntSushi/ripgrep | https://github.com/phiresky/ripgrep-all
- Replace `"your-api-key"`, `"your-base-url"`, `"your-model-name"` and `/path/to/documents` with your actual values.


### Command Line Interface

Sirchmunk provides a powerful CLI for server management and search operations.

#### Installation

```bash
pip install "sirchmunk[web]"

# or install via UV
uv pip install "sirchmunk[web]"
```


#### Initialize

```bash
# Initialize Sirchmunk with default settings (Default work path: `~/.sirchmunk/`)
sirchmunk init

# Alternatively, initialize with custom work path
sirchmunk init --work-path /path/to/workspace
```

#### Start Server

```bash
# Start backend API server only
sirchmunk serve

# Custom host and port
sirchmunk serve --host 0.0.0.0 --port 8000
```

#### Search

```bash
# Search in current directory
sirchmunk search "How does authentication work?"

# Search in specific paths
sirchmunk search "find all API endpoints" ./src ./docs

# Quick filename search
sirchmunk search "config" --mode FILENAME_ONLY

# Output as JSON
sirchmunk search "database schema" --output json

# Use API server (requires running server)
sirchmunk search "query" --api --api-url http://localhost:8584
```

#### Available Commands

| Command | Description |
|---------|-------------|
| `sirchmunk init` | Initialize working directory, .env, and MCP config |
| `sirchmunk serve` | Start the backend API server |
| `sirchmunk search` | Perform search queries |
| `sirchmunk web init` | Build WebUI frontend (requires Node.js 18+) |
| `sirchmunk web serve` | Start API + WebUI (single port) |
| `sirchmunk web serve --dev` | Start API + Next.js dev server (hot-reload) |
| `sirchmunk mcp serve` | Start the MCP server (stdio/HTTP) |
| `sirchmunk mcp version` | Show MCP version information |
| `sirchmunk version` | Show version information |

---

## 🔌 MCP Server

Sirchmunk provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that exposes its intelligent search capabilities as MCP tools. This enables seamless integration with AI assistants like **Claude Desktop** and **Cursor IDE**.

### Quick Start

```bash
# Install with MCP support
pip install sirchmunk[mcp]

# Initialize (generates .env and mcp_config.json)
sirchmunk init

# Edit ~/.sirchmunk/.env with your LLM API key

# Test with MCP Inspector
npx @modelcontextprotocol/inspector sirchmunk mcp serve
```

### `mcp_config.json` Configuration

After running `sirchmunk init`, a `~/.sirchmunk/mcp_config.json` file is generated. Copy it to your MCP client configuration directory.

**Example:**

```json
{
  "mcpServers": {
    "sirchmunk": {
      "command": "sirchmunk",
      "args": ["mcp", "serve"],
      "env": {
        "SIRCHMUNK_SEARCH_PATHS": "/path/to/your_docs,/another/path"
      }
    }
  }
}
```

| Parameter | Description |
|---|---|
| `command` | The command to start the MCP server. Use full path (e.g. `/path/to/venv/bin/sirchmunk`) if running in a virtual environment. |
| `args` | Command arguments. `["mcp", "serve"]` starts the MCP server in stdio mode. |
| `env.SIRCHMUNK_SEARCH_PATHS` | Default document search directories (comma-separated). Supports both English `,` and Chinese `，` as delimiters. When set, these paths are used as default if no `paths` parameter is provided during tool invocation. |

> **Tip**: MCP Inspector is a great way to test the integration before connecting to your AI assistant.
> In MCP Inspector: **Connect** → **Tools** → **List Tools** → `sirchmunk_search` → Input parameters (`query` and `paths`, e.g. `["/path/to/your_docs"]`) → **Run Tool**.

### Features

- **Multi-Mode Search**: DEEP mode for comprehensive analysis, FILENAME_ONLY for fast file discovery
- **Knowledge Cluster Management**: Automatic extraction, storage, and reuse of knowledge
- **Standard MCP Protocol**: Works with stdio and Streamable HTTP transports

📖 **For detailed documentation, see [Sirchmunk MCP README](src/sirchmunk_mcp/README.md)**.

---

## 🖥️ Web UI

The web UI is built for fast, transparent workflows: chat, knowledge analytics, and system monitoring in one place.

<div align="center">
  <img src="assets/pic/Sirchmunk_Home.png" alt="Sirchmunk Home" width="85%">
  <p><sub>Home — Chat with streaming logs, file-based RAG, and session management.</sub></p>
</div>

<div align="center">
  <img src="assets/pic/Sirchmunk_Monitor.png" alt="Sirchmunk Monitor" width="85%">
  <p><sub>Monitor — System health, chat activity, knowledge analytics, and LLM usage.</sub></p>
</div>

### Option 1: Single-Port Mode (Recommended)

Build the frontend once, then serve everything from a single port — no Node.js needed at runtime.

```bash
# Build WebUI frontend (requires Node.js 18+ at build time)
sirchmunk web init

# Start server with embedded WebUI
sirchmunk web serve
```

**Access:** http://localhost:8584 (API + WebUI on the same port)

### Option 2: Development Mode

For frontend development with hot-reload:

```bash
# Start backend + Next.js dev server
sirchmunk web serve --dev
```

**Access:**
   - Frontend (hot-reload): http://localhost:8585
   - Backend APIs: http://localhost:8584/docs

### Option 3: Legacy Script

```bash
# Start frontend and backend via script
python scripts/start_web.py 

# Stop all services
python scripts/stop_web.py
```

**Configuration:**

- Access `Settings` → `Envrionment Variables` to configure LLM API, and other parameters.


---

## 🏗️ How it Works

### Sirchmunk Framework

<div align="center">
  <img src="assets/pic/Sirchmunk_Architecture.png" alt="Sirchmunk Architecture" width="85%">
</div>

### Core Components

| Component             | Description                                                              |
|:----------------------|:-------------------------------------------------------------------------|
| **AgenticSearch**     | Search orchestrator with LLM-enhanced retrieval capabilities             |
| **KnowledgeBase**     | Transforms raw results into structured knowledge clusters with evidences |
| **EvidenceProcessor** | Evidence processing based on the MonteCarlo Importance Sampling          |
| **GrepRetriever**     | High-performance _indexless_ file search with parallel processing        |
| **OpenAIChat**        | Unified LLM interface supporting streaming and usage tracking            |
| **MonitorTracker**    | Real-time system and application metrics collection                      |

### Monte Carlo Evidence Sampling

Traditional retrieval systems read entire documents or rely on fixed-size chunks, leading to either wasted tokens or lost context. Sirchmunk takes a fundamentally different approach inspired by **Monte Carlo methods** — treating evidence extraction as a **sampling problem** rather than a parsing problem.

<div align="center">
  <img src="assets/pic/Sirchmunk_MonteCarloSamplingAlgo.png" alt="Monte Carlo Evidence Sampling" width="85%">
  <p><sub>Monte Carlo Evidence Sampling — A three-phase exploration-exploitation strategy for extracting relevant evidence from large documents.</sub></p>
</div>

The algorithm operates in three phases:

1. **Phase 1 — Cast the Net (Exploration):** Fuzzy anchor matching combined with stratified random sampling. The system identifies seed regions of potential relevance while maintaining broad coverage through randomized probing — ensuring no high-value region is missed.

2. **Phase 2 — Focus (Exploitation):** Gaussian importance sampling centered around high-scoring seeds from Phase 1. The sampling density concentrates on the most promising regions, extracting surrounding context and scoring each snippet for relevance.

3. **Phase 3 — Synthesize:** The top-K scored snippets are passed to the LLM, which synthesizes them into a coherent Region of Interest (ROI) summary with a confidence flag — enabling the pipeline to decide whether evidence is sufficient or a ReAct agent should be invoked for deeper exploration.

**Key properties:**

- **Document-agnostic:** The same algorithm works equally well on a 2-page memo and a 500-page technical manual — no document-specific chunking heuristics needed.
- **Token-efficient:** Only the most relevant regions are sent to the LLM, dramatically reducing token consumption compared to full-document approaches.
- **Exploration-exploitation balance:** Random exploration prevents tunnel vision, while importance sampling ensures depth where it matters most.

### Self-Evolving Knowledge Clusters

Sirchmunk does not discard search results after answering a query. Instead, every search produces a **KnowledgeCluster** — a structured, reusable knowledge unit that grows smarter over time. This is what makes the system _self-evolving_.

#### What is a KnowledgeCluster?

A KnowledgeCluster is a richly annotated object that captures the full cognitive output of a single search cycle:

| Field | Purpose |
|:------|:--------|
| **Evidences** | Source-linked snippets extracted via Monte Carlo sampling, each with file path, summary, and raw text |
| **Content** | LLM-synthesized markdown with structured analysis and references |
| **Patterns** | 3–5 distilled design principles or mechanisms identified from the evidence |
| **Confidence** | A consensus score \[0, 1\] indicating the reliability of the cluster |
| **Queries** | Historical queries that contributed to or reused this cluster (FIFO, max 5) |
| **Hotness** | Activity score reflecting query frequency and recency |
| **Embedding** | 384-dim vector derived from accumulated queries, enabling semantic retrieval |

#### Lifecycle: From Creation to Evolution

```
 ┌─────── New Query ───────┐
 │                          ▼
 │     ┌──────────────────────────────┐
 │     │  Phase 0: Semantic Reuse     │──── Match found ──→ Return cached cluster
 │     │  (cosine similarity ≥ 0.85)  │                     + update hotness/queries/embedding
 │     └──────────┬───────────────────┘
 │           No match
 │                ▼
 │     ┌──────────────────────────────┐
 │     │  Phase 1–3: Full Search      │
 │     │  (keywords → retrieval →     │
 │     │   Monte Carlo → LLM synth)   │
 │     └──────────┬───────────────────┘
 │                ▼
 │     ┌──────────────────────────────┐
 │     │  Build New Cluster           │
 │     │  Deterministic ID: C{sha256} │
 │     └──────────┬───────────────────┘
 │                ▼
 │     ┌──────────────────────────────┐
 │     │  Phase 5: Persist            │
 │     │  Embed queries → DuckDB →    │
 │     │  Parquet (atomic sync)       │
 └─────└──────────────────────────────┘
```

1. **Reuse Check (Phase 0):** Before any retrieval, the query is embedded and compared against all stored clusters via cosine similarity. If a high-confidence match is found, the existing cluster is returned instantly — saving LLM tokens and search time entirely.

2. **Creation (Phase 1–3):** When no reuse match is found, the full pipeline runs: keyword extraction, file retrieval, Monte Carlo evidence sampling, and LLM synthesis produce a new `KnowledgeCluster`.

3. **Persistence (Phase 5):** The cluster is stored in an in-memory DuckDB table and periodically flushed to Parquet files. Atomic writes and mtime-based reload ensure multi-process safety.

4. **Evolution on Reuse:** Each time a cluster is reused, the system:
   - Appends the new query to the cluster's query history (FIFO, max 5)
   - Increases hotness (`+0.1`, capped at 1.0)
   - Recomputes the embedding from the updated query set — broadening the cluster's semantic catchment area
   - Updates version and timestamp

#### Key Properties

- **Zero-cost acceleration:** Repeated or semantically similar queries are answered from cached clusters without any LLM inference, making subsequent searches near-instantaneous.
- **Query-driven embeddings:** Cluster embeddings are derived from _queries_ rather than content, ensuring that retrieval aligns with how users actually ask questions — not how documents are written.
- **Semantic broadening:** As diverse queries reuse the same cluster, its embedding drifts to cover a wider semantic neighborhood, naturally improving recall for related future queries.
- **Lightweight persistence:** DuckDB in-memory + Parquet on disk — no external database infrastructure required. Background daemon sync with configurable flush intervals keeps overhead minimal.

---


### Data Storage

All persistent data is stored in the configured `SIRCHMUNK_WORK_PATH` (default: `~/.sirchmunk/`):

```
{SIRCHMUNK_WORK_PATH}/
  ├── .cache/
    ├── history/              # Chat session history (DuckDB)
    │   └── chat_history.db
    ├── knowledge/            # Knowledge clusters (Parquet)
    │   └── knowledge_clusters.parquet
    └── settings/             # User settings (DuckDB)
        └── settings.db

```

---

## 🔗 HTTP Client Access (Search API)

When the server is running (`sirchmunk serve` or `sirchmunk web serve`), the Search API is accessible via any HTTP client.

<details>
<summary><b>API Endpoints</b></summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/search` | Execute a search query |
| `GET` | `/api/v1/search/status` | Check server and LLM configuration status |

**Interactive Docs:** http://localhost:8584/docs (Swagger UI)

</details>

<details>
<summary><b>cURL Examples</b></summary>

```bash
# Basic search (DEEP mode)
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "paths": ["/path/to/project"],
    "mode": "DEEP"
  }'

# Filename search (fast, no LLM required)
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "config",
    "paths": ["/path/to/project"],
    "mode": "FILENAME_ONLY"
  }'

# Full parameters
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection pooling",
    "paths": ["/path/to/project/src"],
    "mode": "DEEP",
    "max_depth": 10,
    "top_k_files": 20,
    "keyword_levels": 3,
    "include_patterns": ["*.py", "*.java"],
    "exclude_patterns": ["*test*", "*__pycache__*"],
    "return_cluster": true
  }'

# Check server status
curl http://localhost:8584/api/v1/search/status
```

</details>

<details>
<summary><b>Python Client Examples</b></summary>

**Using `requests`:**

```python
import requests

response = requests.post(
    "http://localhost:8584/api/v1/search",
    json={
        "query": "How does authentication work?",
        "paths": ["/path/to/project"],
        "mode": "DEEP"
    },
    timeout=300  # DEEP mode may take a while
)

data = response.json()
if data["success"]:
    print(data["data"]["result"])
```

**Using `httpx` (async):**

```python
import httpx
import asyncio

async def search():
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "http://localhost:8584/api/v1/search",
            json={
                "query": "find all API endpoints",
                "paths": ["/path/to/project"],
                "mode": "DEEP"
            }
        )
        data = resp.json()
        print(data["data"]["result"])

asyncio.run(search())
```

</details>

<details>
<summary><b>JavaScript Client Example</b></summary>

```javascript
const response = await fetch("http://localhost:8584/api/v1/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "How does authentication work?",
    paths: ["/path/to/project"],
    mode: "DEEP"
  })
});

const data = await response.json();
if (data.success) {
  console.log(data.data.result);
}
```

</details>

<details>
<summary><b>Request Parameters</b></summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `string` | *required* | Search query or question |
| `paths` | `string[]` | *required* | Directories or files to search (min 1) |
| `mode` | `string` | `"DEEP"` | `DEEP` or `FILENAME_ONLY` |
| `max_depth` | `int` | `null` | Maximum directory depth |
| `top_k_files` | `int` | `null` | Number of top files to return |
| `keyword_levels` | `int` | `null` | Keyword granularity levels |
| `include_patterns` | `string[]` | `null` | File glob patterns to include |
| `exclude_patterns` | `string[]` | `null` | File glob patterns to exclude |
| `return_cluster` | `bool` | `false` | Return full KnowledgeCluster object |

> **Note:** `FILENAME_ONLY` mode does not require an LLM API key. `DEEP` mode requires a configured LLM.

</details>

---

## ❓ FAQ

<details>
<summary><b>How is this different from traditional RAG systems?</b></summary>

Sirchmunk takes an **indexless approach**:

1. **No pre-indexing**: Direct file search without vector database setup
2. **Self-evolving**: Knowledge clusters evolve based on search patterns
3. **Multi-level retrieval**: Adaptive keyword granularity for better recall
4. **Evidence-based**: Monte Carlo sampling for precise content extraction

</details>

<details>
<summary><b>What LLM providers are supported?</b></summary>

Any OpenAI-compatible API endpoint, including (but not limited too):
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Local models served via Ollama, llama.cpp, vLLM, SGLang etc.
- Claude via API proxy

</details>

<details>
<summary><b>How do I add documents to search?</b></summary>

Simply specify the path in your search query:

```python
result = await searcher.search(
    query="Your question",
    paths=["/path/to/folder", "/path/to/file.pdf"]
)
```

No pre-processing or indexing required!

</details>

<details>
<summary><b>Where are knowledge clusters stored?</b></summary>

Knowledge clusters are persisted in Parquet format at:
```
{SIRCHMUNK_WORK_PATH}/.cache/knowledge/knowledge_clusters.parquet
```

You can query them using DuckDB or the `KnowledgeManager` API.

</details>

<details>
<summary><b>How do I monitor LLM token usage?</b></summary>

1. **Web Dashboard**: Visit the Monitor page for real-time statistics
2. **API**: `GET /api/v1/monitor/llm` returns usage metrics
3. **Code**: Access `searcher.llm_usages` after search completion

</details>

---

## 📋 Roadmap

- [x] Text-retrieval from raw files
- [x] Knowledge structuring & persistence
- [x] Real-time chat with RAG
- [x] Web UI support
- [ ] Web search integration
- [ ] Multi-modal support (images, videos)
- [ ] Distributed search across nodes
- [ ] Knowledge visualization and deep analytics
- [ ] More file type support

---

## 🤝 Contributing

We welcome [contributions](https://github.com/modelscope/sirchmunk/pulls) !

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">

**[ModelScope](https://github.com/modelscope)** · [⭐ Star us](https://github.com/modelscope/sirchmunk/stargazers) · [🐛 Report a bug](https://github.com/modelscope/sirchmunk/issues) · [💬 Discussions](https://github.com/modelscope/sirchmunk/discussions)

*✨ Sirchmunk: Raw data to self-evolving intelligence, real-time.*

</div>

<p align="center">
  <em> ❤️ Thanks for Visiting ✨ Sirchmunk !</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=modelscope.sirchmunk&style=for-the-badge&color=00d4ff" alt="Views">
</p>
