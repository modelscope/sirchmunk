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
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=flat-square&logo=ollama&logoColor=white)](https://ollama.com/)
[![Kreuzberg](https://img.shields.io/badge/Kreuzberg-Text_Extraction-4CAF50?style=flat-square&logo=python&logoColor=white)](https://github.com/zalando/kreuzberg)


[**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Core Modules**](#-core-modules) · [**API Reference**](#-api-reference) · [**FAQ**](#-faq)

[🇨🇳 中文](README_zh.md)

</div>

<div align="center">

🔍 **Agentic Search** &nbsp;•&nbsp; 🧠 **Knowledge Clustering** &nbsp;•&nbsp; 📊 **Monte Carlo Evidence Sampling**<br>
⚡ **Indexless Retrieval** &nbsp;•&nbsp; 🔄 **Self-Evolving Knowledge Base** &nbsp;•&nbsp; 💬 **Real-time Chat**

</div>

---

## 🌰 Why “Sirchmunk”？

Intelligence pipelines built upon vector-based retrieval can be _rigid and brittle_. They rely on static vector embeddings that are **expensive to compute, blind to real-time changes, and detached from the raw context**. We introduce **Sirchmunk** to usher in a more agile paradigm, where data is no longer treated as a snapshot, and insights can evolve together with the data.

---

## ✨ Key Features

### 🔍 Agentic Search with Multi-Level Keyword Extraction

- **Intelligent Query Understanding**: LLM-powered keyword extraction with configurable granularity levels (coarse → fine)
- **Priority-Hit Retrieval**: Sequential search across keyword levels, stopping at first successful match for optimal efficiency
- **TF-IDF Scoring**: Advanced document ranking with customizable weighting algorithms

### 🧠 Self-Evolving Knowledge Clusters

- **Automatic Knowledge Structuring**: Raw search results transformed into structured `KnowledgeCluster` objects
- **Evidence-Based Learning**: Monte Carlo sampling for relevant region identification with LLM evaluation
- **Lifecycle Management**: Track knowledge states (`STABLE`, `EMERGING`, `CONTESTED`, `DEPRECATED`)
- **Persistent Storage**: DuckDB + Parquet for efficient knowledge persistence and retrieval
- **Dynamic Knowledge Generation**: New clusters are built and updated continuously as search patterns evolve

### 📚 Large-Scale Document Understanding

- **High-Volume Coverage**: Handle large repositories without pre-indexing
- **Granular Evidence Tracing**: Extracts and scores precise spans from massive documents
- **Fast Multi-Level Recall**: Coarse-to-fine keyword tiers improve hit rate on long or noisy corpora

### ⚡ Indexless Real-Time Retrieval

- **No Pre-indexing Required**: Direct `grep`-based retrieval on raw files
- **Multi-Format Support**: PDF, DOCX, TXT, Markdown, code files, and more
- **Blazing Fast**: Parallel file scanning with configurable concurrency

### 💬 Interactive Chat Interface

- **WebSocket Streaming**: Real-time response streaming with search log visualization
- **RAG Integration**: Seamless knowledge base augmented generation
- **Session Management**: Persistent chat history with DuckDB storage
- **LLM Usage Tracking**: Real-time token consumption monitoring

### 📊 Comprehensive Monitoring Dashboard

- **System Metrics**: CPU, memory, disk usage tracking
- **Chat Analytics**: Session statistics and activity monitoring
- **Knowledge Analytics**: BI-style visualization for knowledge clusters
- **LLM Usage Statistics**: Token consumption tracking by model

---

## 🏗️ Architecture

<div align="center">
  <img src="assets/pic/Sirchmunk_Architecture.png" alt="Sirchmunk Architecture" width="80%">
</div>

### Core Components

| Component | Description |
|:---|:---|
| **AgenticSearch** | Main search orchestrator with LLM-powered keyword extraction and retrieval |
| **KnowledgeBase** | Transforms raw results into structured knowledge clusters with evidence sampling |
| **KnowledgeManager** | Persistent storage layer using DuckDB and Parquet format |
| **GrepRetriever** | High-performance indexless file search with parallel processing |
| **OpenAIChat** | Unified LLM interface supporting streaming and usage tracking |
| **MonitorTracker** | Real-time system and application metrics collection |

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (for web interface)
- **LLM API Key** (OpenAI-compatible endpoint)

### Step 1: Clone & Configure

```bash
# Clone the repository
git clone https://github.com/modelscope/sirchmunk.git
cd sirchmunk

# Create environment file
cp .env.example .env
# Edit .env with your LLM API credentials
```

<details>
<summary>📋 <b>Environment Variables Reference</b></summary>

| Variable | Required | Description |
|:---|:---:|:---|
| `LLM_BASE_URL` | **Yes** | LLM API endpoint (e.g., `https://api.openai.com/v1`) |
| `LLM_API_KEY` | **Yes** | Your LLM API key |
| `LLM_MODEL_NAME` | **Yes** | Model name (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `WORK_PATH` | No | Working directory for data storage (default: current directory) |
| `GREP_CONCURRENT_LIMIT` | No | Parallel grep workers (default: `10`) |

</details>

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
conda create -n sirchmunk python=3.10 && conda activate sirchmunk
# Or: python -m venv venv && source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install web dependencies
npm install --prefix web
```

### Step 3: Launch

```bash
# Start both backend and frontend
python scripts/start_web.py

# Or start separately:
# Backend: python src/api/run_server.py
# Frontend: cd web && npm run dev
```

### Access URLs

| Service | URL | Description |
|:---:|:---|:---|
| **Web Interface** | http://localhost:3000 | Main chat and dashboard |
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |

---

## 🖥️ Web Experience

The web UI is built for fast, transparent workflows: chat, knowledge analytics, and system monitoring in one place.

<div align="center">
  <img src="assets/pic/Sirchmunk_Home.png" alt="Sirchmunk Home" width="80%">
  <p><sub>Home — Chat with streaming logs, file-based RAG, and session management.</sub></p>
</div>

<div align="center">
  <img src="assets/pic/Sirchmunk_Monitor.png" alt="Sirchmunk Monitor" width="80%">
  <p><sub>Monitor — System health, chat activity, knowledge analytics, and LLM usage.</sub></p>
</div>

---

## 📦 Core Modules

<details>
<summary><b>🔍 AgenticSearch</b></summary>

> **Intelligent search engine** with multi-level keyword extraction, priority-hit retrieval, and automatic knowledge clustering.

**Core Features**

| Feature | Description |
|:---:|:---|
| Multi-Level Keywords | Extract keywords at configurable granularity levels (1-N) |
| Priority-Hit Search | Stop search as soon as results are found at any level |
| TF-IDF Scoring | Advanced document ranking with length penalty |
| Knowledge Persistence | Auto-save search results as KnowledgeCluster objects |

**Python API**

```python
import asyncio
from sirchmunk import AgenticSearch

async def main():
    search = AgenticSearch()
    
    result = await search.search(
        query="How does transformer attention work?",
        search_paths=["/path/to/documents"],
        keyword_levels=3,  # Coarse → Medium → Fine
        top_k_files=5,
        max_depth=10
    )
    
    print(result)

asyncio.run(main())
```

</details>

---

<details>
<summary><b>🧠 KnowledgeBase</b></summary>

> **Evidence processor** that transforms raw search results into structured knowledge clusters using Monte Carlo sampling and LLM evaluation.

**Core Features**

| Feature | Description |
|:---:|:---|
| Monte Carlo Sampling | Identify relevant regions in large documents through iterative sampling |
| LLM Evidence Evaluation | Score and validate evidence snippets with reasoning |
| Fuzzy Anchoring | RapidFuzz-based pre-filtering for efficient sampling |
| Structured Output | Generate KnowledgeCluster with evidences, patterns, and constraints |

**Knowledge Cluster Schema**

```python
@dataclass
class KnowledgeCluster:
    id: str
    name: str
    description: List[str]
    content: Union[str, List[str]]
    evidences: List[EvidenceUnit]
    patterns: List[str]
    constraints: List[Constraint]
    confidence: float
    abstraction_level: AbstractionLevel
    lifecycle: Lifecycle  # STABLE, EMERGING, CONTESTED, DEPRECATED
    hotness: float
    search_results: List[str]
```

</details>

---

<details>
<summary><b>💾 KnowledgeManager</b></summary>

> **Persistent storage layer** for knowledge clusters using DuckDB and Parquet format.

**Core Features**

| Feature | Description |
|:---:|:---|
| CRUD Operations | Full create, read, update, delete support |
| Fuzzy Search | Find clusters by name, description, or content |
| Merge & Split | Combine or divide knowledge clusters |
| Statistics | Get analytics and distribution metrics |

**Python API**

```python
from sirchmunk.storage import KnowledgeManager

# Initialize manager
km = KnowledgeManager(work_path="/path/to/workspace")

# Insert a cluster
await km.insert(cluster)

# Search clusters
results = await km.find("transformer attention")

# Get statistics
stats = km.get_stats()
print(f"Total clusters: {stats['custom_stats']['total_clusters']}")
```

**Storage Location**

```
{WORK_PATH}/
└── .cache/
    └── knowledge/
        └── knowledge_clusters.parquet
```

</details>

---

<details>
<summary><b>⚡ GrepRetriever</b></summary>

> **High-performance indexless retriever** using parallel grep for real-time file search.

**Core Features**

| Feature | Description |
|:---:|:---|
| Parallel Processing | Configurable concurrent workers |
| Multi-Format Support | PDF, DOCX, TXT, MD, code files, and more |
| Regex Support | Full regular expression pattern matching |
| Result Merging | Deduplicate and merge results across files |

**Supported File Types**

- Documents: PDF, DOCX, TXT, Markdown
- Code: Python, JavaScript, TypeScript, Java, Go, Rust, C/C++
- Data: JSON, YAML, XML, CSV
- Archives: ZIP, TAR (with extraction)

</details>

---

<details>
<summary><b>💬 Chat API</b></summary>

> **WebSocket-based chat interface** with RAG integration and real-time streaming.

**Chat Modes**

| Mode | Description |
|:---:|:---|
| Pure Chat | Direct LLM conversation without retrieval |
| Chat + RAG | Knowledge-augmented generation from local files |
| Chat + Web | Web search augmented responses (coming soon) |
| Chat + RAG + Web | Combined knowledge and web search |

**WebSocket Message Format**

```json
{
  "type": "message",
  "content": "Your question here",
  "session_id": "uuid",
  "enable_rag": true,
  "kb_name": "/path/to/documents"
}
```

**Response Types**

```json
// Streaming content
{"type": "content", "content": "..."}

// Search logs
{"type": "search_log", "level": "info", "message": "...", "is_streaming": false}

// Status updates
{"type": "status", "stage": "generating", "message": "..."}

// Completion
{"type": "done", "message_id": "uuid", "sources": {...}}
```

</details>

---

## 📂 Data Storage

All persistent data is stored in the configured `WORK_PATH`:

```
{WORK_PATH}/
├── .cache/
│   ├── history/              # Chat session history (DuckDB)
│   │   └── chat_history.db
│   ├── knowledge/            # Knowledge clusters (Parquet)
│   │   └── knowledge_clusters.parquet
│   └── settings/             # User settings (DuckDB)
│       └── settings.db
└── logs/                     # Application logs
```

---

## 🔧 API Reference

### REST Endpoints

| Endpoint | Method | Description |
|:---|:---:|:---|
| `/api/v1/chat/sessions` | GET | List all chat sessions |
| `/api/v1/chat/session/{id}` | GET | Get session details |
| `/api/v1/chat/ws` | WS | WebSocket chat endpoint |
| `/api/v1/knowledge/list` | GET | List knowledge clusters |
| `/api/v1/knowledge/stats` | GET | Get knowledge statistics |
| `/api/v1/knowledge/search` | POST | Search knowledge clusters |
| `/api/v1/monitor/overview` | GET | Get system overview |
| `/api/v1/monitor/llm` | GET | Get LLM usage statistics |
| `/api/v1/settings` | GET/POST | Manage settings |

### Python SDK

```python
from sirchmunk import AgenticSearch
from sirchmunk.llm import OpenAIChat
from sirchmunk.storage import KnowledgeManager

# Initialize with custom LLM
llm = OpenAIChat(
    base_url="https://api.openai.com/v1",
    api_key="your-key",
    model="gpt-4o"
)

search = AgenticSearch(llm=llm)
result = await search.search(query="...", search_paths=["..."])
```

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

Any OpenAI-compatible API endpoint, including:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Azure OpenAI
- Local models via Ollama, vLLM, or LM Studio
- Claude via API proxy

</details>

<details>
<summary><b>How do I add documents to search?</b></summary>

Simply specify the path in your search query:

```python
result = await search.search(
    query="Your question",
    search_paths=["/path/to/folder", "/path/to/file.pdf"]
)
```

No pre-processing or indexing required!

</details>

<details>
<summary><b>Where are knowledge clusters stored?</b></summary>

Knowledge clusters are persisted in Parquet format at:
```
{WORK_PATH}/.cache/knowledge/knowledge_clusters.parquet
```

You can query them using DuckDB or the `KnowledgeManager` API.

</details>

<details>
<summary><b>How do I monitor LLM token usage?</b></summary>

1. **Web Dashboard**: Visit the Monitor page for real-time statistics
2. **API**: `GET /api/v1/monitor/llm` returns usage metrics
3. **Code**: Access `search.llm_usages` after search completion

</details>

---

## 📋 Roadmap

- [x] Multi-level keyword extraction
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

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

```bash
# Development setup
pip install -r requirements/tests.txt
pytest tests/
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

<div align="center">

**[ModelScope](https://github.com/modelscope)** · [⭐ Star us](https://github.com/modelscope/sirchmunk/stargazers) · [🐛 Report a bug](https://github.com/modelscope/sirchmunk/issues) · [💬 Discussions](https://github.com/modelscope/sirchmunk/discussions)

*✨ Sirchmunk: From raw data to self-evolving real-time intelligence.*

</div>

<p align="center">
  <em> ❤️ Thanks for Visiting ✨ Sirchmunk !</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=modelscope.sirchmunk&style=for-the-badge&color=00d4ff" alt="Views">
</p>
