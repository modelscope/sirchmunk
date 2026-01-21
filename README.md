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
[![Kreuzberg](https://img.shields.io/badge/Kreuzberg-Text_Extraction-4CAF50?style=flat-square)](https://github.com/kreuzberg-dev/kreuzberg)


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

### 1. Embedding-Free: Data in its Purest Form

Traditional RAG (Retrieval-Augmented Generation) forces nuanced files into fixed-dimensional vectors. **Sirchmunk** retrieves directly from **raw data**.

* **Instant Search:** No complex pre-processing pipelines or multi-hour indexing; drop files and search immediately.
* **Full Fidelity:** Zero information loss—no vector approximation, just raw precision.

### 2. Self-Evolving: A Living Index

Data is a stream, not a snapshot. While vector databases become stale the moment data changes, **Sirchmunk** is **dynamic by design**.

* **Context-Aware:** The index evolves in real-time as your files grow and change.
* **LLM-Powered Autonomy:** Designed for Agents that perceive data as it lives, utilizing **token-efficient** reasoning that triggers LLM inference only when necessary to maximize intelligence while minimizing cost.

### 3. Intelligence at Scale: Real-Time & Massive

**Sirchmunk** bridges massive local repositories and the web with **Large-Scale** throughput and **Real-Time** perception. </br>
It provides a unified, intelligent pulse for AI Agents, delivering deep insights across vast datasets with the speed of thought.

---

### How it compares

| Feature | RAG (Chunking+VectorDB)                       | **Sirchmunk** |
| --- |------------------------------| --- |
| **Setup Cost** | High (Models + Infra)        | **Zero (Direct Retrieval)** |
| **Data Freshness** | Stale (Requires re-indexing) | **Instant (Self-evolving)** |
| **Scalability** | Costly/Complex at Scale      | **Native Large-Scale Support** |
| **Accuracy** | Approximate (Probabilistic)  | **Exact & Contextual** |
| **Workflow** | Complex ETL Pipelines        | **Drop-and-Search** |

---


## 🚀 Quick Start

### Prerequisites

- **Python** 3.10+
- **LLM API Key** (OpenAI-compatible endpoint, or Ollama for local models)
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
    
    agent_search = AgenticSearch(llm=llm)
    
    result: str = await agent_search.search(
        query="How does transformer attention work?",
        search_paths=["/path/to/documents"],
    )
    
    print(result)

asyncio.run(main())
```






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

## 🏗️ How it Works

### Architecture

<div align="center">
  <img src="assets/pic/Sirchmunk_Architecture.png" alt="Sirchmunk Architecture" width="85%">
</div>

### Core Components

| Component | Description |
|:---|:---|
| **AgenticSearch** | Main search orchestrator with LLM-powered keyword extraction and retrieval |
| **KnowledgeBase** | Transforms raw results into structured knowledge clusters with evidence sampling |
| **KnowledgeManager** | Persistent storage layer, by default in Parquet format and stored in DuckDB |
| **GrepRetriever** | High-performance _indexless_ file search with parallel processing |
| **OpenAIChat** | Unified LLM interface supporting streaming and usage tracking |
| **MonitorTracker** | Real-time system and application metrics collection |

---

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

*✨ Sirchmunk: Raw data to self-evolving intelligence, real-time.*

</div>

<p align="center">
  <em> ❤️ Thanks for Visiting ✨ Sirchmunk !</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=modelscope.sirchmunk&style=for-the-badge&color=00d4ff" alt="Views">
</p>
