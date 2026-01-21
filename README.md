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

**Sirchmunk** works directly with **raw data** -- bypassing the heavy overhead of squeezing your rich files into fixed-dimensional vectors.

* **Instant Search:** Eliminating complex pre-processing pipelines in hours long indexing; just drop your files and search immediately.
* **Full Fidelity:** Zero information loss —- stay true to your data without vector approximation.

### 2. Self-Evolving: A Living Index

Data is a stream, not a snapshot.  **Sirchmunk** is **dynamic by design**, while vector DB can become obsolete the moment your data changes.

* **Context-Aware:** Evolves in real-time with your data context.
* **LLM-Powered Autonomy:** Designed for Agents that perceive data as it lives, utilizing **token-efficient** reasoning that triggers LLM inference only when necessary to maximize intelligence while minimizing cost.

### 3. Intelligence at Scale: Real-Time & Massive
**Sirchmunk** bridges massive local repositories and the web with **high-scale throughput** and **real-time awareness**. 
It serves as a unified intelligent hub for AI agents, delivering deep insights across vast datasets at the speed of thought.

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
    
    agent_search = AgenticSearch(llm=llm)
    
    result: str = await agent_search.search(
        query="How does transformer attention work?",
        search_paths=["/path/to/documents"],
    )
    
    print(result)

asyncio.run(main())
```

**⚠️ Notes:**
- Upon initialization, AgenticSearch automatically checks if ripgrep-all and ripgrep are installed. If they are missing, it will attempt to install them automatically. If the automatic installation fails, please install them manually.
  - References: https://github.com/BurntSushi/ripgrep | https://github.com/phiresky/ripgrep-all
- Replace `"your-api-key"`, `"your-base-url"`, `"your-model-name"` and `/path/to/documents` with your actual values.





---

## 🖥️ Web UI

The web UI is built for fast, transparent workflows: chat, knowledge analytics, and system monitoring in one place.

<div align="center">
  <img src="assets/gif/Sirchmunk_Web.gif" alt="Sirchmunk WebUI" width="85%">
  <p><sub> Access files directly to start chatting </sub></p>
</div>


### Installation 

```bash
pip install "sirchmunk[web]"
```


### Running the Web UI

```bash
git clone https://github.com/modelscope/sirchmunk.git && cd sirchmunk

# Start frontend and backend
python scripts/start_web.py 

# Stop frontend and backend
python scripts/stop_web.py
```

**Access the web UI at (By default):**
   - Backend APIs:  http://localhost:8584/docs
   - Frontend: http://localhost:8585


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

---


### Data Storage

All persistent data is stored in the configured `WORK_PATH` (default: `~/.sirchmunk/`):

```
{WORK_PATH}/
  ├── .cache/
    ├── history/              # Chat session history (DuckDB)
    │   └── chat_history.db
    ├── knowledge/            # Knowledge clusters (Parquet)
    │   └── knowledge_clusters.parquet
    └── settings/             # User settings (DuckDB)
        └── settings.db

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

Any OpenAI-compatible API endpoint, including (but not limited too):
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
- Local models served via Ollama, llama.cpp, vLLM, SGLang etc.
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
