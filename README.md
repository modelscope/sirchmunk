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

### Traditional RAG vs. Sirchmunk

<table style="width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; border-radius: 12px; font-family: sans-serif; border: 1px solid rgba(128, 128, 128, 0.2);">
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

---


## Demo

The web UI is built for fast, transparent workflows: chat, knowledge analytics, and system monitoring in one place.

<div align="center">
  <img src="assets/gif/Sirchmunk_Web.gif" alt="Sirchmunk WebUI" width="100%">
  <p><sub> Access files directly to start chatting </sub></p>
</div>

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
