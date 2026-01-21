<div align="center">

<img src="web/public/logo-v2.png" alt="Sirchmunk 标志" width="250" style="border-radius: 15px;">

# Sirchmunk：无需向量数据库和预索引的自进化搜索引擎

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat-square&logo=next.js&logoColor=white)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![DuckDB](https://img.shields.io/badge/DuckDB-OLAP-FFF000?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![ripgrep-all](https://img.shields.io/badge/ripgrep--all-Search-E67E22?style=flat-square&logo=rust&logoColor=white)](https://github.com/phiresky/ripgrep-all)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white)](https://github.com/openai/openai-python)
[![Kreuzberg](https://img.shields.io/badge/Kreuzberg-Text_Extraction-4CAF50?style=flat-square)](https://github.com/kreuzberg-dev/kreuzberg)


[**快速开始**](#-快速开始) · [**核心特性**](#-核心特性) · [**Web UI**](#-web-ui) · [**工作原理**](#-工作原理) · [**FAQ**](#-faq)

[English](README.md)

</div>

<div align="center">

🔍 **智能体搜索** &nbsp;•&nbsp; 🧠 **知识聚类** &nbsp;•&nbsp; 📊 **蒙特卡洛证据采样**<br>
⚡ **无索引检索** &nbsp;•&nbsp; 🔄 **自进化知识库** &nbsp;•&nbsp; 💬 **实时对话**

</div>

---

## 🌰 为什么选择 “Sirchmunk”？

基于向量检索的智能流水线往往 _僵硬且脆弱_。它们依赖静态向量嵌入，**计算成本高、对实时变化不敏感，并且脱离原始上下文**。我们引入 **Sirchmunk**，开启更敏捷的范式：数据不再是静态的快照和分块，而是直接从原始数据中洞见所查。

---

## ✨ 核心特性

### 1. 无需向量嵌入和预索引：直接面向原始数据形态

**Sirchmunk** 直接处理 **原始数据** —— 无需将大量而繁杂的文件压缩为固定维度向量，或是构建为图数据库。

* **即开即用搜索：** 不再需要复杂、耗时的预处理与索引；直接添加文件即可检索。
* **全量保真：** 零信息损失，避免向量近似带来的偏差。

### 2. 自进化：实时动态索引

数据是流动的，而非静态快照。**Sirchmunk** 天然具备动态特性。相比之下，向量数据库可能在数据变化的瞬间就过时。

* **上下文感知：** 随数据上下文实时演化。
* **LLM 自主驱动：** 面向智能体设计，通过精心设计的上下文检索技术，仅在必要时触发LLM推理，提高Token使用效率，兼顾智能与成本。

### 3. 规模化：实时与海量数据支持
**Sirchmunk** 具备 **高吞吐** 与 **实时感知** 的特性，能够高效处理本地大型数据集和文件系统。

---

### 传统 RAG vs. Sirchmunk

<div style="display: flex; justify-content: center; width: 100%;">
  <table style="width: 100%; max-width: 900px; border-collapse: separate; border-spacing: 0; overflow: hidden; border-radius: 12px; font-family: sans-serif; border: 1px solid rgba(128, 128, 128, 0.2); margin: 0 auto;">
    <colgroup>
      <col style="width: 25%;">
      <col style="width: 30%;">
      <col style="width: 45%;">
    </colgroup>
    <thead>
      <tr style="background-color: rgba(128, 128, 128, 0.05);">
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(128, 128, 128, 0.2); font-size: 1.3em;">维度</th>
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(128, 128, 128, 0.2); font-size: 1.3em; opacity: 0.7;">传统 RAG</th>
        <th style="text-align: left; padding: 16px; border-bottom: 2px solid rgba(58, 134, 255, 0.5); color: #3a86ff; font-weight: 800; font-size: 1.3em;">✨Sirchmunk</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">💰 搭建成本</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">高开销 <br/>（VectorDB、GraphDB、复杂文档解析器...）</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ 零基础设施 <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">直接面向数据检索，无向量孤岛</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">🕒 数据新鲜度</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">滞后（批量重建索引）</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ 即时 &amp; 动态 <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">自进化索引反映实时变化</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">📈 可扩展性</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">线性成本增长</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ 极低 RAM/CPU 占用 <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">原生弹性支持，高效处理大规模数据集</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">🎯 准确性</td>
        <td style="padding: 16px; opacity: 0.6; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">近似向量匹配</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef; border-bottom: 1px solid rgba(128, 128, 128, 0.1);">
          ✅ 确定性 &amp; 上下文相关 <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">混合逻辑确保语义精度</small>
        </td>
      </tr>
      <tr>
        <td style="padding: 16px; font-weight: 600;">⚙️ 工作流</td>
        <td style="padding: 16px; opacity: 0.6;">复杂 ETL 流水线</td>
        <td style="padding: 16px; background-color: rgba(58, 134, 255, 0.08); color: #4895ef;">
          ✅ 直接检索 <br/>
          <small style="opacity: 0.8; font-size: 0.85em;">零配置集成，快速部署</small>
        </td>
      </tr>
    </tbody>
  </table>
</div>

---


## 演示


<div align="center">
  <img src="assets/gif/Sirchmunk_Web.gif" alt="Sirchmunk WebUI" width="100%">
  <p style="font-size: 1.1em; font-weight: 600; margin-top: 8px; color: #00bcd4;">
    直接访问文件即可开始对话
  </p>
</div>

---


## 🚀 快速开始

### 前置条件

- **Python** 3.10+
- **LLM API Key**（OpenAI 兼容 Endpoint，本地或远程）
- **Node.js** 18+（可选，用于 Web 界面）

### 安装

```bash
# 创建虚拟环境（推荐）
conda create -n sirchmunk python=3.13 -y && conda activate sirchmunk 

pip install sirchmunk

# 或使用 UV：
uv pip install sirchmunk

# 或从源码安装：
git clone https://github.com/modelscope/sirchmunk.git && cd sirchmunk
pip install -e .
```

### Python SDK 使用

```python
import asyncio

from sirchmunk import AgenticSearch
from sirchmunk.llm import OpenAIChat

llm = OpenAIChat(
        api_key="your-api-key",
        base_url="your-base-url",   # 例如 https://api.openai.com/v1
        model="your-model-name"     # 例如 gpt-4o
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

**⚠️ 注意：**
- 初始化时，AgenticSearch 会自动检查是否安装 ripgrep-all 和 ripgrep。如缺失，会尝试自动安装。若自动安装失败，请手动安装。
  - 参考：https://github.com/BurntSushi/ripgrep | https://github.com/phiresky/ripgrep-all
- 将 `"your-api-key"`、`"your-base-url"`、`"your-model-name"` 和 `/path/to/documents` 替换为实际值。


---

## 🖥️ Web UI

Web UI 专为快速、透明的工作流设计：对话、知识分析、系统监控一体化。

<div align="center">
  <img src="assets/pic/Sirchmunk_Home.png" alt="Sirchmunk Home" width="85%">
  <p><sub>Home — 流式日志聊天、基于文件的 RAG 与会话管理。</sub></p>
</div>

<div align="center">
  <img src="assets/pic/Sirchmunk_Monitor.png" alt="Sirchmunk Monitor" width="85%">
  <p><sub>Monitor — 系统健康、聊天活动、知识分析与 LLM 用量。</sub></p>
</div>

### 安装 

```bash
git clone https://github.com/modelscope/sirchmunk.git && cd sirchmunk

pip install ".[web]"

npm install --prefix web
```
- 备注: 需要安装 Node.js 18+


### 运行 Web UI

```bash
# 启动前端和后端
python scripts/start_web.py 

# 停止前端和后端
python scripts/stop_web.py
```

**默认访问地址：**
   - 后端API列表： http://localhost:8584/docs
   - 前端主页： http://localhost:8585

**配置:**

- 访问 `Settings` → `Envrionment Variables` 设置 LLM API Key 和其他环境变量


---

## 🏗️ 工作原理

### Sirchmunk 框架

<div align="center">
  <img src="assets/pic/Sirchmunk_Architecture.png" alt="Sirchmunk 架构" width="85%">
</div>

### 核心组件

| 组件                    | 说明                                                                   |
|:------------------------|:-----------------------------------------------------------------------|
| **AgenticSearch**       | 搜索编排器，具备 LLM 增强检索能力                                       |
| **KnowledgeBase**       | 将原始结果转化为结构化知识聚类并附带证据                               |
| **EvidenceProcessor**   | 基于蒙特卡洛重要性采样的证据处理                                       |
| **GrepRetriever**       | 高性能 _无索引_ 文件检索，支持并行处理                                 |
| **OpenAIChat**          | 统一 LLM 接口，支持流式与用量统计                                       |
| **MonitorTracker**      | 实时系统与应用指标采集                                                 |

---


### 数据存储

所有持久化数据存储在配置的 `WORK_PATH`（默认：`~/.sirchmunk/`）：

```
{WORK_PATH}/
  ├── .cache/
    ├── history/              # 聊天会话历史（DuckDB）
    │   └── chat_history.db
    ├── knowledge/            # 知识聚类（Parquet）
    │   └── knowledge_clusters.parquet
    └── settings/             # 用户设置（DuckDB）
        └── settings.db

```

---

## ❓ FAQ

<details>
<summary><b>这与传统 RAG 系统有什么不同？</b></summary>

Sirchmunk 采用 **无索引** 方法：

1. **无预索引**：无需向量数据库，直接检索文件
2. **自进化**：知识聚类随检索模式演化
3. **多层检索**：自适应关键词粒度提升召回
4. **证据驱动**：蒙特卡洛重要性采样实现精准内容定位和抽取

</details>

<details>
<summary><b>支持哪些 LLM 提供商？</b></summary>

任何 OpenAI 兼容 API 端点，包括但不限于：
- OpenAI（GPT-4、GPT-4o、GPT-3.5）
- 通过 Ollama、llama.cpp、vLLM、SGLang 等托管的本地模型
- 通过 API 代理接入的 Claude

</details>

<details>
<summary><b>如何添加需要检索的文档？</b></summary>

只需在搜索请求中指定路径：

```python
result = await search.search(
    query="Your question",
    search_paths=["/path/to/folder", "/path/to/file.pdf"]
)
```

</details>

<details>
<summary><b>知识聚类存储在哪里？</b></summary>

知识聚类以 Parquet 格式持久化于：
```
{WORK_PATH}/.cache/knowledge/knowledge_clusters.parquet
```

你可以使用 DuckDB 或 `KnowledgeManager` API 查询。

</details>

<details>
<summary><b>如何监控 LLM Token 使用量？</b></summary>

1. **Web 面板**：访问 Monitor 页面查看实时统计
2. **API**：`GET /api/v1/monitor/llm` 返回用量指标
3. **代码**：搜索完成后访问 `search.llm_usages`

</details>

---

## 📋 Roadmap

- [x] 原始文件文本检索
- [x] 知识结构化与持久化
- [x] 基于 RAG 的实时对话
- [x] Web UI 支持
- [ ] Web 搜索集成
- [ ] 多模态支持（图片、视频）
- [ ] 分布式跨节点检索
- [ ] 知识可视化与深度分析
- [ ] 更多文件类型支持

---

## 🤝 贡献

欢迎 [贡献](https://github.com/modelscope/sirchmunk/pulls)！

---

## 📄 许可

本项目采用 [Apache License 2.0](LICENSE)。

---

<div align="center">

**[ModelScope](https://github.com/modelscope)** · [⭐ Star us](https://github.com/modelscope/sirchmunk/stargazers) · [🐛 反馈问题](https://github.com/modelscope/sirchmunk/issues) · [💬 Discussions](https://github.com/modelscope/sirchmunk/discussions)

*✨ Sirchmunk：原始数据到自进化智能，实时。*

</div>

<p align="center">
  <em> ❤️ 感谢访问 ✨ Sirchmunk ！</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=modelscope.sirchmunk&style=for-the-badge&color=00d4ff" alt="Views">
</p>