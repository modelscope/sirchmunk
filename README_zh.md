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
[![MCP](https://img.shields.io/badge/MCP-Python_SDK-8B5CF6?style=flat-square&logo=python&logoColor=white)](https://github.com/modelcontextprotocol/python-sdk)
[![Documentation](https://img.shields.io/badge/Documentation-官方文档-4285F4?style=flat-square&logo=readthedocs&logoColor=white)](https://modelscope.github.io/sirchmunk-web/)


[**快速开始**](#-快速开始) · [**核心特性**](#-核心特性) · [**MCP 服务器**](#-mcp-服务器) · [**Web UI**](#️-web-ui) · [**工作原理**](#️-工作原理) · [**FAQ**](#-faq)

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


|  WeChat Group
|:-------------------------:
|  <img src="asset/ms-agent.jpg" width="200" height="200">

---


## 🎉 News

* 🚀 **2026.2.5**: 发布 **v0.0.2** — MCP 支持、CLI 命令行 & 知识持久化！
  - **MCP 集成**：完整支持 [Model Context Protocol](https://modelcontextprotocol.io)，与 Claude Desktop 和 Cursor IDE 无缝协作。
  - **CLI 命令行**：全新 `sirchmunk` 命令行工具，支持 `init`、`serve`、`search`、`web` 和 `mcp` 命令。
  - **KnowledgeCluster 持久化**：基于 DuckDB 存储，支持 Parquet 导出，高效管理知识聚类。
  - **知识复用**：基于语义相似度的知识聚类检索，通过 embedding 向量加速搜索。

* 🎉🎉 2026.1.22: **Sirchmunk** 初始版本 v0.0.1 现已发布！

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
        paths=["/path/to/documents"],
    )
    
    print(result)

asyncio.run(main())
```

**⚠️ 注意：**
- 初始化时，AgenticSearch 会自动检查是否安装 ripgrep-all 和 ripgrep。如缺失，会尝试自动安装。若自动安装失败，请手动安装。
  - 参考：https://github.com/BurntSushi/ripgrep | https://github.com/phiresky/ripgrep-all
- 将 `"your-api-key"`、`"your-base-url"`、`"your-model-name"` 和 `/path/to/documents` 替换为实际值。


### 命令行界面

Sirchmunk 提供强大的 CLI，用于服务器管理和搜索操作。


#### 安装

```bash
pip install "sirchmunk[web]"

# 或使用UV安装
uv pip install "sirchmunk[web]"
```


#### 初始化

```bash
# 使用默认设置初始化 Sirchmunk，默认工作路径为 `~/.sirchmunk/`
sirchmunk init

# 或者，也可以使用自定义工作路径初始化
sirchmunk init --work-path /path/to/workspace
```

#### 启动服务器

```bash
# 仅启动后端 API 服务器
sirchmunk serve

# 自定义主机和端口
sirchmunk serve --host 0.0.0.0 --port 8000
```

#### 搜索

```bash
# 在当前目录搜索
sirchmunk search "认证是如何工作的？"

# 在指定路径搜索
sirchmunk search "查找所有 API 端点" ./src ./docs

# 快速文件名搜索
sirchmunk search "config" --mode FILENAME_ONLY

# 输出为 JSON 格式
sirchmunk search "数据库模式" --output json

# 通过 API 服务器搜索（需要先启动服务器）
sirchmunk search "查询" --api --api-url http://localhost:8584
```

#### 可用命令

| 命令 | 说明 |
|------|------|
| `sirchmunk init` | 初始化工作目录、.env 及 MCP 配置 |
| `sirchmunk serve` | 仅启动后端 API 服务器 |
| `sirchmunk search` | 执行搜索查询 |
| `sirchmunk web init` | 构建 WebUI 前端（需要 Node.js 18+） |
| `sirchmunk web serve` | 启动 API + WebUI（单端口） |
| `sirchmunk web serve --dev` | 开发模式，Next.js 热重载 |
| `sirchmunk mcp serve` | 启动 MCP 服务器（stdio/HTTP） |
| `sirchmunk mcp version` | 显示 MCP 版本信息 |
| `sirchmunk version` | 显示版本信息 |

---

## 🔌 MCP 服务器

Sirchmunk 提供 [Model Context Protocol (MCP)](https://modelcontextprotocol.io) 服务器，将其智能搜索能力作为 MCP 工具暴露。可与 **Claude Desktop** 和 **Cursor IDE** 等 AI 助手无缝集成。

### 快速开始

```bash
# 安装（含 MCP 支持）
pip install sirchmunk[mcp]

# 初始化（生成 .env 和 mcp_config.json）
sirchmunk init

# 编辑 ~/.sirchmunk/.env 配置 LLM API Key

# 使用 MCP Inspector 测试
npx @modelcontextprotocol/inspector sirchmunk mcp serve
```

### `mcp_config.json` 配置

运行 `sirchmunk init` 后会生成 `~/.sirchmunk/mcp_config.json` 文件。将其复制到你的 MCP 客户端配置目录即可。

**示例：**

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

| 参数 | 说明 |
|---|---|
| `command` | 启动 MCP 服务器的命令。如果在虚拟环境中运行，请使用完整路径（如 `/path/to/venv/bin/sirchmunk`）。 |
| `args` | 命令参数。`["mcp", "serve"]` 以 stdio 模式启动 MCP 服务器。 |
| `env.SIRCHMUNK_SEARCH_PATHS` | 默认文档搜索目录（逗号分隔）。同时支持英文逗号 `,` 和中文逗号 `，` 作为分隔符。设置后，若工具调用时未提供 `paths` 参数，将使用这些路径作为默认值。 |

> **提示**：MCP Inspector 非常适合在连接 AI 助手之前测试集成是否正常。
> 在 MCP Inspector 中：**Connect** → **Tools** → **List Tools** → `sirchmunk_search` → 输入参数（`query` 和 `paths`，如 `["/path/to/your_docs"]`）→ **Run Tool**。

### 特性

- **多模式搜索**：DEEP 模式进行全面分析，FILENAME_ONLY 模式快速发现文件
- **知识聚类管理**：自动提取、存储和复用知识
- **标准 MCP 协议**：支持 stdio 和 Streamable HTTP 传输

📖 **详细文档请参阅 [Sirchmunk MCP README](src/sirchmunk_mcp/README.md)**。

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

### 方式一：单端口模式（推荐）

一次构建前端，随后通过单端口同时提供 API 和 WebUI — 运行时无需 Node.js。

```bash
# 构建 WebUI 前端（构建时需要 Node.js 18+）
sirchmunk web init

# 启动含内嵌 WebUI 的服务器
sirchmunk web serve
```

**访问地址：** http://localhost:8584（API + WebUI 同端口）

### 方式二：开发模式

支持前端热重载的开发环境：

```bash
# 启动后端 + Next.js 开发服务器
sirchmunk web serve --dev
```

**访问地址：**
   - 前端（热重载）：http://localhost:8585
   - 后端 API：http://localhost:8584/docs

### 方式三：传统脚本

```bash
# 通过脚本启动前后端
python scripts/start_web.py 

# 停止所有服务
python scripts/stop_web.py
```

**配置：**

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

### 蒙特卡洛证据采样

传统检索系统要么读取完整文档，要么依赖固定大小的分块，导致 Token 浪费或上下文丢失。Sirchmunk 借鉴**蒙特卡洛方法**，采用了截然不同的策略——将证据提取视为一个**采样问题**而非解析问题。

<div align="center">
  <img src="assets/pic/Sirchmunk_MonteCarloSamplingAlgo.png" alt="蒙特卡洛证据采样" width="85%">
  <p><sub>蒙特卡洛证据采样 — 从大文档中提取相关证据的三阶段启发式探索-利用策略。</sub></p>
</div>

该算法分为三个阶段：

1. **第一阶段 — 撒网（探索）：** 模糊锚定匹配结合分层随机采样。系统在识别潜在相关种子区域的同时，通过随机探测保持广泛覆盖，确保不会遗漏高价值区域。

2. **第二阶段 — 聚焦（利用）：** 以第一阶段高分种子为中心进行高斯重要性采样。采样密度集中在最有前景的区域，提取上下文并对每个片段评分。

3. **第三阶段 — 合成：** 将 Top-K 评分片段传递给 LLM，合成为连贯的兴趣区域（ROI）摘要，并附带置信度标志——使管线能够判断证据是否充分，或是否需要启用 ReAct 智能体进行更深层的自适应检索。

**核心特性：**

- **文档无关性：** 同一算法在 2 页备忘录和 500 页技术手册上同样有效，无需针对特定文档的分块启发式规则。
- **Token 高效：** 仅将最相关的区域发送给 LLM，相比全文档方案大幅降低 Token 消耗。
- **探索-利用平衡：** 随机探索防止视野盲区，重要性采样确保在关键区域深入挖掘。

### 自进化知识聚类（Knowledge Cluster）

Sirchmunk 不会在回答完查询后丢弃搜索结果。相反，每次搜索都会产生一个 **KnowledgeCluster（知识聚类）**——一个结构化、可复用的知识单元，随着使用不断变得更加智能。这正是系统具备_自进化_能力的核心机制。

#### 什么是 KnowledgeCluster？

KnowledgeCluster 是一个丰富标注的对象，完整记录了单次搜索周期的认知产出：

| 字段 | 用途 |
|:-----|:-----|
| **Evidences（证据）** | 通过蒙特卡洛采样提取的源文件片段，包含文件路径、摘要和原始文本 |
| **Content（内容）** | LLM 合成的结构化 Markdown 分析，附带引用 |
| **Patterns（模式）** | 从证据中提炼的 3–5 条设计原则或核心机制 |
| **Confidence（置信度）** | 共识评分 \[0, 1\]，指示聚类的可靠性 |
| **Queries（查询历史）** | 贡献或复用该聚类的历史查询（FIFO，最多 5 条） |
| **Hotness（热度）** | 反映查询频率和时效性的活跃度评分 |
| **Embedding（嵌入向量）** | 由累积查询生成的 384 维向量，用于语义检索 |

#### 生命周期：从创建到进化

```
 ┌─────── 新查询 ───────┐
 │                       ▼
 │     ┌───────────────────────────────┐
 │     │  阶段 0：语义复用             │──── 匹配命中 ──→ 返回缓存聚类
 │     │  （余弦相似度 ≥ 0.85）         │                  + 更新热度/查询/嵌入
 │     └──────────┬────────────────────┘
 │           未匹配
 │                ▼
 │     ┌───────────────────────────────┐
 │     │  阶段 1–3：完整搜索           │
 │     │  （关键词 → 检索 →            │
 │     │   蒙特卡洛采样 → LLM 合成）   │
 │     └──────────┬────────────────────┘
 │                ▼
 │     ┌───────────────────────────────┐
 │     │  构建新聚类                   │
 │     │  确定性 ID: C{sha256}         │
 │     └──────────┬────────────────────┘
 │                ▼
 │     ┌───────────────────────────────┐
 │     │  阶段 5：持久化               │
 │     │  嵌入查询 → DuckDB →         │
 │     │  Parquet（原子写入同步）       │
 └─────└───────────────────────────────┘
```

1. **复用检查（阶段 0）：** 在任何检索开始之前，查询会被嵌入并通过余弦相似度与所有已存储聚类进行比对。若发现高置信度匹配，系统直接返回已有聚类——完全省去 LLM 推理和搜索开销。

2. **创建（阶段 1–3）：** 当无复用匹配时，完整管线运行：关键词提取、文件检索、蒙特卡洛证据采样、LLM 合成，最终生成新的 `KnowledgeCluster`。

3. **持久化（阶段 5）：** 聚类存储在内存中的 DuckDB 表中，并定期刷写为 Parquet 文件。原子写入和基于文件修改时间的重载机制确保多进程安全。

4. **复用时进化：** 每当聚类被复用时，系统会：
   - 将新查询追加到聚类的查询历史中（FIFO，最多 5 条）
   - 提升热度（+0.1，上限 1.0）
   - 基于更新后的查询集重新计算嵌入——扩展聚类的语义覆盖范围
   - 更新版本号和时间戳

#### 核心特性

- **零成本加速：** 重复或语义相似的查询直接从缓存聚类获取答案，无需任何 LLM 推理，后续搜索几乎瞬时完成。
- **查询驱动的嵌入：** 聚类嵌入基于_查询_而非内容生成，确保检索与用户的实际提问方式对齐——而非文档的书写方式。
- **语义拓展：** 随着多样化查询复用同一聚类，其嵌入会漂移以覆盖更广的语义邻域，自然提升相关未来查询的召回率。
- **轻量级持久化：** DuckDB 内存存储 + Parquet 磁盘持久化——无需外部数据库基础设施。后台守护线程同步，可配置刷写间隔，开销极小。

---


### 数据存储

所有持久化数据存储在配置的 `SIRCHMUNK_WORK_PATH`（默认：`~/.sirchmunk/`）：

```
{SIRCHMUNK_WORK_PATH}/
  ├── .cache/
    ├── history/              # 聊天会话历史（DuckDB）
    │   └── chat_history.db
    ├── knowledge/            # 知识聚类（Parquet）
    │   └── knowledge_clusters.parquet
    └── settings/             # 用户设置（DuckDB）
        └── settings.db

```

---

## 🔗 HTTP 客户端访问（Search API）

服务器启动后（`sirchmunk serve` 或 `sirchmunk web serve`），Search API 可通过任何 HTTP 客户端访问。

<details>
<summary><b>API 端点</b></summary>

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/search` | 执行搜索查询 |
| `GET` | `/api/v1/search/status` | 检查服务器和 LLM 配置状态 |

**交互式文档：** http://localhost:8584/docs（Swagger UI）

</details>

<details>
<summary><b>cURL 示例</b></summary>

```bash
# 基础搜索（DEEP 模式）
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "认证是如何工作的？",
    "paths": ["/path/to/project"],
    "mode": "DEEP"
  }'

# 文件名搜索（快速，无需 LLM）
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "config",
    "paths": ["/path/to/project"],
    "mode": "FILENAME_ONLY"
  }'

# 完整参数示例
curl -X POST http://localhost:8584/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "数据库连接池",
    "paths": ["/path/to/project/src"],
    "mode": "DEEP",
    "max_depth": 10,
    "top_k_files": 20,
    "keyword_levels": 3,
    "include_patterns": ["*.py", "*.java"],
    "exclude_patterns": ["*test*", "*__pycache__*"],
    "return_cluster": true
  }'

# 检查服务器状态
curl http://localhost:8584/api/v1/search/status
```

</details>

<details>
<summary><b>Python 客户端示例</b></summary>

**使用 `requests`：**

```python
import requests

response = requests.post(
    "http://localhost:8584/api/v1/search",
    json={
        "query": "认证是如何工作的？",
        "paths": ["/path/to/project"],
        "mode": "DEEP"
    },
    timeout=300  # DEEP 模式可能耗时较长
)

data = response.json()
if data["success"]:
    print(data["data"]["result"])
```

**使用 `httpx`（异步）：**

```python
import httpx
import asyncio

async def search():
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            "http://localhost:8584/api/v1/search",
            json={
                "query": "查找所有 API 端点",
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
<summary><b>JavaScript 客户端示例</b></summary>

```javascript
const response = await fetch("http://localhost:8584/api/v1/search", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "认证是如何工作的？",
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
<summary><b>请求参数说明</b></summary>

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | `string` | *必填* | 搜索查询或问题 |
| `paths` | `string[]` | *必填* | 搜索的目录或文件（至少 1 个） |
| `mode` | `string` | `"DEEP"` | `DEEP` 或 `FILENAME_ONLY` |
| `max_depth` | `int` | `null` | 最大目录深度 |
| `top_k_files` | `int` | `null` | 返回的文件数量 |
| `keyword_levels` | `int` | `null` | 关键词粒度层级 |
| `include_patterns` | `string[]` | `null` | 文件 glob 匹配模式（包含） |
| `exclude_patterns` | `string[]` | `null` | 文件 glob 匹配模式（排除） |
| `return_cluster` | `bool` | `false` | 是否返回完整的 KnowledgeCluster 对象 |

> **注意：** `FILENAME_ONLY` 模式无需 LLM API Key。`DEEP` 模式需要配置 LLM。

</details>

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
    paths=["/path/to/folder", "/path/to/file.pdf"]
)
```

</details>

<details>
<summary><b>知识聚类存储在哪里？</b></summary>

知识聚类以 Parquet 格式持久化于：
```
{SIRCHMUNK_WORK_PATH}/.cache/knowledge/knowledge_clusters.parquet
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