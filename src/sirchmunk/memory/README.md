# Retrieval Memory — 自进化检索记忆系统

## 设计方法论

### 核心理念：检索即学习

传统 RAG 系统对每次检索一视同仁——相同的关键词策略、相同的路径遍历、相同的参数配置。Retrieval Memory 通过**闭环反馈学习**打破这一局限：

```
查询 → 检索 → 反馈 → 学习 → 记忆 → 优化下次检索
  ↑                                       │
  └───────────────────────────────────────┘
```

每次检索完成后，系统自动收集隐式信号（是否找到答案、延迟、Token 消耗、文件命中率等），分发给五个记忆层进行增量学习。经过足够样本积累后，记忆开始反向影响检索策略——越用越准、越用越快。

### 分层记忆架构

设计遵循**认知科学中的多系统记忆模型**：短期工作记忆（SearchContext）负责单次会话状态，长期记忆（RetrievalMemory）负责跨会话的模式积累和知识沉淀。

长期记忆进一步分为五个职责独立、存储独立的层：

```
RetrievalMemory/
├── PatternMemory        查询模式 → 策略映射        (JSON)
├── CorpusMemory         语料库认知：实体-路径索引    (DuckDB + JSON)
├── PathMemory           路径热度与效用统计          (DuckDB)
├── FailureMemory        失败模式：噪声词、死路径     (DuckDB)
└── FeedbackMemory       反馈信号存储               (DuckDB)
```

分层原则：

| 原则 | 说明 |
|------|------|
| **单一职责** | 每层只负责一种认知维度，互不依赖 |
| **存储异构** | JSON 适合人可读、Schema 多变的小数据；DuckDB 适合大规模、需聚合查询的结构化数据 |
| **零耦合** | 任一层失效不影响其他层，也不阻塞主检索管线 |
| **置信度衰减** | 所有记忆都有时间衰减机制，防止过时知识污染决策 |

### 各层设计思路

#### 1. PatternMemory — 查询模式记忆

**方法论**：将查询抽象为特征签名（类型 × 复杂度 × 实体类型），学习每种签名对应的最优检索参数。类似于强化学习中的 state → action 映射。

- **查询分类**：零 LLM 开销的启发式分类（支持中英文），提取 `query_type`（factual / comparison / bridge / definition / procedural）、`complexity`（simple / moderate / complex）、`entity_types`
- **策略建议**：当某模式积累 ≥3 个样本且成功率 ≥40% 时，返回 `StrategyHint`（推荐 mode、top_k_files、max_loops 等）
- **统计更新**：使用指数移动平均（EMA, α=0.3）更新延迟和 Token 统计
- **推理链模板**：存储成功的 ReAct 推理链，作为后续相似查询的 hint 注入

#### 2. CorpusMemory — 语料库认知

**方法论**：构建"实体在哪些文件中出现过"的倒排索引，以及"哪些关键词可以互相替代"的语义桥接表。从成功检索中提取实体-路径映射关系，从关键词切换中学习同义/别名扩展。

- **实体-路径索引**（DuckDB）：`entity → [(path, confidence, hit_count)]`，带置信度衰减
- **语义桥接**（JSON）：`term → [SemanticExpansion(target, relation, confidence)]`，用于关键词自动扩展
- **扩展策略**：仅当 `confidence ≥ 0.4` 且 `hit_count ≥ 2` 时才激活扩展，权重 = 原始权重 × 置信度 × 0.7

#### 3. PathMemory — 路径热度记忆

**方法论**：对每个文件路径维护"热度分数"，反映其在历史检索中的频率、效用率和新鲜度。在 BM25 重排阶段可作为 boost factor。

- **热度公式**：`hot_score = useful_ratio × log₂(1 + total_retrievals) × recency_factor`
- **新鲜度因子**：`recency = max(0, 1 - days_since_last_useful / 90)`，90 天无用则衰减到 0

#### 4. FailureMemory — 失败模式记忆

**方法论**：记住什么不该做——比"记住什么该做"更高效。三个子模块分别追踪噪声关键词、死路径和失败策略组合。

- **噪声关键词**：EMA 跟踪 `avg_files_found` 和 `avg_useful_ratio`，当 useful_ratio ≤ 10% 且样本 ≥ 3 时标记为 `skip`
- **死路径**：当某路径被检索 ≥ 5 次但从未有用时，从候选集中过滤
- **失败策略**：`(pattern_id, params_hash) → failure_count`，避免重复尝试已知失败的参数组合

#### 5. FeedbackMemory — 反馈信号存储

**方法论**：作为闭环的"感知层"，收集每次检索的 `FeedbackSignal`（隐式信号如 answer_found、latency、tokens，显式信号如 user_verdict、EM/F1 分数）。本层只负责存储，信号分发由 Manager 负责。

### 与检索管线的融合

Memory 在 `AgenticSearch.search()` 管线中的介入点：

| 阶段 | 介入操作 | 延迟影响 |
|------|---------|---------|
| `search()` 入口 | `suggest_strategy(query)` — 查询 PatternMemory | ~0 ms（内存字典查找） |
| Phase 1 后 | `expand_keywords(query_keywords)` — 语义桥接扩展 | ~0 ms（内存字典查找） |
| Phase 3 前 | `get_entity_paths(entities)` — 实体路径补充候选 | ~1 ms（DuckDB 查询） |
| Phase 3 中 | `filter_dead_paths(merged_files)` — 过滤死路径 | ~1 ms（DuckDB 查询） |
| Phase 5 后 | `record_feedback(signal)` — 异步记录反馈 | 0 ms（fire-and-forget） |

**关键设计约束**：

- 所有查询路径（Lookup API）是**同步**的，延迟 < 2ms
- 所有写入路径（Record API）是**异步**的，通过 `asyncio.ensure_future` + `run_in_executor` 在后台线程执行，不阻塞搜索响应
- 所有操作都有 try/except 保护，任何异常只 log 不 raise，保证主管线不受影响

### 记忆卫生与治理

| 机制 | 实现 |
|------|------|
| **置信度衰减** | 每层实现 `decay()` 方法，对超期未命中的条目乘以衰减系数 |
| **容量上限** | 每层实现 `cleanup()` 方法，按分数/命中数排序淘汰低价值条目 |
| **自动维护** | Manager 每 50 次反馈自动触发一轮 `decay_all()` + `cleanup_all()` |
| **写入节流** | PatternMemory 使用 dirty flag + 最小写入间隔（5s），避免高频磁盘写入 |
| **原子写入** | JSON 文件通过 temp file + `os.replace` 实现原子写入，防止写入中断导致数据损坏 |
| **优雅降级** | 任一层初始化失败时，其他层继续工作；Manager 的 `close()` 确保最终一致性 |

### 进化曲线预期

```
           准确率 / 速度
               ↑
    成熟期     │           ┌──────────────────
               │          ╱
    加速期     │        ╱
               │      ╱
    冷启动     │────╱
               └──────────────────────────────→  查询次数
               0     50    200    500   1000
```

- **冷启动期** (0–50 queries)：记忆为空，所有查询使用默认参数，但每次检索都在积累信号
- **加速期** (50–500 queries)：高频模式开始形成，实体索引覆盖核心语料，噪声词被识别
- **成熟期** (500+ queries)：策略推荐稳定，语义桥接覆盖常见同义词，热路径优先排序

## 存储布局

```
{work_path}/memory/
├── pattern_memory/
│   ├── query_patterns.json        # 查询模式 → 策略映射
│   └── reasoning_chains.json      # 推理链模板
├── semantic_bridge.json           # 语义桥接（同义词/别名扩展）
├── corpus.duckdb                  # entity_index, path_stats,
│                                  # noise_keywords, dead_paths,
│                                  # failed_strategies
└── feedback.duckdb                # signals（反馈信号时序表）
```

## 快速开始

```python
from sirchmunk.memory import RetrievalMemory, FeedbackSignal

# 初始化（通常由 AgenticSearch 自动完成）
memory = RetrievalMemory(work_path="~/.sirchmunk")

# 查询阶段：策略建议
hint = memory.suggest_strategy("Who invented the telephone?")
if hint and hint.confidence >= 0.5:
    print(f"Suggested mode: {hint.mode}, top_k: {hint.top_k_files}")

# 查询阶段：关键词扩展
keywords = {"telephone": 0.9, "inventor": 0.7}
expanded = memory.expand_keywords(keywords)

# 检索阶段：实体路径 + 死路径过滤
entity_paths = memory.get_entity_paths(["Bell", "Edison"])
safe_paths = memory.filter_dead_paths(candidate_paths)

# 反馈阶段：记录检索结果
signal = FeedbackSignal(
    query="Who invented the telephone?",
    mode="DEEP", answer_found=True,
    files_read=["/data/bell.txt"], keywords_used=["Bell", "telephone"],
)
await memory.record_feedback(signal)

# 维护
memory.decay_all()
memory.cleanup_all()
print(memory.stats())
```

## 扩展新的记忆层

实现 `MemoryStore` 抽象基类，注册到 `RetrievalMemory._stores` 列表即可：

```python
from sirchmunk.memory.base import MemoryStore

class MyCustomMemory(MemoryStore):
    @property
    def name(self) -> str:
        return "MyCustomMemory"

    def initialize(self) -> None: ...
    def decay(self, now=None) -> int: ...
    def cleanup(self, max_entries=None) -> int: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...
```
