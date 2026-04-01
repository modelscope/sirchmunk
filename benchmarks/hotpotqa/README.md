# HotpotQA Fullwiki Benchmark for Sirchmunk

End-to-end RAG evaluation on the **HotpotQA Fullwiki** setting ([Yang et al., 2018](https://arxiv.org/pdf/1809.09600)).
Tests the full **retrieval → reasoning → generation** pipeline against a global Wikipedia corpus (~5.3M article abstracts).

**Leaderboard-aligned metrics** (Yang et al., 2018 §5.2, Table 4):
- **Ans** (EM, F1): Answer exact match and token-level F1
- **Sup** (EM, F1): Supporting-fact set-level EM and F1
- **Joint** (EM, F1): Combined answer × supporting-fact metrics

Additional diagnostic metrics aligned with [LinearRAG](https://arxiv.org/pdf/2510.10114) (Zhuang et al., 2025):
Contain-Match Accuracy, GPT-Eval Accuracy, Evidence Recall.

## Data

### Questions (parquet)

Local parquet files under the dataset root: `{HOTPOT_DATASET_DIR}/{setting}/{split}*.parquet`.
Each row: `id, question, answer, type, level, supporting_facts, context`.

Dataset root is configured in `.env.hotpotqa` via `HOTPOT_DATASET_DIR` (default: `/Users/jason/work/github/sirchmunk_work/data/hotpotqa_dataset`).

```
hotpotqa_dataset/
├── fullwiki/
│   ├── validation-00000-of-00001.parquet  (7405 rows)
│   ├── train-*.parquet
│   └── test-*.parquet
└── distractor/
    ├── validation-*.parquet
    └── train-*.parquet
```

### Wiki Corpus

The fullwiki setting searches the **entire** Wikipedia corpus (first paragraphs of all articles).

**Download** (3.2GB compressed → 11GB decompressed):

```bash
wget https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
tar xjf enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -C hotpotqa_dataset/

# Decompress individual bz2 files (required for AgenticSearch)
find hotpotqa_dataset/enwiki-20171001-pages-meta-current-withlinks-abstracts -name "*.bz2" -print0 | xargs -0 -P 8 bunzip2
```

Each decompressed file contains JSON lines (one per article):
```json
{"id": "12", "title": "Anarchism", "text": ["sentence1", "sentence2", ...], ...}
```

156 subdirectories × ~100 files each = 15,517 files, ~5.3M articles total.

## Setup

```bash
pip install duckdb

cp .env.hotpotqa.example .env.hotpotqa
# Set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
```

## Usage

All experiment settings are read from **.env.hotpotqa** (unified config). Edit that file to change limit, mode, paths, etc.

```bash
# Run with default .env.hotpotqa in script directory
python run_benchmark.py

# Use a different env file (e.g. for another experiment)
python run_benchmark.py --env /path/to/.env.hotpotqa

# Resume from a previous results file (checkpoint/resume)
python run_benchmark.py --resume output/results_fullwiki_validation_*.json
```

### Checkpoint / Resume

For long-running evaluations, use `--resume` to continue from a previous run:
- Already-evaluated questions are loaded from the checkpoint file and skipped
- New results are merged with cached results
- A fresh report is generated with all samples
- Intermediate checkpoints are saved automatically during evaluation

Examples of .env.hotpotqa: set `HOTPOT_LIMIT=20` for a quick test, `HOTPOT_LIMIT=0` for full validation set, `HOTPOT_MODE=FAST`, `HOTPOT_WIKI_CORPUS_DIR=/path/to/enwiki-abstracts`, `HOTPOT_EXTRACT_ANSWER=false`, `HOTPOT_ENABLE_GPT_EVAL=false`, etc. See ".env.hotpotqa Keys" below.

## Metrics

### Leaderboard Metrics (Official HotpotQA)

Fully aligned with the official HotpotQA evaluation (Yang et al., 2018 §5.2, Table 4):

| Metric | Description |
|--------|-------------|
| **Ans EM** | Answer exact match (normalized) |
| **Ans F1** | Answer token-level F1 |
| **Sup EM** | Supporting-fact set-level exact match |
| **Sup F1** | Supporting-fact set-level F1 |
| **Joint EM** | 1 iff Ans EM = 1 AND Sup EM = 1 |
| **Joint F1** | Combined F1: P_joint = P_ans × P_sup, R_joint = R_ans × R_sup |

### Supporting Facts Prediction

AgenticSearch now extracts predicted supporting facts from retrieved wiki files:
- Each wiki file contains JSON lines (one per article)
- For every article in files read during search, we extract `(title, sent_id)` pairs
- These predictions are compared against gold supporting facts for Sup EM/F1
- Joint metrics combine answer and supporting-fact scores per the paper's definition

### Diagnostic Metrics (LinearRAG-aligned)

| Metric | Description |
|--------|-------------|
| **Contain-Match Acc** | 1 if normalized gold answer appears in normalized prediction (bidirectional) |
| **GPT-Eval Acc** | LLM judges semantic equivalence between prediction and gold |
| **Evidence Recall** | Fraction of gold SP titles found in retrieved articles |

### Efficiency

| Metric | Description |
|--------|-------------|
| **Avg Latency** | Mean AgenticSearch main-path seconds per query (excludes postprocessing) |
| **Avg Tokens** | Mean LLM tokens per query |
| **Avg Loops** | Mean ReAct iterations (hops) per query |
| **Avg Files Read** | Mean wiki corpus files opened per query |
| **Avg Titles Retr** | Mean article titles in retrieved files |
| **Avg SP Predictions** | Mean supporting-fact (title, sent_id) pairs predicted |

### Fine-grained Breakdown

- **By Type**: `bridge` (multi-hop retrieval) vs `comparison` (logic-heavy)
- **By Level**: `easy`, `medium`, `hard`

### LLM Judge

For samples where EM=0 but F1 >= threshold (default 0.3), the LLM judge
checks semantic equivalence.

## Architecture

```
config.py        — ExperimentConfig (parse-only, no hardcoded defaults)
data_loader.py   — Load parquet, validate wiki corpus
runner.py        — AgenticSearch against global wiki corpus, answer + SP extraction
evaluate.py      — Full leaderboard metrics (Ans/Sup/Joint EM/F1) + diagnostics
llm_judge.py     — LLM Judge (borderline) + GPT-Eval (all samples)
run_benchmark.py — CLI entry point, checkpoint/resume, professional report
```

## How AgenticSearch.search() Works in Fullwiki

Unlike the per-question context approach, the fullwiki setting uses a **single
global wiki corpus** as the search target for ALL questions:

```python
result = await searcher.search(
    query=question,
    paths=[wiki_corpus_dir],   # Global corpus (11GB, 15K+ files)
    mode="DEEP",               # Multi-hop retrieval
    top_k_files=5,             # LinearRAG alignment
    return_context=True,       # Capture telemetry
)
```

For each question:
1. **Search**: `rga` searches the entire wiki corpus for keyword matches
2. **Read**: AgenticSearch reads top-k matching files, extracts evidence
3. **Reason**: LLM synthesizes answer from retrieved evidence (DEEP = multi-hop)
4. **Extract**: Optional LLM call converts verbose briefing → short factoid answer
5. **Evaluate**: Compare prediction with gold, check evidence recall

This tests true end-to-end retrieval over a massive corpus — the agent must
find relevant passages from ~5.3M candidates and produce a correct answer.

## LinearRAG Comparison

This benchmark follows the experimental setup from
[LinearRAG](https://arxiv.org/pdf/2510.10114) (Zhuang et al., 2025):

| Setting | Value |
|---------|-------|
| Questions | 1000 from validation set |
| Retrieval | top-k=5 |
| Primary metrics | Contain-Match Acc, GPT-Eval Acc |
| Leaderboard metrics | Ans EM/F1, Sup EM/F1, Joint EM/F1 |

LinearRAG reports **64.30%** Contain-Acc and **66.50%** GPT-Acc on HotpotQA
(Table 1), which serves as the comparison baseline for AgenticSearch.

## Sample Report Output

```
==============================================================================
  HOTPOTQA FULLWIKI EVALUATION REPORT
==============================================================================

  Leaderboard Metrics
               N  │  Ans EM  Ans F1 │  Sup EM  Sup F1 │  Jnt EM  Jnt F1
  ──────────── ───────┼──────── ────────┼──────── ────────┼──────── ───────
  Overall        50  │  32.00   45.67 │  12.00   38.45 │   8.00   21.34

  By Question Type
               N  │  Ans EM  Ans F1 │  Sup EM  Sup F1 │  Jnt EM  Jnt F1
  ──────────── ───────┼──────── ────────┼──────── ────────┼──────── ───────
  bridge          35  │  34.29   47.12 │  14.29   40.23 │  11.43   24.56
  comparison      15  │  26.67   42.33 │   6.67   34.12 │   0.00   13.78

  Diagnostic Metrics
               N  │ Cont-Acc Ev Recall
  ──────────── ───────┼───────── ─────────
  Overall        50  │   58.00     72.34

  GPT-Eval Accuracy:   62.00%  (N=50)

  Efficiency
  ────────────────────────────────────────
    Avg latency:            8.45 s/query
    Avg tokens:             4523 /query
    Avg loops (hops):        2.3
    Avg files read:         12.4
    Avg titles retrieved:   89
    Avg SP predictions:     356
    Total search time:    422.5 s
    Total wall time:       423.5 s
    Errors:                   0 / 50
```

## Output

Results and reports are saved to `output/`:
- `results_*.json` — per-question predictions + telemetry + retrieved_titles + predicted_sp
- `report_*.json` — leaderboard metrics (Ans/Sup/Joint), diagnostic metrics, efficiency
- `checkpoint_*.json` — intermediate results for crash recovery
- `logs/benchmark_*.log` — full console output with timestamps

## .env.hotpotqa Keys

**All** config is read from `.env.hotpotqa`. `config.py` has zero hardcoded defaults.

| Key | Description |
|-----|-------------|
| **LLM** | |
| `LLM_BASE_URL` | LLM API endpoint (required) |
| `LLM_API_KEY` | API key (required) |
| `LLM_MODEL_NAME` | Model name |
| `LLM_TIMEOUT` | Request timeout (seconds) |
| **Paths** | |
| `HOTPOT_DATASET_DIR` | Dataset root (parquet + wiki corpus) |
| `HOTPOT_WIKI_CORPUS_DIRNAME` | Wiki corpus subdirectory name under dataset root |
| `HOTPOT_WIKI_CORPUS_DIR` | Override full wiki corpus path (auto-derived if unset) |
| `HOTPOT_OUTPUT_DIR` | Where to write results and reports |
| **Experiment** | |
| `HOTPOT_SETTING` | Dataset setting (fullwiki / distractor) |
| `HOTPOT_SPLIT` | Dataset split (validation / train / test) |
| `HOTPOT_LIMIT` | Max samples (0 = all) |
| `HOTPOT_SEED` | Random seed for sampling |
| `HOTPOT_MODE` | Search mode (DEEP / FAST) |
| `HOTPOT_TOP_K_FILES` | Retrieval top-k |
| `HOTPOT_MAX_TOKEN_BUDGET` | Max token budget for search |
| `HOTPOT_ENABLE_DIR_SCAN` | Enable directory scanning |
| `HOTPOT_EXTRACT_ANSWER` | Extract short factoid answer |
| **Evaluation** | |
| `HOTPOT_ENABLE_GPT_EVAL` | Enable GPT-Evaluation Accuracy |
| `HOTPOT_ENABLE_LLM_JUDGE` | Enable LLM judge |
| `HOTPOT_JUDGE_F1_THRESHOLD` | F1 threshold for judge |
| **Concurrency** | |
| `HOTPOT_MAX_CONCURRENT` | Max concurrent search requests |
| `HOTPOT_REQUEST_DELAY` | Delay (s) between requests |
| **Sirchmunk Runtime** | |
| `SIRCHMUNK_SEARCH_PATHS` | Default search paths for AgenticSearch (points to wiki corpus) |
| `SIRCHMUNK_WORK_PATH` | Sirchmunk working directory |
| `SIRCHMUNK_VERBOSE` | Verbose logging |
