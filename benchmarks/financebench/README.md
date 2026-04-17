# FinanceBench Benchmark

FinanceBench evaluation pipeline for **Sirchmunk AgenticSearch**.

## Overview

[FinanceBench](https://arxiv.org/abs/2311.11944) is an open-book financial QA benchmark
with **150 expert-annotated questions** across **40+ US public companies** (10-K/10-Q filings).

### Evaluation Modes

| Mode | Description |
|------|-------------|
| `singleDoc` | Each question searches only its target PDF (standard) |
| `sharedCorpus` | All questions search the full 41-PDF corpus |

### Metrics

- **3-Class Scoring**: Correct / Hallucination / Refusal (per FinanceBench paper)
- **EM / F1**: Exact Match and token-level F1 with financial value normalisation
- **Evidence Recall**: Retrieved pages vs gold evidence pages

## Prerequisites

### Step 1: Install Sirchmunk

Install Sirchmunk from the repository root so that the `sirchmunk` CLI is available:

```bash
# From repository root
pip install -e .
```

Verify the installation:

```bash
sirchmunk --version
```

### Step 2: Prepare Dataset

Download the [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
dataset and place the files under `benchmarks/financebench/data/`:

```
data/
├── financebench_open_source.jsonl   # 150 expert-annotated QA pairs
└── pdfs/                            # 41 SEC-filing PDFs (10-K / 10-Q)
    ├── 3M_2018_10K.pdf
    ├── AMCOR_2023_10K.pdf
    └── ...
```

Each PDF filename must match the `doc_name` field in the JSONL file.

### Step 3: Initialize Experiment Workspace

Initialize an isolated workspace for this experiment. This keeps the knowledge base
and cache separate from the default `~/.sirchmunk`:

```bash
cd benchmarks/financebench
sirchmunk init --work-path .work
```

This creates a `.work/` directory containing a **platform .env** file (`.work/.env`).

**Configure the platform .env** (`.work/.env`):

This file controls the LLM provider used by Sirchmunk's search engine.
You **must** set valid LLM credentials here before proceeding.

| Variable | Required | Description | Example                                             |
|----------|----------|-------------|-----------------------------------------------------|
| `LLM_API_KEY` | **Yes** | API key for the LLM provider | `sk-xxx`                                            |
| `LLM_BASE_URL` | **Yes** | LLM API endpoint | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LLM_MODEL_NAME` | **Yes** | Model name for search & QA | `qwen3.6-plus`                                      |
| `LLM_TIMEOUT` | No | Request timeout in seconds | `120`                                               |

```bash
# Edit the platform .env
vi .work/.env
```

### Step 4: Knowledge Compiling

Compile the PDF corpus into the experiment workspace so that Sirchmunk can search it:

```bash
sirchmunk compile --work-path .work --paths data/pdfs
```

> **Note:** This step parses, chunks, and indexes all PDFs.
> For FinanceBench's ~41 PDFs (10-K/10-Q filings), expect 10–30 minutes.

### Step 5: Configure Experiment

Create the **experiment .env** from the template:

```bash
cp .env.example .env.financebench
```

**Configure the experiment .env** (`.env.financebench`):

This file controls FinanceBench-specific evaluation parameters.

#### Dataset Paths

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FB_WORK_PATH` | No | Isolated workspace path | `./.work` |
| `FB_DATA_DIR` | **Yes** | Directory containing `financebench_open_source.jsonl` | `./data` |
| `FB_PDF_DIR` | **Yes** | Directory containing the 41 PDF files | `./data/pdfs` |
| `FB_OUTPUT_DIR` | No | Results output directory | `./output` |

#### Dataset Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FB_LIMIT` | No | Number of questions to evaluate (`0` = all 150) | `0` |
| `FB_SEED` | No | Random seed for reproducibility | `42` |

#### Search Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FB_MODE` | No | Search mode: `FAST` or `DEEP` | `FAST` |
| `FB_TOP_K_FILES` | No | Max files returned per search | `5` |
| `FB_MAX_TOKEN_BUDGET` | No | Token budget for search context | `128000` |
| `FB_ENABLE_DIR_SCAN` | No | Enable directory-level scanning | `true` |

#### Evaluation Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FB_EVAL_MODE` | No | `singleDoc` (per-PDF) or `sharedCorpus` (all PDFs) | `singleDoc` |
| `FB_ENABLE_LLM_JUDGE` | No | Enable LLM Judge for semantic equivalence | `true` |
| `FB_EXTRACT_ANSWER` | No | Extract short answer from verbose response | `true` |

#### Concurrency Settings

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `FB_MAX_CONCURRENT` | No | Max concurrent evaluation requests | `3` |
| `FB_REQUEST_DELAY` | No | Delay between requests in seconds | `0.5` |

**Optional LLM Override**: If you want this experiment to use a **different** LLM
than the platform config, uncomment the `LLM_*` lines in `.env.financebench`.
Otherwise, the experiment inherits LLM settings from `.work/.env`.

```bash
# Edit the experiment .env
vi .env.financebench
```

## Configuration Architecture

Configuration loads with layered inheritance (highest priority wins):

```
Priority (highest → lowest):
┌──────────────────────────────────┐
│  Command-line args               │  ← --limit N, --env <file>
├──────────────────────────────────┤
│  .env.financebench (experiment)  │  ← FB_* params + optional LLM override
├──────────────────────────────────┤
│  .work/.env (platform)           │  ← LLM_API_KEY, LLM_MODEL_NAME, etc.
├──────────────────────────────────┤
│  Environment variables           │  ← os.environ fallback
├──────────────────────────────────┤
│  Defaults                        │  ← Hard-coded in FinanceBenchConfig
└──────────────────────────────────┘
```

### What Goes Where?

| Setting | Platform `.work/.env` | Experiment `.env.financebench` |
|---------|:---------------------:|:------------------------------:|
| LLM API Key | ✅ (required) | Only if overriding |
| LLM Model | ✅ (required) | Only if overriding |
| LLM Base URL | ✅ (required) | Only if overriding |
| LLM Timeout | Optional | Only if overriding |
| PDF directory | — | ✅ (required) |
| Data directory | — | ✅ (required) |
| Output directory | — | Optional |
| Eval mode | — | Optional |
| Search mode | — | Optional |
| LLM Judge | — | Optional |
| Concurrency | — | Optional |

### 1. Run

```bash
# Run full benchmark (150 questions)
python run_benchmark.py

# Run with custom config and question limit
python run_benchmark.py --env .env.custom --limit 20
```

### 2. Analyze

```bash
# Analyze a completed run
python analyze_results.py output/results_YYYYMMDD_HHMMSS.jsonl

# Show more error cases
python analyze_results.py output/results_*.jsonl --max-errors 50
```

## Data Format

The dataset file `financebench_open_source.jsonl` contains one JSON object per line:

```json
{
  "financebench_id": "financebench_id_00001",
  "question": "What is the FY2018 capital expenditure amount for 3M?",
  "answer": "$1,577.00",
  "doc_name": "3M_2018_10K",
  "company": "3M",
  "question_type": "fact-based-w-numerical-answer",
  "question_reasoning": "retrieve",
  "evidence": [{"evidence_text": "...", "evidence_page_num": 42}]
}
```

## File Structure

```
benchmarks/financebench/
├── .env.example           # Config template (copy to .env.financebench)
├── config.py              # FinanceBenchConfig dataclass
├── data_loader.py         # Dataset + PDF corpus loader
├── evaluate.py            # EM/F1/3-class scoring + aggregation
├── runner.py              # Async batch runner (AgenticSearch)
├── run_benchmark.py       # CLI entry point
├── analyze_results.py     # Post-hoc analysis tool
├── data/
│   ├── financebench_open_source.jsonl
│   └── pdfs/              # 41 SEC-filing PDFs
├── output/                # Results + metrics (auto-created)
└── logs/                  # Run logs (auto-created)
```

## SOTA Reference

| System | Accuracy | Coverage |
|--------|----------|----------|
| Mafin 2.5 (SOTA) | 98.7% | 100% |
| Fintool | 98.0% | 66.7% |
| Quantly | 94.0% | 100% |
| GPT-4 (zero-shot) | 29.3% | 100% |

> Mafin 2.5 uses PageIndex + Agentic Vectorless RAG 3.0 architecture.
