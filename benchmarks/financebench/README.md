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

### 1. Install Sirchmunk

Make sure Sirchmunk is installed and accessible:

```bash
pip install -e .
```

### 2. Prepare Corpus

Download the FinanceBench dataset (PDF files and JSONL) and place them in the appropriate directory.
Update the paths in your `.env.financebench`:

- `FB_PDF_DIR` — path to the directory containing the 10-K/10-Q PDF files
- `FB_QUESTIONS_FILE` — path to `financebench_open_source.jsonl`

### 3. Initialize Workspace

Initialize the Sirchmunk workspace with an experiment-isolated work path:

```bash
cd benchmarks/financebench
sirchmunk init --work-path ./.work
```

This creates a `.work/` directory under the experiment folder, keeping knowledge base
and cache isolated from the default `~/.sirchmunk`.

### 4. Compile Knowledge Base

Compile the PDF corpus into the experiment workspace:

```bash
sirchmunk compile --work-path ./.work --paths <your_pdf_dir>
```

> **Note:** The compile step may take some time depending on the corpus size.
> For FinanceBench's ~41 PDFs (10-K/10-Q filings), expect 10-30 minutes.

### 5. Configure Environment

```bash
cp .env.example .env.financebench
# Edit .env.financebench with your API keys and paths
```

## Quick Start

### Configuration Priority

Configuration loads in this order (later overrides earlier):

1. **Dataclass defaults** — hard-coded in `FinanceBenchConfig`
2. **Platform .env** — `.work/.env` (created by `sirchmunk init`)
3. **Experiment .env** — `.env.financebench`
4. **Command-line** — `--limit N`, `--env <file>`

To reuse platform LLM config, leave `LLM_*` commented in `.env.financebench`.
To override, uncomment and set different values.

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
