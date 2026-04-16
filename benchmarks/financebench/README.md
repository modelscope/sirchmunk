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

## Quick Start

### 1. Setup

```bash
cd benchmarks/financebench

# Copy and edit the config file
cp .env.example .env.financebench
# Edit .env.financebench — set your LLM_API_KEY at minimum

# Download FinanceBench data
# Place financebench_open_source.jsonl in ./data/
# Place PDF corpus (41 files) in ./data/pdfs/
```

### 2. Run

```bash
# Run full benchmark (150 questions)
python run_benchmark.py

# Run with custom config and question limit
python run_benchmark.py --env .env.financebench --limit 20
```

### 3. Analyze

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
