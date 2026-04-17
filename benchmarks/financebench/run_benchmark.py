"""FinanceBench benchmark entry point.

Usage:
    cd benchmarks/financebench
    python run_benchmark.py [--env .env.financebench] [--limit N]

Examples:
    # Run all 150 questions with default config
    python run_benchmark.py

    # Run a quick sanity check with 10 questions
    python run_benchmark.py --limit 10

    # Use a custom .env file
    python run_benchmark.py --env .env.custom --limit 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from config import FinanceBenchConfig
from data_loader import FinanceBenchLoader
from evaluate import compute_metrics
from runner import run_batch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(output_dir: str) -> str:
    """Configure logging to file + console.

    Creates a timestamped log file under ``logs/`` (relative to *output_dir*'s
    parent, i.e. the benchmark root directory).

    Returns:
        Absolute path to the log file.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"benchmark_{ts}.log"

    root_logger = logging.getLogger("financebench")
    root_logger.setLevel(logging.DEBUG)

    # File handler – DEBUG level, full detail
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Console handler – INFO level, concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    )

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

    return str(log_path.resolve())


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def _print_summary(
    results: List[dict],
    metrics: dict,
    total_time: float,
    results_path: Path,
    metrics_path: Path,
    log_path: str,
) -> None:
    """Print a human-readable run summary to stdout."""
    n = len(results)
    acc = metrics.get("accuracy", 0)
    cov = metrics.get("coverage", 0)
    avg_latency = metrics.get("avg_latency", 0)

    token_usage = metrics.get("token_usage", {})
    total_tokens = token_usage.get("total_tokens", 0)
    search_tokens = token_usage.get("search_tokens", 0)
    judge_tokens = token_usage.get("judge_tokens", 0)
    avg_tokens_q = token_usage.get("avg_tokens_per_question", 0)

    print("\n" + "=" * 60)
    print(f"FinanceBench Results ({n} questions)")
    print("=" * 60)
    print(f"  Accuracy (Judge):   {acc:.1f}%")
    print(f"  Coverage (Judge):   {cov:.1f}%")
    print(f"  Avg Latency:        {avg_latency:.1f}s")
    print(f"  Total Time:         {total_time:.1f}s")

    print(f"\n  --- Token Usage ---")
    print(f"  Total Tokens:      {total_tokens:>,}")
    print(f"  Search Tokens:     {search_tokens:>,}")
    print(f"  Judge Tokens:      {judge_tokens:>,}")
    print(f"  Avg per Question:  {avg_tokens_q:>,.0f}")

    print(f"\n  Results:  {results_path}")
    print(f"  Metrics:  {metrics_path}")
    print(f"  Log:      {log_path}")

    # Breakdown by question_type
    by_qt = metrics.get("by_question_type")
    if by_qt:
        print(f"\n  {'Question Type':<28} {'Acc%':>6} {'Cover%':>7} {'N':>5}")
        print("  " + "-" * 48)
        for qt, m in sorted(by_qt.items()):
            qt_acc = m.get("accuracy", 0)
            qt_cov = m.get("coverage", 0)
            qt_n = m.get("n", 0)
            print(f"  {qt:<28} {qt_acc:>5.1f} {qt_cov:>7.1f} {qt_n:>5}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, run the benchmark, and save results."""
    parser = argparse.ArgumentParser(
        description="Run FinanceBench benchmark against Sirchmunk AgenticSearch",
    )
    parser.add_argument(
        "--env",
        default=".env.financebench",
        help="Path to .env config file (default: .env.financebench)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override FB_LIMIT — number of questions to evaluate",
    )
    args = parser.parse_args()

    # 1. Load config
    cfg = FinanceBenchConfig.from_env(args.env)
    if args.limit is not None:
        cfg.limit = args.limit

    # 2. Setup logging
    log_path = setup_logging(cfg.output_dir)
    logger = logging.getLogger("financebench")

    # Print config source info
    work_env = Path(cfg.work_path) / ".env"
    logger.info("=" * 50)
    logger.info("FinanceBench Configuration")
    logger.info("=" * 50)
    logger.info("  Experiment env : %s", args.env)
    logger.info("  Platform env   : %s (%s)", work_env, "found" if work_env.exists() else "not found")
    logger.info("  Work path      : %s", Path(cfg.work_path).resolve())
    logger.info("  LLM            : %s @ %s", cfg.llm_model, cfg.llm_base_url)
    logger.info("  Eval mode      : %s", cfg.eval_mode)
    logger.info("  Search mode    : %s, Top-K: %d", cfg.mode, cfg.top_k_files)
    logger.info("  LLM Judge      : %s", "enabled" if cfg.enable_llm_judge else "disabled")
    logger.info("=" * 50)

    # 3. Load data
    loader = FinanceBenchLoader(cfg.data_dir, cfg.pdf_dir)
    questions = loader.load_questions()
    logger.info("Loaded %d questions from %s", len(questions), cfg.data_dir)

    # 4. Validate corpus
    found, missing = loader.validate_corpus(questions)
    logger.info("PDF corpus: %d found, %d missing", found, len(missing))
    if missing:
        preview = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        logger.warning("Missing PDFs: %s%s", preview, suffix)

    # 5. Apply limit / seed
    if cfg.limit > 0 and cfg.limit < len(questions):
        random.seed(cfg.seed)
        questions = random.sample(questions, cfg.limit)
        logger.info("Sampled %d questions (seed=%d)", len(questions), cfg.seed)

    # 6. Print run config
    logger.info(
        "Config: mode=%s, eval_mode=%s, llm_judge=%s, concurrent=%d, model=%s",
        cfg.mode,
        cfg.eval_mode,
        cfg.enable_llm_judge,
        cfg.max_concurrent,
        cfg.llm_model,
    )

    # 7. Run benchmark
    t0 = time.time()
    results = asyncio.run(run_batch(questions, cfg))
    total_time = time.time() - t0

    # 8. Compute metrics
    metrics = compute_metrics(results)
    metrics["total_time_seconds"] = round(total_time, 2)
    metrics["num_questions"] = len(questions)
    metrics["config"] = {
        "mode": cfg.mode,
        "eval_mode": cfg.eval_mode,
        "model": cfg.llm_model,
        "top_k_files": cfg.top_k_files,
    }

    # 9. Save results (JSONL) + metrics (JSON)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"results_{ts}.jsonl"
    metrics_path = out_dir / f"metrics_{ts}.json"

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", results_path)
    logger.info("Metrics saved to %s", metrics_path)

    # 10. Print summary
    _print_summary(results, metrics, total_time, results_path, metrics_path, log_path)


if __name__ == "__main__":
    main()
