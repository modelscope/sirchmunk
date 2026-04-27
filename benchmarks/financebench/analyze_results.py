"""Analyze FinanceBench benchmark results.

Read a JSONL results file produced by ``run_benchmark.py`` and print a
comprehensive analysis report including per-type breakdowns, per-company
accuracy, error cases, and a SOTA comparison table.

Usage:
    python analyze_results.py output/results_YYYYMMDD_HHMMSS.jsonl
    python analyze_results.py output/results_*.jsonl --max-errors 30
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluate import compute_metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL results file into a list of dicts.

    Args:
        path: Path to a ``.jsonl`` file where each line is a JSON object.

    Returns:
        List of result dicts.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If a line contains invalid JSON.
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found — {path}", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"WARNING: skipping malformed line {lineno}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def print_breakdown(title: str, breakdown: Dict[str, Dict[str, Any]]) -> None:
    """Pretty-print a metrics breakdown table.

    Args:
        title: Section header text.
        breakdown: ``{group_name: {accuracy, hallucination_rate, ...}}``.
    """
    print(f"\n=== Breakdown by {title} ===\n")

    # Determine if judge data is available
    has_judge = any(m.get("llm_judge_accuracy") is not None for m in breakdown.values())

    if has_judge:
        header = f"  {'Group':<30} {'Acc%':>6} {'Hallu%':>7} {'Refuse%':>8} {'Judge%':>7} {'N':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for group, m in sorted(breakdown.items(), key=lambda kv: -kv[1].get("accuracy", 0)):
            acc = m.get("accuracy", 0)
            hal = m.get("hallucination_rate", 0)
            ref = m.get("refusal_rate", 0)
            n = m.get("n", 0)
            jdg = m.get("llm_judge_accuracy")
            jdg_str = f"{jdg:>6.1f}" if jdg is not None else "   N/A"
            print(f"  {group:<30} {acc:>5.1f} {hal:>7.1f} {ref:>7.1f} {jdg_str} {n:>4}")
    else:
        header = f"  {'Group':<30} {'Acc%':>6} {'Hallu%':>7} {'Refuse%':>8} {'N':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for group, m in sorted(breakdown.items(), key=lambda kv: -kv[1].get("accuracy", 0)):
            acc = m.get("accuracy", 0)
            hal = m.get("hallucination_rate", 0)
            ref = m.get("refusal_rate", 0)
            n = m.get("n", 0)
            print(f"  {group:<30} {acc:>5.1f} {hal:>7.1f} {ref:>7.1f} {n:>4}")


def _compute_company_breakdown(
    results: List[Dict[str, Any]],
) -> List[Tuple[str, float, int, int, int]]:
    """Group results by company and return sorted by accuracy ascending.

    Returns:
        List of ``(company, accuracy, correct, total, halluc)`` tuples,
        sorted by accuracy ascending (worst first).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        company = r.get("company", "unknown") or "unknown"
        groups[company].append(r)

    rows: list[tuple[str, float, int, int, int]] = []
    for company, items in groups.items():
        n = len(items)
        correct = sum(1 for r in items if r.get("classification") == "correct")
        halluc = sum(1 for r in items if r.get("classification") == "hallucination")
        acc = (correct / n * 100) if n else 0.0
        rows.append((company, acc, correct, n, halluc))

    rows.sort(key=lambda x: x[1])  # worst first
    return rows


def print_company_breakdown(results: List[Dict[str, Any]], top_n: int = 10) -> None:
    """Print per-company accuracy table, showing worst *top_n* companies.

    Args:
        results: List of per-question result dicts.
        top_n: Number of worst-performing companies to display.
    """
    rows = _compute_company_breakdown(results)
    if not rows:
        return

    print(f"\n=== Worst {top_n} Companies by Accuracy ===\n")
    header = f"  {'Company':<40} {'Acc%':>6} {'Correct':>8} {'Hallu':>6} {'N':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for company, acc, correct, n, halluc in rows[:top_n]:
        print(f"  {company:<40} {acc:>5.1f} {correct:>8} {halluc:>6} {n:>4}")


def print_error_cases(results: List[Dict[str, Any]], max_show: int = 20) -> None:
    """Print detailed listing of error cases (hallucination + refusal).

    Args:
        results: List of per-question result dicts.
        max_show: Maximum number of error cases to display.
    """
    errors = [r for r in results if r.get("classification") != "correct"]
    if not errors:
        print("\n=== Error Cases ===\n  None — perfect score!")
        return

    print(f"\n=== Error Cases ({len(errors)} total, showing up to {max_show}) ===\n")

    for i, r in enumerate(errors[:max_show], 1):
        fb_id = r.get("financebench_id", "?")
        cls = r.get("classification", "?")
        question = r.get("question", "")[:100]
        pred = r.get("prediction", "")[:80]
        gold = r.get("gold_answer", "")[:80]
        company = r.get("company", "")
        em = r.get("em", False)
        f1 = r.get("f1", 0.0)

        print(f"  [{i:>2}] {fb_id}  [{cls.upper()}]")
        print(f"       Company:    {company}")
        print(f"       Question:   {question}{'...' if len(r.get('question', '')) > 100 else ''}")
        print(f"       Predicted:  {pred}{'...' if len(r.get('prediction', '')) > 80 else ''}")
        print(f"       Gold:       {gold}{'...' if len(r.get('gold_answer', '')) > 80 else ''}")
        print(f"       EM={em}  F1={f1:.3f}")
        if r.get("error"):
            print(f"       Error:      {r['error'][:120]}")
        print()

    if len(errors) > max_show:
        print(f"  ... and {len(errors) - max_show} more error(s) not shown.\n")


def print_comparison_with_sota(metrics: Dict[str, Any]) -> None:
    """Compare with published SOTA results on FinanceBench.

    Reference baselines from the FinanceBench leaderboard and recent papers.
    """
    print("\n=== Comparison with SOTA ===\n")
    header = f"  {'System':<30} {'Accuracy':>10} {'Coverage':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'Mafin 2.5 (SOTA)':<30} {'98.7%':>10} {'100%':>10}")
    print(f"  {'Fintool':<30} {'98.0%':>10} {'66.7%':>10}")
    print(f"  {'Quantly':<30} {'94.0%':>10} {'100%':>10}")
    print(f"  {'GPT-4 (zero-shot)':<30} {'29.3%':>10} {'100%':>10}")

    acc = metrics.get("accuracy", 0)
    n = metrics.get("n", 0)
    coverage = min(100.0, n / 150.0 * 100)
    print(f"  {'Sirchmunk (This Run)':<30} {f'{acc:.1f}%':>10} {f'{coverage:.0f}%':>10}")

    # Show Judge Accuracy in SOTA table if available
    judge_acc = metrics.get("llm_judge_accuracy")
    if judge_acc is not None:
        print(f"  {'Sirchmunk (Judge Acc)':<30} {f'{judge_acc:.1f}%':>10} {f'{coverage:.0f}%':>10}")

    print(f"\n  (This run evaluated {n} questions)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and generate a full analysis report."""
    parser = argparse.ArgumentParser(
        description="Analyze FinanceBench benchmark results from a JSONL file",
    )
    parser.add_argument(
        "results_file",
        help="Path to the results JSONL file produced by run_benchmark.py",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of error cases to display (default: 20)",
    )
    parser.add_argument(
        "--top-companies",
        type=int,
        default=10,
        help="Number of worst-performing companies to show (default: 10)",
    )
    args = parser.parse_args()

    # Load
    results = load_results(args.results_file)
    if not results:
        print("ERROR: no results loaded.", file=sys.stderr)
        sys.exit(1)

    # Compute metrics
    metrics = compute_metrics(results)

    # --- Overall summary ---
    n = metrics.get("n", 0)
    acc = metrics.get("accuracy", 0)
    hallu = metrics.get("hallucination_rate", 0)
    refuse = metrics.get("refusal_rate", 0)
    avg_em = metrics.get("avg_em", 0)
    avg_f1 = metrics.get("avg_f1", 0)
    ev_recall = metrics.get("evidence_recall")
    avg_latency = metrics.get("avg_latency", 0)

    print(f"\n{'=' * 60}")
    print(f"  FinanceBench Analysis ({n} questions)")
    print(f"{'=' * 60}")
    print(f"  Accuracy:           {acc:.1f}%")
    print(f"  Hallucination Rate: {hallu:.1f}%")
    print(f"  Refusal Rate:       {refuse:.1f}%")
    print(f"  Avg EM:             {avg_em:.3f}")
    print(f"  Avg F1:             {avg_f1:.3f}")
    if metrics.get("avg_evidence_recall") is not None:
        print(f"  Evidence Recall:    {metrics['avg_evidence_recall']:.3f}")
    else:
        print(f"  Evidence Recall:    N/A (page-level telemetry unavailable)")
    print(f"  Avg Latency:        {avg_latency:.1f}s")

    # LLM Judge independent metrics
    if metrics.get("llm_judge_accuracy") is not None:
        print(f"\n  --- LLM Judge (Independent Evaluation) ---")
        print(f"  Judge Accuracy:    {metrics['llm_judge_accuracy']:.1f}%")
        print(f"  Judge Correct:     {metrics['llm_judge_correct']}/{metrics['llm_judge_count']}")

    # --- Breakdowns ---
    if "by_question_type" in metrics:
        print_breakdown("Question Type", metrics["by_question_type"])

    if "by_question_reasoning" in metrics:
        print_breakdown("Question Reasoning", metrics["by_question_reasoning"])

    # --- Per-company breakdown (worst performers) ---
    print_company_breakdown(results, top_n=args.top_companies)

    # --- Error cases ---
    print_error_cases(results, max_show=args.max_errors)

    # --- Judge-Rule Discrepancies ---
    discrepancies = [r for r in results
                     if r.get("llm_judge_correct") is not None
                     and r.get("classification") != "correct"
                     and r.get("llm_judge_correct") is True]
    if discrepancies:
        print(f"\n=== Judge-Rule Discrepancies ({len(discrepancies)} cases) ===")
        print("  (Cases where LLM Judge says correct but EM/F1 says wrong)")
        for r in discrepancies[:10]:
            print(f"  {r.get('financebench_id', 'N/A')}: pred='{r.get('prediction', '')[:50]}' gold='{r.get('gold_answer', '')[:50]}'")
            print(f"    classification={r.get('classification')}, judge_reasoning={r.get('llm_judge_reasoning', '')[:80]}")
        if len(discrepancies) > 10:
            print(f"  ... and {len(discrepancies) - 10} more discrepancy(ies) not shown.")

    # --- SOTA comparison ---
    print_comparison_with_sota(metrics)

    print(f"\n{'=' * 60}")
    print(f"  Source: {args.results_file}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
