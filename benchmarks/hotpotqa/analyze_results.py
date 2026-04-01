#!/usr/bin/env python3
"""Analyze HotpotQA results files: per-sample details, worst performers, sp_em stats, elapsed time."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import get_config
from data_loader import load_samples
from evaluate import evaluate_predictions


def trunc(s: str, n: int = 60) -> str:
    return (s[:n] + "…") if len(s) > n else s


def _load_results(results_path: Path) -> list:
    """Load results from JSONL or legacy JSON format."""
    suffix = results_path.suffix.lower()
    if suffix == ".jsonl":
        results = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    with open(results_path) as f:
        return json.load(f)


def analyze_file(results_path: Path, samples: list, label: str) -> dict:
    results = _load_results(results_path)

    n_total = len(results)
    n_errors = sum(1 for r in results if r.get("error"))
    ok_results = [r for r in results if not r.get("error")]

    sample_map = {s["_id"]: s for s in samples}
    metrics = evaluate_predictions(ok_results, samples)
    per_sample = metrics["per_sample"]

    # Build id->per_sample map (per_sample uses sample order, may not match results order)
    ps_by_id = {s["_id"]: s for s in per_sample}

    rows = []
    for r in results:
        qid = r["_id"]
        ps = ps_by_id.get(qid)
        telemetry = r.get("telemetry", {})
        row = {
            "_id": qid,
            "question": trunc(r.get("question", ""), 60),
            "prediction": trunc(r.get("prediction", ""), 60),
            "gold_answer": sample_map.get(qid, {}).get("answer", r.get("gold_answer", "")),
            "ans_em": ps["ans_em"] if ps else None,
            "ans_f1": round(ps["ans_f1"], 4) if ps else None,
            "sp_em": ps["sp_em"] if ps else None,
            "sp_f1": round(ps["sp_f1"], 4) if ps else None,
            "elapsed": r.get("telemetry", {}).get("stage_seconds", {}).get("search", r.get("elapsed", 0)),
            "loops": telemetry.get("loop_count"),
            "total_tokens": telemetry.get("total_tokens"),
            "error": r.get("error"),
        }
        rows.append(row)

    # Worst performers: ans_em=0 and ans_f1 < 0.3
    worst = [r for r in rows if r["ans_em"] == 0 and r["ans_f1"] is not None and r["ans_f1"] < 0.3]

    # sp_em=0 stats
    sp_em_zero = [r for r in rows if r["sp_em"] == 0 and r["sp_f1"] is not None]
    sp_f1_for_sp_em_zero = [r["sp_f1"] for r in sp_em_zero]
    pct_sp_em_zero = 100 * len(sp_em_zero) / len(rows) if rows else 0
    avg_sp_f1_sp_em_zero = sum(sp_f1_for_sp_em_zero) / len(sp_f1_for_sp_em_zero) if sp_f1_for_sp_em_zero else 0

    # Elapsed time distribution
    elapsed_vals = [r["elapsed"] for r in rows if r["elapsed"] is not None]
    elapsed_vals.sort()
    n_el = len(elapsed_vals)
    min_el = min(elapsed_vals) if elapsed_vals else None
    max_el = max(elapsed_vals) if elapsed_vals else None
    med_el = elapsed_vals[n_el // 2] if elapsed_vals else None
    # Outliers: > 1.5 * IQR above Q3
    q1 = elapsed_vals[n_el // 4] if n_el >= 4 else None
    q3 = elapsed_vals[3 * n_el // 4] if n_el >= 4 else None
    iqr = (q3 - q1) if (q1 is not None and q3 is not None) else None
    outlier_thresh = q3 + 1.5 * iqr if iqr else None
    outliers = [e for e in elapsed_vals if outlier_thresh and e > outlier_thresh] if elapsed_vals else []

    return {
        "label": label,
        "n_total": n_total,
        "n_errors": n_errors,
        "rows": rows,
        "worst": worst,
        "sp_em_zero_count": len(sp_em_zero),
        "pct_sp_em_zero": pct_sp_em_zero,
        "avg_sp_f1_sp_em_zero": avg_sp_f1_sp_em_zero,
        "sp_f1_for_sp_em_zero": sp_f1_for_sp_em_zero,
        "elapsed_min": min_el,
        "elapsed_max": max_el,
        "elapsed_median": med_el,
        "elapsed_outliers": outliers,
    }


def main():
    cfg = get_config()
    samples = load_samples(cfg)

    file1 = Path(__file__).parent / "output" / "results_fullwiki_validation_DEEP_qwen3.5-plus_20q_20260317_125859.json"
    file2 = Path(__file__).parent / "output" / "results_fullwiki_validation_DEEP_qwen3.5-plus_20q_20260317_124732.json"

    a1 = analyze_file(file1, samples, "enable_thinking=false (latest)")
    a2 = analyze_file(file2, samples, "enable_thinking=true (second latest)")

    for a in [a1, a2]:
        print("\n" + "=" * 100)
        print(f"  {a['label']}")
        print("=" * 100)
        print(f"\nTotal samples: {a['n_total']}  |  Errors: {a['n_errors']}")

        print("\n--- Per-sample table ---")
        print(f"{'_id':<26} {'question (60ch)':<62} {'pred (60ch)':<62} {'gold':<20} {'ans_em':<6} {'ans_f1':<8} {'sp_em':<6} {'sp_f1':<8} {'elapsed':<8} {'loops':<6} {'tokens':<10} {'error'}")
        print("-" * 260)
        for r in a["rows"]:
            err = trunc(str(r["error"]), 20) if r["error"] else ""
            print(f"{r['_id']:<26} {r['question']:<62} {r['prediction']:<62} {str(r['gold_answer']):<20} {str(r['ans_em']):<6} {str(r['ans_f1']):<8} {str(r['sp_em']):<6} {str(r['sp_f1']):<8} {str(r['elapsed']):<8} {str(r['loops']):<6} {str(r['total_tokens']):<10} {err}")

        print("\n--- Worst performers (ans_em=0, ans_f1<0.3) ---")
        for w in a["worst"]:
            print(f"  _id={w['_id']}  q: {w['question']}  pred: {w['prediction']}  gold: {w['gold_answer']}  ans_f1={w['ans_f1']}")

        print(f"\n--- sp_em=0 stats ---")
        print(f"  Percentage with sp_em=0: {a['pct_sp_em_zero']:.1f}% ({a['sp_em_zero_count']}/{a['n_total']})")
        print(f"  Typical sp_f1 for those: avg={a['avg_sp_f1_sp_em_zero']:.4f}  values={[round(x, 4) for x in a['sp_f1_for_sp_em_zero']]}")

        print(f"\n--- Elapsed time distribution ---")
        print(f"  min={a['elapsed_min']}  max={a['elapsed_max']}  median={a['elapsed_median']}")
        if a["elapsed_outliers"]:
            print(f"  Outliers (>{a['elapsed_median'] and a['elapsed_median'] * 2 or 'N/A'}): {a['elapsed_outliers']}")

    print("\n" + "=" * 100)
    print("  SUMMARY COMPARISON")
    print("=" * 100)
    print(f"\n{'Metric':<30} {'enable_thinking=false':<25} {'enable_thinking=true':<25}")
    print("-" * 80)
    print(f"{'Total samples':<30} {a1['n_total']:<25} {a2['n_total']:<25}")
    print(f"{'Errors':<30} {a1['n_errors']:<25} {a2['n_errors']:<25}")
    print(f"{'sp_em=0 %':<30} {a1['pct_sp_em_zero']:.1f}%{'':<20} {a2['pct_sp_em_zero']:.1f}%")
    print(f"{'Worst performers count':<30} {len(a1['worst']):<25} {len(a2['worst']):<25}")
    print(f"{'Elapsed min (s)':<30} {a1['elapsed_min']:<25} {a2['elapsed_min']:<25}")
    print(f"{'Elapsed max (s)':<30} {a1['elapsed_max']:<25} {a2['elapsed_max']:<25}")
    print(f"{'Elapsed median (s)':<30} {a1['elapsed_median']:<25} {a2['elapsed_median']:<25}")


if __name__ == "__main__":
    main()
