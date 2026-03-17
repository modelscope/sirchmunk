#!/usr/bin/env python3
"""
HotpotQA Fullwiki Benchmark for AgenticSearch
==============================================
End-to-end RAG evaluation on the HotpotQA Fullwiki setting.

Search target: global Wikipedia corpus (~5.3M article abstracts, 11GB).
Each question searches the SAME corpus — no per-question document prep.

Leaderboard-aligned metrics (Yang et al. 2018 §5.2, Table 4):
  - Ans  (EM, F1):  Answer exact match and token-level F1
  - Sup  (EM, F1):  Supporting-fact set-level EM and F1
  - Joint(EM, F1):  Combined answer × supporting-fact metrics

Additional diagnostic metrics:
  - Contain-Match Accuracy  (LinearRAG-style, Zhuang et al. 2025)
  - GPT-Evaluation Accuracy
  - Evidence Recall (title-level SP coverage)
  - Efficiency: latency, tokens, loops

Checkpoint / Resume:
  --resume PATH   Load a previous results JSON and skip already-evaluated
                  questions.  New results are merged and a fresh report is
                  generated.  Useful for long-running evaluations.

Config: all options from .env.hotpotqa (unified experiment config).

Ref:
  HotpotQA paper: https://arxiv.org/pdf/1809.09600
  LinearRAG paper: https://arxiv.org/pdf/2510.10114
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WORK_DIR = Path(__file__).resolve().parent
LOGS_DIR = WORK_DIR / "logs"


class _Tee:
    """Write to multiple streams (e.g. stdout + log file)."""

    def __init__(self, *streams):
        self._streams = streams
        self._closed = False

    def write(self, data):
        if self._closed:
            return
        for s in self._streams:
            try:
                if not s.closed:
                    s.write(data)
                    s.flush()
            except (ValueError, OSError):
                pass

    def flush(self):
        if self._closed:
            return
        for s in self._streams:
            try:
                if not s.closed:
                    s.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        self._closed = True


def _setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    return log_path, log_file


def _install_log_tee(log_file):
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee_out = _Tee(orig_stdout, log_file)
    tee_err = _Tee(orig_stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err
    return orig_stdout, orig_stderr, tee_out, tee_err


def _restore_stdout_stderr(orig_stdout, orig_stderr):
    sys.stdout, sys.stderr = orig_stdout, orig_stderr


from config import get_config
from data_loader import load_samples, validate_wiki_corpus
from runner import run_batch, run_batch_llm_only
from evaluate import evaluate_predictions
from llm_judge import run_llm_judge, run_gpt_eval


_DEFAULT_ENV = Path(__file__).resolve().parent / ".env.hotpotqa"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HotpotQA Fullwiki benchmark — config from .env.hotpotqa")
    p.add_argument(
        "--env",
        type=Path,
        default=_DEFAULT_ENV,
        help=f"Path to env file (default: {_DEFAULT_ENV})",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="Resume from a previous results JSON file. "
             "Already-evaluated questions are skipped; "
             "new results are merged for a fresh report.",
    )
    return p.parse_args()


def _pct(v: float) -> str:
    return f"{v * 100:6.2f}"


def _load_checkpoint(path: Path) -> dict:
    """Load a previous results JSON for checkpoint/resume.

    Returns {qid: result_dict} for all successfully evaluated questions.
    """
    if not path.exists():
        print(f"[resume] Warning: checkpoint file not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    results_map = {}
    for r in data:
        qid = r.get("_id")
        if qid and not r.get("error"):
            results_map[qid] = r
    return results_map


def _save_intermediate(results: list, output_dir: Path, tag: str):
    """Save intermediate results for crash recovery."""
    path = output_dir / f"checkpoint_{tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return path


# =========================================================================
# Professional report formatting
# =========================================================================

_W = 78  # report width
_SEP_THICK = "=" * _W
_SEP_THIN = "-" * _W


def _print_header(title: str, char: str = "="):
    border = char * _W
    print(f"\n{border}")
    print(f"  {title}")
    print(border)


def _print_leaderboard_table(
    title: str,
    rows: list,
    row_label_key: str = "label",
):
    """Print a table in leaderboard format:
       Label   N  | Ans EM  Ans F1 | Sup EM  Sup F1 | Joint EM  Joint F1
    """
    print(f"\n  {title}")
    print(f"  {'':12s} {'N':>5s}  │ {'Ans EM':>7s} {'Ans F1':>7s}"
          f" │ {'Sup EM':>7s} {'Sup F1':>7s}"
          f" │ {'Jnt EM':>7s} {'Jnt F1':>7s}")
    print(f"  {'─' * 12} {'─' * 5}──┼─{'─' * 7}─{'─' * 7}"
          f"─┼─{'─' * 7}─{'─' * 7}"
          f"─┼─{'─' * 7}─{'─' * 7}")
    for r in rows:
        label = r[row_label_key]
        n = r["count"]
        print(f"  {label:<12s} {n:>5d}  │"
              f" {_pct(r['ans_em']):>7s} {_pct(r['ans_f1']):>7s} │"
              f" {_pct(r['sp_em']):>7s} {_pct(r['sp_f1']):>7s} │"
              f" {_pct(r['joint_em']):>7s} {_pct(r['joint_f1']):>7s}")


def _print_diagnostic_table(title: str, rows: list, row_label_key: str = "label"):
    """Print diagnostic metrics (Contain-Acc, Ev Recall)."""
    print(f"\n  {title}")
    print(f"  {'':12s} {'N':>5s}  │ {'Cont-Acc':>8s} {'Ev Recall':>9s}")
    print(f"  {'─' * 12} {'─' * 5}──┼─{'─' * 8}─{'─' * 9}")
    for r in rows:
        print(f"  {r[row_label_key]:<12s} {r['count']:>5d}  │"
              f" {_pct(r['contain']):>8s} {_pct(r['ev_recall']):>9s}")


def print_report(metrics, results, total_time, judge_results, gpt_acc, cfg):
    """Print leaderboard-aligned evaluation report."""
    _print_header("HOTPOTQA FULLWIKI EVALUATION REPORT")

    ov = metrics["overall"]

    # -- Leaderboard metrics: Overall --
    overall_rows = [{"label": "Overall", **ov}]
    _print_leaderboard_table("Leaderboard Metrics", overall_rows)

    # -- Leaderboard metrics: By Type --
    bt = metrics.get("by_type", {})
    if bt:
        type_rows = [{"label": t, **v} for t, v in sorted(bt.items())]
        _print_leaderboard_table("By Question Type", type_rows)

    # -- Leaderboard metrics: By Level --
    bl = metrics.get("by_level", {})
    if bl:
        level_rows = [{"label": lv, **v} for lv, v in sorted(bl.items())]
        _print_leaderboard_table("By Difficulty Level", level_rows)

    # -- Diagnostic metrics --
    diag_rows = [{"label": "Overall", **ov}]
    if bt:
        diag_rows.extend({"label": t, **v} for t, v in sorted(bt.items()))
    _print_diagnostic_table("Diagnostic Metrics", diag_rows)

    # -- GPT-Eval --
    if gpt_acc is not None:
        print(f"\n  GPT-Eval Accuracy:  {_pct(gpt_acc)}%  "
              f"(N={ov['count']})")

    # -- Efficiency --
    ok = [r for r in results if not r.get("error")]
    n = len(ok) or 1
    errors = len(results) - len(ok)
    avg_time = sum(r["elapsed"] for r in ok) / n
    total_tokens = sum(
        r.get("telemetry", {}).get("total_tokens", 0) for r in ok)
    avg_tokens = total_tokens / n
    avg_loops = sum(
        r.get("telemetry", {}).get("loop_count", 0) for r in ok) / n
    avg_files = sum(
        len(r.get("telemetry", {}).get("files_read", [])) for r in ok) / n
    avg_titles = sum(len(r.get("retrieved_titles", [])) for r in ok) / n
    avg_sp = sum(len(r.get("predicted_sp", [])) for r in ok) / n

    print(f"\n  Efficiency")
    print(f"  {'─' * 40}")
    print(f"    Avg latency:         {avg_time:>8.2f} s/query")
    print(f"    Avg tokens:          {avg_tokens:>8.0f} /query")
    print(f"    Avg loops (hops):    {avg_loops:>8.1f}")
    print(f"    Avg files read:      {avg_files:>8.1f}")
    print(f"    Avg titles retrieved:{avg_titles:>8.0f}")
    print(f"    Avg SP predictions:  {avg_sp:>8.0f}")
    print(f"    Total wall time:     {total_time:>8.1f} s")
    print(f"    Errors:              {errors:>5d} / {len(results)}")

    # -- LLM Judge --
    if judge_results:
        correct = sum(1 for j in judge_results if j["llm_correct"])
        print(f"\n  LLM Judge  (EM=0, F1 >= {cfg.judge_f1_threshold})")
        print(f"  {'─' * 40}")
        print(f"    Candidates:          {len(judge_results)}")
        print(f"    Judged correct:      {correct} / {len(judge_results)}")
        if ov["count"]:
            adjusted_em = (ov["ans_em"] * ov["count"] + correct) / ov["count"]
            print(f"    Adjusted Ans EM:     {_pct(adjusted_em)}%")

    print(f"\n{_SEP_THICK}")


async def main():
    args = parse_args()
    log_path, log_file = _setup_logging()
    orig_stdout, orig_stderr, tee_out, tee_err = _install_log_tee(log_file)
    try:
        return await _main_impl(args, log_path, log_file, orig_stdout, orig_stderr)
    finally:
        tee_out.close()
        tee_err.close()
        _restore_stdout_stderr(orig_stdout, orig_stderr)
        log_file.flush()
        log_file.close()


async def _main_impl(args, log_path, log_file, orig_stdout, orig_stderr):
    cfg = get_config(env_file=args.env)

    if not cfg.llm_api_key:
        print(f"[ERROR] LLM_API_KEY not set in {args.env}")
        return
    if not cfg.dataset_dir or not Path(cfg.dataset_dir).exists():
        print(f"[ERROR] HOTPOT_DATASET_DIR not set or missing: {cfg.dataset_dir}")
        return

    n_corpus = 0
    if not cfg.llm_only:
        try:
            n_corpus = validate_wiki_corpus(cfg.wiki_corpus_dir)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[ERROR] {e}")
            return

    _bench_title = ("HotpotQA Fullwiki Benchmark — LLM-Only (Ablation)"
                    if cfg.llm_only
                    else "HotpotQA Fullwiki Benchmark — AgenticSearch")
    _print_header(_bench_title)
    print(f"  Config:       {args.env.resolve()}")
    if cfg.llm_only:
        print(f"  Mode:         LLM-ONLY (no retrieval)")
    else:
        print(f"  Wiki Corpus:  {cfg.wiki_corpus_dir}")
        print(f"  Corpus Files: {n_corpus}")
    print(f"  Setting:      {cfg.setting}")
    print(f"  Split:        {cfg.split}")
    print(f"  Limit:        {cfg.limit or 'ALL'}")
    if not cfg.llm_only:
        print(f"  Mode:         {cfg.mode}")
        print(f"  Top-K:        {cfg.top_k_files}")
    print(f"  Model:        {cfg.llm_model}")
    print(f"  Concurrent:   {cfg.max_concurrent}")
    if not cfg.llm_only:
        print(f"  Token Budget: {cfg.max_token_budget}")
        print(f"  Dir Scan:     {'ON' if cfg.enable_dir_scan else 'OFF'}")
    print(f"  Extract:      {'ON' if cfg.extract_answer else 'OFF'}")
    print(f"  GPT-Eval:     {'ON' if cfg.enable_gpt_eval else 'OFF'}")
    print(f"  LLM Judge:    {'ON' if cfg.enable_llm_judge else 'OFF'}")
    print(f"  Thinking:     {'ON' if cfg.enable_thinking else 'OFF'}")
    if args.resume:
        print(f"  Resume from:  {args.resume}")
    print(_SEP_THICK)

    # --- Load data ---
    samples = load_samples(cfg)
    if not samples:
        print("[ERROR] No samples loaded. "
              "Check hotpotqa_dataset/{setting}/{split}*.parquet")
        return

    # --- Type/level distribution ---
    type_dist: dict = {}
    level_dist: dict = {}
    for s in samples:
        type_dist[s["type"]] = type_dist.get(s["type"], 0) + 1
        level_dist[s["level"]] = level_dist.get(s["level"], 0) + 1
    print(f"\n[data] Type distribution:  {type_dist}")
    print(f"[data] Level distribution: {level_dist}")

    # --- Checkpoint / Resume ---
    prior_results: dict = {}
    if args.resume:
        prior_results = _load_checkpoint(args.resume)
        print(f"[resume] Loaded {len(prior_results)} completed results "
              f"from {args.resume}")

    # Partition samples into done (from checkpoint) and pending
    pending_samples = []
    resumed_results = []
    for s in samples:
        qid = s["_id"]
        if qid in prior_results:
            resumed_results.append(prior_results[qid])
        else:
            pending_samples.append(s)
    print(f"[resume] {len(resumed_results)} cached, "
          f"{len(pending_samples)} pending")

    # --- Run predictions on pending samples ---
    new_results = []
    t0 = time.time()
    if pending_samples:
        if cfg.llm_only:
            print(f"\n[run] Starting LLM-only evaluation on "
                  f"{len(pending_samples)} questions ...")
            new_results = await run_batch_llm_only(pending_samples, cfg)
        else:
            print(f"\n[run] Starting evaluation on {len(pending_samples)} questions "
                  f"against wiki corpus ({n_corpus} files) ...")
            new_results = await run_batch(pending_samples, cfg)

        # Save intermediate checkpoint after each batch
        all_results_so_far = resumed_results + new_results
        ckpt_path = _save_intermediate(
            all_results_so_far, cfg.output_dir,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        print(f"[checkpoint] Saved intermediate results: {ckpt_path}")
    else:
        print(f"\n[run] All {len(samples)} questions already evaluated "
              f"(loaded from checkpoint)")

    total_time = time.time() - t0
    results = resumed_results + new_results

    # --- Evaluate ---
    metrics = evaluate_predictions(results, samples)

    # --- GPT-Eval ---
    gpt_acc = None
    gpt_details = []
    if cfg.enable_gpt_eval and metrics["per_sample"]:
        print(f"\n[gpt-eval] Running GPT-Evaluation on "
              f"{len(metrics['per_sample'])} samples ...")
        gpt_acc, gpt_details = await run_gpt_eval(metrics["per_sample"], cfg)

    # --- LLM Judge ---
    judge_results = []
    if cfg.enable_llm_judge:
        candidates = [
            s for s in metrics["per_sample"]
            if s["ans_em"] == 0 and s["ans_f1"] >= cfg.judge_f1_threshold
        ]
        if candidates:
            print(f"\n[judge] Running LLM judge on "
                  f"{len(candidates)} borderline samples ...")
            judge_results = await run_llm_judge(candidates, cfg)

    # --- Report ---
    print_report(metrics, results, total_time, judge_results, gpt_acc, cfg)

    # --- Save artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _mode_tag = "LLM-ONLY" if cfg.llm_only else cfg.mode
    tag = (f"{cfg.setting}_{cfg.split}_{_mode_tag}_"
           f"{cfg.llm_model}_{len(results)}q_{timestamp}")

    def _round_metric(v):
        return round(v * 100, 2) if isinstance(v, float) else v

    ov = metrics["overall"]
    report = {
        "config": {
            "setting": cfg.setting, "split": cfg.split, "limit": cfg.limit,
            "mode": "LLM-ONLY" if cfg.llm_only else cfg.mode,
            "llm_only": cfg.llm_only,
            "model": cfg.llm_model, "seed": cfg.seed,
            "top_k_files": cfg.top_k_files,
            "max_token_budget": cfg.max_token_budget,
            "enable_dir_scan": cfg.enable_dir_scan,
            "extract_answer": cfg.extract_answer,
            "enable_thinking": cfg.enable_thinking,
            "wiki_corpus_dir": str(cfg.wiki_corpus_dir),
            "resumed_from": str(args.resume) if args.resume else None,
            "resumed_count": len(resumed_results),
        },
        "leaderboard_metrics": {
            "overall": {
                "ans_em": _round_metric(ov.get("ans_em", 0)),
                "ans_f1": _round_metric(ov.get("ans_f1", 0)),
                "sp_em": _round_metric(ov.get("sp_em", 0)),
                "sp_f1": _round_metric(ov.get("sp_f1", 0)),
                "joint_em": _round_metric(ov.get("joint_em", 0)),
                "joint_f1": _round_metric(ov.get("joint_f1", 0)),
                "count": ov.get("count", 0),
            },
            "by_type": {
                t: {k: _round_metric(v) for k, v in d.items()}
                for t, d in metrics["by_type"].items()
            },
            "by_level": {
                lv: {k: _round_metric(v) for k, v in d.items()}
                for lv, d in metrics["by_level"].items()
            },
        },
        "diagnostic_metrics": {
            "contain_acc": _round_metric(ov.get("contain", 0)),
            "ev_recall": _round_metric(ov.get("ev_recall", 0)),
        },
        "gpt_eval": {
            "accuracy": round(gpt_acc * 100, 2) if gpt_acc is not None else None,
            "correct": sum(1 for d in gpt_details if d["llm_correct"]),
            "total": len(gpt_details),
        } if cfg.enable_gpt_eval else None,
        "efficiency": {
            "avg_latency_sec": round(
                sum(r["elapsed"] for r in results) / max(len(results), 1), 2),
            "avg_tokens": round(
                sum(r.get("telemetry", {}).get("total_tokens", 0)
                    for r in results) / max(len(results), 1)),
            "avg_loops": round(
                sum(r.get("telemetry", {}).get("loop_count", 0)
                    for r in results) / max(len(results), 1), 1),
            "total_time_sec": round(total_time, 2),
            "errors": sum(1 for r in results if r.get("error")),
        },
        "llm_judge": {
            "candidates": len(judge_results),
            "correct": sum(1 for j in judge_results if j["llm_correct"]),
            "details": judge_results,
        } if judge_results else None,
        "timestamp": timestamp,
    }

    results_path = cfg.output_dir / f"results_{tag}.json"
    report_path = cfg.output_dir / f"report_{tag}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {results_path}")
    print(f"  Report:  {report_path}")
    print(f"  Log:     {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
