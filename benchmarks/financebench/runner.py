"""Run AgenticSearch on FinanceBench questions.

Supports two evaluation modes:
- **singleDoc**: each question searches only its target PDF directory.
- **sharedCorpus**: all questions search the full PDF corpus.

All evaluation (Accuracy + Coverage) is driven by LLM Judge.
"""
from __future__ import annotations

import asyncio
import json as json_mod
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import FinanceBenchConfig
from data_loader import FinanceBenchLoader
from evaluate import compute_metrics

logger = logging.getLogger("financebench.runner")


# ------------------------------------------------------------------
# Single question execution
# ------------------------------------------------------------------


async def run_single(
    entry: Dict[str, Any],
    loader: FinanceBenchLoader,
    searcher: Any,
    llm: Any,
    cfg: FinanceBenchConfig,
    semaphore: asyncio.Semaphore,
    judge: Any = None,
) -> Dict[str, Any]:
    """Execute one FinanceBench question end-to-end."""
    fb_id = entry.get("financebench_id", "")
    question = entry["question"]
    gold = entry.get("answer", "")

    async with semaphore:
        t0 = time.time()
        error: str | None = None
        raw_answer = ""
        telemetry: dict[str, Any] = {}

        try:
            # Determine search paths based on eval mode
            if cfg.eval_mode == "singleDoc":
                pdf_path = loader.get_pdf_path(entry.get("doc_name", ""))
                if pdf_path:
                    search_paths = [pdf_path]
                else:
                    logger.warning(
                        "PDF not found for %s, falling back to full corpus",
                        entry.get("doc_name", ""),
                    )
                    search_paths = [cfg.pdf_dir]
            else:
                search_paths = [cfg.pdf_dir]

            result = await searcher.search(
                query=question,
                paths=search_paths,
                mode=cfg.mode,
                top_k_files=cfg.top_k_files,
                max_token_budget=cfg.max_token_budget,
                enable_dir_scan=cfg.enable_dir_scan,
                return_context=True,
            )

            raw_answer = getattr(result, "answer", "") or str(result)

            # Collect telemetry
            read_files = list(getattr(result, "read_file_ids", None) or set())
            telemetry = {
                "read_file_ids": read_files,
                "total_tokens": getattr(result, "total_llm_tokens", 0),
                "loop_count": getattr(result, "loop_count", 0),
                "llm_calls": len(getattr(result, "llm_usages", None) or []),
                "num_files_read": len(read_files),
            }

        except Exception as exc:
            error = str(exc)
            logger.error("Error on %s: %s", fb_id, error)

        elapsed = time.time() - t0

        # Delay between requests
        if cfg.request_delay > 0:
            await asyncio.sleep(cfg.request_delay)

    # --- LLM Judge evaluation (Accuracy + Coverage) ---
    judge_correct = False
    judge_reasoning = ""
    judge_tokens = 0
    has_coverage = False
    coverage_reasoning = ""

    if judge is not None:
        # Accuracy evaluation
        try:
            judge_result = await judge.judge(
                prediction=raw_answer,
                gold_answer=gold,
                question=question,
            )
            judge_correct = judge_result.get("equivalent", False)
            judge_reasoning = judge_result.get("reasoning", "")
            judge_tokens += judge_result.get("tokens_used", 0)
        except Exception as e:
            logger.warning("LLM Judge (accuracy) failed for %s: %s", fb_id, e)

        # Coverage evaluation
        try:
            coverage_result = await judge.judge_coverage(
                prediction=raw_answer,
                question=question,
            )
            has_coverage = coverage_result.get("has_coverage", False)
            coverage_reasoning = coverage_result.get("reasoning", "")
            judge_tokens += coverage_result.get("tokens_used", 0)
        except Exception as e:
            logger.warning("LLM Judge (coverage) failed for %s: %s", fb_id, e)

    return {
        "financebench_id": fb_id,
        "question": question,
        "raw_prediction": raw_answer,
        "gold_answer": gold,
        "company": entry.get("company", ""),
        "doc_name": entry.get("doc_name", ""),
        "question_type": entry.get("question_type", ""),
        "question_reasoning": entry.get("question_reasoning", ""),
        "elapsed": round(elapsed, 2),
        "telemetry": telemetry,
        "judge_correct": judge_correct,
        "judge_reasoning": judge_reasoning,
        "coverage": has_coverage,
        "coverage_reasoning": coverage_reasoning,
        "judge_tokens": judge_tokens,
        "error": error,
    }


# ------------------------------------------------------------------
# Batch execution
# ------------------------------------------------------------------


async def run_batch(
    samples: List[Dict[str, Any]],
    cfg: FinanceBenchConfig,
) -> List[Dict[str, Any]]:
    """Run all *samples* concurrently and persist results incrementally."""
    from sirchmunk.llm.openai_chat import OpenAIChat
    from sirchmunk.search import AgenticSearch

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    work_path = str(Path(cfg.work_path).resolve())
    searcher = AgenticSearch(llm=llm, work_path=work_path, reuse_knowledge=False, verbose=False)
    loader = FinanceBenchLoader(data_dir=cfg.data_dir, pdf_dir=cfg.pdf_dir)
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    # Initialise LLM Judge
    judge = None
    if cfg.enable_llm_judge:
        from judge import FinanceBenchLLMJudge
        judge = FinanceBenchLLMJudge(llm=llm)
        logger.info("LLM Judge enabled (drives Accuracy + Coverage)")

    # Prepare output directory / file
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"financebench_{ts}.jsonl"

    results: list[dict] = []
    completed = 0
    total = len(samples)

    async def _run_and_record(entry: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal completed
        res = await run_single(entry, loader, searcher, llm, cfg, semaphore, judge=judge)
        # Incremental save
        with open(out_path, "a", encoding="utf-8") as fp:
            fp.write(json_mod.dumps(res, ensure_ascii=False) + "\n")
        completed += 1
        acc_tag = "\u2713" if res["judge_correct"] else "\u2717"
        cov_tag = "cov" if res["coverage"] else "no-cov"
        logger.info(
            "[%d/%d] %s  [acc:%s] [%s]  %.1fs",
            completed,
            total,
            res["financebench_id"],
            acc_tag,
            cov_tag,
            res["elapsed"],
        )
        return res

    tasks = [asyncio.create_task(_run_and_record(s)) for s in samples]
    results = await asyncio.gather(*tasks)

    # Write aggregate metrics
    metrics = compute_metrics(list(results))
    metrics_path = out_dir / f"financebench_{ts}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json_mod.dump(metrics, fp, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", metrics_path)
    logger.info(
        "Accuracy=%.2f%%  Coverage=%.2f%%",
        metrics.get("accuracy", 0),
        metrics.get("coverage", 0),
    )

    return list(results)
