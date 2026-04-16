"""Run AgenticSearch on FinanceBench questions.

Supports two evaluation modes:
- **singleDoc**: each question searches only its target PDF directory.
- **sharedCorpus**: all questions search the full PDF corpus.

After search, an optional LLM extraction step converts the verbose
briefing into a short factoid answer suitable for EM/F1.
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
from evaluate import (
    classify_answer,
    compute_metrics,
    exact_match,
    evidence_recall,
    f1_score,
    normalize_answer,
)

logger = logging.getLogger("financebench.runner")

# ------------------------------------------------------------------
# Answer extraction prompt (financial domain)
# ------------------------------------------------------------------

_EXTRACT_PROMPT = """\
Given the financial question and a verbose response, extract ONLY the short factoid answer.
Rules:
- Output ONLY the answer value/phrase (1-20 words). No explanation.
- If the response says it cannot find the answer, output: unknown
- For monetary values, keep the currency format (e.g., $1,577.00)
- For percentages, keep the % sign (e.g., 15.3%)
- For yes/no questions, output: yes or no

Question: {question}
Response: {response}

Short answer:"""


# NOTE: _normalize_prediction removed — use evaluate.normalize_answer instead.


# ------------------------------------------------------------------
# LLM short-answer extraction
# ------------------------------------------------------------------


async def _extract_short_answer(
    question: str,
    verbose: str,
    llm: Any,
) -> str:
    """Use *llm* to distil *verbose* into a short factoid answer."""
    prompt = _EXTRACT_PROMPT.format(question=question, response=verbose[:4000])
    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.content.strip()
    except Exception:
        logger.warning("Short-answer extraction failed; falling back to raw answer.")
        return verbose


# ------------------------------------------------------------------
# Page extraction helper
# ------------------------------------------------------------------


def _try_extract_pages(telemetry: Dict[str, Any]) -> List[int]:
    """Best-effort extraction of retrieved page numbers from telemetry.

    Current limitation: Sirchmunk's ``read_file_ids`` contains plain file
    paths without page-level suffixes, so this function will typically
    return an empty list.  When empty, callers should treat evidence
    recall as *unavailable* (``None``) rather than zero.
    """
    pages: list[int] = []
    for fid in telemetry.get("read_file_ids", []):
        # Convention: page indices may be embedded in file IDs
        if isinstance(fid, str) and "_page_" in fid:
            try:
                pages.append(int(fid.rsplit("_page_", 1)[-1]))
            except (ValueError, IndexError):
                pass
    return pages


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
    gold_evidence = entry.get("evidence", [])

    async with semaphore:
        t0 = time.time()
        error: str | None = None
        raw_answer = ""
        answer = ""
        telemetry: dict[str, Any] = {}
        retrieved_pages: list[int] = []

        try:
            # Determine search paths based on eval mode
            if cfg.eval_mode == "singleDoc":
                pdf_path = loader.get_pdf_path(entry.get("doc_name", ""))
                if pdf_path:
                    search_paths = [pdf_path]  # pass the single PDF file directly
                else:
                    logger.warning("PDF not found for %s, falling back to full corpus", entry.get("doc_name", ""))
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
            retrieved_pages = _try_extract_pages(telemetry)

            # Answer extraction
            if cfg.extract_answer and raw_answer:
                answer = await _extract_short_answer(question, raw_answer, llm)
                answer = normalize_answer(answer)
            else:
                answer = normalize_answer(raw_answer)

        except Exception as exc:
            error = str(exc)
            logger.error("Error on %s: %s", fb_id, error)

        elapsed = time.time() - t0

        # Delay between requests
        if cfg.request_delay > 0:
            await asyncio.sleep(cfg.request_delay)

    # --- Evaluation ---
    is_no_result = not answer or answer.lower() in ("unknown", "")
    em = exact_match(answer, gold)
    f1 = f1_score(answer, gold)
    classification = classify_answer(answer, gold, is_no_result=is_no_result)
    if retrieved_pages:  # only compute when page-level data is available
        ev_recall = evidence_recall(retrieved_pages, gold_evidence)
    else:
        ev_recall = None  # mark as unavailable, avoid false 0

    # LLM Judge — independent evaluation dimension
    # Skip judge for refusals (no point calling LLM on non-answers)
    llm_judge_correct = None
    llm_judge_reasoning = None
    if judge is not None and classification != "refusal":
        try:
            judge_result = await judge.judge(
                prediction=answer,
                gold_answer=gold,
                question=question,
            )
            llm_judge_correct = judge_result.get("equivalent", False)
            llm_judge_reasoning = judge_result.get("reasoning", "")
        except Exception as e:
            logger.warning("LLM Judge failed for %s: %s", fb_id, e)
    elif judge is not None and classification == "refusal":
        llm_judge_correct = False
        llm_judge_reasoning = "Skipped: prediction classified as refusal"

    return {
        "financebench_id": fb_id,
        "question": question,
        "prediction": answer,
        "raw_prediction": raw_answer,
        "gold_answer": gold,
        "company": entry.get("company", ""),
        "doc_name": entry.get("doc_name", ""),
        "question_type": entry.get("question_type", ""),
        "question_reasoning": entry.get("question_reasoning", ""),
        "elapsed": round(elapsed, 2),
        "telemetry": telemetry,
        "classification": classification,
        "em": em,
        "f1": round(f1, 4),
        "evidence_recall": round(ev_recall, 4) if ev_recall is not None else None,
        "llm_judge_correct": llm_judge_correct,  # None if judge disabled
        "llm_judge_reasoning": llm_judge_reasoning,
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

    # Initialise LLM Judge (uses the same test model)
    judge = None
    if cfg.enable_llm_judge:
        from judge import FinanceBenchLLMJudge
        judge = FinanceBenchLLMJudge(llm=llm)
        logger.info("LLM Judge enabled (independent evaluation dimension)")

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
        status = res["classification"]
        judge_tag = ""
        if res.get("llm_judge_correct") is not None:
            judge_tag = " [judge:\u2713]" if res["llm_judge_correct"] else " [judge:\u2717]"
        logger.info(
            "[%d/%d] %s  %s  EM=%s  F1=%.2f  %.1fs%s",
            completed,
            total,
            res["financebench_id"],
            status,
            res["em"],
            res["f1"],
            res["elapsed"],
            judge_tag,
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
        "Accuracy=%.2f%%  Hallucination=%.2f%%  Refusal=%.2f%%",
        metrics.get("accuracy", 0),
        metrics.get("hallucination_rate", 0),
        metrics.get("refusal_rate", 0),
    )

    return list(results)
