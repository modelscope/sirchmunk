"""Run AgenticSearch on HotpotQA samples against the full wiki corpus.

Fullwiki setting (Yang et al., 2018 §3): all questions search the same
global Wikipedia corpus (~5.3M article abstracts), testing end-to-end
retrieval + reasoning + generation.

When extract_answer=True, a lightweight LLM call converts the verbose
AgenticSearch briefing into a short factoid answer suitable for EM/F1.

After search, article titles are extracted from read wiki files for
evidence recall computation.
"""

import asyncio
import json as json_mod
import time
from typing import Any, Dict, List, Set

from config import ExperimentConfig

_EXTRACT_PROMPT = """\
Given the question and a verbose response, extract ONLY the short factoid answer.
Rules:
- Output ONLY the answer phrase (1-10 words). No explanation.
- If the response says it cannot find the answer, output: unknown
- For yes/no questions, output: yes or no
- For dates, use the format that appears in the response.

Question: {question}
Response:
{response}

Short answer:"""


def _normalize_prediction(pred: str) -> str:
    """Post-process prediction to improve EM/F1 matching.

    Strips markdown formatting, quotation marks, trailing periods,
    and common wrapper phrases that LLMs add.
    """
    import re
    s = pred.strip()
    # Remove markdown bold/italic
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    # Remove surrounding quotes
    if len(s) >= 2 and s[0] in ('"', "'", "\u201c") and s[-1] in ('"', "'", "\u201d"):
        s = s[1:-1].strip()
    # Remove trailing period / colon
    s = s.rstrip(".:")
    # Remove common LLM wrapper phrases
    for prefix in [
        "The answer is ", "The short answer is ",
        "Answer: ", "Short answer: ",
    ]:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].strip()
    return s


async def _extract_short_answer(
    question: str,
    verbose: str,
    llm: Any,
) -> str:
    """Extract short factoid answer from verbose AgenticSearch output."""
    prompt = _EXTRACT_PROMPT.format(
        question=question,
        response=verbose[:3000],
    )
    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.content.strip()
    except Exception:
        return verbose


def _extract_titles_from_files(files_read: List[str]) -> Set[str]:
    """Parse wiki corpus files to extract article titles from read files.

    Each wiki file contains JSON lines (one per article). Extracts all
    article titles from the files that AgenticSearch opened during search.
    """
    titles: Set[str] = set()
    for fpath in files_read:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json_mod.loads(line)
                        t = obj.get("title")
                        if t:
                            titles.add(t)
                    except (json_mod.JSONDecodeError, TypeError):
                        continue
        except (FileNotFoundError, PermissionError):
            continue
    return titles


async def run_single(
    entry: Dict[str, Any],
    searcher: Any,
    llm: Any,
    cfg: ExperimentConfig,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run search on one HotpotQA question against the global wiki corpus."""
    qid = entry["_id"]
    question = entry["question"]

    async with semaphore:
        t0 = time.time()
        error = None
        raw_answer = ""
        answer = ""
        telemetry: Dict[str, Any] = {}
        retrieved_titles: List[str] = []

        try:
            result = await searcher.search(
                query=question,
                paths=[str(cfg.wiki_corpus_dir)],
                mode=cfg.mode,
                top_k_files=cfg.top_k_files,
                max_token_budget=cfg.max_token_budget,
                enable_dir_scan=cfg.enable_dir_scan,
                return_context=True,
            )

            raw_answer = getattr(result, "answer", "") or str(result)
            files_read = list(getattr(result, "read_file_ids", None) or set())

            # Supplement from cluster evidences (Phase 3 may read files
            # without populating read_file_ids in older code paths)
            cluster = getattr(result, "cluster", None)
            if cluster:
                for ev in (getattr(cluster, "evidences", None) or []):
                    fp = str(getattr(ev, "file_or_url", ""))
                    if fp and fp not in files_read:
                        files_read.append(fp)

            telemetry = {
                "total_tokens": getattr(result, "total_llm_tokens", 0),
                "loop_count": getattr(result, "loop_count", 0),
                "files_read": files_read,
                "llm_calls": len(getattr(result, "llm_usages", None) or []),
            }

            titles = _extract_titles_from_files(files_read)
            retrieved_titles = list(titles)

            if cfg.extract_answer and raw_answer:
                answer = await _extract_short_answer(question, raw_answer, llm)
                answer = _normalize_prediction(answer)
            else:
                answer = _normalize_prediction(raw_answer)
        except Exception as e:
            error = str(e)

        elapsed = time.time() - t0
        if cfg.request_delay > 0:
            await asyncio.sleep(cfg.request_delay)

    return {
        "_id": qid,
        "question": question,
        "prediction": answer,
        "raw_prediction": raw_answer,
        "gold_answer": entry.get("answer") or "",
        "type": entry.get("type", ""),
        "level": entry.get("level", ""),
        "elapsed": round(elapsed, 2),
        "telemetry": telemetry,
        "retrieved_titles": retrieved_titles,
        "error": error,
    }


async def run_batch(
    samples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """Run AgenticSearch on all samples with progress tracking."""
    from sirchmunk.search import AgenticSearch
    from sirchmunk.llm.openai_chat import OpenAIChat

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    searcher = AgenticSearch(llm=llm, reuse_knowledge=False, verbose=False)

    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    total = len(samples)
    completed = 0

    async def _tracked(entry):
        nonlocal completed
        r = await run_single(entry, searcher, llm, cfg, semaphore)
        completed += 1
        status = "OK" if not r["error"] else f"ERR: {r['error'][:60]}"
        t = r.get("telemetry", {})
        n_titles = len(r.get("retrieved_titles", []))
        extract_tag = " [ext]" if r.get("prediction") != r.get("raw_prediction") else ""
        print(f"  [{completed}/{total}] {r['_id']}  "
              f"{r['elapsed']:.1f}s  tok={t.get('total_tokens', 0)}  "
              f"loops={t.get('loop_count', 0)}  titles={n_titles}  "
              f"{status}{extract_tag}")
        return r

    tasks = [_tracked(s) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)
