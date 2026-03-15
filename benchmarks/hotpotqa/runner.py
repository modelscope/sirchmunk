"""Run AgenticSearch on HotpotQA samples against the full wiki corpus.

Fullwiki setting (Yang et al., 2018 §3): all questions search the same
global Wikipedia corpus (~5.3M article abstracts), testing end-to-end
retrieval + reasoning + generation.

When extract_answer=True, a lightweight LLM call converts the verbose
AgenticSearch briefing into a short factoid answer suitable for EM/F1.

After search, article titles and supporting-fact predictions are
extracted from wiki files read during the search.  Predicted SP follows
the (title, sent_id) format of the official evaluation so that
Sup EM/F1 and Joint EM/F1 can be computed.

SP prediction uses evidence-based relevance filtering: only articles
whose title or sentence content appears in the search context (question,
answer, evidence snippets) are included.  Without this filter, every
article in every read JSONL file (~360 articles/file × ~21 files)
would be predicted, yielding ~17K SP pairs vs. ~4 gold pairs —
collapsing SP precision to near zero.
"""

import asyncio
import json as json_mod
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from config import ExperimentConfig

_EXTRACT_PROMPT = """\
Given the question and a verbose response, extract ONLY the short factoid answer.
Rules:
- Output ONLY the answer phrase (1-10 words). No explanation.
- If the response says it cannot find the answer, output: unknown
- For yes/no questions, output: yes or no
- For dates, use the format that appears in the response.
- For person names, use the full name as it appears in the response.

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
    # Remove surrounding quotes (ASCII and Unicode)
    if len(s) >= 2 and s[0] in ('"', "'", "\u201c", "\u2018") and s[-1] in ('"', "'", "\u201d", "\u2019"):
        s = s[1:-1].strip()
    # Remove trailing period / colon
    s = s.rstrip(".:")
    # Remove common LLM wrapper phrases (case-insensitive)
    _PREFIXES = [
        "the answer is ", "the short answer is ",
        "answer: ", "short answer: ",
        "based on the information, ",
        "based on the context, ",
        "according to the documents, ",
    ]
    s_lower = s.lower()
    for prefix in _PREFIXES:
        if s_lower.startswith(prefix):
            s = s[len(prefix):].strip()
            s_lower = s.lower()
    # Remove trailing parenthetical clarifications: "Paris (the capital)"
    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()
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


# ---------------------------------------------------------------------------
# Evidence-based SP extraction
# ---------------------------------------------------------------------------

def _collect_evidence_texts(cluster: Any) -> List[str]:
    """Collect textual content from a knowledge cluster for SP matching.

    Gathers cluster content, description, evidence summaries, and raw
    evidence snippets — the combined text represents what the system
    actually "looked at" during the search.
    """
    texts: List[str] = []
    if not cluster:
        return texts

    content = getattr(cluster, "content", None)
    if content:
        texts.append(str(content))

    for desc in (getattr(cluster, "description", None) or []):
        if desc:
            texts.append(str(desc))

    for ev in (getattr(cluster, "evidences", None) or []):
        summary = getattr(ev, "summary", "")
        if summary:
            texts.append(str(summary))
        for snip in (getattr(ev, "snippets", None) or []):
            if isinstance(snip, dict):
                s = snip.get("snippet", "")
            elif isinstance(snip, str):
                s = snip
            else:
                continue
            if s:
                texts.append(s)

    return texts


def _extract_titles_and_sp(
    files_read: List[str],
    question: str = "",
    answer: str = "",
    evidence_texts: Optional[List[str]] = None,
) -> Tuple[Set[str], List[List]]:
    """Parse wiki JSONL files to extract article titles and predicted SP.

    Returns (all_titles, predicted_sp) where:
      - all_titles: ALL article titles from all read files
        (broad set for Evidence Recall diagnostic metric)
      - predicted_sp: Only (title, sent_id) pairs from articles that are
        demonstrably relevant to the query (narrow set for Sup EM/F1)

    Relevance filtering (for predicted_sp only):
      1. Article title appears in the question, answer, or evidence text
      2. Article sentence content (first 60 chars) appears in the evidence

    Without this filter, every article in every read JSONL would be
    predicted (~17K SP pairs), collapsing SP precision to near zero.
    """
    all_titles: Set[str] = set()
    predicted_sp: List[List] = []

    # Build combined context for relevance matching
    _ev = evidence_texts or []
    context_lower = "\n".join([question, answer] + _ev).lower()

    for fpath in files_read:
        try:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json_mod.loads(line)
                        title = obj.get("title")
                        if not title:
                            continue
                        all_titles.add(title)

                        text = obj.get("text")
                        if not text:
                            continue
                        sentences = (
                            text[0] if text and isinstance(text[0], list)
                            else text
                        )
                        if not sentences:
                            continue

                        # --- Relevance check ---
                        relevant = title.lower() in context_lower

                        if not relevant:
                            for sent in sentences:
                                s = sent.strip() if isinstance(sent, str) else ""
                                if len(s) > 20 and s[:60].lower() in context_lower:
                                    relevant = True
                                    break

                        if relevant:
                            for sid in range(len(sentences)):
                                predicted_sp.append([title, sid])
                    except (json_mod.JSONDecodeError, TypeError, IndexError):
                        continue
        except (FileNotFoundError, PermissionError):
            continue

    return all_titles, predicted_sp


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
        predicted_sp: List[List] = []

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

            # Supplement from cluster evidences
            cluster = getattr(result, "cluster", None)
            if cluster:
                for ev in (getattr(cluster, "evidences", None) or []):
                    fp = str(getattr(ev, "file_or_url", ""))
                    if fp and fp not in files_read:
                        files_read.append(fp)

            # Supplement from keyword search retrieval logs
            _seen = set(files_read)
            for log_entry in (getattr(result, "retrieval_logs", None) or []):
                meta = getattr(log_entry, "metadata", None) or {}
                for p in meta.get("files_discovered", []):
                    if p and p not in _seen:
                        files_read.append(p)
                        _seen.add(p)

            telemetry = {
                "total_tokens": getattr(result, "total_llm_tokens", 0),
                "loop_count": getattr(result, "loop_count", 0),
                "files_read": files_read,
                "llm_calls": len(getattr(result, "llm_usages", None) or []),
            }

            # Collect evidence texts for relevance-based SP filtering
            evidence_texts = _collect_evidence_texts(cluster)

            titles, predicted_sp = _extract_titles_and_sp(
                files_read, question, raw_answer, evidence_texts,
            )
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
        "predicted_sp": predicted_sp,
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
        n_sp = len(r.get("predicted_sp", []))
        extract_tag = " [ext]" if r.get("prediction") != r.get("raw_prediction") else ""
        print(f"  [{completed}/{total}] {r['_id']}  "
              f"{r['elapsed']:.1f}s  tok={t.get('total_tokens', 0)}  "
              f"loops={t.get('loop_count', 0)}  titles={n_titles}  "
              f"sp={n_sp}  {status}{extract_tag}")
        return r

    tasks = [_tracked(s) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)
