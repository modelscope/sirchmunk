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
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikipedia title → filepath index
# ---------------------------------------------------------------------------
# Global index: normalized_title -> list of file paths containing articles with that title
_WIKI_TITLE_INDEX: Dict[str, List[str]] = {}
_TITLE_INDEX_BUILT = False


def _normalize_title(title: str) -> str:
    """Normalize title for index lookup (lowercase, strip whitespace)."""
    return title.lower().strip()


def build_title_index(wiki_corpus_dir: Path, progress_interval: int = 1000) -> int:
    """Build title→filepath index from Wikipedia JSONL files.

    Scans all .jsonl files in the corpus directory, parses article titles,
    and builds a mapping for efficient lookup. Called once at startup.

    Args:
        wiki_corpus_dir: Root directory containing Wikipedia JSONL files.
        progress_interval: Print progress every N files.

    Returns:
        Number of titles indexed.
    """
    global _WIKI_TITLE_INDEX, _TITLE_INDEX_BUILT

    if _TITLE_INDEX_BUILT:
        return len(_WIKI_TITLE_INDEX)

    if not wiki_corpus_dir.exists():
        print(f"[TitleIndex] Wiki corpus dir not found: {wiki_corpus_dir}")
        _TITLE_INDEX_BUILT = True
        return 0

    print(f"[TitleIndex] Building title→filepath index from: {wiki_corpus_dir}")
    start_time = time.time()
    files_processed = 0
    titles_indexed = 0

    # Find all JSONL files
    jsonl_files = list(wiki_corpus_dir.rglob("*.jsonl"))
    print(f"[TitleIndex] Found {len(jsonl_files)} JSONL files to index")

    for jsonl_path in jsonl_files:
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json_mod.loads(line)
                        title = obj.get("title")
                        if title:
                            norm_title = _normalize_title(title)
                            if norm_title not in _WIKI_TITLE_INDEX:
                                _WIKI_TITLE_INDEX[norm_title] = []
                            fp_str = str(jsonl_path)
                            if fp_str not in _WIKI_TITLE_INDEX[norm_title]:
                                _WIKI_TITLE_INDEX[norm_title].append(fp_str)
                            titles_indexed += 1
                    except (json_mod.JSONDecodeError, TypeError):
                        continue
            files_processed += 1
            if files_processed % progress_interval == 0:
                print(f"[TitleIndex] Processed {files_processed}/{len(jsonl_files)} files, "
                      f"{titles_indexed} titles indexed")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            continue

    elapsed = time.time() - start_time
    print(f"[TitleIndex] Complete: {len(_WIKI_TITLE_INDEX)} unique titles from "
          f"{files_processed} files in {elapsed:.1f}s")

    _TITLE_INDEX_BUILT = True
    return len(_WIKI_TITLE_INDEX)


def lookup_title_files(title: str) -> List[str]:
    """Look up file paths containing a given title.

    Args:
        title: Article title to look up (case-insensitive).

    Returns:
        List of file paths containing articles with this title.
    """
    return _WIKI_TITLE_INDEX.get(_normalize_title(title), [])

_EXTRACT_PROMPT = """\
Given the question and a verbose response, extract ONLY the short factoid answer.
Rules:
- Output ONLY the answer phrase (1-10 words). No explanation.
- Look for bold text, named entities, dates, numbers, or proper nouns that directly answer the question.
- For yes/no or comparison questions, output: yes or no
- For dates, use the format that appears in the response.
- For person names, use the full name as it appears in the response.
- Even if the response expresses uncertainty or says information is incomplete, extract any specific entity/fact that partially answers the question.
- Output "unknown" ONLY if absolutely no relevant entity, date, name, or fact can be found anywhere in the response.

Question: {question}
Response:
{response}

Short answer:"""


def _normalize_prediction(pred: str) -> str:
    """Post-process prediction to improve EM/F1 matching.

    Strips markdown formatting, quotation marks, trailing periods,
    common wrapper phrases, Wikipedia disambiguation patterns, and
    normalizes common formatting differences (ampersand, ordinals,
    hedging prefixes).
    """
    s = pred.strip()
    # Remove markdown bold/italic
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    # Remove quotes
    if len(s) >= 2 and s[0] in ('"', "'", "\u201c", "\u2018") and s[-1] in ('"', "'", "\u201d", "\u2019"):
        s = s[1:-1].strip()
    s = s.rstrip(".:")

    # Wikipedia disambiguation removal - patterns like "Foo (film)" -> "Foo"
    # Only remove if disambiguation suffix is at the end
    s = re.sub(r"\s*\((?:film|movie|band|singer|actor|actress|musician|album|song|novel|book|TV series|series|disambiguation|person|place|company|organization|politician|athlete|writer|artist|painter|director|composer|scientist|mathematician|physicist|chemist|biologist|philosopher|economist|historian|psychologist|sociologist|anthropologist)\)\s*$", "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', s)

    # Remove common wrapper prefixes
    _PREFIXES = [
        "the answer is: ", "the answer is ",
        "the short answer is: ", "the short answer is ",
        "answer: ", "short answer: ",
        "based on the information, ",
        "based on the context, ",
        "according to the documents, ",
        "approximately ", "about ", "around ", "roughly ",
        "nearly ", "almost ", "more than ", "less than ",
        "at least ", "up to ",
    ]
    s_lower = s.lower()
    for prefix in _PREFIXES:
        if s_lower.startswith(prefix):
            s = s[len(prefix):].strip()
            s_lower = s.lower()

    # Remove trailing parenthetical (often metadata or Wikipedia disambiguation)
    s = re.sub(r"\s*\(.*?\)\s*$", "", s).strip()

    # Normalize full-width to half-width common chars
    s = s.replace("：", ":").replace("，", ",").replace("。", ".")

    return s


_BOLD_RE = re.compile(r"\*\*([^*]{2,80})\*\*")


_BOLD_BLACKLIST = frozenset((
    "missing", "unknown", "not found", "no data", "insufficient",
    "not specify", "not available", "no information", "do not",
    "query", "question", "overview", "summary", "briefing",
    "executive", "context", "status", "result", "event",
    "subject", "key finding", "confirmed", "note",
    "objective", "aucun", "estado", "résultat", "dato",
))


def _extract_bold_entities(text: str) -> List[str]:
    """Extract bold-text entities from markdown as candidate answers.

    Filters out field labels (bold text immediately followed by ':')
    and known non-answer keywords.
    """
    out = []
    for m in _BOLD_RE.finditer(text):
        val = m.group(1).strip()
        val_lower = val.lower()
        if len(val) < 2 or any(kw in val_lower for kw in _BOLD_BLACKLIST):
            continue
        after_pos = m.end()
        if after_pos < len(text) and text[after_pos:after_pos + 1] == ":":
            continue
        out.append(val)
    return out


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
        answer = resp.content.strip()
    except Exception:
        answer = ""

    if answer and answer.lower() != "unknown":
        return answer

    candidates = _extract_bold_entities(verbose)
    if candidates:
        return candidates[0]

    return answer or verbose


# ---------------------------------------------------------------------------
# Evidence-based SP extraction
# ---------------------------------------------------------------------------

def _collect_evidence_texts(cluster: Any, result: Any = None) -> List[str]:
    """Collect textual content from a search result for SP matching.

    Gathers text from two sources:

    1. **Knowledge cluster** — cluster content, description, evidence
       summaries, and raw evidence snippets.
    2. **Search context** (``result``) — search history entries that
       capture the ReAct reasoning text.  This is critical for cases
       where the final ``answer`` is terse (e.g. "1967") but the
       intermediate reasoning mentions article titles ("Daler Mehndi",
       "Tunak Tunak Tun") that are needed for SP matching.
    """
    texts: List[str] = []

    # --- Cluster-level text ---
    if cluster:
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

    # --- Search context text (ReAct reasoning, search queries) ---
    if result is not None:
        for entry in (getattr(result, "search_history", None) or []):
            if isinstance(entry, str) and entry:
                texts.append(entry)
        for entry in (getattr(result, "reasoning_texts", None) or []):
            if isinstance(entry, str) and entry:
                texts.append(entry)

    return texts


_MAX_SENTS_PER_ARTICLE = 3
_MAX_SP_PAIRS_PER_QUERY = 8
_SENT_OVERLAP_THRESHOLD = 0.5
_SENT_MIN_OVERLAP_TOKENS = 4

_SP_STOP = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'and',
    'or', 'but', 'if', 'because', 'while', 'that', 'this',
    'which', 'what', 'it', 'its', 'he', 'she', 'they', 'them',
    'his', 'her', 'their', 'who', 'whom', 'whose', 'not', 'no',
    'so', 'than', 'too', 'very', 'just', 'also', 'only',
})
_WORD_RE = re.compile(r'\b\w+\b')


def _content_tokens(text: str) -> Set[str]:
    """Extract meaningful content tokens for overlap-based matching."""
    return {
        t for t in _WORD_RE.findall(text.lower())
        if t not in _SP_STOP and len(t) >= 2
    }


def _strip_disambiguation(title: str) -> str:
    """Remove Wikipedia disambiguation suffix, e.g. 'Foo (film)' → 'Foo'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def _is_title_relevant(
    title: str,
    context_lower: str,
    context_tokens: Optional[Set[str]] = None,
    files_read: Optional[List[str]] = None,
) -> bool:
    """Determine if a Wikipedia article title is relevant to the search context.

    Checks (in order of priority):
      1. Index lookup: title exists in title index AND file was read.
      2. Exact substring match (full title or without disambiguation suffix).
      3. Token coverage: all meaningful title words appear in context.
         Only applied when the title has ≥ 2 content tokens to avoid
         spurious matches on single common words.
    """
    t = title.lower().strip()
    if len(t) < 3:
        return False

    # Priority 1: Check if title is in index AND was read during search
    if files_read:
        index_files = lookup_title_files(title)
        if index_files:
            # Title found in index - check if any indexed file was read
            files_read_set = set(files_read)
            for idx_file in index_files:
                if idx_file in files_read_set:
                    return True

    # Priority 2: Exact substring match
    if t in context_lower:
        return True
    stripped = _strip_disambiguation(t)
    if stripped != t and len(stripped) >= 3 and stripped in context_lower:
        return True

    # Priority 3: Token coverage (relaxed)
    if context_tokens is not None:
        title_toks = _content_tokens(t)
        if len(title_toks) >= 3 and title_toks.issubset(context_tokens):
            return True

    return False


def _select_relevant_sentences(
    sentences: List[str],
    context_lower: str,
    answer_lower: str = "",
    context_tokens: Optional[Set[str]] = None,
) -> List[int]:
    """Choose which sentence IDs to include in the SP prediction.

    Strategy:
      1. Substring match: include sentences whose leading text appears
         verbatim in the context.
      2. Token overlap: include sentences sharing enough content tokens
         with the evidence/question/answer context — catches bridging
         facts at any position regardless of paraphrasing.
      3. Answer match: include sentences containing the answer text.
    Capped at ``_MAX_SENTS_PER_ARTICLE`` to keep SP precision high.
    """
    selected: Set[int] = set()

    if context_tokens is None:
        context_tokens = _content_tokens(context_lower)

    for sid, sent in enumerate(sentences):
        s = sent.strip() if isinstance(sent, str) else ""
        if not s or len(s) <= 15:
            continue
        s_lower = s.lower()

        prefix = s_lower[:120].rstrip(".,;:!?")
        if len(prefix) > 20 and prefix in context_lower:
            selected.add(sid)
            continue

        sent_toks = _content_tokens(s)
        if sent_toks:
            common = sent_toks & context_tokens
            if (len(common) >= _SENT_MIN_OVERLAP_TOKENS
                    and len(common) / len(sent_toks) >= _SENT_OVERLAP_THRESHOLD):
                selected.add(sid)
                continue

        if answer_lower and len(answer_lower) >= 3 and answer_lower in s_lower:
            selected.add(sid)

    return sorted(selected)[:_MAX_SENTS_PER_ARTICLE]


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
      1. Article title appears as exact substring in the search context
      2. Article sentence content (first 80 chars) appears in the context

    Without this filter, every article in every read JSONL would be
    predicted (~17K SP pairs), collapsing SP precision to near zero.
    """
    all_titles: Set[str] = set()
    predicted_sp: List[List] = []

    _ev = evidence_texts or []
    context_lower = "\n".join([question, answer] + _ev).lower()
    context_tokens = _content_tokens(context_lower)
    from evaluate import normalize_answer
    answer_lower = normalize_answer(answer) if answer else ""

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

                        # --- Relevance check (with index support) ---
                        relevant = _is_title_relevant(
                            title, context_lower, context_tokens,
                            files_read=files_read,
                        )

                        if not relevant:
                            for sent in sentences:
                                s = sent.strip() if isinstance(sent, str) else ""
                                if len(s) > 20 and s[:80].lower() in context_lower:
                                    relevant = True
                                    break

                        if relevant:
                            for sid in _select_relevant_sentences(
                                sentences, context_lower, answer_lower,
                                context_tokens,
                            ):
                                predicted_sp.append([title, sid])
                    except (json_mod.JSONDecodeError, TypeError, IndexError):
                        continue
        except (FileNotFoundError, PermissionError):
            continue

    if len(predicted_sp) > _MAX_SP_PAIRS_PER_QUERY:
        predicted_sp = _prioritize_sp(predicted_sp, answer_lower, context_lower)

    return all_titles, predicted_sp


def _prioritize_sp(
    sp_pairs: List[List],
    answer_lower: str,
    context_lower: str,
) -> List[List]:
    """Rank SP pairs by relevance: answer-bearing > context-dense > rest."""
    answer_toks = _content_tokens(answer_lower) if answer_lower else set()

    def _score(pair: List) -> float:
        title, sid = pair[0], pair[1]
        s = 0.0
        t_lower = title.lower()
        if answer_lower and answer_lower in t_lower:
            s += 10.0
        if answer_toks:
            t_toks = _content_tokens(t_lower)
            s += len(answer_toks & t_toks) * 2.0
        if sid == 0:
            s += 0.5
        return s

    scored = sorted(sp_pairs, key=_score, reverse=True)
    return scored[:_MAX_SP_PAIRS_PER_QUERY]


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
                enable_thinking=cfg.enable_thinking,
                enable_cross_lingual=cfg.enable_cross_lingual,
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
            evidence_texts = _collect_evidence_texts(cluster, result)

            titles, predicted_sp = _extract_titles_and_sp(
                files_read, question, raw_answer, evidence_texts,
            )
            retrieved_titles = list(titles)

            if cfg.extract_answer and raw_answer:
                # Short answers (< 80 chars) are likely already factoid-level;
                # skip the extra LLM call to avoid corruption and latency.
                if len(raw_answer.strip()) > 80:
                    answer = await _extract_short_answer(question, raw_answer, llm)
                    answer = _normalize_prediction(answer)
                    if not answer:
                        answer = _normalize_prediction(raw_answer)
                else:
                    answer = _normalize_prediction(raw_answer)
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

    # Build title→filepath index at startup (one-time cost)
    build_title_index(cfg.wiki_corpus_dir)

    # Resolve ugrep corpus path: explicit config or fall back to wiki_corpus_dir
    _ugrep_cp = cfg.ugrep_corpus_path or cfg.wiki_corpus_dir
    if _ugrep_cp:
        from sirchmunk.retrieve.text_retriever import GrepRetriever
        GrepRetriever.ensure_ugrep_index(_ugrep_cp)

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    searcher = AgenticSearch(
        llm=llm,
        reuse_knowledge=cfg.reuse_knowledge,
        verbose=False,
        enable_memory=cfg.enable_memory,
        rga_max_count=cfg.rga_max_count,
        ugrep_corpus_path=_ugrep_cp,
        highfreq_file_threshold=cfg.highfreq_file_threshold,
        rga_max_parse_lines=cfg.rga_max_parse_lines,
        merge_max_files=cfg.merge_max_files,
    )

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
