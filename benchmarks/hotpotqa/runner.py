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


def _find_corpus_files(wiki_corpus_dir: Path) -> List[Path]:
    """Discover indexable corpus files in the wiki directory.

    Supports two layouts:
      1. ``*.jsonl`` files (explicit extension)
      2. HotpotQA wiki dump: ``<subdir>/wiki_*`` files with no extension,
         where each line is a JSON object with a ``"title"`` key.
    """
    jsonl_files = list(wiki_corpus_dir.rglob("*.jsonl"))
    if jsonl_files:
        return jsonl_files

    wiki_files = sorted(wiki_corpus_dir.rglob("wiki_*"))
    return [f for f in wiki_files if f.is_file() and f.suffix not in (".bz2",)]


def build_title_index(wiki_corpus_dir: Path, progress_interval: int = 2000) -> int:
    """Build title->filepath index from Wikipedia corpus files.

    Scans corpus files (JSONL or HotpotQA wiki_* dumps), parses article
    titles, and builds a normalised title -> filepath mapping for O(1)
    lookup.  Called once at startup.

    Returns:
        Number of unique titles indexed.
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

    corpus_files = _find_corpus_files(wiki_corpus_dir)
    print(f"[TitleIndex] Found {len(corpus_files)} corpus files to index")

    if not corpus_files:
        print("[TitleIndex] No corpus files found — title_lookup will be unavailable")
        _TITLE_INDEX_BUILT = True
        return 0

    for fpath in corpus_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        obj = json_mod.loads(line)
                        title = obj.get("title")
                        if title:
                            norm_title = _normalize_title(title)
                            if norm_title not in _WIKI_TITLE_INDEX:
                                _WIKI_TITLE_INDEX[norm_title] = []
                            fp_str = str(fpath)
                            if fp_str not in _WIKI_TITLE_INDEX[norm_title]:
                                _WIKI_TITLE_INDEX[norm_title].append(fp_str)
                            titles_indexed += 1
                    except (json_mod.JSONDecodeError, TypeError):
                        continue
            files_processed += 1
            if files_processed % progress_interval == 0:
                elapsed_so_far = time.time() - start_time
                print(f"[TitleIndex] Processed {files_processed}/{len(corpus_files)} files, "
                      f"{titles_indexed} titles ({elapsed_so_far:.1f}s)")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
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
- If uncertain, provide your best guess rather than saying "unknown".
- NEVER output "unknown". Always extract the most relevant entity or fact.

Question: {question}
Response:
{response}

Short answer:"""


_YES_NO_RE = re.compile(
    r"^(yes|no)\b",
    re.IGNORECASE,
)


def _normalize_prediction(pred: str) -> str:
    """Post-process prediction to improve EM/F1 matching.

    Strips markdown formatting, quotation marks, trailing periods,
    common wrapper phrases, Wikipedia disambiguation patterns, and
    normalizes common formatting differences (ampersand, ordinals,
    hedging prefixes).  Also collapses verbose yes/no answers to
    just "yes" or "no".
    """
    s = pred.strip()

    # Collapse verbose yes/no answers (e.g. "Yes, Stefan Edberg..." → "yes")
    m = _YES_NO_RE.match(s)
    if m and len(s) > 10:
        return m.group(1).lower()

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
    reasoning_texts: Optional[List[str]] = None,
) -> str:
    """Extract short factoid answer from verbose AgenticSearch output."""
    # Use head+tail to preserve both context opening and final answer portion
    _MAX_EXTRACT_CHARS = 3000
    if len(verbose) <= _MAX_EXTRACT_CHARS:
        trimmed = verbose
    else:
        _head = _MAX_EXTRACT_CHARS * 2 // 3
        _tail = _MAX_EXTRACT_CHARS - _head
        trimmed = verbose[:_head] + "\n...[truncated]...\n" + verbose[-_tail:]
    prompt = _EXTRACT_PROMPT.format(
        question=question,
        response=trimmed,
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

    # Fallback 1: extract bold entities from verbose response
    candidates = _extract_bold_entities(verbose)
    if candidates:
        return candidates[0]

    # Fallback 2: extract bold entities from reasoning_texts (ReAct chain)
    if reasoning_texts:
        for rt in reasoning_texts:
            rt_candidates = _extract_bold_entities(rt)
            if rt_candidates:
                return rt_candidates[0]

    return answer or verbose


# ---------------------------------------------------------------------------
# Evidence-grounded answer extraction
# ---------------------------------------------------------------------------

_YES_NO_STARTERS = frozenset((
    "is", "are", "was", "were", "did", "do", "does",
    "has", "have", "had", "can", "could", "will", "would", "should",
))

_GROUNDED_EXTRACT_PROMPT = """\
Extract the precise answer to the question from the evidence sentences below.

Rules:
- Your answer MUST be grounded in the evidence — use the exact phrasing \
from the evidence whenever possible.
- The answer should be {answer_format}.
- For yes/no questions, output ONLY "yes" or "no".
- For entity or name questions, use the FULL name exactly as it appears \
in the evidence.
- If the evidence is insufficient, output "INSUFFICIENT".
- Output ONLY the answer (1-10 words). No explanation.

Question: {question}

Evidence sentences:
{evidence}

Additional context from search:
{raw_summary}

Answer:"""


def _detect_answer_format(question: str) -> str:
    """Heuristically detect the expected answer format from a question.

    Returns a human-readable format hint for use in extraction prompts.
    This is independent of any upstream decomposition — works on any
    question regardless of the search pipeline used.
    """
    q = question.strip()
    first_word = q.split()[0].lower().rstrip(",.?") if q else ""

    if first_word in _YES_NO_STARTERS:
        return "yes or no"
    if q.lower().startswith(("how many", "how much", "how long", "how old", "how far")):
        return "a number or quantity"
    if first_word == "when":
        return "a date or time period"
    if first_word == "where":
        return "a place name"
    if first_word == "who" or first_word == "whom":
        return "a person or entity name"
    return "a short factual phrase (1-10 words)"


def _collect_sp_evidence_lines(
    predicted_sp: List[List],
    candidates: Dict[str, List[str]],
) -> List[str]:
    """Build evidence lines from predicted SP for extraction / reflection."""
    evidence_lines: List[str] = []
    for sp_item in predicted_sp:
        title = sp_item[0] if isinstance(sp_item, (list, tuple)) and len(sp_item) >= 2 else ""
        sid = int(sp_item[1]) if isinstance(sp_item, (list, tuple)) and len(sp_item) >= 2 else -1
        sents = candidates.get(title, [])
        if 0 <= sid < len(sents):
            sent = sents[sid].strip()
            if sent:
                evidence_lines.append(f"[{title}]: {sent[:300]}")
    return evidence_lines


def _raw_answer_in_evidence(raw_answer: str, evidence_lines: List[str]) -> bool:
    """Check if the (normalised) raw answer appears in any evidence line."""
    norm = _normalize_prediction(raw_answer)
    if not norm:
        return False
    evidence_blob = " ".join(evidence_lines).lower()
    return norm.lower() in evidence_blob


async def _extract_grounded_answer(
    question: str,
    predicted_sp: List[List],
    candidates: Dict[str, List[str]],
    raw_answer: str,
    llm: Any,
) -> str:
    """Extract an answer grounded in supporting-evidence sentences.

    Uses the predicted supporting fact sentences as the primary evidence
    source, forcing the LLM to return an exact span from the evidence
    rather than generating freely.  This reduces form mismatches and
    hallucinated entities.

    After extraction, applies a **safety gate**: if the extracted answer
    diverges from the ReAct chain's raw answer *and* the raw answer is
    already present in the evidence, the raw answer is preferred — the
    multi-step ReAct reasoning is trusted over a single-shot extraction
    that may be misled by surface evidence mentions.

    Returns the extracted answer, or empty string on failure.
    """
    evidence_lines = _collect_sp_evidence_lines(predicted_sp, candidates)
    if not evidence_lines:
        return ""

    answer_format = _detect_answer_format(question)
    raw_summary = raw_answer[:500] if len(raw_answer) > 500 else raw_answer

    prompt = _GROUNDED_EXTRACT_PROMPT.format(
        question=question,
        answer_format=answer_format,
        evidence="\n".join(evidence_lines),
        raw_summary=raw_summary,
    )

    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        answer = (resp.content or "").strip()

        if not answer or answer.upper() == "INSUFFICIENT":
            return ""

        raw_norm = _normalize_prediction(raw_answer)
        ext_norm = _normalize_prediction(answer)

        if raw_norm and ext_norm and raw_norm.lower() != ext_norm.lower():
            if _raw_answer_in_evidence(raw_answer, evidence_lines):
                logger.info(
                    "Safety gate: keeping raw answer %r (grounded in evidence) "
                    "over extraction %r",
                    raw_norm, ext_norm,
                )
                return raw_norm

        return answer
    except Exception as exc:
        logger.warning("Grounded extraction failed: %s", exc)
        return ""


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


_MAX_CANDIDATE_ARTICLES = 30
_MAX_SENTS_PER_CANDIDATE = 10

_STRIP_DISAMBIG_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _parse_wiki_articles(
    files_read: List[str],
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Parse wiki JSONL files into article titles and their sentences.

    Returns (all_titles, articles) where:
      - all_titles: every article title seen (for Evidence Recall metric)
      - articles: {title → [sentence_0, sentence_1, ...]} for articles
        whose sentences could be parsed
    """
    all_titles: Set[str] = set()
    articles: Dict[str, List[str]] = {}

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
                        if isinstance(text[0], list):
                            sentences = [
                                s for para in text
                                for s in (para if isinstance(para, list) else [para])
                            ]
                        else:
                            sentences = list(text)
                        if sentences and title not in articles:
                            articles[title] = sentences
                    except (json_mod.JSONDecodeError, TypeError, IndexError):
                        continue
        except (FileNotFoundError, PermissionError):
            continue

    return all_titles, articles


def _select_candidate_articles(
    articles: Dict[str, List[str]],
    context: str,
) -> Dict[str, List[str]]:
    """Lightweight pre-filter: keep articles whose title appears in context."""
    ctx_lower = context.lower()
    candidates: Dict[str, List[str]] = {}

    for title, sents in articles.items():
        t_lower = title.lower()
        if t_lower in ctx_lower:
            candidates[title] = sents
            continue
        stripped = _STRIP_DISAMBIG_RE.sub("", t_lower).strip()
        if stripped != t_lower and len(stripped) >= 3 and stripped in ctx_lower:
            candidates[title] = sents

    return dict(list(candidates.items())[:_MAX_CANDIDATE_ARTICLES])


def _format_candidates_for_llm(
    candidates: Dict[str, List[str]],
) -> str:
    """Format candidate articles into a compact string for the LLM prompt."""
    parts: List[str] = []
    for title, sents in candidates.items():
        lines = [f'["{title}"]']
        for sid, sent in enumerate(sents[:_MAX_SENTS_PER_CANDIDATE]):
            s = sent.strip() if isinstance(sent, str) else ""
            if s:
                lines.append(f"  {sid}: {s[:200]}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


_SP_EXTRACT_PROMPT = """\
Given a multi-hop question, its answer, and the search reasoning, identify \
the supporting facts from the Wikipedia articles below.

Rules:
- A supporting fact is a [title, sentence_index] pair where sentence_index \
is the 0-based index shown before each sentence.
- Typically 2-5 supporting facts are needed for multi-hop questions.
- Include ONLY sentences that directly provide evidence for answering the question.
- For comparison questions, include the key defining sentence from EACH entity.
- Do NOT include tangential or background sentences.

Question: {question}
Answer: {answer}

Search reasoning (excerpts):
{reasoning}

Candidate articles:
{candidates}

Output ONLY a JSON array of [title, sentence_index] pairs.
Example: [["Article A", 0], ["Article B", 2]]
JSON:"""


async def _extract_sp_with_llm(
    question: str,
    answer: str,
    files_read: List[str],
    llm: Any,
    evidence_texts: Optional[List[str]] = None,
    reasoning_texts: Optional[List[str]] = None,
) -> Tuple[Set[str], List[List], Dict[str, List[str]]]:
    """Extract supporting facts using LLM post-processing.

    Replaces heuristic substring matching with a single LLM call that
    understands the question semantics and selects precise (title, sent_id)
    pairs in the same format as the official gold data.

    Returns (all_titles, predicted_sp, candidates_dict).
    """
    from pathlib import Path as _Path
    files_read = [str(_Path(f).resolve()) for f in files_read]

    all_titles, articles = _parse_wiki_articles(files_read)

    _ev = evidence_texts or []
    _rt = reasoning_texts or []
    context = "\n".join([question, answer] + _ev + _rt)

    candidates = _select_candidate_articles(articles, context)

    if not candidates:
        return all_titles, [], {}

    candidates_text = _format_candidates_for_llm(candidates)

    reasoning_summary = "\n".join(_rt[-5:])
    if len(reasoning_summary) > 2000:
        reasoning_summary = reasoning_summary[:1000] + "\n...[truncated]...\n" + reasoning_summary[-1000:]

    prompt = _SP_EXTRACT_PROMPT.format(
        question=question,
        answer=answer,
        reasoning=reasoning_summary or "(no reasoning available)",
        candidates=candidates_text,
    )

    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        raw = resp.content.strip()

        bracket_start = raw.find("[")
        bracket_end = raw.rfind("]")
        if bracket_start >= 0 and bracket_end > bracket_start:
            raw = raw[bracket_start:bracket_end + 1]

        parsed = json_mod.loads(raw)
        if not isinstance(parsed, list):
            return all_titles, []

        predicted_sp: List[List] = []
        valid_titles = set(candidates.keys())
        for item in parsed:
            if (isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], (int, float))):
                title, sid = item[0], int(item[1])
                if title in valid_titles and 0 <= sid < len(candidates[title]):
                    predicted_sp.append([title, sid])

        return all_titles, predicted_sp, candidates

    except Exception as e:
        logger.warning("SP LLM extraction failed: %s", e)
        return all_titles, [], candidates


# ---------------------------------------------------------------------------
# Answer reflection (optional post-processing)
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT = """\
You are verifying whether an extracted answer is correct and well-formed.

Question: {question}
Current answer: {answer}

Supporting evidence (sentences identified as relevant):
{evidence}

Instructions:
1. Check if the current answer is CONSISTENT with the supporting evidence.
2. Check if the answer is in the correct FORM:
   - If the question asks "who", the answer should be a person/entity name (not an adjective).
   - If the question asks "where", the answer should be a place name.
   - Nationality questions: use the NOUN form of the country (e.g., "Armenia" not "Armenian") \
unless the question explicitly asks for the adjective.
   - Use the FULL name as it appears in the evidence, not abbreviated or partial.
3. Check if the answer is COMPLETE — not truncated or missing key parts.

Output rules:
- If the answer is correct and well-formed, output it EXACTLY as-is.
- If the answer needs correction, output ONLY the corrected answer (1-10 words).
- Do NOT add explanations. Output ONLY the answer.

Verified answer:"""


def _should_reflect(
    question: str,
    answer: str,
    raw_answer: str,
) -> bool:
    """Decide whether reflection is worthwhile for this sample.

    Triggers reflection only when there is meaningful ambiguity or
    form-correction opportunity, avoiding unnecessary LLM calls for
    samples where the answer is already clean and unambiguous.
    """
    if not answer:
        return False

    raw_norm = _normalize_prediction(raw_answer) if raw_answer else ""
    ans_norm = _normalize_prediction(answer)

    # 1) Post-processing changed the answer — verify the change
    if raw_norm and ans_norm and raw_norm.lower() != ans_norm.lower():
        return True

    # 2) Verbose answer that may still contain noise
    if len(answer.split()) > 15:
        return True

    # 3) Format mismatch — yes/no question with non-yes/no answer
    q_lower = question.strip().lower()
    first_word = q_lower.split()[0].rstrip(",.?") if q_lower else ""
    if first_word in _YES_NO_STARTERS:
        if ans_norm.lower() not in ("yes", "no"):
            return True

    return False


async def _reflect_on_answer(
    question: str,
    answer: str,
    predicted_sp: List[List],
    candidates: Dict[str, List[str]],
    llm: Any,
) -> str:
    """Verify and optionally rewrite the answer using supporting evidence.

    Uses the predicted supporting fact sentences as grounding evidence to
    check answer consistency, form correctness, and completeness.  Returns
    the original answer if reflection fails or produces no improvement.
    """
    if not answer or not predicted_sp or not candidates:
        return answer

    evidence_lines = _collect_sp_evidence_lines(predicted_sp, candidates)
    if not evidence_lines:
        return answer

    evidence_text = "\n".join(evidence_lines)
    prompt = _REFLECTION_PROMPT.format(
        question=question,
        answer=answer,
        evidence=evidence_text,
    )

    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        reflected = (resp.content or "").strip()

        if not reflected or reflected.lower() in ("unknown", ""):
            return answer

        reflected = _normalize_prediction(reflected)
        if not reflected:
            return answer

        if reflected.lower() != answer.lower():
            logger.info(
                "Reflection rewrote answer: %r -> %r", answer, reflected,
            )

        return reflected

    except Exception as exc:
        logger.warning("Answer reflection failed: %s", exc)
        return answer


_SAMPLE_RETRYABLE_KEYWORDS = ("429", "RateLimitError", "rate limit",
                              "APIConnectionError", "APITimeoutError",
                              "InternalServerError", "502", "503", "504")
_SAMPLE_RETRY_BASE_DELAY = 30.0


def _is_retryable_error(error_str: str) -> bool:
    """Check whether a sample-level error is transient and worth retrying."""
    e_lower = error_str.lower()
    return any(kw.lower() in e_lower for kw in _SAMPLE_RETRYABLE_KEYWORDS)


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

    max_attempts = cfg.sample_max_retries + 1

    for attempt in range(max_attempts):
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

                evidence_texts = _collect_evidence_texts(cluster, result)
                _reasoning_texts = list(getattr(result, "reasoning_texts", None) or [])

                titles, predicted_sp, sp_candidates = await _extract_sp_with_llm(
                    question, raw_answer, files_read, llm,
                    evidence_texts=evidence_texts,
                    reasoning_texts=_reasoning_texts,
                )
                retrieved_titles = list(titles)

                if cfg.extract_answer and raw_answer:
                    answer = ""

                    # Primary: evidence-grounded extraction (uses SP sentences)
                    if predicted_sp and sp_candidates:
                        answer = await _extract_grounded_answer(
                            question, predicted_sp, sp_candidates,
                            raw_answer, llm,
                        )

                    # Fallback: LLM extraction from verbose raw answer
                    if not answer and len(raw_answer.strip()) > 80:
                        answer = await _extract_short_answer(
                            question, raw_answer, llm,
                            reasoning_texts=_reasoning_texts,
                        )

                    # Last resort: use raw answer directly
                    if not answer:
                        answer = raw_answer

                    answer = _normalize_prediction(answer)
                    if not answer:
                        answer = _normalize_prediction(raw_answer)
                else:
                    answer = _normalize_prediction(raw_answer)

                if cfg.enable_reflection and answer:
                    if _should_reflect(question, answer, raw_answer):
                        answer = await _reflect_on_answer(
                            question, answer, predicted_sp,
                            sp_candidates, llm,
                        )
            except Exception as e:
                error = str(e)

            elapsed = time.time() - t0
            if cfg.request_delay > 0:
                await asyncio.sleep(cfg.request_delay)

        # Retry decision (outside semaphore so slot is released for others)
        if not error or attempt >= max_attempts - 1:
            break
        if not _is_retryable_error(error):
            break

        delay = _SAMPLE_RETRY_BASE_DELAY * (2 ** attempt)
        logging.getLogger(__name__).warning(
            "[run_single] %s attempt %d/%d failed (%s), retrying in %.0fs",
            qid, attempt + 1, max_attempts, error[:80], delay,
        )
        await asyncio.sleep(delay)

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
) -> Tuple[List[Dict[str, Any]], Any]:
    """Run AgenticSearch on all samples with progress tracking.

    Returns (results, searcher) so callers can inject evaluation feedback
    and flush memory after evaluation completes.
    """
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
        max_retries=4,
        retry_base_delay=2.0,
        retry_max_delay=60.0,
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
        title_lookup_fn=lookup_title_files if _WIKI_TITLE_INDEX else None,
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

    return list(results), searcher


# ---------------------------------------------------------------------------
# LLM-Only ablation mode
# ---------------------------------------------------------------------------

_LLM_ONLY_PROMPT = """\
Answer the following question using ONLY your own knowledge. \
Do NOT say you need to search or look anything up. \
Provide a short, direct factoid answer (1-10 words).
For yes/no or comparison questions, answer with: yes or no

Question: {question}

Short answer:"""


async def run_single_llm_only(
    entry: Dict[str, Any],
    llm: Any,
    cfg: ExperimentConfig,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Answer a HotpotQA question using LLM parametric knowledge only."""
    qid = entry["_id"]
    question = entry["question"]

    max_attempts = cfg.sample_max_retries + 1

    for attempt in range(max_attempts):
        async with semaphore:
            t0 = time.time()
            error = None
            raw_answer = ""
            answer = ""
            usage_tokens = 0

            try:
                resp = await llm.achat(
                    messages=[{"role": "user", "content": _LLM_ONLY_PROMPT.format(question=question)}],
                    stream=False,
                    enable_thinking=cfg.enable_thinking,
                )
                raw_answer = (resp.content or "").strip()
                usage = resp.usage or {}
                usage_tokens = (
                    usage.get("total_tokens", 0)
                    or usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
                )

                if raw_answer:
                    answer = _normalize_prediction(raw_answer)
            except Exception as e:
                error = str(e)

            elapsed = time.time() - t0
            if cfg.request_delay > 0:
                await asyncio.sleep(cfg.request_delay)

        if not error or attempt >= max_attempts - 1:
            break
        if not _is_retryable_error(error):
            break

        delay = _SAMPLE_RETRY_BASE_DELAY * (2 ** attempt)
        logging.getLogger(__name__).warning(
            "[llm_only] %s attempt %d/%d failed (%s), retrying in %.0fs",
            qid, attempt + 1, max_attempts, error[:80], delay,
        )
        await asyncio.sleep(delay)

    return {
        "_id": qid,
        "question": question,
        "prediction": answer,
        "raw_prediction": raw_answer,
        "gold_answer": entry.get("answer") or "",
        "type": entry.get("type", ""),
        "level": entry.get("level", ""),
        "elapsed": round(elapsed, 2),
        "telemetry": {
            "total_tokens": usage_tokens,
            "loop_count": 0,
            "files_read": [],
            "llm_calls": 1,
        },
        "retrieved_titles": [],
        "predicted_sp": [],
        "error": error,
    }


async def run_batch_llm_only(
    samples: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """Run LLM-only (no retrieval) on all samples for ablation study."""
    from sirchmunk.llm.openai_chat import OpenAIChat

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        max_retries=4,
        retry_base_delay=2.0,
        retry_max_delay=60.0,
    )

    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    total = len(samples)
    completed = 0

    async def _tracked(entry):
        nonlocal completed
        r = await run_single_llm_only(entry, llm, cfg, semaphore)
        completed += 1
        status = "OK" if not r["error"] else f"ERR: {r['error'][:60]}"
        t = r.get("telemetry", {})
        print(f"  [{completed}/{total}] {r['_id']}  "
              f"{r['elapsed']:.1f}s  tok={t.get('total_tokens', 0)}  "
              f"{status}")
        return r

    tasks = [_tracked(s) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)
