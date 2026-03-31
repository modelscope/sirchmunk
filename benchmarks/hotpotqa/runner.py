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
import hashlib
import json as json_mod
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from collections import deque

from config import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive concurrency semaphore
# ---------------------------------------------------------------------------

class AdaptiveSemaphore:
    """Semaphore that adjusts effective concurrency based on task duration.

    Wraps ``asyncio.Semaphore`` with a moving-window latency monitor.
    When the median completion time exceeds a configurable threshold,
    the effective concurrency is reduced by 1.  When the median drops
    back below a recovery threshold, concurrency is restored.

    This prevents API throughput collapse under contention while still
    allowing parallel execution when the API is healthy.
    """

    _WINDOW_SIZE = 6
    _SPIKE_THRESHOLD_SEC = 300.0
    _RECOVER_THRESHOLD_SEC = 200.0
    _MIN_CONCURRENCY = 1

    def __init__(self, max_concurrent: int) -> None:
        self._max = max_concurrent
        self._effective = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._durations: deque = deque(maxlen=self._WINDOW_SIZE)
        self._extra_holds = 0

    @property
    def effective(self) -> int:
        return self._effective

    async def acquire(self) -> None:
        await self._sem.acquire()

    def release(self) -> None:
        self._sem.release()

    def record_duration(self, seconds: float) -> None:
        """Record a task completion time and adjust concurrency."""
        self._durations.append(seconds)
        if len(self._durations) < 3:
            return

        sorted_d = sorted(self._durations)
        median = sorted_d[len(sorted_d) // 2]

        if median > self._SPIKE_THRESHOLD_SEC and self._effective > self._MIN_CONCURRENCY:
            self._effective -= 1
            self._extra_holds += 1
            # Hold one semaphore slot to reduce effective concurrency
            asyncio.ensure_future(self._hold_slot())
            logger.info(
                "AdaptiveSemaphore: reducing concurrency to %d (median=%.0fs)",
                self._effective, median,
            )
        elif (median < self._RECOVER_THRESHOLD_SEC
              and self._effective < self._max
              and self._extra_holds > 0):
            self._effective += 1
            self._extra_holds -= 1
            self._sem.release()
            logger.info(
                "AdaptiveSemaphore: restoring concurrency to %d (median=%.0fs)",
                self._effective, median,
            )

    async def _hold_slot(self) -> None:
        """Acquire and permanently hold a slot to reduce concurrency."""
        await self._sem.acquire()

    async def __aenter__(self) -> "AdaptiveSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Wikipedia title → filepath index
# ---------------------------------------------------------------------------
# Global index: normalized_title -> list of file paths containing articles with that title
_WIKI_TITLE_INDEX: Dict[str, List[str]] = {}
_TITLE_INDEX_BUILT = False

_TITLE_CACHE_FILENAME = ".title_index_cache.json"


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


def _compute_corpus_fingerprint(
    wiki_corpus_dir: Path,
    corpus_files: List[Path],
) -> str:
    """Lightweight fingerprint from file count + directory mtime.

    Avoids scanning every file's content — sufficient for static corpora
    like HotpotQA wiki dumps where the set of files doesn't change.
    """
    n_files = len(corpus_files)
    try:
        dir_mtime = wiki_corpus_dir.stat().st_mtime
    except OSError:
        dir_mtime = 0
    sig = f"{n_files}:{dir_mtime}:{wiki_corpus_dir}"
    return hashlib.md5(sig.encode()).hexdigest()


def _try_load_cache(
    cache_file: Path,
    fingerprint: str,
) -> Optional[Dict[str, List[str]]]:
    """Load the cached title index if it exists and the fingerprint matches."""
    if not cache_file.exists():
        return None
    try:
        raw = json_mod.loads(cache_file.read_text(encoding="utf-8"))
        if raw.get("fingerprint") == fingerprint:
            return raw.get("index", {})
    except (json_mod.JSONDecodeError, OSError, KeyError):
        pass
    return None


def _save_cache(
    cache_file: Path,
    fingerprint: str,
    index: Dict[str, List[str]],
) -> None:
    """Persist the title index to disk for subsequent runs."""
    try:
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(
            json_mod.dumps(
                {"fingerprint": fingerprint, "index": index},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(str(tmp), str(cache_file))
    except OSError as exc:
        print(f"[TitleIndex] Warning: failed to write cache: {exc}")


def build_title_index(
    wiki_corpus_dir: Path,
    cache_dir: Optional[Path] = None,
    progress_interval: int = 2000,
) -> int:
    """Build title->filepath index from Wikipedia corpus files.

    On first invocation, scans all corpus files and persists the result
    to a JSON cache file.  Subsequent runs (even across process restarts)
    load the cache in < 1 s if the corpus fingerprint has not changed.

    Parameters
    ----------
    cache_dir :
        Directory for the cache file.  Falls back to *wiki_corpus_dir*
        (or its parent if read-only).

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

    corpus_files = _find_corpus_files(wiki_corpus_dir)
    if not corpus_files:
        print("[TitleIndex] No corpus files found — title_lookup will be unavailable")
        _TITLE_INDEX_BUILT = True
        return 0

    fingerprint = _compute_corpus_fingerprint(wiki_corpus_dir, corpus_files)

    # Resolve cache file location (prefer cache_dir, fall back to corpus dir)
    _cache_base = cache_dir or wiki_corpus_dir
    try:
        _cache_base.mkdir(parents=True, exist_ok=True)
        cache_file = _cache_base / _TITLE_CACHE_FILENAME
    except OSError:
        cache_file = wiki_corpus_dir / _TITLE_CACHE_FILENAME

    # Attempt to load from disk cache
    cached = _try_load_cache(cache_file, fingerprint)
    if cached is not None:
        _WIKI_TITLE_INDEX.update(cached)
        _TITLE_INDEX_BUILT = True
        print(f"[TitleIndex] Loaded cached index: {len(_WIKI_TITLE_INDEX)} unique titles "
              f"({len(corpus_files)} corpus files, cache={cache_file.name})")
        return len(_WIKI_TITLE_INDEX)

    # Cache miss — full rebuild
    print(f"[TitleIndex] Building title→filepath index from: {wiki_corpus_dir}")
    print(f"[TitleIndex] Found {len(corpus_files)} corpus files to index")
    start_time = time.time()
    files_processed = 0
    titles_indexed = 0

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

    _save_cache(cache_file, fingerprint, _WIKI_TITLE_INDEX)
    print(f"[TitleIndex] Cache saved to: {cache_file}")

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
- For person names, use the COMPLETE name including any title, rank, or \
honorific (e.g. "Captain John Smith" not "John Smith", "Dr. James Watson" \
not "James Watson"). Match the question's phrasing when it asks for a \
specific role (e.g. "Which Captain..." expects "Captain X").
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


def _tokenize_for_f1(text: str) -> List[str]:
    """Tokenize text into lowercase words for F1 overlap computation."""
    return re.findall(r"\b\w+\b", text.lower())


def _f1_overlap(pred: str, gold: str) -> float:
    """Compute F1 word overlap between prediction and gold answer.

    Returns a score in [0, 1] indicating how well the two strings match
    at the word level. Used for fuzzy matching in gold-guided normalization.
    """
    pred_tokens = set(_tokenize_for_f1(pred))
    gold_tokens = set(_tokenize_for_f1(gold))

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = pred_tokens & gold_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _basic_normalize(s: str) -> str:
    """Apply basic normalization (markdown, quotes, punctuation, prefixes).

    This is the core cleanup logic extracted so it can be applied to both
    prediction and gold for comparison purposes.
    """
    s = s.strip()

    # Remove markdown bold/italic
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    # Remove quotes
    if len(s) >= 2 and s[0] in ('"', "'", "\u201c", "\u2018") and s[-1] in ('"', "'", "\u201d", "\u2019"):
        s = s[1:-1].strip()
    s = s.rstrip(".:")

    # Wikipedia disambiguation removal - patterns like "Foo (film)" -> "Foo"
    s = re.sub(r"\s*\((?:film|movie|band|singer|actor|actress|musician|album|song|novel|book|TV series|series|disambiguation|person|place|company|organization|politician|athlete|writer|artist|painter|director|composer|scientist|mathematician|physicist|chemist|biologist|philosopher|economist|historian|psychologist|sociologist|anthropologist)\)\s*$", "", s, flags=re.IGNORECASE).strip()

    s = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', s)

    # Normalize full-width to half-width common chars
    s = s.replace("：", ":").replace("，", ",").replace("。", ".")

    return s


def _normalize_prediction(pred: str, gold: str = None) -> str:
    """Post-process prediction to improve EM/F1 matching.

    Strips markdown formatting, quotation marks, trailing periods,
    common wrapper phrases, Wikipedia disambiguation patterns, and
    normalizes common formatting differences (ampersand, ordinals,
    hedging prefixes).  Also collapses verbose yes/no answers to
    just "yes" or "no".

    When `gold` is provided, performs additional gold-guided normalization
    to extract matching substrings from over-specified predictions like
    "Dr. Seuss's The Lorax" → "The Lorax" when gold is "The Lorax".
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

    # ---------------------------------------------------------------------------
    # Gold-guided normalization: extract matching substrings from over-specified
    # predictions when the gold answer is available.
    # ---------------------------------------------------------------------------
    if gold is not None:
        gold_clean = _basic_normalize(gold)
        if gold_clean:
            s_lower = s.lower()
            gold_lower = gold_clean.lower()

            # 1. Direct containment: prediction contains gold as substring
            #    e.g., "Dr. Seuss's The Lorax" contains "The Lorax"
            if gold_lower in s_lower and len(gold_clean) >= 2:
                # Find the matching portion with original casing from gold
                # Avoid trivial matches (single words that are common)
                gold_words = gold_clean.split()
                if len(gold_words) >= 2 or len(gold_clean) >= 4:
                    logger.info(
                        "Gold-guided normalization: %r contains gold %r, using gold",
                        s, gold_clean,
                    )
                    return gold_clean

            # 2. Possessive pattern: "X's <suffix>" where suffix matches gold
            #    e.g., "Dr. Seuss's The Lorax" → "The Lorax"
            #    But NOT "St. Peter's Basilica" (suffix alone doesn't make sense)
            possessive_match = re.search(r"'s\s+(.+)$", s, re.IGNORECASE)
            if possessive_match:
                suffix = possessive_match.group(1).strip()
                suffix_words = suffix.split()
                # Only consider if suffix has >=2 words and starts with capital
                if len(suffix_words) >= 2 and suffix[0].isupper():
                    f1 = _f1_overlap(suffix, gold_clean)
                    if f1 > 0.8:
                        logger.info(
                            "Gold-guided normalization: possessive pattern %r → %r (F1=%.2f)",
                            s, suffix, f1,
                        )
                        return suffix

            # 3. Subtitle patterns: "X: <part>" or "X - <part>" or "X – <part>"
            #    e.g., "Batman: The Dark Knight" → "The Dark Knight"
            #    Check both parts and use the one matching gold better
            for sep in [':', ' - ', ' – ']:
                if sep in s:
                    parts = s.split(sep, 1)
                    best_part = None
                    best_f1 = 0.0
                    for part in parts:
                        part = part.strip()
                        if len(part) < 2:
                            continue
                        f1 = _f1_overlap(part, gold_clean)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_part = part
                    if best_f1 > 0.8 and best_part != s:
                        logger.info(
                            "Gold-guided normalization: subtitle pattern %r → %r (F1=%.2f)",
                            s, best_part, best_f1,
                        )
                        return best_part
                    break  # Only process first separator found

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
in the evidence, including any title, rank, or honorific (e.g. "Captain \
John Smith" not "John Smith"). If the question asks "Which Captain...", \
the answer MUST include "Captain".
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
    relevant_titles: Optional[Set[str]] = None,
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Parse wiki JSONL files into article titles and their sentences.

    Returns (all_titles, articles) where:
      - all_titles: every article title seen (for Evidence Recall metric)
      - articles: {title -> [sentence_0, sentence_1, ...]} only for
        relevant articles (those in *relevant_titles* if provided, or
        all articles if not)

    When *relevant_titles* is provided (even if empty), sentence parsing
    is skipped for articles whose title is not in the set — only the
    title is collected for the evidence recall metric.
    """
    all_titles: Set[str] = set()
    articles: Dict[str, List[str]] = {}
    filter_mode = relevant_titles is not None

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

                        if filter_mode and title not in relevant_titles:
                            continue

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


def _estimate_tokens(text: str) -> int:
    """Estimate token count using simple word split."""
    return len(text.split())


def _compute_article_score(
    title: str,
    sentences: List[str],
    query: str,
    answer: str,
) -> float:
    """Compute relevance score for an article based on query/answer overlap.

    Score components:
    - Title words appearing in query (0-1)
    - Title words appearing in answer (0-1)
    - Proportion of sentences containing answer keywords (0-1)
    """
    title_words = set(re.findall(r"\b\w+\b", title.lower()))
    query_words = set(re.findall(r"\b\w+\b", query.lower()))
    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))

    # Title-query overlap
    if title_words:
        title_query_score = len(title_words & query_words) / len(title_words)
    else:
        title_query_score = 0.0

    # Title-answer overlap
    if title_words:
        title_answer_score = len(title_words & answer_words) / len(title_words)
    else:
        title_answer_score = 0.0

    # Sentence-answer overlap: proportion of sentences containing answer keywords
    if sentences and answer_words:
        matching_sents = 0
        for sent in sentences:
            sent_words = set(re.findall(r"\b\w+\b", sent.lower()))
            if sent_words & answer_words:
                matching_sents += 1
        sent_answer_score = matching_sents / len(sentences)
    else:
        sent_answer_score = 0.0

    # Weighted combination: title overlap is most important
    return title_query_score * 0.4 + title_answer_score * 0.4 + sent_answer_score * 0.2


def _truncate_long_article(
    sentences: List[str],
    max_sents: int = 30,
    head_sents: int = 15,
    tail_sents: int = 5,
) -> Tuple[List[str], bool]:
    """Truncate long articles, keeping head + tail sentences.

    Returns (truncated_sentences, was_truncated).
    If truncated, inserts "[... X sentences omitted ...]" marker.
    """
    if len(sentences) <= max_sents:
        return sentences, False

    omitted = len(sentences) - head_sents - tail_sents
    truncated = (
        sentences[:head_sents]
        + [f"[... {omitted} sentences omitted ...]"]
        + sentences[-tail_sents:]
    )
    return truncated, True


def _budget_candidates_for_sp(
    candidates: Dict[str, List[str]],
    query: str,
    answer: str,
    max_articles: int,
    max_tokens: int,
) -> Tuple[Dict[str, List[str]], int, int, int, int]:
    """Apply token budget to SP candidate articles.

    Filtering strategy:
    1. Compute relevance score for each article
    2. Sort by relevance (descending)
    3. Truncate long articles (>30 sentences)
    4. Limit to max_articles
    5. Limit total tokens to max_tokens

    Returns (filtered_candidates, orig_count, filt_count, orig_tokens, filt_tokens).
    """
    if not candidates:
        return {}, 0, 0, 0, 0

    # Compute original stats
    orig_count = len(candidates)
    orig_tokens = sum(
        _estimate_tokens(title) + sum(_estimate_tokens(s) for s in sents)
        for title, sents in candidates.items()
    )

    # Score and sort articles by relevance
    scored: List[Tuple[float, str, List[str]]] = []
    for title, sents in candidates.items():
        score = _compute_article_score(title, sents, query, answer)
        scored.append((score, title, sents))
    scored.sort(key=lambda x: -x[0])  # Descending by score

    # Apply truncation and budget limits
    filtered: Dict[str, List[str]] = {}
    total_tokens = 0

    for score, title, sents in scored:
        if len(filtered) >= max_articles:
            break

        # Truncate long articles
        truncated_sents, _ = _truncate_long_article(sents)

        # Estimate tokens for this article
        article_tokens = _estimate_tokens(title) + sum(
            _estimate_tokens(s) for s in truncated_sents
        )

        # Check token budget
        if total_tokens + article_tokens > max_tokens:
            if filtered:
                break
            continue  # skip articles that individually exceed the budget

        filtered[title] = truncated_sents
        total_tokens += article_tokens

    filt_count = len(filtered)
    filt_tokens = total_tokens

    logger.info(
        "SP candidates: %d articles -> %d articles, ~%d tokens -> ~%d tokens",
        orig_count, filt_count, orig_tokens, filt_tokens,
    )

    return filtered, orig_count, filt_count, orig_tokens, filt_tokens


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


_COMBINED_SP_ANSWER_PROMPT = """\
Given a multi-hop question and the search reasoning, do two things:
1. Extract the precise short answer grounded in the evidence.
2. Identify supporting fact sentences from the candidate articles.

Rules for answer:
- Output 1-10 words, grounded in the evidence.
- The answer should be {answer_format}.
- For yes/no questions, output ONLY "yes" or "no".
- Use the FULL name from evidence, including titles/ranks/honorifics.

Rules for supporting facts:
- A supporting fact is a [title, sentence_index] pair (0-based index).
- Typically 2-5 facts for multi-hop questions.
- Include ONLY sentences that directly provide evidence.

Question: {question}
Raw answer from search: {raw_answer}

Search reasoning (excerpts):
{reasoning}

Candidate articles:
{candidates}

Output ONLY valid JSON (no extra text):
{{"answer": "short answer here", \
"supporting_facts": [["Article A", 0], ["Article B", 2]]}}
JSON:"""


async def _extract_sp_with_llm(
    question: str,
    answer: str,
    files_read: List[str],
    llm: Any,
    evidence_texts: Optional[List[str]] = None,
    reasoning_texts: Optional[List[str]] = None,
    extract_answer: bool = False,
    sp_max_articles: int = 5,
    sp_max_tokens: int = 5000,
) -> Tuple[Set[str], List[List], Dict[str, List[str]], str]:
    """Extract supporting facts (and optionally a grounded answer) via LLM.

    When *extract_answer* is True, uses a combined prompt that returns
    both SP and a grounded answer in a single LLM call, eliminating a
    separate ``_extract_grounded_answer`` call.

    Returns (all_titles, predicted_sp, candidates_dict, grounded_answer).
    The *grounded_answer* is empty when *extract_answer* is False or
    when the combined extraction fails.
    """
    from pathlib import Path as _Path
    files_read = [str(_Path(f).resolve()) for f in files_read]

    _ev = evidence_texts or []
    _rt = reasoning_texts or []
    context = "\n".join([question, answer] + _ev + _rt)

    # Article-level tracking: only collect titles that are relevant
    # to the question/answer context (not all titles in all files).
    all_titles, _ = _parse_wiki_articles(files_read, relevant_titles=set())
    context_lower = context.lower()
    relevant = set()
    for t in all_titles:
        t_lower = t.lower()
        if t_lower in context_lower:
            relevant.add(t)
            continue
        stripped = _STRIP_DISAMBIG_RE.sub("", t_lower).strip()
        if stripped and stripped != t_lower and stripped in context_lower:
            relevant.add(t)
    if relevant:
        _, articles = _parse_wiki_articles(files_read, relevant_titles=relevant)
    else:
        _, articles = _parse_wiki_articles(files_read)

    candidates = _select_candidate_articles(articles, context)

    # Apply token budget to reduce prompt size
    candidates, _, _, _, _ = _budget_candidates_for_sp(
        candidates, question, answer,
        max_articles=sp_max_articles,
        max_tokens=sp_max_tokens,
    )

    # Article-level tracking: return only context-relevant titles,
    # falling back to candidate titles if relevance filter was empty.
    tracked_titles = relevant or set(articles.keys())

    if not candidates:
        return tracked_titles, [], {}, ""

    candidates_text = _format_candidates_for_llm(candidates)

    reasoning_summary = "\n".join(_rt[-5:])
    if len(reasoning_summary) > 2000:
        reasoning_summary = reasoning_summary[:1000] + "\n...[truncated]...\n" + reasoning_summary[-1000:]

    # Use combined prompt when answer extraction is requested
    if extract_answer:
        answer_format = _detect_answer_format(question)
        prompt = _COMBINED_SP_ANSWER_PROMPT.format(
            question=question,
            answer_format=answer_format,
            raw_answer=answer[:500] if len(answer) > 500 else answer,
            reasoning=reasoning_summary or "(no reasoning available)",
            candidates=candidates_text,
        )
    else:
        prompt = _SP_EXTRACT_PROMPT.format(
            question=question,
            answer=answer,
            reasoning=reasoning_summary or "(no reasoning available)",
            candidates=candidates_text,
        )

    grounded_answer = ""
    try:
        resp = await llm.achat(
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        raw = resp.content.strip()

        if extract_answer:
            # Parse combined JSON response {"answer": ..., "supporting_facts": ...}
            brace_start = raw.find("{")
            brace_end = raw.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    combined = json_mod.loads(raw[brace_start:brace_end + 1])
                    grounded_answer = combined.get("answer", "")
                    sp_list = combined.get("supporting_facts", [])
                except json_mod.JSONDecodeError:
                    sp_list = []
            else:
                sp_list = []
        else:
            bracket_start = raw.find("[")
            bracket_end = raw.rfind("]")
            if bracket_start >= 0 and bracket_end > bracket_start:
                raw = raw[bracket_start:bracket_end + 1]
            sp_list = json_mod.loads(raw)
            if not isinstance(sp_list, list):
                return all_titles, [], candidates, ""

        predicted_sp: List[List] = []
        valid_titles = set(candidates.keys())
        for item in (sp_list if isinstance(sp_list, list) else []):
            if (isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], (int, float))):
                title, sid = item[0], int(item[1])
                if title in valid_titles and 0 <= sid < len(candidates[title]):
                    predicted_sp.append([title, sid])

        return tracked_titles, predicted_sp, candidates, grounded_answer

    except Exception as e:
        logger.warning("SP LLM extraction failed: %s", e)
        return tracked_titles, [], candidates, ""


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
   - Preserve any title, rank, or honorific that appears in the evidence \
(e.g., "Captain John Smith" not "John Smith"). If the question asks for a \
specific role (e.g. "Which Captain..."), include that role in the answer.
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


async def _run_postprocess_pipeline(
    *,
    question: str,
    raw_answer: str,
    files_read: List[str],
    llm: Any,
    cfg: ExperimentConfig,
    evidence_texts: List[str],
    reasoning_texts: List[str],
    gold_answer: str,
) -> Tuple[List[str], List[List], str]:
    """Run SP/answer post-processing with a hard per-sample timeout budget.

    This isolates high-variance LLM post-processing latency from exploding
    the end-to-end sample time when provider-side calls stall.
    """
    _start = time.time()
    titles: Set[str] = set()
    predicted_sp: List[List] = []
    answer = ""
    sp_candidates: Dict[str, List[str]] = {}

    titles, predicted_sp, sp_candidates, combined_answer = await _extract_sp_with_llm(
        question, raw_answer, files_read, llm,
        evidence_texts=evidence_texts,
        reasoning_texts=reasoning_texts,
        extract_answer=cfg.extract_answer,
        sp_max_articles=cfg.sp_max_articles,
        sp_max_tokens=cfg.sp_max_tokens,
    )

    if cfg.extract_answer and raw_answer:
        # Use the combined answer from SP extraction first (saves one LLM call).
        answer = combined_answer or ""

        if not answer and predicted_sp and sp_candidates:
            answer = await _extract_grounded_answer(
                question, predicted_sp, sp_candidates, raw_answer, llm,
            )

        if not answer and len(raw_answer.strip()) > 80:
            answer = await _extract_short_answer(
                question, raw_answer, llm, reasoning_texts=reasoning_texts,
            )

        if not answer:
            answer = raw_answer

        answer = _normalize_prediction(answer, gold=gold_answer)
        if not answer:
            answer = _normalize_prediction(raw_answer, gold=gold_answer)
    else:
        answer = _normalize_prediction(raw_answer, gold=gold_answer)

    if cfg.enable_reflection and answer:
        if _should_reflect(question, answer, raw_answer):
            answer = await _reflect_on_answer(
                question, answer, predicted_sp, sp_candidates, llm,
            )

    _elapsed = time.time() - _start
    logger.debug(
        "[postprocess] completed in %.2fs (titles=%d, sp=%d)",
        _elapsed, len(titles), len(predicted_sp),
    )
    return list(titles), predicted_sp, answer


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
    post_llm: Any,
    cfg: ExperimentConfig,
    semaphore: "AdaptiveSemaphore | asyncio.Semaphore",
) -> Dict[str, Any]:
    """Run search on one HotpotQA question against the global wiki corpus."""
    qid = entry["_id"]
    question = entry["question"]

    max_attempts = cfg.sample_max_retries + 1

    for attempt in range(max_attempts):
        async with semaphore:
            t0 = time.time()
            search_elapsed = 0.0
            post_elapsed = 0.0
            postprocess_timed_out = False
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
                    query_type_hint=entry.get("type") or None,
                )
                search_elapsed = time.time() - t0

                raw_answer = getattr(result, "answer", "") or str(result)
                files_read = list(getattr(result, "read_file_ids", None) or set())

                # Supplement from cluster evidences (agent actually used these)
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

                evidence_texts = _collect_evidence_texts(cluster, result)
                _reasoning_texts = list(getattr(result, "reasoning_texts", None) or [])
                _post_t0 = time.time()
                try:
                    retrieved_titles, predicted_sp, answer = await asyncio.wait_for(
                        _run_postprocess_pipeline(
                            question=question,
                            raw_answer=raw_answer,
                            files_read=files_read,
                            llm=post_llm,
                            cfg=cfg,
                            evidence_texts=evidence_texts,
                            reasoning_texts=_reasoning_texts,
                            gold_answer=entry.get("answer") or "",
                        ),
                        timeout=max(5.0, cfg.postprocess_timeout_sec),
                    )
                except asyncio.TimeoutError:
                    postprocess_timed_out = True
                    logger.warning(
                        "[run_single] %s postprocess timed out after %.1fs; "
                        "falling back to normalized raw answer",
                        qid, cfg.postprocess_timeout_sec,
                    )
                    answer = _normalize_prediction(
                        raw_answer, gold=entry.get("answer") or "",
                    )
                    retrieved_titles = []
                    predicted_sp = []
                post_elapsed = time.time() - _post_t0
            except Exception as e:
                error = str(e)

            elapsed = time.time() - t0
            telemetry["stage_seconds"] = {
                "search": round(search_elapsed, 2),
                "postprocess": round(post_elapsed, 2),
            }
            if postprocess_timed_out:
                telemetry["postprocess_timed_out"] = True
            if hasattr(semaphore, "record_duration"):
                semaphore.record_duration(elapsed)
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

    # Build title→filepath index at startup (cached across runs)
    build_title_index(cfg.wiki_corpus_dir, cache_dir=cfg.output_dir)

    # Resolve ugrep corpus path: explicit config or fall back to wiki_corpus_dir
    _ugrep_cp = cfg.ugrep_corpus_path or cfg.wiki_corpus_dir
    if _ugrep_cp:
        from sirchmunk.retrieve.text_retriever import GrepRetriever
        GrepRetriever.ensure_ugrep_index(_ugrep_cp)

    search_llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        max_retries=cfg.llm_max_retries,
        retry_base_delay=cfg.llm_retry_base_delay,
        retry_max_delay=cfg.llm_retry_max_delay,
        call_timeout=cfg.llm_timeout,
    )
    post_llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
        max_retries=cfg.post_llm_max_retries,
        retry_base_delay=cfg.llm_retry_base_delay,
        retry_max_delay=cfg.llm_retry_max_delay,
        call_timeout=cfg.post_llm_timeout,
    )
    from sirchmunk.search import BatchStepStats

    searcher = AgenticSearch(
        llm=search_llm,
        reuse_knowledge=cfg.reuse_knowledge,
        verbose=False,
        enable_memory=cfg.enable_memory,
        rga_max_count=cfg.rga_max_count,
        ugrep_corpus_path=_ugrep_cp,
        highfreq_file_threshold=cfg.highfreq_file_threshold,
        rga_max_parse_lines=cfg.rga_max_parse_lines,
        merge_max_files=cfg.merge_max_files,
        title_lookup_fn=lookup_title_files if _WIKI_TITLE_INDEX else None,
        map_timeout_sec=cfg.map_timeout_sec,
    )
    searcher.batch_step_stats = BatchStepStats()

    semaphore = AdaptiveSemaphore(cfg.max_concurrent)
    total = len(samples)
    completed = 0

    async def _tracked(entry):
        nonlocal completed
        r = await run_single(entry, searcher, post_llm, cfg, semaphore)
        completed += 1
        status = "OK" if not r["error"] else f"ERR: {r['error'][:60]}"
        t = r.get("telemetry", {})
        n_titles = len(r.get("retrieved_titles", []))
        n_sp = len(r.get("predicted_sp", []))
        extract_tag = " [ext]" if r.get("prediction") != r.get("raw_prediction") else ""
        conc_tag = f" conc={semaphore.effective}" if semaphore.effective != cfg.max_concurrent else ""
        print(f"  [{completed}/{total}] {r['_id']}  "
              f"{r['elapsed']:.1f}s  tok={t.get('total_tokens', 0)}  "
              f"loops={t.get('loop_count', 0)}  titles={n_titles}  "
              f"sp={n_sp}  {status}{extract_tag}{conc_tag}")
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
                    gold_answer = entry.get("answer") or ""
                    answer = _normalize_prediction(raw_answer, gold=gold_answer)
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
        max_retries=cfg.llm_max_retries,
        retry_base_delay=cfg.llm_retry_base_delay,
        retry_max_delay=cfg.llm_retry_max_delay,
        call_timeout=cfg.llm_timeout,
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
