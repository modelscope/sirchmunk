"""FinanceBench evaluation metrics.

Implements the three-class scoring scheme from the FinanceBench paper
(Islam et al., 2023): **correct**, **hallucination**, **refusal**.

Financial-value normalisation handles currency symbols, thousand separators,
trailing zeros, and percentage signs so that ``$1,577.00`` matches ``1577``.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_REFUSAL_PHRASES: list[str] = [
    "i cannot",
    "i can't",
    "i could not",
    "i couldn't",
    "no results found",
    "unable to",
    "not able to",
    "i don't know",
    "i do not know",
    "information is not available",
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "insufficient data",
    "no relevant information",
    "data not found",
    "unknown",
    "i'm not able to",
    "i am not able to",
    "the document does not contain",
    "the document doesn't contain",
    "this information is not disclosed",
    "not disclosed",
    "could not find",
    "couldn't find",
    "no mention of",
    "no information about",
    "not provided in",
    "not found in the document",
    "i was unable to",
    "unable to determine",
    "unable to find",
    "unable to locate",
    "there is no data",
    "no data available",
    "not available in",
    "not specified",
]

_F1_CORRECT_THRESHOLD: float = 0.8

# Markdown / wrapper patterns compiled once
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"\*(.+?)\*")
_RE_QUOTES = re.compile(r'^["\u201c\u201d\']+|["\u201c\u201d\']+$')
_RE_ANSWER_PREFIX = re.compile(
    r"^(the\s+(short\s+)?answer\s+is\s*:?\s*|answer\s*:\s*|short\s+answer\s*:\s*)",
    re.IGNORECASE,
)
# Financial value helpers
_RE_DOLLAR = re.compile(r"^\$\s*")
_RE_THOUSAND_SEP = re.compile(r",(\d{3})")
_RE_TRAILING_ZEROS = re.compile(r"\.0+$")


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------


def normalize_answer(answer: str) -> str:
    """Normalise an answer string for comparison.

    Steps:
    1. Strip Markdown bold / italic.
    2. Strip surrounding quotes.
    3. Strip trailing punctuation (``.``, ``:``).
    4. Remove common LLM wrapper phrases.
    5. Financial value normalisation (currency, commas, trailing zeros).
    6. Lowercase.
    """
    s = answer.strip()
    if not s:
        return ""

    # 1. Markdown
    s = _RE_BOLD.sub(r"\1", s)
    s = _RE_ITALIC.sub(r"\1", s)

    # 2. Quotes
    s = _RE_QUOTES.sub("", s).strip()

    # 3. Trailing punctuation
    s = s.rstrip(".:")

    # 4. Wrapper phrases
    s = _RE_ANSWER_PREFIX.sub("", s).strip()

    # 5. Financial normalisation
    s = _normalize_financial_value(s)

    # 6. Lowercase
    return s.lower().strip()


def _normalize_financial_value(text: str) -> str:
    """Normalise financial figures for robust comparison.

    - ``$1,577.00`` → ``1577``
    - ``15.3%``     → ``15.3%``
    - ``$1577``     → ``1577``
    - ``1,577``     → ``1577``
    - ``($500)``    → ``-500``
    - ``-$500``     → ``-500``
    """
    s = text.strip()

    # Handle accounting bracket notation for negatives: ($500) → -$500
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    # Handle negative sign: remember it, strip it for processing
    negative = False
    if s.startswith("-"):
        negative = True
        s = s[1:]

    # Detect if value looks numeric (possibly with $ / % / commas)
    stripped_for_check = _RE_DOLLAR.sub("", s)
    stripped_for_check = stripped_for_check.replace(",", "").rstrip("%").strip()
    try:
        float(stripped_for_check)
    except ValueError:
        # Not a numeric value – restore negative sign and return as-is
        return ("-" + s) if negative else s

    # Remove dollar sign
    s = _RE_DOLLAR.sub("", s)

    # Remember and temporarily strip percentage
    has_pct = s.endswith("%")
    if has_pct:
        s = s[:-1].strip()

    # Remove thousand-separator commas
    s = s.replace(",", "")

    # Remove trailing decimal zeros: 1577.00 → 1577, 15.30 → 15.3
    if "." in s:
        s = s.rstrip("0").rstrip(".")

    # Re-attach percentage
    if has_pct:
        s = s + "%"

    # Re-attach negative sign
    if negative and not s.startswith("-"):
        s = "-" + s

    return s


# ------------------------------------------------------------------
# Matching helpers
# ------------------------------------------------------------------


def exact_match(prediction: str, gold: str) -> bool:
    """Return ``True`` when normalised strings are identical."""
    return normalize_answer(prediction) == normalize_answer(gold)


def f1_score(prediction: str, gold: str) -> float:
    """Compute token-level F1 between *prediction* and *gold*.

    Tokenisation is simple whitespace splitting after normalisation.
    Each token is further normalised as a financial value so that
    ``$1577`` matches ``1577`` at the token level.
    Returns 0.0 when either side is empty.
    """
    pred_tokens = [_normalize_financial_value(t) for t in normalize_answer(prediction).split()]
    gold_tokens = [_normalize_financial_value(t) for t in normalize_answer(gold).split()]
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------
# Three-class classification
# ------------------------------------------------------------------


def classify_answer(
    prediction: str,
    gold: str,
    *,
    is_no_result: bool = False,
    f1_threshold: float = _F1_CORRECT_THRESHOLD,
) -> str:
    """Classify a prediction into ``correct``, ``refusal``, or ``hallucination``.

    Classification logic (faithful to FinanceBench paper):
    1. If the system explicitly refused (``is_no_result=True``) or the
       prediction contains a refusal phrase → **refusal**.
    2. If EM passes or token-level F1 ≥ *f1_threshold* → **correct**.
    3. Otherwise → **hallucination**.
    """
    norm_pred = normalize_answer(prediction)

    # --- Refusal ---
    if is_no_result:
        return "refusal"
    pred_lower = norm_pred.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in pred_lower:
            return "refusal"

    # --- Correct ---
    if exact_match(prediction, gold):
        return "correct"
    if f1_score(prediction, gold) >= f1_threshold:
        return "correct"

    # --- Hallucination ---
    return "hallucination"


# ------------------------------------------------------------------
# Evidence recall
# ------------------------------------------------------------------


def evidence_recall(
    retrieved_pages: List[int],
    gold_evidence: List[Dict[str, Any]],
) -> float:
    """Compute page-level evidence recall.

    ``gold_evidence`` entries carry ``evidence_page_num`` (0-indexed).
    Returns 1.0 when there is no gold evidence (vacuously true).
    """
    if not gold_evidence:
        return 1.0

    gold_pages = {
        int(e["evidence_page_num"])
        for e in gold_evidence
        if "evidence_page_num" in e
    }
    if not gold_pages:
        return 1.0

    retrieved_set = set(retrieved_pages)
    hits = gold_pages & retrieved_set
    return len(hits) / len(gold_pages)


# ------------------------------------------------------------------
# Aggregate metrics
# ------------------------------------------------------------------


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question results into benchmark-level metrics.

    Expected keys per result dict: ``classification``, ``em``, ``f1``,
    ``elapsed``, ``telemetry``, ``question_type``, ``question_reasoning``,
    ``evidence_recall`` (optional).

    Returns a dict with overall stats plus breakdowns by *question_type*
    and *question_reasoning*.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    # --- Overall counts ---
    correct = sum(1 for r in results if r.get("classification") == "correct")
    halluc = sum(1 for r in results if r.get("classification") == "hallucination")
    refusal = sum(1 for r in results if r.get("classification") == "refusal")

    em_sum = sum(1 for r in results if r.get("em"))
    f1_sum = sum(r.get("f1", 0.0) for r in results)

    latencies = [r["elapsed"] for r in results if "elapsed" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    token_counts = [
        r.get("telemetry", {}).get("total_tokens", 0) for r in results
    ]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0

    ev_recalls = [r["evidence_recall"] for r in results if r.get("evidence_recall") is not None]
    avg_ev_recall = sum(ev_recalls) / len(ev_recalls) if ev_recalls else None

    overall = {
        "n": n,
        "accuracy": round(correct / n * 100, 2),
        "hallucination_rate": round(halluc / n * 100, 2),
        "refusal_rate": round(refusal / n * 100, 2),
        "correct": correct,
        "hallucination": halluc,
        "refusal": refusal,
        "avg_em": em_sum / n,
        "avg_f1": f1_sum / n,
        "avg_latency": round(avg_latency, 2),
        "avg_tokens": round(avg_tokens, 1),
    }
    if avg_ev_recall is not None:
        overall["evidence_recall"] = round(avg_ev_recall, 4)

    # --- LLM Judge metrics (independent dimension, NOT fallback) ---
    judge_results = [r for r in results if r.get("llm_judge_correct") is not None]
    if judge_results:
        judge_correct = sum(1 for r in judge_results if r["llm_judge_correct"])
        overall["llm_judge_accuracy"] = round(judge_correct / len(judge_results) * 100, 2)
        overall["llm_judge_count"] = len(judge_results)
        overall["llm_judge_correct"] = judge_correct
    else:
        overall["llm_judge_accuracy"] = None
        overall["llm_judge_count"] = 0
        overall["llm_judge_correct"] = 0

    # --- Breakdowns ---
    overall["by_question_type"] = _breakdown(results, "question_type")
    overall["by_question_reasoning"] = _breakdown(results, "question_reasoning")

    return overall


def _breakdown(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """Compute per-group accuracy / hallucination / refusal breakdown."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        group = r.get(key, "unknown")
        groups[group].append(r)

    out: dict[str, dict] = {}
    for group, items in sorted(groups.items()):
        g_n = len(items)
        g_correct = sum(1 for r in items if r.get("classification") == "correct")
        g_halluc = sum(
            1 for r in items if r.get("classification") == "hallucination"
        )
        g_refusal = sum(1 for r in items if r.get("classification") == "refusal")
        group_dict: dict[str, Any] = {
            "n": g_n,
            "accuracy": round(g_correct / g_n * 100, 2) if g_n else 0.0,
            "hallucination_rate": round(g_halluc / g_n * 100, 2) if g_n else 0.0,
            "refusal_rate": round(g_refusal / g_n * 100, 2) if g_n else 0.0,
            "correct": g_correct,
            "hallucination": g_halluc,
            "refusal": g_refusal,
        }
        # LLM Judge breakdown
        g_judge = [r for r in items if r.get("llm_judge_correct") is not None]
        if g_judge:
            g_jc = sum(1 for r in g_judge if r["llm_judge_correct"])
            group_dict["llm_judge_accuracy"] = round(g_jc / len(g_judge) * 100, 2)
            group_dict["llm_judge_count"] = len(g_judge)
        out[group] = group_dict
    return out
