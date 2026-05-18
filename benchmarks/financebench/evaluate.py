"""FinanceBench evaluation metrics — LLM Judge driven.

All correctness evaluation (Accuracy, Coverage) is driven by the LLM Judge.
This module aggregates per-question judge results into benchmark-level metrics.

The ``normalize_answer`` helper is retained for quick short-circuit checks
inside the judge (exact-match bypass before calling the LLM).
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

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
# Aggregate metrics
# ------------------------------------------------------------------


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question results into benchmark-level metrics.

    All correctness evaluation is driven by LLM Judge results stored in
    each result dict (``judge_correct``, ``coverage``).

    Returns a dict with overall stats plus breakdown by *question_type*.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    # --- Accuracy (Judge) ---
    judge_correct = sum(1 for r in results if r.get("judge_correct"))

    # --- Coverage (Judge) ---
    coverage_true = sum(1 for r in results if r.get("coverage"))

    # --- Latency ---
    latencies = [r["elapsed"] for r in results if "elapsed" in r]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    total_time = sum(latencies)

    # --- Token usage ---
    search_tokens = sum(
        r.get("telemetry", {}).get("total_tokens", 0) for r in results
    )
    judge_tokens = sum(r.get("judge_tokens", 0) for r in results)
    total_tokens = search_tokens + judge_tokens
    avg_tokens_per_question = total_tokens / n if n else 0

    overall: Dict[str, Any] = {
        "n": n,
        "accuracy": round(judge_correct / n * 100, 2),
        "coverage": round(coverage_true / n * 100, 2),
        "avg_latency": round(avg_latency, 2),
        "total_time_seconds": round(total_time, 2),
        "token_usage": {
            "total_tokens": total_tokens,
            "search_tokens": search_tokens,
            "judge_tokens": judge_tokens,
            "avg_tokens_per_question": round(avg_tokens_per_question, 1),
        },
        "judge_correct": judge_correct,
        "coverage_true": coverage_true,
        "by_question_type": _breakdown(results, "question_type"),
    }

    return overall


def _breakdown(
    results: List[Dict[str, Any]], key: str
) -> Dict[str, Dict[str, Any]]:
    """Compute per-group accuracy / coverage breakdown."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        group = r.get(key) or "unknown"
        groups[group].append(r)

    out: dict[str, dict] = {}
    for group, items in sorted(
        groups.items(), key=lambda x: (x[0] is None, x[0] or "")
    ):
        g_n = len(items)
        g_correct = sum(1 for r in items if r.get("judge_correct"))
        g_coverage = sum(1 for r in items if r.get("coverage"))
        out[group] = {
            "n": g_n,
            "accuracy": round(g_correct / g_n * 100, 2) if g_n else 0.0,
            "coverage": round(g_coverage / g_n * 100, 2) if g_n else 0.0,
            "judge_count": g_n,
            "judge_correct": g_correct,
        }
    return out
