"""HotpotQA evaluation metrics aligned with the official leaderboard.

Implements the six core metrics from Yang et al. (2018) §5.2, Table 4:
  - Ans  (EM, F1): Answer exact match and token-level F1
  - Sup  (EM, F1): Supporting-fact set-level EM and F1
  - Joint(EM, F1): Combined answer × supporting-fact metrics

Joint metrics follow the paper's definition:
  P_joint = P_ans × P_sup,  R_joint = R_ans × R_sup
  Joint_F1 = 2 × P_joint × R_joint / (P_joint + R_joint)
  Joint_EM = 1 iff Ans_EM = 1 AND Sup_EM = 1

Additional (non-leaderboard) metrics retained for diagnostic purposes:
  - Contain-Match Accuracy (LinearRAG-style bidirectional substring match)
  - Evidence Recall (title-level SP coverage proxy)

Reference: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Paper:     https://arxiv.org/pdf/1809.09600
"""

import re
import string
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Answer-level helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text, remove articles / punctuation / extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """Token-level F1 with yes/no/noanswer special-casing.

    Returns (f1, precision, recall) — all three are needed for Joint metrics.

    Per the official HotpotQA eval script, if either the prediction or the
    ground truth is one of {yes, no, noanswer} and they don't match after
    normalisation, return (0, 0, 0).
    """
    ZERO = (0.0, 0.0, 0.0)
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    _SPECIAL = {"yes", "no", "noanswer"}
    if norm_pred in _SPECIAL and norm_pred != norm_gt:
        return ZERO
    if norm_gt in _SPECIAL and norm_pred != norm_gt:
        return ZERO

    pred_tokens = norm_pred.split()
    gt_tokens = norm_gt.split()
    if not pred_tokens or not gt_tokens:
        return ZERO

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def contain_match_score(prediction: str, ground_truth: str) -> int:
    """Bidirectional Contain-Match (LinearRAG-style)."""
    np_ = normalize_answer(prediction)
    ng = normalize_answer(ground_truth)
    if not np_ or not ng:
        return 0
    return int(ng in np_ or np_ in ng)


# ---------------------------------------------------------------------------
# Supporting-fact helpers
# ---------------------------------------------------------------------------

def _to_sp_set(sp_list) -> Set[Tuple[str, int]]:
    """Normalise supporting-fact data into a set of (title, sent_id) tuples.

    Accepts:
      - list of [title, sent_id] pairs  (official JSON format)
      - dict with parallel "title" / "sent_id" arrays  (parquet format)
    """
    if isinstance(sp_list, dict):
        titles = sp_list.get("title", [])
        sids = sp_list.get("sent_id", [])
        return {(t, int(s)) for t, s in zip(titles, sids)}
    if isinstance(sp_list, (list, tuple)):
        return {(t, int(s)) for t, s in sp_list}
    return set()


def sp_prec_recall_f1(
    predicted_sp: Set[Tuple[str, int]],
    gold_sp: Set[Tuple[str, int]],
) -> Tuple[float, float, float, float]:
    """Set-level EM, precision, recall, F1 for supporting facts.

    Aligned with official ``update_sp`` in hotpot_evaluate_v1.py:
    tp/fp/fn counting → prec, recall, f1, em.

    Returns (sp_em, sp_prec, sp_recall, sp_f1).
    """
    tp = len(predicted_sp & gold_sp)
    fp = len(predicted_sp - gold_sp)
    fn = len(gold_sp - predicted_sp)

    prec = 1.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    sp_em = 1.0 if (fp + fn) == 0 else 0.0

    return sp_em, prec, recall, f1


# ---------------------------------------------------------------------------
# Title-level diagnostics (non-leaderboard)
# ---------------------------------------------------------------------------

def _normalize_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _strip_disambiguation(t: str) -> str:
    """Remove Wikipedia disambiguation suffixes like '(film)', '(band)'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()


def _fuzzy_title_match(pred_title: str, gold_title: str) -> bool:
    """Check if two titles refer to the same entity with lenient matching.

    Handles disambiguation suffixes and containment, e.g.:
      - "Universal Soldier (franchise)" vs "Universal Soldier: Day of Reckoning"
      - "The Outsiders (film)" vs "The Outsiders"
    """
    np = _normalize_title(pred_title)
    ng = _normalize_title(gold_title)
    if np == ng:
        return True
    np_stripped = _normalize_title(_strip_disambiguation(pred_title))
    ng_stripped = _normalize_title(_strip_disambiguation(gold_title))
    if np_stripped and ng_stripped and np_stripped == ng_stripped:
        return True
    if np_stripped and ng_stripped:
        if np_stripped in ng_stripped or ng_stripped in np_stripped:
            return True
    return False


def _sp_title_prec_recall_f1(
    predicted_sp: Set[Tuple[str, int]],
    gold_sp: Set[Tuple[str, int]],
) -> Tuple[float, float, float]:
    """Title-level fuzzy SP precision, recall, F1 (diagnostic only).

    Ignores sent_id — only checks whether the predicted title set
    overlaps the gold title set using fuzzy matching.
    """
    pred_titles = {t for t, _ in predicted_sp}
    gold_titles = {t for t, _ in gold_sp}

    if not gold_titles and not pred_titles:
        return 1.0, 1.0, 1.0
    if not gold_titles or not pred_titles:
        return 0.0, 0.0, 0.0

    tp_pred = sum(
        1 for pt in pred_titles
        if any(_fuzzy_title_match(pt, gt) for gt in gold_titles)
    )
    tp_gold = sum(
        1 for gt in gold_titles
        if any(_fuzzy_title_match(pt, gt) for pt in pred_titles)
    )

    prec = tp_pred / len(pred_titles) if pred_titles else 0.0
    recall = tp_gold / len(gold_titles) if gold_titles else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    return prec, recall, f1


def _evidence_recall(
    retrieved_titles: List[str],
    supporting_facts,
) -> float:
    """Fraction of gold SP titles found among retrieved article titles."""
    if isinstance(supporting_facts, dict):
        sf_titles = supporting_facts.get("title", [])
    elif isinstance(supporting_facts, (list, tuple)):
        sf_titles = [pair[0] for pair in supporting_facts]
    else:
        return 0.0
    if not sf_titles:
        return 0.0
    unique_gold: Set[str] = set(sf_titles)
    norm_retrieved: Set[str] = {_normalize_title(t) for t in retrieved_titles}
    found = sum(
        1 for g in unique_gold if _normalize_title(g) in norm_retrieved
    )
    return found / len(unique_gold)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    results: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all leaderboard metrics plus diagnostic metrics.

    Leaderboard (per-sample, then averaged):
      - Ans EM / F1
      - Sup EM / F1
      - Joint EM / F1

    Diagnostic:
      - Contain-Match Accuracy
      - Evidence Recall (title-level)

    Returns dict with keys: overall, by_type, by_level, per_sample.
    """
    sample_map = {s["_id"]: s for s in samples}

    METRIC_KEYS = [
        "ans_em", "ans_f1", "ans_prec", "ans_recall",
        "sp_em", "sp_f1", "sp_prec", "sp_recall",
        "joint_em", "joint_f1", "joint_prec", "joint_recall",
        "contain", "ev_recall",
        "sp_title_prec", "sp_title_recall", "sp_title_f1",
    ]
    overall: Dict[str, list] = {k: [] for k in METRIC_KEYS}
    by_type: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in METRIC_KEYS})
    by_level: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {k: [] for k in METRIC_KEYS})
    per_sample: List[Dict[str, Any]] = []

    for r in results:
        qid = r["_id"]
        sample = sample_map.get(qid)
        if not sample or r.get("error"):
            continue
        gold = sample.get("answer", "")
        pred = r.get("prediction", "")
        if not gold:
            continue

        # --- Answer metrics ---
        ans_em = exact_match_score(pred, gold)
        ans_f1, ans_prec, ans_recall = f1_score(pred, gold)

        # --- Supporting-fact metrics ---
        gold_sp = _to_sp_set(sample.get("supporting_facts", {}))
        pred_sp = _to_sp_set(r.get("predicted_sp", []))
        sp_em, sp_prec, sp_recall, sp_f1 = sp_prec_recall_f1(pred_sp, gold_sp)

        # --- Joint metrics (paper §5.2) ---
        joint_prec = ans_prec * sp_prec
        joint_recall = ans_recall * sp_recall
        joint_f1 = (
            (2 * joint_prec * joint_recall / (joint_prec + joint_recall))
            if (joint_prec + joint_recall) > 0 else 0.0
        )
        joint_em = float(ans_em and sp_em)

        # --- Title-level fuzzy SP (diagnostic) ---
        sp_t_prec, sp_t_recall, sp_t_f1 = _sp_title_prec_recall_f1(
            pred_sp, gold_sp,
        )

        # --- Diagnostic metrics ---
        contain = contain_match_score(pred, gold)
        retrieved_titles = r.get("retrieved_titles", [])
        sf = sample.get("supporting_facts", {})
        ev_recall = _evidence_recall(retrieved_titles, sf)

        q_type = sample.get("type", "unknown")
        q_level = sample.get("level", "unknown")

        row = {
            "ans_em": ans_em, "ans_f1": ans_f1,
            "ans_prec": ans_prec, "ans_recall": ans_recall,
            "sp_em": sp_em, "sp_f1": sp_f1,
            "sp_prec": sp_prec, "sp_recall": sp_recall,
            "joint_em": joint_em, "joint_f1": joint_f1,
            "joint_prec": joint_prec, "joint_recall": joint_recall,
            "contain": contain, "ev_recall": ev_recall,
            "sp_title_prec": sp_t_prec, "sp_title_recall": sp_t_recall,
            "sp_title_f1": sp_t_f1,
        }

        for bucket in [overall, by_type[q_type], by_level[q_level]]:
            for k in METRIC_KEYS:
                bucket[k].append(row[k])

        per_sample.append({
            "_id": qid,
            "question": r.get("question", ""),
            "gold": gold,
            "pred": pred,
            "type": q_type,
            "level": q_level,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in row.items()},
        })

    def _agg(bucket: Dict[str, list], n: int = 0) -> Dict[str, Any]:
        """Aggregate a metric bucket into averages.

        When *n* is provided (> 0) it is used as the denominator instead
        of the number of evaluated samples.  This aligns with the official
        eval which divides by ``len(gold)`` — missing / errored samples
        implicitly receive 0 for every metric.
        """
        evaluated = len(bucket["ans_em"]) if bucket["ans_em"] else 0
        denom = n if n > 0 else evaluated
        if denom == 0:
            return {k: 0.0 for k in METRIC_KEYS} | {"count": 0}
        return {
            k: sum(bucket[k]) / denom for k in METRIC_KEYS
        } | {"count": denom}

    total_n = len(samples)
    return {
        "overall": _agg(overall, n=total_n),
        "by_type": {t: _agg(v) for t, v in by_type.items()},
        "by_level": {lv: _agg(v) for lv, v in by_level.items()},
        "per_sample": per_sample,
    }
