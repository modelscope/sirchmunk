#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Build the anonymized code-and-data supplement for the LENS submission.

The supplement is assembled from the single evaluation run that backs the
paper's three tables (``benchmarks/hotpotqa/output/dynamic_eval``) and is
deliberately *not* a repository dump. Three properties are enforced:

1. **Paper-anchored scope.** Only the stages the paper reports (``G_125`` and
   ``G_250``) are shipped, and every artifact maps to a specific claim
   (sample-ID / frozen-order / corpus checksums, stratification fidelity,
   lifecycle cost, query budget).
2. **Anonymity.** Local paths, the real package name, the account name and any
   credential-shaped value are rewritten before writing. A hard gate re-scans
   every emitted byte and aborts the build on a single leak.
3. **Self-verifiability.** Metrics are recomputed from the shipped per-question
   records and compared cell-by-cell against the numbers printed in the paper,
   so a reviewer (and this script) can tell whether the two agree.

The raw corpus is intentionally excluded: snapshots are reconstructible from the
official HotpotQA release plus the shipped title lists, and the shipped
per-document checksums let a reviewer prove the rebuild is byte-identical.

Usage::

    python scripts/build_supplement.py [--exclude-react] [--no-zip]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = REPO_ROOT / "benchmarks" / "hotpotqa" / "output" / "dynamic_eval"
OUT_DIR = (
    REPO_ROOT / "temp" / "papers" / "overleaf_version" / "lens_submission" / "code_and_data"
)

STAGES: List[str] = ["G_125_D_125", "G_250_D_250"]
CORPUS_STAGES: List[str] = ["D_125", "D_250"]

# Artifact file stem -> display name used by the paper / tables.
SYSTEMS: Dict[str, str] = {
    "bm25_rag": "BM25-RAG",
    "hybrid_rag": "Hybrid-RAG",
    "ablation_lens_full": "LENS",
    "react": "ReAct-Search",
}
PAPER_SYSTEMS = ("BM25-RAG", "Hybrid-RAG", "LENS")

# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------

# Longest paths first so that nested prefixes are rewritten unambiguously.
PATH_REWRITES: List[Tuple[str, str]] = [
    (str(REPO_ROOT), "<PROJECT_ROOT>"),
    (str(REPO_ROOT.parent), "<WORKSPACE_ROOT>"),
    (str(Path.home()), "<HOME>"),
]
TOKEN_REWRITES: List[Tuple[str, str]] = [
    ("sirchmunk", "lens"),
    ("Sirchmunk", "LENS"),
    ("SIRCHMUNK", "LENS"),
    ("wangxingjun778", "anon"),
    ("ModelScope Contributors", "Anonymous Authors"),
    ("ModelScope Team", "Anonymous Authors"),
]
# Values under these keys never travel, regardless of content. Matching is
# segment-exact so that budget knobs such as ``MAX_TOTAL_TOKENS`` -- which the
# paper cites and a reviewer must be able to read -- are not mistaken for
# credentials merely because they contain the substring "token".
SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:api_?key|secret|password|passwd|access_key|base_url|token)(?:$|_)",
    re.I,
)
COMMIT_KEY_RE = re.compile(r"(git_commit|commit_hash|revision)", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# The shipped configuration must be readable by any reviewer, so its Chinese
# comments are translated rather than dropped: they carry the operational meaning
# of knobs the paper cites (loading order, concurrency, cold cache). The mapping
# is exhaustive by construction -- an untranslated line aborts the build -- so a
# future edit upstream cannot silently ship non-English text.
CONFIG_COMMENT_TRANSLATIONS: Dict[str, str] = {
    "# benchmarks/.env.global.example \u2014 Layer 0 \u5168\u5c40\u5171\u4eab\u914d\u7f6e\u6a21\u677f":
        "# benchmarks/.env.global.example - Layer 0 shared global configuration template",
    "# \u7528\u9014\uff1a": "# Purpose:",
    "#   \u6b64\u6587\u4ef6\u662f\u6240\u6709 benchmark \u7684\u6700\u4f4e\u4f18\u5148\u7ea7\u914d\u7f6e\u5c42\uff08Layer 0\uff09\u3002":
        "#   This is the lowest-priority configuration layer (Layer 0) for every benchmark.",
    "#   \u5c06\u6b64\u6587\u4ef6\u590d\u5236\u4e3a benchmarks/.env.global \u5e76\u586b\u5199\u771f\u5b9e API key \u540e\uff0c":
        "#   Copy it to benchmarks/.env.global and fill in a real API key; every benchmark",
    "#   \u6240\u6709 benchmark \u5c06\u81ea\u52a8\u7ee7\u627f\u8fd9\u4e9b\u914d\u7f6e\uff0c":
        "#   then inherits these settings automatically,",
    "#   \u65e0\u9700\u5728\u6bcf\u4e2a benchmark \u7684 profile env \u6587\u4ef6\u4e2d\u91cd\u590d\u586b\u5199\u3002":
        "#   with no need to repeat them in each benchmark's profile env file.",
    "# \u4f18\u5148\u7ea7\uff08\u4ece\u4f4e\u5230\u9ad8\uff09\uff1a": "# Priority (lowest to highest):",
    "#   \u4f8b\u5982 HotpotQA: .env.global < .env.hotpotqa.base < .env.hotpotqa.frozen < os.environ\u3002":
        "#   For example, HotpotQA: .env.global < .env.hotpotqa.base < .env.hotpotqa.frozen < os.environ.",
    "# \u4f55\u65f6\u5e94\u4fee\u6539\u6b64\u6587\u4ef6\uff1a": "# When to edit this file:",
    "#   - \u5207\u6362 LLM \u63d0\u4f9b\u5546\uff08\u5982\u4ece DashScope \u6362\u5230 OpenAI\uff09":
        "#   - Switching between OpenAI-compatible LLM providers",
    "#   - \u66f4\u65b0\u5171\u4eab API key": "#   - Updating the shared API key",
    "#   - \u4fee\u6539 Embedding \u6a21\u578b": "#   - Changing the embedding model",
    "# \u26a0\ufe0f  Layer 0 \u53d8\u66f4\u63d0\u9192\uff1a": "# Layer 0 change warning:",
    "#   \u6b64\u6587\u4ef6\u4e2d\u7684\u4efb\u4f55\u53d8\u66f4\u90fd\u5c5e\u4e8e Layer 0 GLOBAL \u53d8\u66f4\uff0c\u5c06\u5f71\u54cd\u6240\u6709 benchmark\u3002":
        "#   Any change here is a Layer 0 GLOBAL change and affects every benchmark.",
    "#   \u5982\u4f7f\u7528 run_research_loop.py\uff0cImprovementAdvisor \u4f1a\u81ea\u52a8\u6807\u6ce8\u5e76\u63d0\u793a\u3002":
        "#   When run_research_loop.py is used, ImprovementAdvisor flags such changes.",
    "# ===== LLM \u8bbe\u7f6e\uff08\u5168\u5c40\u5171\u4eab\uff09 =====": "# ===== LLM settings (shared globally) =====",
    "# \u5404 benchmark \u53ef\u5728\u81ea\u5df1\u7684 .env \u6587\u4ef6\u4e2d\u7528 LLM_MODEL_NAME \u8986\u76d6\u6b64\u503c":
        "# Each benchmark may override this with LLM_MODEL_NAME in its own .env file.",
    "# ===== Embedding \u8bbe\u7f6e\uff08\u53ef\u9009\uff0c\u5168\u5c40\u5171\u4eab\uff09 =====":
        "# ===== Embedding settings (optional, shared globally) =====",
    "# \u82e5\u4e0d\u586b\uff0c\u5404 benchmark \u4f7f\u7528\u9ed8\u8ba4\u503c\u3002":
        "# If left empty, each benchmark falls back to its own default.",
    "# LightRAG v1.3.6 SDK baseline \u4e5f\u4f1a\u8bfb\u53d6 EMBEDDING_MODEL_ID\uff1b\u8bf7\u8bbe\u7f6e\u4e3a":
        "# The LightRAG v1.3.6 SDK baseline also reads EMBEDDING_MODEL_ID; set it to an",
    "# \u5f53\u524d LLM_BASE_URL \u652f\u6301\u7684 OpenAI-compatible embedding model\u3002":
        "# OpenAI-compatible embedding model served by the configured LLM_BASE_URL.",
    "# \u793a\u4f8b\uff1aOpenAI \u53ef\u7528 text-embedding-3-small\uff1bDashScope \u53ef\u6309\u8d26\u53f7\u53ef\u7528\u6a21\u578b\u586b\u5199\u3002":
        "# Example: text-embedding-3-small, or any model the configured account exposes.",
    "# LightRAG \u7684 query mode / working_dir / max files \u7b49\u5b9e\u9a8c\u53c2\u6570\u901a\u8fc7 CLI \u4f20\u9012\uff0c":
        "# LightRAG experiment knobs (query mode, working_dir, max files) are passed on",
    "# \u4e0d\u653e\u5728\u5168\u5c40 env \u4e2d\u3002":
        "# the command line rather than stored in this global env file.",
}

# Tokens that must not survive into the package. Note that ``modelscope`` is
# deliberately absent: it appears only as a third-party import, and rewriting a
# dependency name would break the shipped code without adding anonymity, whereas
# the copyright attribution that *is* identifying is rewritten above. The account
# name is matched only in a path-like context, because it is also a common
# Wikipedia article name that legitimately occurs in the corpus and in answers.
FORBIDDEN_RE = re.compile(
    r"sirchmunk|wangxingjun|ModelScope\s+(?:Contributors|Team)|/Users/|"
    r"[/\\~]" + re.escape(Path.home().name) + r"\b",
    re.I,
)


def scrub_text(text: str) -> str:
    """Rewrite local paths and identifying tokens in a text blob."""
    for src, dst in PATH_REWRITES:
        text = text.replace(src, dst)
    for src, dst in TOKEN_REWRITES:
        text = text.replace(src, dst)
    return text


def scrub_obj(obj: Any) -> Any:
    """Recursively scrub a decoded JSON value, redacting sensitive keys."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            new_key = scrub_text(str(key))
            if SECRET_KEY_RE.search(str(key)) and isinstance(value, str) and value:
                out[new_key] = "<REDACTED>"
            elif COMMIT_KEY_RE.search(str(key)) and isinstance(value, str):
                out[new_key] = "<REDACTED>"
            else:
                out[new_key] = scrub_obj(value)
        return out
    if isinstance(obj, list):
        return [scrub_obj(v) for v in obj]
    if isinstance(obj, str):
        return scrub_text(obj)
    return obj


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    """Write scrubbed JSON, compactly for bulk records where readability is moot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scrubbed = scrub_obj(payload)
    if compact:
        text = json.dumps(scrubbed, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(scrubbed, ensure_ascii=False, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Slimming rules
# ---------------------------------------------------------------------------

# Dropped from every embedded corpus manifest: absolute per-file path lists and
# the title-resolution trace. Both are large and reconstructible; the title list
# and the per-document checksums (shipped separately) carry the audit value.
CORPUS_MANIFEST_DROP = {"selected_document_paths", "title_resolution_report"}

RESULT_TOP_KEYS = (
    "result_schema_version", "baseline_name", "citation_name", "adapter_class",
    "config_hash", "sample_id", "system_name", "question", "gold_answer",
    "prediction", "judge_correct", "coverage", "evidence_recall", "elapsed",
    "tokens_used", "judge_tokens", "question_type", "error", "failure_reason",
)
RESULT_TELEMETRY_KEYS = (
    "official_em", "official_f1", "official_exact_match", "official_f1_correct",
    "normalized_prediction", "normalized_gold", "short_prediction",
    "total_tokens", "judge_tokens", "cached", "llm_judge_used", "equivalent",
    "confidence", "failure_reason", "error_type", "failure_phase",
    "query_budget", "supporting_fact_titles", "retrieved_titles",
    "matched_supporting_fact_titles", "missing_supporting_fact_titles",
    "supporting_fact_hit", "supporting_fact_count", "supporting_sentence_count",
    "matched_supporting_sentence_count", "supporting_fact_title_recall",
    "supporting_sentence_recall", "supporting_sentence_completion_rate",
    "evidence_recall", "answer_source_grounded", "read_file_ids",
    "zero_hit_rescue_used", "react_explore_fallback_used",
    "span_calibration_used",
)
RESULT_METADATA_KEYS = (
    "baseline_type", "index_required", "rebuild_required",
    "baseline_cache_identity", "setup_metrics",
)


def slim_corpus_manifest(manifest: Any) -> Any:
    if not isinstance(manifest, dict):
        return manifest
    return {k: v for k, v in manifest.items() if k not in CORPUS_MANIFEST_DROP}


BINDING_IDENTITY_KEYS = (
    "stage_name", "g_stage", "d_stage",
    "sample_id_checksum", "frozen_order_checksum", "corpus_checksum",
)


def dereference_binding(binding: Any) -> Any:
    """Reduce an embedded stage binding to its identity fields."""
    if not isinstance(binding, dict):
        return binding
    out = {k: binding[k] for k in BINDING_IDENTITY_KEYS if k in binding}
    out["full_binding"] = "artifacts/stage_bindings.json"
    return out


def slim_metadata(metadata: Any, *, dereference_corpus: bool = False) -> Any:
    """Slim the ``metadata`` block shared by bindings and stage records.

    With *dereference_corpus*, the embedded corpus manifest and stage binding are
    replaced by pointers to the artifacts that carry them once. Both describe the
    snapshot rather than the system that queried it, so embedding them per
    (stage, system) pair duplicates megabytes without adding evidence: the
    retained checksums already tie the record to that snapshot.
    """
    if not isinstance(metadata, dict):
        return metadata
    out = dict(metadata)
    manifest = out.get("corpus_manifest")
    if isinstance(manifest, dict):
        if dereference_corpus:
            out["corpus_manifest"] = {
                "stage_name": manifest.get("stage_name"),
                "sample_set_id": manifest.get("sample_set_id"),
                "corpus_checksum": manifest.get("corpus_checksum"),
                "article_count": manifest.get("article_count"),
                "full_manifest": (
                    "artifacts/corpus_provenance/"
                    f"{manifest.get('stage_name')}/ and artifacts/stage_bindings.json"
                ),
            }
        else:
            out["corpus_manifest"] = slim_corpus_manifest(manifest)
    if dereference_corpus and "stage_binding" in out:
        out["stage_binding"] = dereference_binding(out["stage_binding"])
    return out


def slim_result_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the fields that make a single question auditable, drop bulk text.

    Dropped: retrieved chunk bodies, gold sentence bodies and free-form judge
    reasoning. The first two would redistribute corpus text, and none of the
    three participates in any reported metric.
    """
    out = {k: row[k] for k in RESULT_TOP_KEYS if k in row}
    telemetry = row.get("telemetry") or {}
    out["telemetry"] = {k: telemetry[k] for k in RESULT_TELEMETRY_KEYS if k in telemetry}
    metadata = row.get("metadata") or {}
    slim_meta = {k: metadata[k] for k in RESULT_METADATA_KEYS if k in metadata}
    if slim_meta:
        out["metadata"] = slim_meta
    if "setup_metrics" in row:
        out["setup_metrics"] = row["setup_metrics"]
    return out


# ---------------------------------------------------------------------------
# Paper reference values (Tables 1-3 of the submitted manuscript)
# ---------------------------------------------------------------------------

PAPER_TABLES: Dict[Tuple[str, str], Dict[str, float]] = {
    ("G_125_D_125", "BM25-RAG"): dict(
        em=13.6, f1=25.4, ev_rec=71.6, ev_gnd=96.8,
        index_s=1.8, store_mb=2.5, read=5.0, tok=1100, lat=2.7, oracle=0.0, search=0.0),
    ("G_125_D_125", "Hybrid-RAG"): dict(
        em=25.6, f1=37.1, ev_rec=81.3, ev_gnd=94.4,
        index_s=1.9, store_mb=13.9, read=5.0, tok=1200, lat=3.2, oracle=0.0, search=0.0),
    ("G_125_D_125", "LENS"): dict(
        em=42.4, f1=54.2, ev_rec=64.1, ev_gnd=77.6,
        index_s=0.0, store_mb=0.0, read=7.0, tok=33200, lat=28.0, oracle=3.2, search=7.0),
    ("G_250_D_250", "BM25-RAG"): dict(
        em=12.8, f1=24.3, ev_rec=71.1, ev_gnd=96.0,
        index_s=2.2, store_mb=5.0, read=5.0, tok=1100, lat=2.4, oracle=0.0, search=0.0),
    ("G_250_D_250", "Hybrid-RAG"): dict(
        em=28.0, f1=38.2, ev_rec=77.8, ev_gnd=90.4,
        index_s=3.2, store_mb=27.7, read=5.0, tok=1200, lat=4.0, oracle=0.0, search=0.0),
    ("G_250_D_250", "LENS"): dict(
        em=41.9, f1=54.7, ev_rec=66.8, ev_gnd=79.4,
        index_s=0.0, store_mb=0.0, read=6.9, tok=35600, lat=34.3, oracle=3.4, search=6.9),
}
# Per-metric tolerance, chosen to absorb the paper's display rounding only.
TOLERANCE = dict(
    em=0.05, f1=0.05, ev_rec=0.05, ev_gnd=0.05, index_s=0.05, store_mb=0.05,
    read=0.05, tok=100.0, lat=0.05, oracle=0.05, search=0.05,
)


def recompute(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Recompute the reported metrics from shipped per-question records."""
    n = len(rows)
    tel = [r.get("telemetry", {}) for r in rows]
    budgets = [t.get("query_budget", {}) or {} for t in tel]
    setups = [r.get("setup_metrics", {}) or {} for r in rows]

    def mean(values: Iterable[float]) -> float:
        vals = [v for v in values if isinstance(v, (int, float))]
        return sum(vals) / n if n and vals else 0.0

    return dict(
        em=mean(t.get("official_em", 0) for t in tel) * 100,
        f1=mean(t.get("official_f1", 0) for t in tel) * 100,
        ev_rec=mean(r.get("evidence_recall", 0) for r in rows) * 100,
        ev_gnd=sum(1 for t in tel if t.get("supporting_fact_hit")) / n * 100 if n else 0.0,
        index_s=max([s.get("index_build_seconds", 0) or 0 for s in setups] or [0]),
        store_mb=max([s.get("storage_bytes", 0) or 0 for s in setups] or [0]) / 1e6,
        read=mean(b.get("read_calls", 0) for b in budgets),
        tok=mean(t.get("total_tokens", 0) for t in tel),
        lat=mean(r.get("elapsed", 0) for r in rows),
        oracle=mean(b.get("oracle_calls", 0) for b in budgets),
        search=mean(b.get("search_calls", 0) for b in budgets),
    )


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------


def copy_artifacts(out: Path, systems: Dict[str, str]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Emit sampling, binding, table and per-question artifacts."""
    art = out / "artifacts"

    # 1) Sampling: sample IDs + nested manifest (backs the sample-ID checksum
    #    and the nested G_125 subset G_250 claim).
    for name in ("G_125_sample_ids.json", "G_250_sample_ids.json", "nested_sample_manifest.json"):
        write_json(art / "sampling" / name, json.loads((RUN_DIR / "sampling" / name).read_text()))

    # 2) Stage bindings: the triple binding key per (stage, system).
    bindings = json.loads((RUN_DIR / "stage_bindings.json").read_text())
    slim_bindings = []
    for entry in bindings:
        item = dict(entry)
        item["metadata"] = slim_metadata(item.get("metadata"))
        slim_bindings.append(item)
    write_json(art / "stage_bindings.json", slim_bindings)

    # 3) Run manifest.
    manifest = json.loads((RUN_DIR / "dynamic_eval_manifest.json").read_text())
    manifest["corpus_snapshots"] = [
        slim_corpus_manifest(s) for s in manifest.get("corpus_snapshots", [])
    ]
    write_json(art / "run_manifest.json", manifest)

    # 4) Generated tables (json/md/tex) exactly as the evaluator emitted them.
    for src in sorted((RUN_DIR / "tables").glob("*")):
        dst = art / "tables" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".json":
            write_json(dst, json.loads(src.read_text()))
        else:
            dst.write_text(scrub_text(src.read_text()), encoding="utf-8")

    # 5) Per-question records + stage execution records.
    measured: Dict[Tuple[str, str], Dict[str, float]] = {}
    for stage in STAGES:
        for stem, display in systems.items():
            src = RUN_DIR / "runs" / stage / "baselines" / f"baseline_{stem}.jsonl"
            if not src.exists():
                continue
            rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
            slim = [slim_result_row(r) for r in rows]
            measured[(stage, display)] = recompute(slim)
            dst = art / "runs" / stage / f"results_{stem}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w", encoding="utf-8") as fh:
                for row in slim:
                    fh.write(json.dumps(scrub_obj(row), ensure_ascii=False) + "\n")

            rec_src = RUN_DIR / "runs" / stage / "stage_records" / f"{stem}_stage_execution_record.json"
            if rec_src.exists():
                record = json.loads(rec_src.read_text())
                record["metadata"] = slim_metadata(
                    record.get("metadata"), dereference_corpus=True
                )
                write_json(art / "runs" / stage / f"stage_record_{stem}.json", record)

    # 6) Corpus provenance: everything needed to rebuild and then *prove* the
    #    rebuild matches, without shipping the corpus itself.
    bulk = {"document_checksum_records.json", "selected_article_titles.json"}
    for stage in CORPUS_STAGES:
        stage_dir = RUN_DIR / "corpus" / stage
        for name in (
            "stage_manifest.json",
            "validation_manifest.json",
            "background_selection_manifest.json",
            "selected_article_titles.json",
            "document_checksum_records.json",
            "sample_ids.json",
        ):
            src = stage_dir / name
            if src.exists():
                write_json(
                    art / "corpus_provenance" / stage / name,
                    json.loads(src.read_text()),
                    compact=name in bulk,
                )

    return measured


def copy_code(out: Path) -> None:
    """Copy the implementation as ``.py`` sources under the anonymized name."""
    code = out / "code"

    def copy_tree(src: Path, dst: Path) -> None:
        for path in sorted(src.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(scrub_text(path.read_text(encoding="utf-8")), encoding="utf-8")

    copy_tree(REPO_ROOT / "src" / "sirchmunk", code / "lens")
    for name in ("framework", "baselines", "evaluation", "ablations"):
        copy_tree(REPO_ROOT / "benchmarks" / name, code / "benchmarks" / name)

    hotpot_src = REPO_ROOT / "benchmarks" / "hotpotqa"
    hotpot_dst = code / "benchmarks" / "hotpotqa"
    hotpot_dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(hotpot_src.glob("*.py")):
        (hotpot_dst / path.name).write_text(
            scrub_text(path.read_text(encoding="utf-8")), encoding="utf-8"
        )

    for path in sorted((REPO_ROOT / "benchmarks").glob("run_*.py")):
        (code / "benchmarks" / path.name).write_text(
            scrub_text(path.read_text(encoding="utf-8")), encoding="utf-8"
        )

    req_dst = code / "requirements"
    req_dst.mkdir(parents=True, exist_ok=True)
    for path in sorted((REPO_ROOT / "requirements").glob("*.txt")):
        (req_dst / path.name).write_text(
            scrub_text(path.read_text(encoding="utf-8")), encoding="utf-8"
        )


def redact_env(text: str, *, source: str) -> str:
    """Keep every knob visible while removing endpoints, credentials and CJK.

    Comment lines written in Chinese are replaced by their English equivalents
    from :data:`CONFIG_COMMENT_TRANSLATIONS`. An unmapped Chinese line raises,
    because silently shipping it would leave the reviewer-facing configuration
    partly unreadable.
    """
    lines = []
    untranslated: List[str] = []
    for line in scrub_text(text).splitlines():
        if CJK_RE.search(line):
            translated = CONFIG_COMMENT_TRANSLATIONS.get(line.rstrip())
            if translated is None:
                untranslated.append(line.rstrip())
                continue
            line = translated
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            lines.append(line)
            continue
        key, _, value = line.partition("=")
        if SECRET_KEY_RE.search(key) and value.strip():
            lines.append(f"{key}=<REDACTED>")
        else:
            lines.append(line)
    if untranslated:
        raise ValueError(
            f"{source}: {len(untranslated)} Chinese line(s) have no English "
            "mapping in CONFIG_COMMENT_TRANSLATIONS:\n  "
            + "\n  ".join(untranslated)
        )
    return "\n".join(lines) + "\n"


def copy_config(out: Path) -> None:
    """Ship the exact frozen configuration, credentials removed."""
    cfg = out / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    sources = {
        "env.global.frozen": REPO_ROOT / "benchmarks" / ".env.global",
        "env.hotpotqa.frozen": REPO_ROOT / "benchmarks" / "hotpotqa" / ".env.hotpotqa.frozen",
    }
    for name, src in sources.items():
        if src.exists():
            (cfg / name).write_text(
                redact_env(src.read_text(encoding="utf-8"), source=name),
                encoding="utf-8",
            )


def check_consistency(
    measured: Dict[Tuple[str, str], Dict[str, float]]
) -> List[str]:
    """Compare recomputed metrics against the paper, cell by cell."""
    diffs: List[str] = []
    for (stage, system), expected in PAPER_TABLES.items():
        got = measured.get((stage, system))
        if got is None:
            diffs.append(f"{stage}/{system}: MISSING from package")
            continue
        for metric, want in expected.items():
            have = got[metric]
            if abs(have - want) > TOLERANCE[metric]:
                diffs.append(
                    f"{stage}/{system}.{metric}: paper={want} artifacts={have:.4g}"
                )
    return diffs


def guard(out: Path) -> List[str]:
    """Abort-level scan: no identifying token may survive anywhere.

    The shipped configuration is additionally required to be English-only, so a
    reviewer can read every knob the paper cites without translation.
    """
    leaks: List[str] = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            leaks.append(f"{path.relative_to(out)}: unreadable as text")
            continue
        hits = FORBIDDEN_RE.findall(text)
        if hits:
            leaks.append(f"{path.relative_to(out)}: {len(hits)} hit(s), e.g. {hits[0]!r}")
        if path.parent.name == "config":
            cjk = CJK_RE.findall(text)
            if cjk:
                leaks.append(
                    f"{path.relative_to(out)}: {len(cjk)} non-English character(s), "
                    f"e.g. {cjk[0]!r}"
                )
    return leaks


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-react", action="store_true",
        help="omit the ReAct reference system (not reported in the paper)",
    )
    parser.add_argument("--no-zip", action="store_true", help="skip archive creation")
    args = parser.parse_args()

    if not RUN_DIR.is_dir():
        print(f"error: evaluation run not found: {RUN_DIR}", file=sys.stderr)
        return 2

    systems = dict(SYSTEMS)
    if args.exclude_react:
        systems.pop("react", None)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    print(f"[1/6] artifacts  <- {RUN_DIR.name}")
    measured = copy_artifacts(OUT_DIR, systems)
    print(f"[2/6] code       <- src/, benchmarks/")
    copy_code(OUT_DIR)
    print(f"[3/6] config     <- frozen env (credentials redacted, English-only)")
    copy_config(OUT_DIR)

    print("[4/6] README")
    write_readme(OUT_DIR, measured, systems)

    print("[5/6] anonymity gate")
    leaks = guard(OUT_DIR)
    if leaks:
        print("BUILD ABORTED — identifying tokens found:", file=sys.stderr)
        for leak in leaks[:20]:
            print(f"  {leak}", file=sys.stderr)
        return 1
    print("       clean: no local paths, package name, account name or CJK config text")

    print("[6/6] paper consistency")
    diffs = check_consistency(measured)
    if diffs:
        print("       MISMATCH between paper and shipped artifacts:")
        for diff in diffs:
            print(f"         {diff}")
    else:
        print("       all reported cells reproduce from shipped records")

    if not args.no_zip:
        archive = shutil.make_archive(str(OUT_DIR), "zip", root_dir=OUT_DIR)
        print(f"\narchive: {Path(archive).name} ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    print(f"package: {OUT_DIR} ({dir_size(OUT_DIR) / 1e6:.1f} MB)")
    return 0


def write_readme(
    out: Path,
    measured: Dict[Tuple[str, str], Dict[str, float]],
    systems: Dict[str, str],
) -> None:
    """Emit a reviewer-facing README (imported late to keep main() readable)."""
    from supplement_readme import render  # type: ignore

    (out / "README.md").write_text(
        render(measured, systems, STAGES, PAPER_SYSTEMS), encoding="utf-8"
    )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
