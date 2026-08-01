#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""README renderer for the code-and-data supplement.

Kept separate from :mod:`build_supplement` so that the reviewer-facing prose can
be edited without touching the packaging and anonymity logic. The measured
column is filled from the shipped records, so the README can never drift from
the artifacts it describes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

HEADER = """# Code and Data Supplement

This supplement accompanies the paper *LENS: Budgeted Evidence Localization over
Dynamic Raw Documents*. It is a scoped, anonymized package rather than a
repository snapshot: every artifact included here maps to a specific claim in
the experimental section, and nothing that would identify the authors or their
machine is present.

## 1. What this package establishes

The paper's central protocol claim is that all systems answer *identical*
questions over *byte-identical* corpus snapshots, bound by three checksums
(sample-ID, frozen-order, corpus), and that the evaluator refuses to emit
publication-ready tables when those checksums disagree. The artifacts below let
a reviewer verify that claim end to end without trusting the reported numbers.

| Path | Backs which claim |
| --- | --- |
| `artifacts/sampling/G_{125,250}_sample_ids.json` | The frozen question sets, in evaluation order |
| `artifacts/sampling/nested_sample_manifest.json` | Nesting (G125 subset of G250), 8 strata, seed, realized-vs-population drift |
| `artifacts/stage_bindings.json` | The triple binding key per (stage, system) |
| `artifacts/run_manifest.json` | Run-level inventory: protocol, snapshots, table paths |
| `artifacts/tables/*` | The three reported tables, exactly as emitted (`.json`/`.md`/`.tex`) |
| `artifacts/runs/<stage>/results_<system>.jsonl` | One record per question: prediction, official EM/F1, evidence recall, per-question budget |
| `artifacts/runs/<stage>/stage_record_<system>.json` | Per-stage execution record carrying the three checksums, config hash and cache mode |
| `artifacts/corpus_provenance/<snapshot>/` | Snapshot composition, title lists and per-document checksums |
| `code/` | Implementation of the method, the baselines and the evaluation protocol |
| `config/env.*.frozen` | The exact frozen configuration, credentials removed |

## 2. Verifying the reported numbers

Every cell of the three tables is recomputable from
`artifacts/runs/<stage>/results_<system>.jsonl` alone:

- **EM / F1** — mean of `telemetry.official_em` / `telemetry.official_f1`.
- **Evidence recall** — mean of `evidence_recall`.
- **Evidence-grounded rate** — fraction with `telemetry.supporting_fact_hit`.
- **Index time / storage** — `setup_metrics.index_build_seconds` /
  `setup_metrics.storage_bytes` (identical across a stage's questions, since
  preparation happens once per snapshot).
- **Oracle / search / read calls, tokens, latency** — `telemetry.query_budget`
  and `elapsed`.

Recomputed from the records shipped here:

"""

CORPUS_SECTION = """
## 3. Corpus snapshots

The raw corpus is **not** shipped: the two snapshots hold {counts} single-article
text files and redistributing them would add hundreds of megabytes of text that
is already public. They are instead reconstructible, and the reconstruction is
verifiable:

1. Obtain the official HotpotQA fullwiki distribution (validation split and the
   accompanying Wikipedia dump) from the dataset's canonical source.
2. Materialize each snapshot with the shipped code
   (`code/benchmarks/run_dynamic_evaluation.py`), using the seed, stage sizes,
   strata, background ratio and background seed recorded in
   `artifacts/run_manifest.json`. Article selection is deterministic given those
   values, and `artifacts/corpus_provenance/<snapshot>/selected_article_titles.json`
   lists exactly which articles must appear.
3. Verify the result against
   `artifacts/corpus_provenance/<snapshot>/document_checksum_records.json`
   (per-document checksums) and the snapshot-level `corpus_checksum` in
   `stage_manifest.json`. A matching corpus checksum is what licenses comparing
   any system in this package against any other.

This is the same mechanism the evaluator itself uses, which is why the protocol
tolerates a corpus that changes between stages: no system is allowed to answer
over a snapshot whose checksum does not match its binding key.

## 4. Anonymization

Local absolute paths, the implementation's package name, the account name and
all credential-shaped values were rewritten before packaging; the build aborts
if any survive. Two consequences are worth flagging so that nothing looks
inconsistent to a reviewer:

- The Python package is named `lens` here, and environment variables are
  prefixed accordingly. Internal references are rewritten consistently, so the
  code remains coherent, but it will not match any public release verbatim.
- Paths inside JSON artifacts appear as `<PROJECT_ROOT>/...` placeholders.
  Checksums were computed over file *contents*, so they are unaffected by this
  rewriting and remain verifiable.

Model endpoints and API credentials are redacted. The chat backend is identified
in the paper by architecture and parameter count rather than by endpoint, which
is sufficient to reproduce the comparison: all systems in a stage share one
backend, so the comparison is internally controlled even under a different
provider.

## 5. Layout

```
code_and_data/
  README.md
  artifacts/
    sampling/            frozen question sets and nesting manifest
    stage_bindings.json  triple binding key per (stage, system)
    run_manifest.json    run-level inventory
    tables/              the reported tables as emitted
    runs/<stage>/        per-question records and per-stage execution records
    corpus_provenance/   snapshot composition, titles, per-document checksums
  code/
    lens/                method implementation
    benchmarks/          protocol, baselines, evaluation, ablations, entry points
    requirements/        pinned dependency sets
  config/                frozen configuration, credentials redacted
```
"""


def _fmt(value: float, metric: str) -> str:
    if metric == "tok":
        return f"{value / 1000:.1f}K"
    if metric == "store_mb":
        return "0" if value < 0.05 else f"{value:.1f}MB"
    return f"{value:.1f}"


def render(
    measured: Dict[Tuple[str, str], Dict[str, float]],
    systems: Dict[str, str],
    stages: List[str],
    paper_systems: Iterable[str],
) -> str:
    """Build the README text, filling the measured table from the artifacts."""
    lines = [HEADER.rstrip("\n"), ""]
    lines.append("| System | Stage | EM | F1 | Ev.Rec | EvGnd | Index (s) | Store | Oracle | Search | Read | Tok/Q | Lat (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    order = list(paper_systems) + [
        name for name in systems.values() if name not in set(paper_systems)
    ]
    for stage in stages:
        stage_label = stage.split("_D_")[0].replace("G_", "G")
        for system in order:
            row = measured.get((stage, system))
            if row is None:
                continue
            cells = " | ".join(
                _fmt(row[m], m)
                for m in ("em", "f1", "ev_rec", "ev_gnd", "index_s", "store_mb",
                          "oracle", "search", "read", "tok", "lat")
            )
            lines.append(f"| {system} | {stage_label} | {cells} |")

    extra = [name for name in systems.values() if name not in set(paper_systems)]
    if extra:
        lines.append("")
        lines.append(
            "The paper compares LENS against the two index-centric baselines. "
            + ", ".join(extra)
            + " is included here as an additional reference system: it was run under the "
            "identical protocol and binding keys, and it illustrates the same "
            "recall-versus-accuracy divergence the paper analyses from the "
            "opposite direction, reaching high exact match while retrieving the "
            "fewest gold evidence documents. It is reported here rather than "
            "omitted so that the package reflects everything the protocol "
            "measured."
        )

    counts = "5,416 and 10,808"
    lines.append(CORPUS_SECTION.format(counts=counts).rstrip("\n"))
    lines.append("")
    return "\n".join(lines)
