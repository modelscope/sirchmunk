# AGENTS.md

This document defines the collaboration rules that AI agents, automated
assistants, and human contributors MUST follow when modifying code,
documentation, experiment configuration, or the web interface in the Sirchmunk
project. Its purpose is to protect the project structure, the core retrieval
algorithm pipeline, public interfaces, experiment reproducibility, and the web
user experience, and to prevent unapproved breaking changes.

## Normative Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "MAY", and "OPTIONAL" in this document are to be interpreted as
described in RFC 2119 and RFC 8174: they denote explicit requirement levels, not
stylistic preferences. "Owner" means the project owner (or an explicitly
delegated maintainer). "Agent" means any AI or automated contributor acting on
the repository.

## 1. General Principles

- Preserving the existing architecture, directory boundaries, public
  interfaces, and experiment protocols is the default. Changes to them MUST be
  justified before implementation.
- Any change affecting project boundaries, core behavior, public APIs, web
  layout, or paper-grade experiment protocols MUST first state its scope of
  impact, risks, and rollback plan, and MUST obtain the approvals defined in
  Sections 2 and 3.
- Configuration values, paths, thresholds, model names, dataset paths, ports,
  secrets, and runtime-environment assumptions MUST NOT be hardcoded. They MUST
  be supplied via parameters, configuration files, environment variables, or
  centralized constants.
- Real secrets, tokens, private data paths, and non-redacted environment
  snapshots MUST NOT be committed.
- Quality gates MUST NOT be weakened, verification skipped, critical checks
  removed, or errors silently swallowed in order to make tests pass.
- Implementations MUST generalize. They MUST NOT overfit to any specific
  dataset or benchmark at the cost of generalization ability (see Section 9,
  and in particular the prohibition in Section 9.5).

## 2. Changes Requiring Explicit Owner Approval

The following changes MUST obtain explicit approval from the owner before
implementation. Absent explicit authorization in the current task, the agent
MUST stop and propose a plan rather than editing directly.

### 2.1 Project Structure and Module Boundaries

- Adding, deleting, moving, or renaming top-level directories or core module
  directories.
- Changing the responsibility boundaries of directories such as `src/`,
  `benchmarks/`, `web/`, `requirements/`, `config/`, `scripts/`, `docker/`.
- Changing the Python package name, import root paths, entry points, build
  configuration, or release configuration.
- Large-scale refactoring of file layout, extraction of shared modules, or
  merging/splitting modules.

### 2.2 Core Algorithm Pipeline

These are high-risk core paths. Owner approval and a written technical plan are
REQUIRED before any edit:

- The FAST / DEEP / FILENAME_ONLY primary retrieval pipelines in
  `src/sirchmunk/search.py`.
- DEEP-mode parallel retrieval, evidence sampling, ReAct refinement,
  self-correction, tree navigation, knowledge reuse, and compiled-artifact
  reuse.
- Logic affecting answer generation, evidence selection, file selection,
  telemetry, token budget, or fallback strategy.
- Logic affecting source fidelity, raw-corpus retrieval, extensionless
  plain-text corpora, or dataset-specific raw-shard reading.
- Any algorithmic strategy that could change benchmark headline results, paper
  metrics, or user-visible search behavior.

### 2.3 Public Interfaces and Entry Points

Owner approval is REQUIRED for:

- Public Python APIs, SDK entry points, class names, function signatures, return
  structures, or exception semantics.
- CLI arguments, defaults, output formats, exit codes, or script
  responsibilities.
- MCP server, API routes, HTTP endpoints, schemas, payload formats, or
  authentication behavior.
- Release-related changes in `pyproject.toml`, `setup.py`, entry points, package
  data, or dependency ranges.
- Entry points relied upon by the README, paper experiments, benchmark
  workflows, or external users.

### 2.4 Web Feature Modules and Layout

Owner approval is REQUIRED for:

- `web/app/` routing structure, page hierarchy, layout, navigation, global
  styles, or responsive layout.
- Core interactive components, charts, monitoring panels, upload/search entry
  points, and status components in `web/components/`.
- Logic in `web/context/`, `web/hooks/`, `web/lib/` affecting global state,
  request wrapping, data flow, or error handling.
- User-visible interaction flows, loading states, error messages, result
  presentation, or the meaning of monitoring metrics.
- Any change that could break the consistency of existing screenshots, demo
  videos, README presentation, or the product narrative.

### 2.5 Benchmark and ResearchOps Experiment Governance

Owner approval is REQUIRED for:

- Core benchmark sampling protocols, GoldenSet, sample IDs, checksums,
  diagnostic subsets, and fixed-sample reproduction logic.
- Frozen-evaluation gates, validator error/warning levels, artifact schema, and
  report schema.
- Baseline lifecycle, setup/index/storage cost, failure classification, import
  coverage, and paired statistics.
- The responsibilities or key parameters of `run_quickstart.py`,
  `run_sampling.py`, `run_evaluation.py`, `run_lifecycle_eval.py`,
  `run_scaling_study.py`.
- Any change to default behavior that would alter paper headline tables, sample
  pairing, frozen stages, cache policy, experiment statistical definitions, or
  reproducibility.

## 3. High-Risk Changes Requiring Secondary Confirmation

Even when the owner has approved the overall direction, the following changes
REQUIRE a secondary confirmation immediately before editing. The confirmation
MUST list the specific files, functions, behavioral changes, verification plan,
and rollback plan.

- DEEP-mode primary pipeline, evidence sampling, ReAct, self-correction, and
  tree navigation.
- Breaking changes to public API / CLI / MCP / web routes.
- Web layout, navigation structure, or core page interactions.
- Paper-grade benchmark protocol, GoldenSet, validator error gate, sample
  checksum, or primary experiment statistics.
- Removing compatibility logic, migrating data formats, or changing default
  configuration or default model behavior.
- Large-scale auto-formatting, bulk renaming, or bulk file moves.

## 4. Low-Risk Changes That MAY Proceed Directly

Provided none of the high-risk areas above are touched, the following MAY
proceed directly:

- Clear typo, comment, or localized documentation wording fixes.
- Behavior-preserving local type annotations, lint fixes, or formatting fixes.
- Adding tests, fixtures, or example scripts that are not enabled by default.
- Adding explanatory notes to existing configuration items without changing
  their defaults.
- Small, single-file edits explicitly specified by the user in the current task.

Even for low-risk changes, the impact surface MUST be kept minimal; unrelated
code MUST NOT be refactored opportunistically.

## 5. Pre-Implementation Checklist

Before editing, the agent SHOULD:

- Confirm the current branch, working-tree state, and whether uncommitted user
  changes exist.
- Determine whether the change triggers the owner-approval or
  secondary-confirmation rules.
- Read the existing implementation of the relevant modules; do not modify by
  guesswork.
- For benchmark / ResearchOps changes, confirm the impact on frozen stages,
  sample IDs, validator gates, report artifacts, and statistical definitions.
- For web changes, confirm the impact on layout, routing, global state, or
  user-visible flows.

## 6. Post-Implementation Verification

After editing, run the minimal-but-sufficient verification for the change scope:

- Python code: prefer `py_compile`, the relevant CLI `--help`, localized smoke
  tests, or the corresponding unit tests.
- Web changes: prefer type checking, lint, build, or a local page smoke test.
- Benchmark changes: MUST verify sample-ID consistency, manifest/checksum,
  validator output, key CLI arguments, and statistical definitions.
- Documentation changes: confirm paths, titles, terminology, and consistency
  with the latest implementation.
- If verification cannot be run, the final report MUST state the reason and the
  residual risk.

## 7. Communication and Reporting

- For high-risk changes, present the plan, impact surface, risks, and
  verification plan, then wait for owner approval.
- For secondary-confirmation changes, restate the exact files and functions to
  be modified; a vague description is not acceptable.
- The final reply MUST describe what was changed, the verification results, any
  unverified items, and the residual risks.
- Failures MUST NOT be hidden, skipped, or reported as success.

## 8. Protected Project Claims

The core experimental claim of Sirchmunk / LENS is: under dynamic raw-data
conditions, it sustains competitive quality within preprocessing-free and
source-fidelity constraints, while explicitly reporting the full lifecycle cost
of setup, indexing, storage, update, and query.

Any change that would weaken the following capabilities REQUIRES owner approval:

- The raw-corpus / indexless / embedding-free core narrative.
- Source fidelity and evidence traceability.
- The paper-grade raw-corpus protocol.
- Frozen stratified subsets, paired statistics, and sample checksums.
- The lifecycle-feasibility comparison against external index-heavy baselines.
- Consistency of how the above claims are expressed in the web UI and
  documentation.

## 9. Anti-Hardcoding and Generalization-First Rules

This section targets the "hardcoded hard rules" most likely to appear in the
retrieval pipeline and answer handling. Such rules typically hardcode entity
naming, column positions, field formats, fixed word lists, or language
assumptions in order to pass one specific benchmark; they do not generalize to
real corpora, and they MUST be avoided and refactored with priority.

### 9.1 Detection: What Counts as a Hard Rule to Refactor

A change exhibiting any of the following MUST be treated as a hardcoded hard
rule and refactored toward a general solution:

- Dependence on a specific dataset's entity naming or ID shape (e.g.,
  `AggUnit-\d+`, a specific prefix, or a specific digit count).
- Assuming fixed table column positions, delimiter layout, or field order to
  extract values.
- Using hardcoded natural-language word lists (stop words, trigger words, unit
  words) that cover only a single language or domain.
- Using regex/string rules to reconstruct a judgment that should be made
  semantically (e.g., "is this number the answer", "does this row belong to
  this entity").
- Rules that fail as soon as they leave the current corpus's naming, format, or
  language.

### 9.2 Preferred General Patterns

When refactoring, prefer the following over stacking special cases:

- Semantic/arithmetic division of labor: let the LLM perform the semantics it is
  good at (selecting operands, judging relevant rows, naming the operation), and
  let Python perform the deterministic computation it is reliable at (sum, mean,
  comparison, count). Prefer folding the structured disclosure into an existing
  LLM call rather than adding a round trip.
- Structured trace + grounding check: the model emits a machine-readable
  structure (e.g., `<COMPUTATION_TRACE>{operation, operands, result}`), which is
  trusted and recomputed deterministically only when the operands are grounded
  (matched by value) in the evidence, so that no correction is fabricated.
- Corpus-adaptive statistics instead of fixed word lists: decide
  "discriminative / stop" using in-corpus statistics such as document frequency,
  which is inherently cross-language and cross-domain, replacing hardcoded
  English stop words.
- Dependency injection instead of embedded rules: tokenizers, thresholds, and
  lexical policies MUST be injectable and configurable, with a general default
  implementation, rather than hardcoded inside a module.
- Intent-level vocabulary instead of entity-level special cases: recognize
  intents such as sum/mean/count/min/max/difference, mapping synonyms and common
  CJK expressions onto the same primitive, rather than matching a specific
  entity name.

### 9.3 Implementation Requirements

- Every replacement behavior MUST have an environment switch (e.g.,
  `LENS_COMPUTATION_TRACE`), default to the new implementation, allow
  single-item rollback on failure, and be registered centrally in
  `config/env.example`.
- An implementation MUST NOT fall back to an entity-level or format-level
  special case merely to meet a benchmark target. When a temporary special case
  is genuinely unavoidable, it MUST explicitly annotate its applicability
  boundary, and the plan MUST state why it cannot generalize.
- After refactoring, a real-corpus regression MUST confirm that the target query
  types are no worse than the prior implementation before the new behavior is
  kept enabled by default.
- Unit tests MUST verify generalization: use entity names, column layouts, and
  languages that differ from the target benchmark, and cover the negative case
  where grounding fails and therefore no correction is applied.

### 9.4 Canonical Example

- Removed: `_deterministic_aggregation_sum` (hardcoded `AggUnit-\d+` and
  fixed-column summation).
- Replaced by: `_verify_computation_trace` + evidence grounding +
  `_reduce_operation`; operands are disclosed by the model within the same
  answer-synthesis call, and the ReAct layer captures `<COMPUTATION_TRACE>` into
  telemetry before answer sanitization.
- Removed: `_TOPIC_STOP_WORDS` (fixed English stop words) and the embedded
  `_TOPIC_TOKEN_RE` in `corpus_topic_map`.
- Replaced by: an injectable tokenizer (with a general default implementation)
  plus corpus document-frequency adaptive stop-word pruning.

### 9.5 Prohibition of Dataset/Benchmark Overfitting

Overfitting to a specific dataset or benchmark at the expense of generalization
ability is PROHIBITED. This is a first-class rule; Sections 9.1–9.4 are specific
instances of it.

- Definition. A change is considered overfitting when its correctness, or its
  measured improvement, depends on incidental properties of a particular
  evaluation set — its naming scheme, value distribution, file layout, language,
  question templates, or the identities of its gold samples — rather than on the
  general semantics of the task. Such a change is expected to degrade or
  silently break on a different but equivalent corpus.
- Prohibited practices (non-exhaustive): keying logic on known sample IDs, gold
  answers, or answer positions; branching on dataset-specific entity or
  file-name patterns; tuning thresholds or rules directly on the test/holdout
  split; special-casing the exact question phrasings of a benchmark; and adding
  any "detector" whose only purpose is to recognize a benchmark's items.
- Required safeguards. A feature MAY be developed and inspected on a benchmark,
  but its logic MUST depend only on general, corpus-agnostic signals; thresholds
  MUST be calibrated on a dedicated calibration split, never on the frozen test
  set, as governed by Section 2.5; and the change MUST be validated on inputs
  that differ from the benchmark (different entities, layouts, and languages)
  together with adversarial and negative cases.
- Reviewer test. Before a change is kept enabled by default, apply this check:
  "If the entity names, file paths, column order, and language of the evaluation
  set were replaced with equivalent but different ones, would this change still
  be correct?" If the honest answer is no, the change overfits and MUST be
  redesigned per Section 9.2.
- Reporting. Any residual, unavoidable dataset-specific assumption MUST be
  documented explicitly at the call site and surfaced in the change's final
  report, together with its applicability boundary and the generalization risk
  it carries.

### 9.6 Retrieval Cost Invariants (Query Hot Path)

The query hot path MUST have a bounded per-file and per-query cost that does not
grow unbounded with corpus size or shape. These invariants are cost/capability
policies (corpus-agnostic), enforced centrally in `GrepRetriever` and configured
in `config/env.example`:

- Per-file size cap: `GREP_MAX_FILESIZE_MB` skips any file over the cap,
  regardless of type.
- Bounded adapters only: `GREP_RGA_ADAPTERS` keeps bounded document extractors
  (poppler/pandoc) and disables the unbounded recursive/streaming adapters
  (decompress/zip/tar/sqlite/ffmpeg) so archives are never inline-decompressed
  during a query.
- Tiered scan: `GREP_TIERED_SCAN` runs a fast native-rg pass over all files
  unioned with an rga pass restricted to `GREP_RICH_EXTENSIONS`, so rga's
  per-file adapter dispatch never walks the whole tree.
- Fail-fast budgets: the rg text pass uses `GREP_TEXT_TIMEOUT`; the rga rich
  pass uses `GREP_TIMEOUT`; on timeout the search degrades to native rg rather
  than hanging.
- Offline-only container recursion: archive/container/compression adapters MAY
  be enabled ONLY in an offline compile/extraction step (never the query hot
  path), by overriding `GREP_RGA_ADAPTERS` in that context. Extracted content
  is then searched as normal files.
- Amortized rich extraction: `GrepRetriever.prewarm_rich_cache` MAY be called
  offline to populate the rga cache for pdf/docx-heavy corpora so the first
  query is warm.

A change that can make a single file or a single query cost grow without bound
(e.g., enabling inline decompression on the hot path, or pointing rga's adapter
engine at an unbounded raw tree) violates these invariants and MUST be
redesigned.
