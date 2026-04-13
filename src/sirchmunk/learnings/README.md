# Sirchmunk Learnings Module

The `sirchmunk/learnings` module implements **knowledge compilation and continuous learning** capabilities. It houses the core logic for transforming raw document collections into structured, searchable knowledge networks.

## Architecture Overview

```
learnings/
├── __init__.py              # Public API exports
├── knowledge_base.py        # Runtime knowledge builder (search-time)
├── evidence_processor.py    # Monte Carlo evidence sampling
├── compiler.py              # Offline knowledge compiler (compile-time)
├── tree_indexer.py           # PageIndex-style document tree indexer
├── lint.py                  # Knowledge network health checks
└── README.md                # This file
```

### Design Philosophy

The module fuses insights from three frameworks:

1. **PageIndex** (VectifyAI) — Hierarchical tree indexing replaces brute-force vector search with LLM reasoning-based navigation. The key insight: *similarity ≠ relevance*.

2. **LLM Wiki** (Karpathy) — Documents are not merely "indexed" but "compiled" into an interlinked knowledge network that compounds over time. Knowledge clusters grow richer with each compile cycle.

3. **NotebookLM** (Google) — Strict source grounding ensures every claim traces back to original evidence. The `EvidenceUnit` system provides full provenance.

### Compile vs. Search

| Aspect | Compile (offline) | Search (runtime) |
|--------|-------------------|-------------------|
| **When** | `sirchmunk compile` | `sirchmunk search` |
| **Speed** | Minutes (batch) | Seconds (interactive) |
| **Purpose** | Build indices + knowledge | Answer queries |
| **Module** | `compiler.py` (uses `tree_indexer.py`) | `knowledge_base.py`, `evidence_processor.py` |
| **Required** | Optional | Always available |

Compile products are automatically leveraged by search when present, but search functions independently without them.

---

## Components

### DocumentTreeIndexer (`tree_indexer.py`)

Builds hierarchical JSON tree indices for structured long documents.

**Key concepts:**
- Only triggers for documents ≥ 50KB in eligible formats (PDF, DOCX, MD, HTML, etc.)
- LLM analyzes document structure recursively (up to 4 levels deep)
- Each node stores: title, summary, character range
- Query-time navigation: LLM selects relevant branches instead of scanning everything

**Data structures:**
- `TreeNode` — Single node with `node_id`, `title`, `summary`, `char_range`, `children`
- `DocumentTree` — Complete tree for a document, JSON-serializable, cached by file hash

**Usage:**
```python
indexer = DocumentTreeIndexer(llm=llm, cache_dir=cache_path)

# Build (async, LLM-powered)
tree = await indexer.build_tree(file_path, content, max_depth=4)

# Navigate (async, LLM-powered branch selection)
leaves = await indexer.navigate(tree, query="How does X work?")
for leaf in leaves:
    relevant_text = content[leaf.char_range[0]:leaf.char_range[1]]

# Cache check (sync)
if indexer.has_tree(file_path):
    tree = indexer.load_tree(file_path)
```

### KnowledgeCompiler (`compiler.py`)

Orchestrates the unified compile pipeline.

**Four-phase pipeline:**
1. **File Discovery & Change Detection** — Scans paths, compares with manifest for incremental processing
2. **Per-File Compile** — Unified pipeline per file: tree-if-eligible → summary → topics → rich evidence
3. **Knowledge Aggregation** — Merges into existing clusters or creates new ones (three-tier similarity)
4. **Cross-Reference Building** — Creates `WeakSemanticEdge` links between related clusters

**Unified single-file pipeline:**
For each file, the compiler runs a single pipeline instead of separate "tree" and "wiki" modes:
- If the file is ≥ 50KB and in an eligible format, a tree is built first. The root node's summary is synthesized from children's section summaries via LLM, and `EvidenceUnit` snippets + `tree_path` are populated directly from tree leaves.
- If the file is small or `shallow=True`, a direct LLM summary is generated instead.
- In both cases, topics are extracted and a `KnowledgeCluster` is created/merged.

**Three-tier similarity strategy:**
| Similarity | Action |
|-----------|--------|
| ≥ 0.80 | Merge into existing cluster, re-compute embedding |
| 0.50 – 0.79 | Create new cluster + build `embed_sim` weak edges |
| < 0.50 | Create standalone cluster |

**Importance probability sampling** (`ImportanceSampler`):
For large datasets, select a representative subset using weighted random sampling:
- File size (log-scaled): larger files contain more information
- Novelty: uncompiled files get 4× weight over already-compiled ones
- Extension diversity: structured formats (PDF, DOCX) get 1.5× boost

**Key data structures:**
- `CompileManifest` — Tracks file hashes and cluster associations for incremental compile
- `FileManifestEntry` — Per-file state (hash, compile timestamp, tree flag, cluster IDs)
- `CompileReport` — Statistics from a compile run
- `CompileStatus` — Quick status snapshot

### KnowledgeLint (`lint.py`)

Health checks for the knowledge network (inspired by LLM Wiki's Lint operation).

**Checks performed:**
- **Empty clusters** — Clusters with minimal or no content
- **Stale evidence** — Evidence pointing to files that no longer exist
- **Orphan clusters** — Clusters with no evidence and no queries
- **Isolated clusters** — Clusters with no cross-references
- **Orphan trees** — Tree cache files without matching manifest entries
- **Stale manifest** — Manifest entries pointing to deleted files

**Auto-fix capabilities:**
- Deprecate clusters where all evidence sources are gone
- Remove orphan tree cache files

### KnowledgeBase (`knowledge_base.py`)

Runtime knowledge builder used during search operations.

**Tree-aware evidence extraction:**
When a tree index exists for a file, `_extract_evidence_for_file()` navigates to relevant sections first, then runs Monte Carlo sampling within those narrowed regions. This dramatically improves precision for large documents.

### MonteCarloEvidenceSampling (`evidence_processor.py`)

Statistical sampling for finding relevant regions in documents. Used both at compile-time and search-time.

---

## CLI Interface

```bash
# Compile documents (optional, after sirchmunk init)
sirchmunk compile --paths /data/docs /data/reports

# Incremental compile (default, skips unchanged files)
sirchmunk compile --paths /data/docs

# Full recompile
sirchmunk compile --paths /data/docs --full

# Importance sampling for large datasets
sirchmunk compile --paths /data/docs --max-files 100

# Shallow mode: skip tree indexing, use direct LLM summarisation
sirchmunk compile --paths /data/docs --shallow

# Check compile status
sirchmunk compile --paths /data/docs --status

# Run health checks
sirchmunk compile --paths /data/docs --lint
sirchmunk compile --paths /data/docs --lint --fix
```

## Python SDK

```python
from sirchmunk.search import AgenticSearch

searcher = AgenticSearch(work_path="~/.sirchmunk")

# Compile
report = await searcher.compile(
    paths=["/data/docs"],
    incremental=True,
    shallow=False,       # set True to skip tree indexing
    max_files=100,       # importance sampling
    concurrency=3,
)

# Status
status = await searcher.compile_status(paths=["/data/docs"])

# Lint
lint_report = await searcher.compile_lint(auto_fix=True)

# Search (automatically uses compile products when available)
result = await searcher.search("query", paths=["/data/docs"])
```

---

## Cache Layout

```
{work_path}/.cache/
├── compile/
│   ├── manifest.json                    # Compile manifest (incremental state)
│   └── trees/
│       ├── {file_hash_1}.json           # Tree index for document 1
│       └── {file_hash_2}.json           # Tree index for document 2
└── knowledge/
    └── knowledge_clusters.parquet       # Persistent cluster storage (DuckDB + Parquet)
```

## Schema Extensions

The compile feature extends existing schemas:

- **`EvidenceUnit`** — Added `tree_path` (node IDs from tree navigation) and `page_range` (character offsets)
- **`KnowledgeCluster`** — Added `merge_count` (tracks compile-time merge frequency for lifecycle promotion: ≥ 3 merges → `STABLE`)

## Design Principles

- **SOLID compliance**: Each class has a single responsibility; dependencies are injected via constructor
- **Optional by design**: Compile never breaks existing search functionality
- **Incremental**: Only processes changed files; manifest tracks state across runs
- **Production-ready**: Bounded concurrency, error isolation per file, graceful schema migration
