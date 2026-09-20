# Copyright (c) ModelScope Contributors. All rights reserved.
import os

# Limits and timeouts for grep/ripgrep-all subprocesses.
GREP_CONCURRENT_LIMIT = max(1, int(os.getenv("GREP_CONCURRENT_LIMIT", "5")))
GREP_KEYWORD_CONCURRENT_LIMIT = max(
    1, int(os.getenv("GREP_KEYWORD_CONCURRENT_LIMIT", "2"))
)
GREP_FALLBACK_CONCURRENT_LIMIT = max(
    1, int(os.getenv("GREP_FALLBACK_CONCURRENT_LIMIT", "2"))
)
GREP_TIMEOUT = max(1.0, float(os.getenv("GREP_TIMEOUT", "60.0")))
GREP_QUEUE_TIMEOUT = max(1.0, float(os.getenv("GREP_QUEUE_TIMEOUT", "10.0")))
GREP_FALLBACK_TIMEOUT = max(
    1.0, float(os.getenv("GREP_FALLBACK_TIMEOUT", "15.0"))
)
GREP_PROCESS_KILL_TIMEOUT = max(
    0.1, float(os.getenv("GREP_PROCESS_KILL_TIMEOUT", "5.0"))
)
GREP_RGA_BACKOFF_SECONDS = max(
    0.0, float(os.getenv("GREP_RGA_BACKOFF_SECONDS", "60.0"))
)
GREP_FALLBACK_TO_RG = os.getenv("GREP_FALLBACK_TO_RG", "true").lower() == "true"

# Per-file cost bound (type-agnostic): skip files whose on-disk size exceeds
# this many megabytes, regardless of extension. Bounds the worst case a single
# huge blob (dump/log/media/archive) can impose on the query hot path. 0 = off.
GREP_MAX_FILESIZE_MB = max(0, int(os.getenv("GREP_MAX_FILESIZE_MB", "64")))
# Capability-based rga adapter whitelist for the query hot path. Keeps bounded
# document extractors (poppler=pdf, pandoc=docx/epub/odt/html) and drops the
# unbounded recursive/streaming adapters (decompress/zip/tar/sqlite/ffmpeg) so
# archives are never inline-decompressed on a query. This is a cost-class
# policy, not a dataset-specific extension blacklist (see AGENTS.md §9.5).
# Empty string restores rga's default (all adapters enabled).
GREP_RGA_ADAPTERS = os.getenv(
    "GREP_RGA_ADAPTERS", "poppler,pandoc,postprocpagebreaks"
).strip()

# P1 tiered scan: run a fast native-rg pass over all files (rg auto-detects and
# skips binaries and never decompresses archives, so it is O(bytes) fast and
# covers every plain-text/markup/code format, including extensionless files)
# unioned with an rga pass restricted to the binary document formats that
# genuinely need extraction. This bounds cost on large / many-file corpora
# without losing rich-format recall, instead of letting rga's per-file adapter
# dispatch walk the whole tree. Set to false to restore the single rga pass.
GREP_TIERED_SCAN = os.getenv("GREP_TIERED_SCAN", "true").lower() == "true"
# Binary document formats routed to the rga rich pass. Kept to formats that a
# plain rg cannot read (poppler extracts pdf; pandoc extracts docx/epub/odt).
# Plain-text/markup such as html/csv/json/log are covered by the rg text pass.
GREP_RICH_EXTENSIONS = tuple(
    ext.strip().lstrip(".").lower()
    for ext in os.getenv("GREP_RICH_EXTENSIONS", "pdf,docx,epub,odt").split(",")
    if ext.strip()
)
# Fail-fast timeout (seconds) for the rg text pass; the rga rich pass keeps the
# longer GREP_TIMEOUT because bounded document extraction is inherently slower.
GREP_TEXT_TIMEOUT = max(1.0, float(os.getenv("GREP_TEXT_TIMEOUT", "15.0")))
# Cap matches emitted per file (rg/rga -m/--max-count). Bounds JSON output
# volume so a broad query over a huge corpus cannot blow the scan budget with
# millions of match lines. Ranking only needs presence + a few snippets per
# file, so this does not affect which files are found. 0 = unlimited.
GREP_MAX_MATCHES_PER_FILE = max(0, int(os.getenv("GREP_MAX_MATCHES_PER_FILE", "5")))

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-5.2")

# Sirchmunk Working Directory Configuration
DEFAULT_SIRCHMUNK_WORK_PATH = os.path.expanduser("~/.sirchmunk")
# Expand ~ in environment variable if set
_env_work_path = os.getenv("SIRCHMUNK_WORK_PATH")
SIRCHMUNK_WORK_PATH = os.path.expanduser(_env_work_path) if _env_work_path else DEFAULT_SIRCHMUNK_WORK_PATH
