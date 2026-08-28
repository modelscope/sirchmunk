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

# Adaptive directory discovery and evidence reading.
SCOPE_PLANNER_ENABLED = (
    os.getenv("SIRCHMUNK_SCOPE_PLANNER_ENABLED", "true").lower() == "true"
)
SCOPE_PLANNER_CONFIDENCE = min(
    1.0, max(0.0, float(os.getenv("SIRCHMUNK_SCOPE_PLANNER_CONFIDENCE", "0.75")))
)
SCOPE_PLANNER_MAX_PATHS = max(
    1, int(os.getenv("SIRCHMUNK_SCOPE_PLANNER_MAX_PATHS", "4"))
)
DIRECTORY_PROFILE_MAX_ENTRIES = max(
    100, int(os.getenv("SIRCHMUNK_DIRECTORY_PROFILE_MAX_ENTRIES", "100000"))
)
DIRECTORY_PROFILE_MAX_DEPTH = max(
    1, int(os.getenv("SIRCHMUNK_DIRECTORY_PROFILE_MAX_DEPTH", "8"))
)
FILE_LIST_PAGE_SIZE = max(
    1, int(os.getenv("SIRCHMUNK_FILE_LIST_PAGE_SIZE", "100"))
)
FILE_READ_MAX_CHARS = max(
    1000, int(os.getenv("SIRCHMUNK_FILE_READ_MAX_CHARS", "60000"))
)
FILE_READ_SMALL_FILE_CHARS = max(
    1000, int(os.getenv("SIRCHMUNK_FILE_READ_SMALL_FILE_CHARS", "30000"))
)
FILE_READ_WINDOW_LINES = max(
    1, int(os.getenv("SIRCHMUNK_FILE_READ_WINDOW_LINES", "80"))
)

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-5.2")

# Sirchmunk Working Directory Configuration
DEFAULT_SIRCHMUNK_WORK_PATH = os.path.expanduser("~/.sirchmunk")
# Expand ~ in environment variable if set
_env_work_path = os.getenv("SIRCHMUNK_WORK_PATH")
SIRCHMUNK_WORK_PATH = os.path.expanduser(_env_work_path) if _env_work_path else DEFAULT_SIRCHMUNK_WORK_PATH
