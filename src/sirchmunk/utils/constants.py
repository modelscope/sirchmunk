import os
from pathlib import Path

# Limit for concurrent RGA requests
GREP_CONCURRENT_LIMIT = int(os.getenv("GREP_CONCURRENT_LIMIT", "5"))

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3-max")

# Search Configuration
WORK_PATH = os.getenv("WORK_PATH", Path.cwd() / "sirchmunk_work")
