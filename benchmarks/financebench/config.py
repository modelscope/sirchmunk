"""FinanceBench benchmark configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FinanceBenchConfig:
    """All settings for a FinanceBench evaluation run."""

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.5-plus"
    llm_timeout: int = 120

    # Data paths
    data_dir: str = "./data"
    pdf_dir: str = "./data/pdfs"
    output_dir: str = "./output"

    # Dataset
    limit: int = 0  # 0 = all 150
    seed: int = 42

    # Search
    mode: str = "FAST"
    top_k_files: int = 5
    max_token_budget: int = 128000
    enable_dir_scan: bool = True

    # Evaluation
    eval_mode: str = "singleDoc"  # singleDoc / sharedCorpus
    enable_llm_judge: bool = True  # Use LLM to judge semantic equivalence (independent metric)
    extract_answer: bool = True
    judge_f1_threshold: float = 0.8  # F1 threshold for 'correct' classification

    # Concurrency
    max_concurrent: int = 3
    request_delay: float = 0.5

    @classmethod
    def from_env(cls, env_path: str = ".env.financebench") -> "FinanceBenchConfig":
        """Load config from .env file with ``os.environ`` fallback."""
        # Read .env file
        env_vars: dict[str, str] = {}
        p = Path(env_path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip()

        def _get(key: str, default: str = "") -> str:
            return env_vars.get(key, os.environ.get(key, default))

        def _bool(key: str, default: bool = False) -> bool:
            v = _get(key, str(default)).lower()
            return v in ("true", "1", "yes")

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(_get(key, str(default)))
            except (ValueError, TypeError):
                return default

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(_get(key, str(default)))
            except (ValueError, TypeError):
                return default

        return cls(
            llm_api_key=_get("LLM_API_KEY"),
            llm_base_url=_get(
                "LLM_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            llm_model=_get("LLM_MODEL_NAME", "qwen3.5-plus"),
            llm_timeout=_int("LLM_TIMEOUT", 120),
            data_dir=_get("FB_DATA_DIR", "./data"),
            pdf_dir=_get("FB_PDF_DIR", "./data/pdfs"),
            output_dir=_get("FB_OUTPUT_DIR", "./output"),
            limit=_int("FB_LIMIT", 0),
            seed=_int("FB_SEED", 42),
            mode=_get("FB_MODE", "FAST"),
            top_k_files=_int("FB_TOP_K_FILES", 5),
            max_token_budget=_int("FB_MAX_TOKEN_BUDGET", 128000),
            enable_dir_scan=_bool("FB_ENABLE_DIR_SCAN", True),
            eval_mode=_get("FB_EVAL_MODE", "singleDoc"),
            enable_llm_judge=_bool("FB_ENABLE_LLM_JUDGE", True),
            extract_answer=_bool("FB_EXTRACT_ANSWER", True),
            max_concurrent=_int("FB_MAX_CONCURRENT", 3),
            request_delay=_float("FB_REQUEST_DELAY", 0.5),
        )
