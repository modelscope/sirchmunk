"""FinanceBench benchmark configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_env_file(path: str) -> dict[str, str]:
    """Parse a .env file into a dict, handling comments, blank lines, and quotes."""
    result: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        # Strip surrounding quotes
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        result[k.strip()] = v
    return result


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
    enable_llm_judge: bool = True  # LLM Judge drives Accuracy + Coverage evaluation

    # Concurrency
    max_concurrent: int = 3
    request_delay: float = 0.5

    # Experiment isolation
    work_path: str = "./.work"  # Isolated workspace for this experiment

    @classmethod
    def from_env(cls, env_path: str = ".env.financebench") -> "FinanceBenchConfig":
        """Load config with layer inheritance.

        Priority (highest to lowest):
        1. Experiment .env (.env.financebench)
        2. Platform .env (<work_path>/.env, if exists)
        3. os.environ
        4. Dataclass defaults
        """
        # Step 0: Pre-read experiment env to determine work_path
        experiment_vars = _parse_env_file(env_path)
        work_path = experiment_vars.get(
            "FB_WORK_PATH", os.environ.get("FB_WORK_PATH", "./.work")
        )

        # Step 1: Load platform-level env (<work_path>/.env)
        platform_env_path = Path(work_path) / ".env"
        platform_vars = _parse_env_file(str(platform_env_path))

        # Step 2: Merge — experiment > platform > os.environ > defaults
        merged = {**platform_vars, **experiment_vars}

        def _get(key: str, default: str = "") -> str:
            return merged.get(key, os.environ.get(key, default))

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
            max_concurrent=_int("FB_MAX_CONCURRENT", 3),
            request_delay=_float("FB_REQUEST_DELAY", 0.5),
            work_path=work_path,
        )
