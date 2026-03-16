"""Experiment configuration for HotpotQA Fullwiki benchmarking.

All values are read from .env.hotpotqa. ExperimentConfig only parses
and encapsulates — no hardcoded defaults in this module.

Ref: HotpotQA https://arxiv.org/pdf/1809.09600
     LinearRAG https://arxiv.org/pdf/2510.10114
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional


WORK_DIR = Path(__file__).resolve().parent
ENV_FILE = WORK_DIR / ".env.hotpotqa"


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def load_env(env_file: Optional[Path] = None) -> None:
    path = Path(env_file) if env_file else ENV_FILE
    _load_env(path)


def _int_or_none(val: Optional[str], default: Optional[int]) -> Optional[int]:
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _bool_env(val: Optional[str], default: bool) -> bool:
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes")


@dataclass
class ExperimentConfig:
    """Parsed and encapsulated config; all values supplied by get_config()."""

    # Dataset
    setting: Literal["fullwiki", "distractor"]
    split: str
    limit: Optional[int]
    seed: int

    # Search
    mode: Literal["FAST", "DEEP"]
    top_k_files: int
    max_token_budget: int
    enable_dir_scan: bool
    enable_cross_lingual: bool
    rga_max_count: Optional[int]

    # LLM
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout: float

    # Post-processing
    extract_answer: bool

    # Evaluation
    enable_llm_judge: bool
    judge_f1_threshold: float
    enable_gpt_eval: bool

    # Paths
    dataset_dir: Path
    wiki_corpus_dir: Path
    output_dir: Path

    # LLM reasoning
    enable_thinking: bool

    # Memory
    enable_memory: bool

    # Knowledge reuse
    reuse_knowledge: bool

    # Concurrency
    max_concurrent: int
    request_delay: float

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir).resolve()
        self.wiki_corpus_dir = Path(self.wiki_corpus_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def parquet_dir(self) -> Path:
        return self.dataset_dir / self.setting


def get_config(
    env_file: Optional[Path] = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Build ExperimentConfig purely from .env.hotpotqa env vars."""
    load_env(env_file)

    dataset_dir = os.getenv("HOTPOT_DATASET_DIR", "")
    wiki_corpus_dirname = os.getenv(
        "HOTPOT_WIKI_CORPUS_DIRNAME",
        "enwiki-20171001-pages-meta-current-withlinks-abstracts",
    )
    wiki_corpus_dir = os.getenv("HOTPOT_WIKI_CORPUS_DIR") or (
        str(Path(dataset_dir) / wiki_corpus_dirname) if dataset_dir else ""
    )
    output_dir = os.getenv("HOTPOT_OUTPUT_DIR") or str(WORK_DIR / "output")

    defaults: Dict[str, Any] = {
        "setting": os.getenv("HOTPOT_SETTING", "fullwiki"),
        "split": os.getenv("HOTPOT_SPLIT", "validation"),
        "limit": _int_or_none(os.getenv("HOTPOT_LIMIT"), 1000),
        "seed": int(os.getenv("HOTPOT_SEED", "42")),
        "mode": os.getenv("HOTPOT_MODE", "DEEP"),
        "top_k_files": int(os.getenv("HOTPOT_TOP_K_FILES", "5")),
        "max_token_budget": int(os.getenv("HOTPOT_MAX_TOKEN_BUDGET", "128000")),
        "enable_dir_scan": _bool_env(os.getenv("HOTPOT_ENABLE_DIR_SCAN"), True),
        "enable_cross_lingual": _bool_env(os.getenv("HOTPOT_ENABLE_CROSS_LINGUAL"), False),
        "rga_max_count": _int_or_none(os.getenv("HOTPOT_RGA_MAX_COUNT"), 50),
        "llm_base_url": os.getenv("LLM_BASE_URL", ""),
        "llm_api_key": os.getenv("LLM_API_KEY", ""),
        "llm_model": os.getenv("LLM_MODEL_NAME", ""),
        "llm_timeout": float(os.getenv("LLM_TIMEOUT", "120")),
        "extract_answer": _bool_env(os.getenv("HOTPOT_EXTRACT_ANSWER"), True),
        "enable_llm_judge": _bool_env(os.getenv("HOTPOT_ENABLE_LLM_JUDGE"), True),
        "judge_f1_threshold": float(os.getenv("HOTPOT_JUDGE_F1_THRESHOLD", "0.3")),
        "enable_gpt_eval": _bool_env(os.getenv("HOTPOT_ENABLE_GPT_EVAL"), True),
        "dataset_dir": dataset_dir,
        "wiki_corpus_dir": wiki_corpus_dir,
        "output_dir": output_dir,
        "enable_thinking": _bool_env(os.getenv("SIRCHMUNK_ENABLE_THINKING"), False),
        "enable_memory": _bool_env(os.getenv("SIRCHMUNK_ENABLE_MEMORY"), False),
        "reuse_knowledge": _bool_env(os.getenv("HOTPOT_REUSE_KNOWLEDGE"), False),
        "max_concurrent": int(os.getenv("HOTPOT_MAX_CONCURRENT", "5")),
        "request_delay": float(os.getenv("HOTPOT_REQUEST_DELAY", "0.5")),
    }

    for k, v in overrides.items():
        if v is not None or k == "limit":
            defaults[k] = v
    if defaults.get("limit") == 0:
        defaults["limit"] = None

    return ExperimentConfig(**defaults)
