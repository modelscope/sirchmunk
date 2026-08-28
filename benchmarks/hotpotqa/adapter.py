"""benchmarks/hotpotqa/adapter.py — HotpotQAAdapter

HotpotQA benchmark adapter for the ResearchOps framework.
It delegates loading, judging, evidence evaluation, and metrics to dedicated
modules so fullwiki runs can be tracked with reproducible artifacts.
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── sys.path injection ──────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()      # benchmarks/hotpotqa/
_BENCHMARKS_ROOT = _HERE.parent              # benchmarks/
_PROJECT_ROOT = _BENCHMARKS_ROOT.parent      # sirchmunk/
_SRC = _PROJECT_ROOT / "src"

# Optional path to the Layer 0 shared global config file
_GLOBAL_ENV = _BENCHMARKS_ROOT / ".env.global"

# Optional Layer 1 HotpotQA shared config path, overridable per profile
_HOTPOT_BASE_ENV = _HERE / ".env.hotpotqa.base"

# HotpotQA-specific work_path, fixed and independent of the CWD
_HOTPOT_WORK_PATH = str(_HERE / ".work")

for _p in (str(_SRC),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_BENCHMARKS_ROOT))
from framework.adapter import BenchmarkAdapter  # noqa: E402
from framework.answer_policy import policy_from_env  # noqa: E402
from framework.protocol import default_protocol  # noqa: E402
from framework.schema import BenchmarkSample    # noqa: E402
from evaluation.sampling_protocol import extract_sample_ids  # noqa: E402
from hotpotqa.evidence import evaluate_supporting_facts  # noqa: E402
from hotpotqa.judge import HotpotQAJudge  # noqa: E402
from hotpotqa.loader import (  # noqa: E402
    build_dataset_manifest,
    compute_split_checksum,
    describe_hotpotqa_split,
    load_hotpotqa_samples,
    validate_hotpotqa_corpus,
)
from hotpotqa.metrics import compute_hotpotqa_metrics  # noqa: E402
# ─────────────────────────────────────────────────────────────────────


def _load_env_file(path: str | Path) -> Dict[str, str]:
    """Parse a .env file into a dict without depending on python-dotenv."""
    result: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return result
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def _default_base_env_path() -> Path:
    override = os.environ.get("HOTPOT_BASE_ENV_FILE", "").strip()
    return Path(override).expanduser().resolve() if override else _HOTPOT_BASE_ENV


def _load_env_layers(profile_env_file: str) -> tuple[Dict[str, str], List[str]]:
    """Load the layered HotpotQA env with precedence global < base < profile < os.environ."""
    env: Dict[str, str] = {}
    sources: List[str] = []
    for path in (_GLOBAL_ENV, _default_base_env_path(), Path(profile_env_file)):
        layer = _load_env_file(path)
        if layer:
            env.update(layer)
            sources.append(str(Path(path).resolve()))
    return env, sources


class HotpotQAAdapter(BenchmarkAdapter):
    """HotpotQA adapter.

    Current state: the interface is fully implemented and data loading reads parquet
    files. For a complete run, make sure HOTPOT_DATASET_DIR is configured correctly
    and the pyarrow / pandas dependencies are installed.

    Usage::

        adapter = HotpotQAAdapter(
            env_file="benchmarks/hotpotqa/.env.hotpotqa.frozen"
        )
    """

    def __init__(self, env_file: str) -> None:
        # HotpotQA keeps an isolated work_path to avoid cache pollution.
        self._env_file = str(Path(env_file).resolve())
        self._env, self._env_sources = _load_env_layers(self._env_file)
        self._searcher = None
        self._judge = None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, default: str = "") -> str:
        value = os.environ.get(key)
        if value is not None and value != "":
            return value
        value = self._env.get(key)
        if value is not None and value != "":
            return value
        return default

    def _get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self._get(key, str(default)))
        except (ValueError, TypeError):
            return default

    def _get_bool(self, key: str, default: bool = False) -> bool:
        return self._get(key, str(default)).lower() in ("true", "1", "yes")

    def _judge_model_name(self, *, required: bool = True) -> str:
        """Return the explicitly configured HotpotQA judge model.

        Rejects a judge model equal to the model under evaluation. Asking a model
        whether its own phrasing matches the gold is not an independent rating,
        and the resulting score would flatter whichever system shares the
        judge's answer style.
        """
        model = self._get("HOTPOT_JUDGE_MODEL_NAME", "")
        if model:
            system_model = self._get("LLM_MODEL_NAME", "")
            if system_model and model.strip() == system_model.strip():
                raise RuntimeError(
                    f"HOTPOT_JUDGE_MODEL_NAME ({model}) equals LLM_MODEL_NAME, so the "
                    "judge would be scoring its own output. Configure a different "
                    "model for judging."
                )
            return model
        if not required:
            return model
        raise RuntimeError(
            "HOTPOT_JUDGE_MODEL_NAME is required when HOTPOT_ENABLE_LLM_JUDGE=true. "
            "It configures the HotpotQA judge model separately from LLM_MODEL_NAME."
        )

    def get_profile_limit(self, default: int = 0) -> int:
        return self._get_int("HOTPOT_LIMIT", default)

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def env_file(self) -> str:
        return self._env_file

    # ------------------------------------------------------------------
    # BenchmarkAdapter interface
    # ------------------------------------------------------------------

    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        """Load HotpotQA samples from parquet files via HotpotQALoader.

        If HOTPOT_SAMPLE_IDS_FILE is configured, the adapter returns exactly
        those sample IDs in file order.  This lets Sirchmunk frozen runs execute
        the same sampled GoldenSet used later by run_evaluation.py.
        """
        samples = self.load_sampling_population(seed=seed)
        fixed_sample_ids_file = self._sample_ids_file()
        if self._require_context_answerable():
            samples = [s for s in samples if self._is_context_answerable(s)]
        if fixed_sample_ids_file:
            samples = self._filter_samples_by_ids(samples, fixed_sample_ids_file)
        elif limit > 0 and limit < len(samples):
            random.seed(seed)
            samples = random.sample(samples, limit)
        return samples

    def load_sampling_population(self, seed: int = 42) -> List[BenchmarkSample]:
        """Load the full declared split for sampling protocol generation.

        This intentionally ignores HOTPOT_SAMPLE_IDS_FILE and context-answerable
        smoke filters so GoldenSet creation always sees the 7405-example
        fullwiki validation population.
        """
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        return load_hotpotqa_samples(
            dataset_dir,
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
            limit=0,
            seed=seed,
        )

    def validate_corpus(self) -> Tuple[int, List[str]]:
        """Validate raw wiki dump inventory (shard count, not title closure)."""
        return validate_hotpotqa_corpus(Path(self._wiki_dir()))

    def get_corpus_index_cache_dir(self) -> str:
        """Directory holding cached raw-corpus title indexes.

        This lives directly under ``work_path`` (not under ``.cache``) on
        purpose: the raw-corpus title index is a deterministic artifact keyed by
        the immutable dump fingerprint, not per-run evaluation state, so it must
        survive cold-cache clearing. Cold runs clear ``.cache`` and its nested
        caches; keeping the dump-derived title index outside that scope avoids a
        full dump rescan on every frozen run and keeps evidence-recall title
        resolution available after a cold clear.
        """
        path = Path(self.get_work_path()) / "corpus_index"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def get_wiki_fingerprint(self) -> Dict[str, Any]:
        """Return the raw wiki dump identity used to bind sampling artifacts."""
        from hotpotqa.corpus_index import compute_wiki_fingerprint
        return compute_wiki_fingerprint(self._wiki_dir()).to_dict()

    def evaluate_corpus_sync(
        self,
        samples: Any = None,
        *,
        seed: int = 42,
        force_rebuild: bool = False,
    ):
        """Check that every referenced article of *samples* exists in the dump.

        Defaults to the full declared split so sampling entry points can gate on
        closure before any sample IDs are frozen. The resulting report is cached
        by dump fingerprint, so repeat calls cost a lookup instead of a scan.
        """
        from hotpotqa.corpus_index import evaluate_corpus_sync as _evaluate

        rows = list(samples) if samples is not None else self.load_sampling_population(seed=seed)
        setting = self._get("HOTPOT_SETTING", "fullwiki")
        split = self._get("HOTPOT_SPLIT", "validation")
        try:
            dataset_checksum = compute_split_checksum(
                Path(self._get("HOTPOT_DATASET_DIR", "")),
                setting=setting,
                split=split,
            )
        except Exception:
            dataset_checksum = ""
        return _evaluate(
            rows,
            self._wiki_dir(),
            cache_dir=self.get_corpus_index_cache_dir(),
            setting=setting,
            split=split,
            dataset_checksum=dataset_checksum,
            force_rebuild=force_rebuild,
        )

    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        """Return search paths for this sample.

        Exploration smoke runs may use the parquet-provided HotpotQA context as
        a per-sample raw-text corpus.  Frozen/fullwiki runs should keep the
        default ``wiki`` mode and search the configured Wikipedia corpus.
        """
        mode = self._context_corpus_mode()
        if mode == "sample":
            return [str(self._materialize_sample_context(sample))]
        if mode == "hybrid":
            return [str(self._materialize_sample_context(sample)), self._wiki_dir()]
        return [self._wiki_dir()]

    def get_run_config(self) -> Dict[str, Any]:
        return {
            "mode":             self._get("HOTPOT_MODE", "DEEP"),
            "top_k_files":      self._get_int("HOTPOT_TOP_K_FILES", 10),
            "max_token_budget": self._get_int("HOTPOT_MAX_TOKEN_BUDGET", 128000),
            "enable_dir_scan":  self._get_bool("HOTPOT_ENABLE_DIR_SCAN", True),
            "enable_memory":    self._get_bool("SIRCHMUNK_ENABLE_MEMORY", False),
            "enable_eval_feedback": self._get_bool("HOTPOT_ENABLE_EVAL_FEEDBACK", False),
            "enable_llm_judge": self._get_bool("HOTPOT_ENABLE_LLM_JUDGE", True),
            "allow_frozen_llm_judge_auxiliary": self._get_bool("ALLOW_FROZEN_LLM_JUDGE_AUXILIARY", False),
            "enable_gpt_eval":  self._get_bool("HOTPOT_ENABLE_GPT_EVAL", False),
            "llm_model":        self._get("LLM_MODEL_NAME", ""),
            "judge_model":      self._judge_model_name(required=False),
            "llm_base_url":     self._get("LLM_BASE_URL", ""),
            "max_concurrent":   self._get_int("HOTPOT_MAX_CONCURRENT", 3),
            "setting":          self._get("HOTPOT_SETTING", "fullwiki"),
            "split":            self._get("HOTPOT_SPLIT", "validation"),
            "limit":            self.get_profile_limit(0),
            "sample_ids_file":   self._sample_ids_file(),
            "context_corpus_mode": self._context_corpus_mode(),
            "require_context_answerable": self._require_context_answerable(),
            "reuse_knowledge":  self._get_bool("HOTPOT_REUSE_KNOWLEDGE", False),
            "cache_mode":       self._get("HOTPOT_CACHE_MODE", "cold"),
            "cache_allow_clear": self._get_bool("CACHE_ALLOW_CLEAR", False),
            "cache_dry_run":    self._get_bool("CACHE_DRY_RUN", False),
            "sample_timeout_seconds": self._get_int("SAMPLE_TIMEOUT_SECONDS", 0),
            "system_timeout_seconds": self._get_int("SYSTEM_TIMEOUT_SECONDS", 0),
            "benchmark_timeout_seconds": self._get_int("BENCHMARK_TIMEOUT_SECONDS", 0),
            "global_timeout_seconds": self._get_int("GLOBAL_TIMEOUT_SECONDS", 0),
            "max_runtime_seconds": self._get_int("MAX_RUNTIME_SECONDS", 0),
            "max_total_tokens": self._get_int("MAX_TOTAL_TOKENS", 0),
            "max_api_cost_usd": float(self._get("MAX_API_COST_USD", "0") or 0),
            "max_disk_usage_bytes": self._get_int("MAX_DISK_USAGE_BYTES", 0),
            "min_free_disk_bytes": self._get_int("MIN_FREE_DISK_BYTES", 0),
            "env_sources": list(self._env_sources),
            "top_k_env_key":    "HOTPOT_TOP_K_FILES",
            "mode_env_key":     "HOTPOT_MODE",
            "judge_model_env_key": "HOTPOT_JUDGE_MODEL_NAME",
            "judge_threshold_env_key": "HOTPOT_JUDGE_F1_THRESHOLD",
        }

    def build_searcher(self) -> Any:
        """Build and cache the AgenticSearch instance.

        work_path comes from self.get_work_path() (benchmarks/hotpotqa/.work) and is
        fully isolated from the .work directory of FinanceBenchAdapter.
        """
        if self._searcher is None:
            from sirchmunk.llm.openai_chat import OpenAIChat
            from sirchmunk.search import AgenticSearch

            llm = OpenAIChat(
                api_key=self._get("LLM_API_KEY", ""),
                base_url=self._get("LLM_BASE_URL", "https://api.openai.com/v1"),
                model=self._get("LLM_MODEL_NAME", "gpt-4o-mini"),
            )
            self._searcher = AgenticSearch(
                llm=llm,
                work_path=self.get_work_path(),  # Use the isolated absolute path
                reuse_knowledge=self._get_bool("HOTPOT_REUSE_KNOWLEDGE", False),
                verbose=False,
                # Evaluation scores a refusal like a wrong answer, so the
                # reporting decision is supplied here instead of being read
                # from the environment inside the retrieval pipeline.
                answer_policy=policy_from_env(),
            )
        return self._searcher

    def build_judge(self) -> Optional[Any]:
        """Build HotpotQA EM/F1 judge with optional LLM semantic fallback."""
        if self._judge is None:
            llm = None
            enable_llm_judge = self._get_bool("HOTPOT_ENABLE_LLM_JUDGE", True)
            if enable_llm_judge:
                judge_model = self._judge_model_name()
                try:
                    from sirchmunk.llm.openai_chat import OpenAIChat

                    llm = OpenAIChat(
                        api_key=self._get("LLM_API_KEY", ""),
                        base_url=self._get("LLM_BASE_URL", "https://api.openai.com/v1"),
                        model=judge_model,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to build HotpotQA judge LLM. "
                        "Check LLM_BASE_URL, LLM_API_KEY, HOTPOT_JUDGE_MODEL_NAME, and benchmark env layering."
                    ) from exc
            self._judge = HotpotQAJudge(
                llm=llm,
                enable_llm_judge=enable_llm_judge,
                llm_fallback_f1_threshold=float(self._get("HOTPOT_JUDGE_F1_THRESHOLD", "0.3")),
            )
        return self._judge

    def get_output_dir(self) -> str:
        """HotpotQA output directory: relative paths resolve against the benchmark dir."""
        raw = self._get("HOTPOT_OUTPUT_DIR", "./output")
        p = Path(raw)
        if p.is_absolute():
            return str(p.resolve())
        return str((_HERE / p).resolve())

    def get_work_path(self) -> str:
        """Return the HotpotQA-specific work_path, fixed to benchmarks/hotpotqa/.work.

        It is unaffected by HOTPOT_OUTPUT_DIR or the CWD and stays fully isolated from
        the FinanceBenchAdapter cache.
        """
        return _HOTPOT_WORK_PATH

    def get_max_concurrent(self) -> int:
        """HOTPOT_MAX_CONCURRENT: LENS sample concurrency plus concurrent competitors.

        It does not affect per-competitor internal sample concurrency, which is
        controlled by HOTPOT_BASELINE_SAMPLE_CONCURRENT; when that is unset, each
        BaselineAdapter keeps its own declaration.
        """
        return self._get_int("HOTPOT_MAX_CONCURRENT", 3)

    def get_baseline_sample_concurrency(self) -> int:
        """HOTPOT_BASELINE_SAMPLE_CONCURRENT: per-competitor sample concurrency, 0 = no override."""
        return self._get_int("HOTPOT_BASELINE_SAMPLE_CONCURRENT", 0)

    def get_request_delay(self) -> float:
        try:
            return float(self._get("HOTPOT_REQUEST_DELAY", "0.5"))
        except ValueError:
            return 0.5

    def get_search_kwargs(self) -> Dict[str, Any]:
        return {
            "mode":             self._get("HOTPOT_MODE", "DEEP"),
            "top_k_files":      self._get_int("HOTPOT_TOP_K_FILES", 10),
            "max_token_budget": self._get_int("HOTPOT_MAX_TOKEN_BUDGET", 128000),
            "enable_dir_scan":  self._get_bool("HOTPOT_ENABLE_DIR_SCAN", True),
        }

    def extra_result_fields(self, sample: BenchmarkSample) -> Dict[str, Any]:
        return {
            "hotpot_id": sample.sample_id,
            "type":      sample.metadata.get("type", ""),
            "level":     sample.metadata.get("level", ""),
            "answer_type": sample.metadata.get("answer_type", ""),
            "supporting_fact_count": sample.metadata.get("supporting_fact_count", 0),
            "supporting_fact_bucket": sample.metadata.get("supporting_fact_bucket", ""),
            "supporting_facts": sample.metadata.get("supporting_facts", []),
        }

    def get_protocol_spec(self, run_id: str, seed: int, limit: int) -> Dict[str, Any]:
        protocol = default_protocol(
            run_id=run_id,
            benchmark=self.name,
            config=self.get_run_config(),
            seed=seed,
        ).to_dict()
        protocol["suite"] = [f"hotpotqa_{self._get('HOTPOT_SETTING', 'fullwiki')}"]
        protocol["metrics"]["answer_quality"] = [
            "official_exact_match",
            "official_f1",
            "llm_assisted_accuracy",
            "coverage",
        ]
        protocol["metrics"]["retrieval"] = ["evidence_recall", "supporting_fact_hit_rate", "source_grounding_accuracy"]
        sample_ids_file = self._sample_ids_file()
        if sample_ids_file:
            protocol["sampling"] = {
                "sample_ids_file": sample_ids_file,
                "method": "fixed_sample_ids",
            }
        protocol["limit"] = limit
        return protocol

    def get_dataset_manifest(self) -> Dict[str, Any]:
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        manifest = build_dataset_manifest(
            dataset_dir,
            Path(self._wiki_dir()),
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
        )
        sample_ids_file = self._sample_ids_file()
        if sample_ids_file:
            manifest["sample_ids_file"] = sample_ids_file
            try:
                ids = extract_sample_ids(sample_ids_file)
                manifest["sample_ids_count"] = len(ids)
            except Exception as exc:
                manifest["sample_ids_error"] = str(exc)
        return manifest

    def get_title_resolver(self):
        """Return a HotpotQA raw wiki title resolver backed by a cached index."""
        from hotpotqa.title_resolver import HotpotQATitleResolver
        return HotpotQATitleResolver(
            self._wiki_dir(),
            index_cache_dir=self.get_corpus_index_cache_dir(),
        )

    def derive_dynamic_sample_sets(self, golden_set: Any, *, stages: List[int], output_dir: str | Path):
        """Derive nested G_n sample-id artifacts from a parent GoldenSet."""
        from hotpotqa.dynamic_corpus import derive_nested_sample_sets
        return derive_nested_sample_sets(golden_set, stages=stages, output_dir=output_dir)

    def build_dynamic_corpus_snapshot_for_stage(
        self,
        samples: List[BenchmarkSample],
        *,
        sample_ids: List[str],
        output_dir: str | Path,
        stage_name: str,
        materialize_mode: str = "symlink",
        background_ratio: float = 3.0,
        background_seed: int = 42,
    ):
        """Build a D_n snapshot aligned with one frozen G_n sample-id set."""
        from hotpotqa.dynamic_corpus import build_dynamic_corpus_snapshot
        return build_dynamic_corpus_snapshot(
            samples,
            sample_ids=sample_ids,
            wiki_dir=self._wiki_dir(),
            output_dir=output_dir,
            stage_name=stage_name,
            materialize_mode=materialize_mode,
            background_ratio=background_ratio,
            background_seed=background_seed,
            resolver=self.get_title_resolver(),
        )

    def enrich_telemetry(self, sample, prediction, telemetry, **kwargs) -> Dict[str, Any]:
        supporting_facts = sample.metadata.get("supporting_facts", [])
        return evaluate_supporting_facts(
            supporting_facts,
            read_file_ids=telemetry.get("read_file_ids", []),
            prediction=prediction,
            retrieval_logs=telemetry.get("retrieval_logs", []),
            evidence_sources=telemetry.get("evidence_sources", []),
            evidence_texts=telemetry.get("evidence_snippets", []),
            context=sample.metadata.get("context"),
            resolved_title_paths=self._resolve_evidence_shard_paths(supporting_facts),
        )

    def _resolve_evidence_shard_paths(self, supporting_facts: Any) -> Dict[str, str]:
        """Map gold supporting-fact titles to their absolute raw-corpus shard.

        The raw enwiki dump packs many articles per shard and the filename
        carries no title, so filename-based title recall is impossible in the
        frozen main experiment. This resolves each gold title to the shard that
        holds it using the read-only cached corpus title index built by the
        raw-corpus synchronization gate. It no-ops (returns ``{}``) when no
        covering cached index exists, e.g. dynamic snapshots or sample smoke
        runs, where per-article filenames already carry the title.
        """
        from hotpotqa.evidence import _extract_facts
        titles = sorted({
            str(fact.get("title")).strip()
            for fact in _extract_facts(supporting_facts)
            if str(fact.get("title") or "").strip()
        })
        if not titles:
            return {}
        index = self._load_evidence_title_index(titles)
        if index is None:
            return {}
        resolved: Dict[str, str] = {}
        wiki_dir = Path(self._wiki_dir())
        for title in titles:
            shard = index.absolute_shard_for(title)
            if shard is None:
                relative = index.shard_for(title)
                shard = (wiki_dir / relative) if relative else None
            if shard is not None:
                resolved[title] = str(shard)
        return resolved

    def _load_evidence_title_index(self, titles: List[str]):
        """Return a cached covering title index, memoized on the adapter.

        Read-only: it never triggers a dump scan. When no covering cached index
        exists (dynamic per-article snapshots, sample smoke runs) it returns
        ``None`` and the caller falls back to filename-based title recall, which
        is already precise for those layouts.
        """
        cached = getattr(self, "_evidence_title_index", "__unset__")
        if cached != "__unset__" and (cached is None or cached.covers(titles)):
            return cached
        try:
            from hotpotqa.corpus_index import load_covering_index
            index = load_covering_index(
                self._wiki_dir(),
                titles,
                cache_dir=self.get_corpus_index_cache_dir(),
            )
        except Exception:
            index = None
        if index is not None:
            self._evidence_title_index = index
        return index

    def get_analysis_schema(self) -> Dict[str, Any]:
        return {
            "primary_group_key": "type",
            "secondary_group_key": "level",
            "evidence_key": "supporting_facts",
            "multi_hop": True,
            "numeric_sensitive": False,
        }

    def get_config_schema(self) -> Dict[str, Any]:
        return {
            "global_keys": [
                "LLM_BASE_URL",
                "LLM_API_KEY",
                "LLM_MODEL_NAME",
                "LLM_TIMEOUT",
                "EMBEDDING_MODEL_ID",
                "EMBEDDING_CACHE_DIR",
                "SIRCHMUNK_WORK_PATH",
                "GREP_CONCURRENT_LIMIT",
                "GREP_KEYWORD_CONCURRENT_LIMIT",
                "GREP_FALLBACK_CONCURRENT_LIMIT",
                "GREP_TIMEOUT",
                "GREP_QUEUE_TIMEOUT",
                "GREP_FALLBACK_TIMEOUT",
                "GREP_PROCESS_KILL_TIMEOUT",
                "GREP_RGA_BACKOFF_SECONDS",
                "GREP_FALLBACK_TO_RG",
                "SIRCHMUNK_SCOPE_PLANNER_ENABLED",
                "SIRCHMUNK_SCOPE_PLANNER_CONFIDENCE",
                "SIRCHMUNK_SCOPE_PLANNER_MAX_PATHS",
                "SIRCHMUNK_DIRECTORY_PROFILE_MAX_ENTRIES",
                "SIRCHMUNK_DIRECTORY_PROFILE_MAX_DEPTH",
                "SIRCHMUNK_FILE_LIST_PAGE_SIZE",
                "SIRCHMUNK_FILE_READ_MAX_CHARS",
                "SIRCHMUNK_FILE_READ_SMALL_FILE_CHARS",
                "SIRCHMUNK_FILE_READ_WINDOW_LINES",
            ],
            "top_k_env_key": "HOTPOT_TOP_K_FILES",
            "mode_env_key": "HOTPOT_MODE",
            "judge_model_env_key": "HOTPOT_JUDGE_MODEL_NAME",
            "judge_threshold_env_key": "HOTPOT_JUDGE_F1_THRESHOLD",
        }

    def get_metric_aggregator(self):
        return compute_hotpotqa_metrics

    def describe_split(self) -> Dict[str, Any]:
        """Return HotpotQA split distribution for sampling protocol design."""
        return describe_hotpotqa_split(
            Path(self._get("HOTPOT_DATASET_DIR", "")),
            setting=self._get("HOTPOT_SETTING", "fullwiki"),
            split=self._get("HOTPOT_SPLIT", "validation"),
        )

    def _sample_ids_file(self) -> str:
        raw = self._get("HOTPOT_SAMPLE_IDS_FILE", "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        candidates = [
            (_HERE / path),
            (_PROJECT_ROOT / path),
            (_BENCHMARKS_ROOT / path),
            (Path.cwd() / path),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())
        return str((_HERE / path).resolve())

    def _filter_samples_by_ids(self, samples: List[BenchmarkSample], sample_ids_file: str) -> List[BenchmarkSample]:
        sample_ids = extract_sample_ids(sample_ids_file)
        by_id = {str(sample.sample_id): sample for sample in samples}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                f"HOTPOT_SAMPLE_IDS_FILE references ids not present in loaded split: "
                f"missing={missing[:10]} total_missing={len(missing)}"
            )
        return [by_id[sample_id] for sample_id in sample_ids]

    def _wiki_dir(self) -> str:
        dataset_dir = Path(self._get("HOTPOT_DATASET_DIR", ""))
        setting = self._get("HOTPOT_SETTING", "fullwiki")
        wiki_dir_override = self._get("HOTPOT_WIKI_CORPUS_DIR", "")
        if wiki_dir_override:
            return wiki_dir_override
        wiki_dirname = self._get(
            "HOTPOT_WIKI_CORPUS_DIRNAME",
            "enwiki-20171001-pages-meta-current-withlinks-abstracts",
        )
        # Support HOTPOT_DATASET_DIR pointing either to the dataset root or
        # directly to the setting directory (e.g. .../hotpotqa_dataset/fullwiki).
        if dataset_dir.name == setting and (dataset_dir.parent / wiki_dirname).exists():
            return str(dataset_dir.parent / wiki_dirname)
        return str(dataset_dir / wiki_dirname)

    def _context_corpus_mode(self) -> str:
        raw = self._get("HOTPOT_CONTEXT_CORPUS_MODE", "wiki").strip().lower()
        if raw in {"sample", "context", "sample_context"}:
            return "sample"
        if raw in {"hybrid", "sample_plus_wiki", "context_plus_wiki"}:
            return "hybrid"
        return "wiki"

    def _require_context_answerable(self) -> bool:
        return self._get_bool("HOTPOT_REQUIRE_CONTEXT_ANSWERABLE", False)

    def _is_context_answerable(self, sample: BenchmarkSample) -> bool:
        context = sample.metadata.get("context") or {}
        if not isinstance(context, dict):
            return False
        titles = context.get("title") or context.get("titles") or []
        sentences = context.get("sentences") or context.get("sentence") or []
        if isinstance(titles, str):
            titles = [titles]
        title_set = {self._normalize_text(str(t)) for t in titles}
        support = sample.metadata.get("supporting_facts") or {}
        support_titles = support.get("title") or support.get("titles") or [] if isinstance(support, dict) else []
        if isinstance(support_titles, str):
            support_titles = [support_titles]
        support_set = {self._normalize_text(str(t)) for t in support_titles if str(t).strip()}
        if support_set and not support_set.issubset(title_set):
            return False

        if isinstance(sentences, list):
            flat_sentences: List[str] = []
            for item in sentences:
                if isinstance(item, list):
                    flat_sentences.extend(str(s) for s in item)
                else:
                    flat_sentences.append(str(item))
            context_text = " ".join(flat_sentences)
        else:
            context_text = str(sentences)
        answer = str(sample.gold_answer or "").strip().lower()
        return bool(answer and answer in context_text.lower())

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().split())

    def _materialize_sample_context(self, sample: BenchmarkSample) -> Path:
        """Materialize parquet context as per-sample raw text files."""
        out_dir = Path(self.get_work_path()) / "context_corpus" / sample.sample_id
        marker = out_dir / ".complete"
        if marker.exists():
            return out_dir

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.txt"):
            try:
                old.unlink()
            except OSError:
                pass

        context = sample.metadata.get("context") or {}
        titles = context.get("title") or context.get("titles") or [] if isinstance(context, dict) else []
        sentences = context.get("sentences") or context.get("sentence") or [] if isinstance(context, dict) else []
        if isinstance(titles, str):
            titles = [titles]
        if not isinstance(titles, list):
            titles = list(titles) if titles is not None else []
        if not isinstance(sentences, list):
            sentences = list(sentences) if sentences is not None else []

        for idx, title in enumerate(titles):
            raw_sentences = sentences[idx] if idx < len(sentences) else []
            if isinstance(raw_sentences, str):
                raw_sentences = [raw_sentences]
            elif not isinstance(raw_sentences, list):
                try:
                    raw_sentences = list(raw_sentences)
                except TypeError:
                    raw_sentences = [str(raw_sentences)]
            text = "\n".join(str(s) for s in raw_sentences if str(s).strip())
            safe_title = self._safe_context_filename(str(title), idx)
            body = f"# {title}\n\n{text}\n" if text else f"# {title}\n"
            (out_dir / f"{idx:02d}_{safe_title}.txt").write_text(
                body,
                encoding="utf-8",
            )

        marker.write_text("ok\n", encoding="utf-8")
        return out_dir

    @staticmethod
    def _safe_context_filename(title: str, idx: int) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
        return (normalized[:120] or f"doc_{idx}")
