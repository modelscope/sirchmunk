"""framework/adapter.py — BenchmarkAdapter abstract base class

Every benchmark must implement this interface to join the research pipeline.
Dependency inversion: UnifiedExperimentRunner depends only on this abstraction, never
on a concrete benchmark.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from .schema import BenchmarkSample


class BenchmarkAdapter(ABC):
    """Abstract base class for benchmark onboarding adapters.

    Implementers must override every abstractmethod.
    Optional hook with a default implementation: ``extra_result_fields``.
    """

    # ------------------------------------------------------------------
    # Must be implemented
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name, used for logging and file naming."""

    @property
    @abstractmethod
    def env_file(self) -> str:
        """Absolute path of the main .env config file."""

    @abstractmethod
    def load_samples(self, limit: int = 0, seed: int = 42) -> List[BenchmarkSample]:
        """Load the question set.

        Args:
            limit: 0 means the full set; >0 samples `limit` questions using `seed`.
            seed:  random seed that keeps sampling reproducible.

        Returns:
            A list of BenchmarkSample.
        """

    @abstractmethod
    def validate_corpus(self) -> Tuple[int, List[str]]:
        """Validate corpus integrity.

        Returns:
            (found_count, missing_doc_names)
        """

    @abstractmethod
    def get_search_paths(self, sample: BenchmarkSample) -> List[str]:
        """Return the list of search paths for the given sample.

        singleDoc mode returns a single PDF path; sharedCorpus mode returns the whole
        corpus directory.
        """

    @abstractmethod
    def get_run_config(self) -> Dict[str, Any]:
        """Return a serializable run-config dict for config_hash and experiment records."""

    @abstractmethod
    def build_searcher(self) -> Any:
        """Build an AgenticSearch instance, or any searcher with a compatible interface."""

    @abstractmethod
    def build_judge(self) -> Optional[Any]:
        """Build the LLM judge instance, or return None when no judge is used."""

    @abstractmethod
    def get_output_dir(self) -> str:
        """Absolute path of the output directory."""

    @abstractmethod
    def get_work_path(self) -> str:
        """Absolute path of the AgenticSearch work_path."""

    # ------------------------------------------------------------------
    # Optional hooks with default implementations
    # ------------------------------------------------------------------

    def extra_result_fields(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Return benchmark-specific fields to write into the result JSONL.

        By default the whole sample.metadata is passed through.
        Subclasses may override for finer control.
        """
        return dict(sample.metadata)

    def get_max_concurrent(self) -> int:
        """Sample-level concurrency, default 3.

        The semantics have two layers that must not be mixed:
        - This system running itself (UnifiedExperimentRunner): how many samples are
          processed concurrently.
        - Competitor evaluation (BaselineEvaluationSuite): how many competitor systems
          are evaluated concurrently.

        It does not control per-competitor internal sample concurrency, which each
        BaselineAdapter declares through its own get_max_concurrent(); the default there
        is sequential.
        """
        return 3

    def get_baseline_sample_concurrency(self) -> int:
        """Unified override for per-competitor sample concurrency, 0 = no override.

        By default nothing is overridden, so each BaselineAdapter keeps its own
        declaration (usually sequential). Setting a positive integer lifts every
        competitor that supports concurrent queries to the same concurrency, which
        changes wall-clock time by an order of magnitude for multi-round LLM competitors
        such as ReAct. The cost is that the measurement condition of the latency column
        changes with it, so the realized concurrency is recorded alongside the results
        for tables and audits.
        """
        return 0

    def get_request_delay(self) -> float:
        """Delay between requests in seconds, default 0.5."""
        return 0.5

    def get_search_kwargs(self) -> Dict[str, Any]:
        """Extra keyword arguments forwarded to searcher.search()."""
        return {}

    def get_protocol_spec(self, run_id: str, seed: int, limit: int) -> Dict[str, Any]:
        """Return an optional machine-readable experiment protocol override."""
        return {}

    def get_dataset_manifest(self) -> Dict[str, Any]:
        """Return optional dataset/corpus provenance metadata."""
        return {}

    def enrich_telemetry(self, sample: BenchmarkSample, prediction: str, telemetry: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Return optional per-sample telemetry additions after judging."""
        return {}

    def get_analysis_schema(self) -> Dict[str, Any]:
        """Return benchmark-specific metadata keys for badcase analysis."""
        return {"primary_group_key": "group"}

    def get_config_schema(self) -> Dict[str, Any]:
        """Return benchmark-specific config keys and global config boundaries."""
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
            "top_k_env_key": "TOP_K_FILES",
            "mode_env_key": "MODE",
        }

    def get_cache_policy(self) -> Dict[str, Any]:
        """Return adapter-specific cache policy hints for P3 queue runs."""
        return {
            "cache_names": [".cache", "knowledge", "history", "compile", "rga"],
            "cache_paths": [".cache/rga", ".cache/knowledge", ".cache/compile"],
            "compiled_markers": ["compile", "compiled"],
        }

    def get_metric_aggregator(self):
        """Return optional benchmark-specific metrics aggregator callable."""
        return None
