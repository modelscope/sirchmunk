"""framework/advisor.py — ImprovementAdvisor

Responsibility: take a BadCaseReport plus the current config and emit
List[ImprovementHypothesis].

Two layers:
1. Signal-driven fast path (no LLM): generate rule suggestions directly from
   root_cause_breakdown and metrics
2. LLM deep analysis (one LLM call): supply context and ask the LLM for deeper hypotheses

Each hypothesis only carries a qualitative impact (low/medium/high); promising concrete
numbers is forbidden. PIPELINE_PATCH / PROMPT_FIX entries only describe the change in
words and never modify code automatically.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .schema import BadCaseReport, ChangeType, ConfigLayer, ImpactLevel, ImprovementHypothesis, RootCause

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config layer classification helpers
# ---------------------------------------------------------------------------

# Layer 0 global config keys: changing them affects every benchmark
_GLOBAL_CONFIG_KEYS: frozenset = frozenset({
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL_NAME", "LLM_TIMEOUT",
    "EMBEDDING_MODEL_ID", "EMBEDDING_CACHE_DIR",
    "SIRCHMUNK_WORK_PATH", "GREP_CONCURRENT_LIMIT",
    "GREP_KEYWORD_CONCURRENT_LIMIT", "GREP_FALLBACK_CONCURRENT_LIMIT",
    "GREP_TIMEOUT", "GREP_QUEUE_TIMEOUT", "GREP_FALLBACK_TIMEOUT",
    "GREP_PROCESS_KILL_TIMEOUT", "GREP_RGA_BACKOFF_SECONDS",
    "GREP_FALLBACK_TO_RG", "SIRCHMUNK_SCOPE_PLANNER_ENABLED",
    "SIRCHMUNK_SCOPE_PLANNER_CONFIDENCE", "SIRCHMUNK_SCOPE_PLANNER_MAX_PATHS",
    "SIRCHMUNK_DIRECTORY_PROFILE_MAX_ENTRIES", "SIRCHMUNK_DIRECTORY_PROFILE_MAX_DEPTH",
    "SIRCHMUNK_FILE_LIST_PAGE_SIZE", "SIRCHMUNK_FILE_READ_MAX_CHARS",
    "SIRCHMUNK_FILE_READ_SMALL_FILE_CHARS", "SIRCHMUNK_FILE_READ_WINDOW_LINES",
})


def _classify_config_layer(hypothesis: ImprovementHypothesis) -> ConfigLayer:
    """Classify a hypothesis into a three-layer config isolation layer.

    Classification rules:
    - PIPELINE_PATCH / PROMPT_FIX touch shared code under src/sirchmunk/ -> Layer 0 GLOBAL
    - CONFIG_CHANGE touching a global key in _GLOBAL_CONFIG_KEYS -> Layer 0 GLOBAL
    - CONFIG_CHANGE touching benchmark-specific config such as HOTPOT_* -> Layer 2 SPECIFIC
    - Default: Layer 2 SPECIFIC (conservative)
    """
    if hypothesis.change_type in (ChangeType.PIPELINE_PATCH, ChangeType.PROMPT_FIX):
        # Code and prompt changes are always global
        return ConfigLayer.GLOBAL

    if hypothesis.change_type == ChangeType.CONFIG_CHANGE:
        keys = set(hypothesis.config_changes.keys())
        if keys & _GLOBAL_CONFIG_KEYS:
            # Contains a global key -> Layer 0
            return ConfigLayer.GLOBAL
        # Only benchmark-specific config -> Layer 2
        return ConfigLayer.SPECIFIC

    return ConfigLayer.SPECIFIC


def _benchmark_env_key(env_file: str, suffix: str) -> str:
    """Infer a benchmark-specific env key from the env file path."""
    lower = (env_file or "").lower()
    if "hotpot" in lower:
        return f"HOTPOT_{suffix}"
    if "setup_cost" in lower or "freshness" in lower or "storage_overhead" in lower or "source_fidelity" in lower or "warm_reuse" in lower:
        return f"MECHANISM_{suffix}"
    return suffix


# ---------------------------------------------------------------------------

_DEEP_ANALYSIS_PROMPT = """\
You are a research engineer helping diagnose and improve a document QA retrieval system (Sirchmunk).

## Current Experiment Metrics
{metrics_summary}

## Failure Analysis Summary
{failure_summary}

## LLM-Identified Failure Patterns
{pattern_summary}

## Current Key Configuration
{config_summary}

## Task
Generate 2-4 concrete improvement hypotheses beyond the rule-based suggestions already made.
Each hypothesis must address a specific root cause observed in the failure analysis.

RULES:
- Do NOT promise specific accuracy numbers
- Clearly label risk (low/medium/high)
- For CONFIG_CHANGE: provide exact env var key and new value
- For PIPELINE_PATCH / PROMPT_FIX: describe the code change location and intent only (no code)
- Each hypothesis must be independent (can be applied without others)

Return ONLY a valid JSON array. Each element:
{{
  "title": "<short title ≤ 10 words>",
  "root_cause": "<retrieval_failure|evidence_partial|synthesis_error|judge_suspect|unknown>",
  "change_type": "<config_change|prompt_fix|pipeline_patch>",
  "description": "<what to change and why, ≤ 50 words>",
  "estimated_impact": "<low|medium|high>",
  "risk_level": "<low|medium|high>",
  "config_changes": {{<env_key>: "<new_value>"}},
  "env_file": "<relative path to .env file or empty>",
  "code_guidance": "<description of code change location and intent, or empty>"
}}
"""


class ImprovementAdvisor:
    """Improvement hypothesis engine.

    Usage::

        advisor = ImprovementAdvisor(llm=llm)
        hypotheses = await advisor.suggest(report, config, env_file)
    """

    def __init__(self, llm: Optional[Any] = None) -> None:
        """
        Args:
            llm: OpenAIChat instance; when None only rule-driven suggestions are produced.
        """
        self._llm = llm
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"H{self._id_counter:03d}"

    async def suggest(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str = "",
    ) -> List[ImprovementHypothesis]:
        """Produce the list of improvement hypotheses.

        Args:
            report:   BadCaseReport from BadCaseAnalyzer.analyze().
            config:   current run-config dict from adapter.get_run_config().
            env_file: relative path of the affected .env file, used by CONFIG_CHANGE.

        Returns:
            A sorted list of ImprovementHypothesis, high impact first.
        """
        hypotheses: List[ImprovementHypothesis] = []

        # ---- Layer 1: signal-driven rule suggestions ----
        hypotheses.extend(self._rule_based_suggestions(report, config, env_file))

        # ---- Layer 2: LLM deep analysis ----
        if self._llm:
            llm_hypotheses = await self._llm_deep_analysis(report, config, env_file)
            # Deduplicate: skip hypotheses with a similar title
            existing_titles = {h.title.lower() for h in hypotheses}
            for h in llm_hypotheses:
                if h.title.lower() not in existing_titles:
                    hypotheses.append(h)
                    existing_titles.add(h.title.lower())

        # Sort by estimated_impact descending
        _impact_order = {ImpactLevel.HIGH: 0, ImpactLevel.MEDIUM: 1, ImpactLevel.LOW: 2}
        hypotheses.sort(key=lambda h: _impact_order.get(h.estimated_impact, 3))

        return hypotheses

    # ------------------------------------------------------------------
    # Layer 1: rule driven
    # ------------------------------------------------------------------

    def _rule_based_suggestions(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str,
    ) -> List[ImprovementHypothesis]:
        """Generate rule suggestions from root_cause_breakdown and metrics."""
        results: List[ImprovementHypothesis] = []
        rc = report.root_cause_breakdown
        total_bad = max(report.total_badcases, 1)

        retrieval_pct = rc.get(RootCause.RETRIEVAL_FAILURE.value, 0) / total_bad
        evidence_partial_pct = rc.get(RootCause.EVIDENCE_PARTIAL.value, 0) / total_bad
        synthesis_pct = rc.get(RootCause.SYNTHESIS_ERROR.value, 0) / total_bad
        no_cov_pct = (report.failure_type_breakdown.get("no_coverage", 0)
                      + report.failure_type_breakdown.get("refusal", 0)) / total_bad

        current_top_k = int(config.get("top_k_files", 5))
        current_mode = config.get("mode", "DEEP")
        top_k_key = str(config.get("top_k_env_key") or _benchmark_env_key(env_file, "TOP_K_FILES"))
        mode_key = str(config.get("mode_env_key") or _benchmark_env_key(env_file, "MODE"))

        # --- Rule 1: high retrieval failure rate -> raise top_k ---
        if retrieval_pct > 0.35 or no_cov_pct > 0.40:
            new_top_k = min(current_top_k + 5, 20)
            if new_top_k > current_top_k:
                h = ImprovementHypothesis(
                    hypothesis_id=self._next_id(),
                    title=f"Increase top_k_files {current_top_k} → {new_top_k}",
                    root_cause=RootCause.RETRIEVAL_FAILURE.value,
                    change_type=ChangeType.CONFIG_CHANGE,
                    description=(
                        f"Retrieval failure accounts for {retrieval_pct:.0%} of badcases. "
                        f"Increasing top_k_files from {current_top_k} to {new_top_k} "
                        f"exposes more candidate files to the search agent."
                    ),
                    estimated_impact=ImpactLevel.MEDIUM,
                    risk_level="low",
                    config_changes={top_k_key: str(new_top_k)},
                    env_file=env_file,
                    estimated_coverage_fraction=retrieval_pct,
                )
                h.config_layer = _classify_config_layer(h)  # Layer 2 (SPECIFIC)
                results.append(h)

        # --- Rule 2: high incomplete-evidence rate -> switch to DEEP mode ---
        if evidence_partial_pct > 0.25 and current_mode == "FAST":
            h = ImprovementHypothesis(
                hypothesis_id=self._next_id(),
                title="Switch search mode FAST → DEEP",
                root_cause=RootCause.EVIDENCE_PARTIAL.value,
                change_type=ChangeType.CONFIG_CHANGE,
                description=(
                    f"Evidence partial failure accounts for {evidence_partial_pct:.0%} of badcases. "
                    "DEEP mode uses multi-phase retrieval and Monte Carlo evidence sampling "
                    "which can surface evidence missed by FAST mode. Note: higher latency and cost."
                ),
                estimated_impact=ImpactLevel.HIGH,
                risk_level="medium",
                config_changes={mode_key: "DEEP"},
                env_file=env_file,
                estimated_coverage_fraction=evidence_partial_pct,
            )
            h.config_layer = _classify_config_layer(h)  # Layer 2 (SPECIFIC)
            results.append(h)

        # --- Rule 3: high synthesis error rate -> tighten answer/evidence validation ---
        if synthesis_pct > 0.30:
            h = ImprovementHypothesis(
                hypothesis_id=self._next_id(),
                title="Strengthen synthesis verification",
                root_cause=RootCause.SYNTHESIS_ERROR.value,
                change_type=ChangeType.PIPELINE_PATCH,
                description=(
                    f"Synthesis errors account for {synthesis_pct:.0%} of badcases. "
                    "Consider strengthening final-answer extraction and evidence-grounding checks."
                ),
                estimated_impact=ImpactLevel.MEDIUM,
                risk_level="medium",
                code_guidance=(
                    "Files: src/sirchmunk/search.py and benchmark-specific judge/evidence modules\n"
                    "Change: ensure final synthesis returns a concise answer span and verify it "
                    "against retrieved evidence before marking coverage as true."
                ),
                estimated_coverage_fraction=synthesis_pct,
            )
            h.config_layer = _classify_config_layer(h)
            results.append(h)

        # --- Rule 4: judge_suspect above threshold -> flag for manual review ---
        if report.judge_suspect_ids:
            pct = len(report.judge_suspect_ids) / total_bad
            if pct > 0.05:
                h = ImprovementHypothesis(
                    hypothesis_id=self._next_id(),
                    title="Review suspected judge false-negatives",
                    root_cause=RootCause.JUDGE_SUSPECT.value,
                    change_type=ChangeType.PIPELINE_PATCH,
                    description=(
                        f"{len(report.judge_suspect_ids)} cases ({pct:.0%} of badcases) "
                        "have numeric overlap between prediction and gold but judge=False. "
                        "Manual review recommended; consider adjusting judge confidence threshold."
                    ),
                    estimated_impact=ImpactLevel.LOW,
                    risk_level="low",
                    code_guidance=(
                        "File: benchmarks/hotpotqa/judge.py or the active benchmark judge module\n"
                        "Class: HotpotQAJudge (or equivalent)\n"
                        "Change: Review the semantic judge confidence threshold and F1 fallback "
                        "policy to reduce false negatives without introducing false positives."
                    ),
                    estimated_coverage_fraction=pct,
                )
                # Benchmark-specific judge modules are treated as SPECIFIC unless the adapter marks otherwise.
                h.config_layer = ConfigLayer.SPECIFIC
                results.append(h)

        return results

    # ------------------------------------------------------------------
    # Layer 2: LLM deep analysis
    # ------------------------------------------------------------------

    async def _llm_deep_analysis(
        self,
        report: BadCaseReport,
        config: Dict[str, Any],
        env_file: str,
    ) -> List[ImprovementHypothesis]:
        """One LLM call that produces supplementary hypotheses."""
        metrics_summary = (
            f"Accuracy: {report.accuracy:.1f}%  Coverage: {report.coverage:.1f}%  "
            f"Total: {report.total_samples}  Badcases: {report.total_badcases}"
        )
        failure_summary = "\n".join(
            f"  {k}: {v} ({v / max(report.total_badcases, 1) * 100:.1f}%)"
            for k, v in sorted(report.root_cause_breakdown.items(), key=lambda x: -x[1])
        ) or "  (no badcases)"

        pattern_summary = report.pattern_summary or "(not available)"

        # Show only the key config fields
        key_config = {
            k: v for k, v in config.items()
            if any(s in k.lower() for s in (
                "mode", "top_k", "budget", "judge", "eval_mode", "model"
            ))
        }
        config_summary = json.dumps(key_config, indent=2, ensure_ascii=False)

        prompt = _DEEP_ANALYSIS_PROMPT.format(
            metrics_summary=metrics_summary,
            failure_summary=failure_summary,
            pattern_summary=pattern_summary,
            config_summary=config_summary,
        )

        try:
            resp = await self._llm.achat(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            raw = (resp.content or "").strip()
            items = self._parse_json_array(raw)
            return [self._dict_to_hypothesis(item, env_file) for item in items if item]
        except Exception as exc:
            logger.warning("[Advisor] LLM deep analysis failed: %s", exc)
            return []

    def _dict_to_hypothesis(self, item: Dict[str, Any], env_file: str) -> ImprovementHypothesis:
        """Convert an LLM output dict into an ImprovementHypothesis and tag config_layer."""
        try:
            ct = ChangeType(item.get("change_type", "config_change"))
        except ValueError:
            ct = ChangeType.CONFIG_CHANGE

        try:
            impact = ImpactLevel(item.get("estimated_impact", "medium"))
        except ValueError:
            impact = ImpactLevel.MEDIUM

        h = ImprovementHypothesis(
            hypothesis_id=self._next_id(),
            title=str(item.get("title", "LLM suggestion"))[:80],
            root_cause=str(item.get("root_cause", "unknown")),
            change_type=ct,
            description=str(item.get("description", ""))[:500],
            estimated_impact=impact,
            risk_level=str(item.get("risk_level", "low")),
            config_changes=dict(item.get("config_changes") or {}),
            env_file=str(item.get("env_file") or env_file),
            code_guidance=str(item.get("code_guidance", "")),
        )
        # Annotate the config layer automatically
        h.config_layer = _classify_config_layer(h)
        return h

    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """Extract the JSON array from an LLM response."""
        import re
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        # Try a direct parse first
        try:
            result = json.loads(cleaned)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
        # Extract the first [...] block
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return []
