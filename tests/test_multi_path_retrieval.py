from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from sirchmunk.learnings.corpus_topic_map import CorpusTopicMap
from sirchmunk.learnings.lens_config import LensConfig
from sirchmunk.llm.openai_chat import LLMTokenBudgetExceeded, OpenAIChat
from sirchmunk.retrieve.confidence import (
    ConfidenceConfig,
    ConfidenceFusionPolicy,
    LexicalConfidenceFeatures,
    calibrate_lexical_features,
)
from sirchmunk.retrieve.text_retriever import GrepRetriever
from sirchmunk.learnings.multi_arm_navigator import MultiArmNavigator
from sirchmunk.learnings.tree_indexer import DocumentTree, DocumentTreeIndexer, TreeNode
from sirchmunk.scan.dir_scanner import DirectoryScanner, FileCandidate, ScanResult
from sirchmunk.schema.query_intelligence import QueryIntelligence
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.search import (
    AgenticSearch,
    CompileArtifacts,
    DataRequirements,
    _PathScope,
    _deep_retrieval_profile,
)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def achat(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.payload), usage={})


class _FakeLogger:
    async def info(self, _message: str) -> None:
        return None

    async def warning(self, _message: str) -> None:
        return None


@pytest.mark.asyncio
async def test_query_intelligence_is_built_in_one_llm_call() -> None:
    llm = _FakeLLM({
        "keywords_level1": {"latent evidence": 8},
        "keywords_level2": {"retrieval": 6},
        "keywords_alt": {"证据检索": 7},
        "intent": "lookup",
        "complexity": "moderate",
        "data_points": ["retrieval method"],
        "likely_sources": ["Method"],
        "expected_answer_type": "entity",
        "target_slot": "method name",
        "hop_type": "single",
        "reformulations": ["evidence retrieval method"],
        "entities": ["LENS"],
        "concepts": ["evidence localization"],
        "expected_doc_type": "report",
        "expected_sections": ["Method"],
        "location_hint": "body",
        "multi_source_score": 0.2,
    })
    search = object.__new__(AgenticSearch)
    search.llm = llm
    search._logger = _FakeLogger()
    search.llm_usages = []

    intelligence = await search._build_query_intelligence("What is the method?")

    assert llm.calls == 1
    assert intelligence.keywords["latent evidence"] == 8.0
    assert intelligence.entities == ["LENS"]
    assert intelligence.expected_sections == ["Method"]


def test_deep_retrieval_profile_is_dynamic(monkeypatch) -> None:
    monkeypatch.setenv("SIRCHMUNK_DEEP_RETRIEVAL_PROFILE", "legacy_keyword")
    assert _deep_retrieval_profile() == "legacy_keyword"
    monkeypatch.setenv("SIRCHMUNK_DEEP_RETRIEVAL_PROFILE", "multipath")
    assert _deep_retrieval_profile() == "multipath"


@pytest.mark.asyncio
async def test_native_format_fallback_finds_precise_log_entity(tmp_path) -> None:
    log_file = tmp_path / "events.log"
    log_file.write_text("noise\nRecord Exact-7788 has value 42\n", encoding="utf-8")
    retriever = GrepRetriever(work_path=tmp_path)

    results = await retriever.retrieve_native_formats(
        ["Exact-7788"], path=[tmp_path], max_depth=2,
    )
    cached_results = await retriever.retrieve_native_formats(
        ["Exact-7788"], path=[tmp_path], max_depth=2,
    )

    assert results[0]["path"] == str(log_file)
    assert "Exact-7788" in results[0]["lines"][0]
    assert results[0]["matches"][0]["_search_cache_hit"] is False
    assert results[0]["matches"][0]["_conversion_elapsed_ms"] >= 0
    assert cached_results[0]["matches"][0]["_search_cache_hit"] is True


@pytest.mark.asyncio
async def test_directory_disk_cache_reuses_metadata(tmp_path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("cached content", encoding="utf-8")
    cache_dir = tmp_path / ".cache"
    first_scanner = DirectoryScanner(
        max_files=20,
        cache_mode="disk",
        cache_dir=cache_dir,
        small_file_threshold=0,
    )
    second_scanner = DirectoryScanner(
        max_files=20,
        cache_mode="disk",
        cache_dir=cache_dir,
        small_file_threshold=0,
    )

    first = await first_scanner.scan([tmp_path])
    second = await second_scanner.scan([tmp_path])

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.candidates[0].path == str(document)


@pytest.mark.asyncio
async def test_directory_scan_cache_is_freshness_safe(tmp_path) -> None:
    document = tmp_path / "alpha.txt"
    document.write_text("alpha content", encoding="utf-8")
    scanner = DirectoryScanner(
        max_files=20,
        cache_mode="memory",
        small_file_threshold=0,
    )

    first = await scanner.scan([tmp_path])
    second = await scanner.scan([tmp_path])
    document.write_text("updated alpha content", encoding="utf-8")
    third = await scanner.scan([tmp_path])

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.files_reused == 1
    assert third.cache_hit is False
    assert third.files_refreshed == 1


def test_query_expansion_preserves_priorities_and_adds_new_signals() -> None:
    qi = QueryIntelligence(
        keywords={"primary phrase": 8.0},
        alt_keywords={"替代术语": 7.0},
        entities=["Named Entity"],
        concepts=["latent evidence"],
        reformulations=["Find the alternate formulation precisely"],
    )

    expanded = AgenticSearch._expand_keywords_from_reformulations(qi)

    assert list(expanded)[0] == "primary phrase"
    assert expanded["Named Entity"] == 9.0
    assert expanded["latent evidence"] == 6.0
    assert "alternate" in expanded


def test_calibrated_confidence_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LENS_CONFIDENCE_GLOBAL_THRESHOLD", raising=False)
    monkeypatch.delenv("LENS_CONFIDENCE_MARGIN", raising=False)

    config = ConfidenceConfig.from_env()

    assert config.global_threshold == 0.90
    assert config.margin_threshold == 0.10


def test_confidence_fusion_fast_tracks_strong_correlated_route() -> None:
    policy = ConfidenceFusionPolicy(ConfidenceConfig(
        mode="soft",
        global_threshold=0.85,
        margin_threshold=0.15,
        min_supporting_routes=2,
    ))

    decision = policy.fuse(
        rankings={
            "lexical": ["target", "other"],
            "entity": [],
            "structure": [],
            "directory": ["target", "other"],
            "existing": [],
        },
        route_scores={
            "lexical": {"target": 0.55, "other": 0.30},
            "directory": {"target": 0.95, "other": 0.40},
        },
        fast_track_eligible=True,
    )

    assert decision.ranked_files[0] == "target"
    assert decision.fast_track is True
    assert decision.dominant_file == "target"
    assert decision.supporting_routes["target"] == 2


def test_unique_exact_entity_can_drive_route_collapse_without_fake_consensus() -> None:
    policy = ConfidenceFusionPolicy(ConfidenceConfig(
        mode="soft", margin_threshold=0.1,
    ))

    decision = policy.fuse(
        rankings={
            "lexical": ["noise"], "entity": ["target"], "structure": [],
            "directory": [], "existing": [],
        },
        route_scores={
            "lexical": {"noise": 0.3},
            "entity": {"target": 0.95},
        },
        fast_track_eligible=True,
        dominant_evidence={"target": "unique_exact_entity"},
        allow_loop_reduction=False,
    )

    assert decision.route_collapse is True
    assert decision.dominant_file == "target"
    assert decision.reasoning_profile == "standard"


def test_lexical_v2_caps_high_frequency_low_coverage() -> None:
    noisy = calibrate_lexical_features(LexicalConfidenceFeatures(
        term_coverage=0.1,
        group_coverage=1.0,
        rarity=0.05,
        density=1.0,
        exact_phrase=False,
    ))
    precise = calibrate_lexical_features(LexicalConfidenceFeatures(
        term_coverage=0.9,
        group_coverage=2 / 3,
        rarity=0.9,
        density=0.5,
        exact_phrase=True,
    ))

    assert noisy < 0.5
    assert precise > 0.8


def test_confidence_fusion_rejects_single_route_overconfidence() -> None:
    policy = ConfidenceFusionPolicy(ConfidenceConfig(mode="soft"))

    decision = policy.fuse(
        rankings={
            "lexical": [], "entity": [], "structure": [],
            "directory": ["target"], "existing": [],
        },
        route_scores={"directory": {"target": 1.0}},
        fast_track_eligible=True,
    )

    assert decision.fast_track is False
    assert decision.reason == "insufficient_route_consensus"


def test_confidence_monitor_mode_never_changes_execution() -> None:
    policy = ConfidenceFusionPolicy(ConfidenceConfig(
        mode="monitor",
        global_threshold=0.1,
        margin_threshold=0.0,
        min_supporting_routes=1,
    ))

    decision = policy.fuse(
        rankings={
            "lexical": ["a", "b"], "entity": [], "structure": [],
            "directory": ["b"], "existing": [],
        },
        route_scores={"directory": {"b": 1.0}},
        fast_track_eligible=True,
    )

    assert decision.fast_track is False
    assert decision.reason == "monitor_only"


def test_fast_track_falls_back_on_absent_evidence() -> None:
    search = object.__new__(AgenticSearch)
    context = SearchContext()
    context.telemetry = {"evidence_sufficiency": "absent"}

    assert search._fast_track_requires_fallback("candidate", context) is True


def test_weighted_rrf_rewards_cross_path_consensus() -> None:
    ranked, scores = AgenticSearch._fuse_retrieval_results(
        keyword_results=["a", "b"],
        phrase_results=["b"],
        structure_results=[("b", 0.8, [(10, 30)]), ("c", 0.7, [])],
        directory_results=[("c", 0.5)],
        existing_results=["a"],
    )

    assert ranked[0] == "b"
    assert scores["b"] > scores["a"]
    assert scores["c"] > 0


def test_directory_semantic_rank_uses_type_and_entity_signals() -> None:
    scan = ScanResult(candidates=[
        FileCandidate(
            path="/repo/src/payment_engine.py",
            filename="payment_engine.py",
            extension=".py",
            title="Payment Engine",
        ),
        FileCandidate(
            path="/repo/reports/annual.pdf",
            filename="annual.pdf",
            extension=".pdf",
            title="Annual Results",
        ),
    ])

    ranked = DirectoryScanner.semantic_rank(
        scan,
        expected_doc_type="code",
        entities=["Payment Engine"],
    )
    detailed = DirectoryScanner.semantic_rank_detailed(
        scan,
        expected_doc_type="code",
        entities=["Payment Engine"],
    )

    assert ranked[0][0].endswith("payment_engine.py")
    assert ranked[0][1] > 0
    assert detailed[0].direct_match is True
    assert detailed[0].signal_count >= 2
    assert detailed[0].confidence >= 0.85


def test_directory_exact_filename_can_reach_dominant_confidence() -> None:
    scan = ScanResult(candidates=[FileCandidate(
        path="/repo/ledgers/ledger_0003.xlsx",
        filename="ledger_0003.xlsx",
        extension=".xlsx",
        title="Financial Ledger",
    )])

    hits = DirectoryScanner.semantic_rank_detailed(
        scan,
        expected_doc_type="data",
        entities=["ledger_0003.xlsx", "AggUnit-8585"],
    )

    assert hits[0].confidence >= 0.85
    assert hits[0].components["filename_exact"] > 0


def test_heuristic_summary_keeps_identifiers_beyond_the_lead() -> None:
    summary = DocumentTreeIndexer._heuristic_summary(
        "introductory text " * 40 + "Asset Brontes-3964 is assigned here."
    )

    assert "Brontes-3964" in summary
    assert len(summary) <= 500


def test_heuristic_tree_v2_accepts_single_container_with_multiple_leaves(monkeypatch) -> None:
    monkeypatch.setenv("LENS_TREE_HEURISTIC_VERSION", "v2")
    content = (
        "# Report\n"
        "## Methods\n" + "method text " * 250
        + "\n## Results\n" + "result text " * 250
    )
    indexer = object.__new__(DocumentTreeIndexer)

    root = indexer._build_tree_heuristic(content)

    assert root is not None
    assert len(root.children) >= 2
    assert len(root.all_leaves()) >= 2


def test_heuristic_tree_v2_bounds_oversegmented_documents(monkeypatch) -> None:
    monkeypatch.setenv("LENS_TREE_HEURISTIC_VERSION", "v2")
    monkeypatch.setenv("LENS_TREE_MAX_NODES", "16")
    content = "\n".join(
        f"# Section {index}\n" + (f"payload {index} " * 20)
        for index in range(80)
    )
    indexer = object.__new__(DocumentTreeIndexer)

    root = indexer._build_tree_heuristic(content)

    assert root is not None
    assert len(root.all_leaves()) <= 16


def test_heuristic_tree_builds_heading_ranges_without_llm() -> None:
    content = (
        "# Introduction\n" + "alpha " * 400
        + "\n# Results\n" + "beta " * 400
        + "\n# Conclusion\n" + "gamma " * 400
    )
    indexer = object.__new__(DocumentTreeIndexer)

    root = indexer._build_tree_heuristic(content)

    assert root is not None
    assert len(root.children) == 3
    assert root.children[0].char_range[1] <= root.children[1].char_range[0]


@dataclass
class _FakeIndexer:
    tree: DocumentTree

    def load_tree(self, _file_path: str) -> DocumentTree:
        return self.tree


@pytest.mark.asyncio
async def test_structure_probe_matches_exact_entity_in_summary(tmp_path) -> None:
    document = tmp_path / "report.txt"
    document.write_text("content", encoding="utf-8")
    root = TreeNode(
        node_id="root",
        title="Document",
        summary="",
        char_range=(0, 100),
        children=[TreeNode(
            node_id="details",
            title="Details",
            summary="Identifiers: Brontes-3964, Division-8054",
            char_range=(20, 80),
            level=1,
        )],
    )
    tree = DocumentTree(
        file_path=str(document), file_hash="hash", created_at="now",
        total_chars=100, root=root,
    )
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    artifacts = CompileArtifacts(
        catalog=[], catalog_map={}, tree_indexer=_FakeIndexer(tree),
        tree_available_paths={str(document)},
    )

    hits = await search._structure_guided_probe(
        QueryIntelligence(entities=["Brontes-3964"]),
        artifacts,
        _PathScope([str(tmp_path)]),
    )

    assert hits[0][0] == str(document)
    assert hits[0][1] >= 0.75
    assert hits[0][2] == [(20, 80)]


@pytest.mark.asyncio
async def test_structure_probe_returns_ranked_char_ranges(tmp_path) -> None:
    document = tmp_path / "report.txt"
    document.write_text("results", encoding="utf-8")
    root = TreeNode(
        node_id="root",
        title="Document",
        summary="",
        char_range=(0, 100),
        children=[TreeNode(
            node_id="results",
            title="Experimental Results",
            summary="retrieval quality and latency",
            char_range=(20, 80),
            level=1,
        )],
    )
    tree = DocumentTree(
        file_path=str(document), file_hash="hash", created_at="now",
        total_chars=100, root=root,
    )
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    artifacts = CompileArtifacts(
        catalog=[], catalog_map={}, tree_indexer=_FakeIndexer(tree),
        tree_available_paths={str(document)},
    )
    qi = QueryIntelligence(
        expected_sections=["Results"], concepts=["retrieval quality"],
    )

    hits = await search._structure_guided_probe(
        qi, artifacts, _PathScope([str(tmp_path)])
    )

    assert hits[0][0] == str(document)
    assert hits[0][2] == [(20, 80)]


def test_corpus_topic_map_discovers_cross_document_topics(tmp_path) -> None:
    root = TreeNode(
        node_id="root",
        title="Document",
        summary="",
        char_range=(0, 100),
        children=[
            TreeNode(
                node_id="cast",
                title="Cast and Characters",
                summary="",
                char_range=(0, 50),
                level=1,
            )
        ],
    )
    tree = DocumentTree(
        file_path="/corpus/film.txt",
        file_hash="hash",
        created_at="now",
        total_chars=100,
        root=root,
    )
    topic_map = CorpusTopicMap.build_from_indexer(
        _FakeIndexer(tree), ["/corpus/film.txt"]
    )

    assert topic_map.search(["cast"])[0][0] == "/corpus/film.txt"
    artifact = tmp_path / "topic-map.json"
    topic_map.save(artifact)
    loaded = CorpusTopicMap.load(artifact)
    assert loaded is not None
    assert loaded.search(["characters"])[0][0] == "/corpus/film.txt"


def test_corpus_topic_map_prunes_high_document_frequency_tokens() -> None:
    """Non-discriminative tokens are dropped by corpus DF, not a fixed list.

    A token shared by every document (here 'overview') carries no routing
    signal and is pruned, while a rare token still routes to its file — without
    any hardcoded stopword vocabulary.
    """
    trees = {}
    for index in range(10):
        path = f"/corpus/doc_{index}.txt"
        children = [
            TreeNode(
                node_id=f"overview-{index}",
                title="Overview",
                summary="",
                char_range=(0, 10),
                level=1,
            )
        ]
        if index == 3:
            children.append(
                TreeNode(
                    node_id="rare",
                    title="Peculiar Turbine Calibration",
                    summary="",
                    char_range=(10, 20),
                    level=1,
                )
            )
        trees[path] = DocumentTree(
            file_path=path,
            file_hash="hash",
            created_at="now",
            total_chars=20,
            root=TreeNode(
                node_id=f"root-{index}",
                title="Document",
                summary="",
                char_range=(0, 20),
                children=children,
            ),
        )

    class _MultiIndexer:
        def load_tree(self, file_path):
            return trees.get(str(file_path))

    topic_map = CorpusTopicMap.build_from_indexer(
        _MultiIndexer(), list(trees), min_documents_for_pruning=8,
        stop_document_fraction=0.5,
    )

    # 'overview' is in all 10 docs -> pruned -> no routing signal.
    assert topic_map.search(["overview"]) == []
    # A rare, discriminative token still routes to its single file.
    assert topic_map.search(["turbine"])[0][0] == "/corpus/doc_3.txt"


@pytest.mark.asyncio
async def test_computation_trace_parsed_from_answer_when_no_telemetry() -> None:
    """Fallback path: a trace embedded in the answer text is still honored."""
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    context = SearchContext()
    answer = (
        "30316\n"
        "<COMPUTATION_TRACE>{\"operation\": \"sum\", "
        "\"operands\": [4996, 3850, 2046], \"result\": 30316}</COMPUTATION_TRACE>"
    )
    evidence = "4996\n3850\n2046"

    corrected_answer, corrected = await search._verify_computation(
        "total", answer, evidence=evidence, context=context,
    )
    assert corrected is True
    assert AgenticSearch._extract_answer_span(corrected_answer) == "10892"


@pytest.mark.asyncio
async def test_structure_anchor_extracts_only_target_region(tmp_path) -> None:
    document = tmp_path / "structured.txt"
    document.write_text("prefix\n" + "target evidence " * 100 + "\nsuffix", encoding="utf-8")
    search = object.__new__(AgenticSearch)

    result = await search._extract_structured_anchor_content(
        str(document), [(7, 107)], max_chars=500, padding=0,
    )

    assert result is not None
    assert "target evidence" in result
    assert len(result) < 600


@pytest.mark.asyncio
async def test_dominant_exact_route_uses_snippets_before_document_extraction() -> None:
    search = object.__new__(AgenticSearch)
    context = SearchContext()

    async def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("exact-match warm start must not pre-read the document")

    search._agentic_retrieve = unexpected_retrieval
    evidence = await search._build_prior_observations(
        "What reference value was assigned to record Halcyon-9807?",
        ["/corpus/report.pdf"],
        {"/corpus/report.pdf": [
            "Record Halcyon-9807 was assigned reference value 391.",
        ]},
        context,
        exact_snippets_only=True,
    )

    assert "reference value 391" in evidence
    assert "/corpus/report.pdf" not in context.read_file_ids
    assert context.retrieval_logs[0].tool_name == "exact_match_snippet"
    assert context.telemetry["warm_start_exact_only"] is True


def test_answer_resolution_rejects_numeric_entity_drift() -> None:
    filename_requirements = DataRequirements(
        data_points=[], likely_sources=[], formula=None, time_period=None,
        intent="lookup", target_slot="full filename",
        answer_constraints=["Must include the file extension"],
    )
    assert AgenticSearch._preserves_target_form(
        "atlas89783", "report_atlas89783_biology.json", filename_requirements,
    ) is False
    assert AgenticSearch._preserves_target_form(
        "report_atlas89783_biology.json",
        "report_atlas89783_biology.json",
        filename_requirements,
    ) is True
    assert AgenticSearch._preserves_numeric_tokens(
        "Division-8054 is headquartered in Brightmoor-522",
        "Brightmoor-723",
    ) is False
    assert AgenticSearch._preserves_numeric_tokens(
        "445 百万美元", "445",
    ) is True
    assert AgenticSearch._is_compact_quantity_with_unit("959 kqps") is True
    assert AgenticSearch._number_answer_requires_unit(
        "What peak throughput did it sustain?", None,
    ) is True


@pytest.mark.asyncio
async def test_answer_resolution_preserves_required_numeric_unit() -> None:
    search = object.__new__(AgenticSearch)
    search.llm = _FakeLLM({
        "relational": "445 百万美元",
        "minimal": "445",
    })
    search._logger = _FakeLogger()
    search.llm_usages = []
    context = SearchContext()
    requirements = DataRequirements(
        data_points=["Cassin-2545 annual revenue"],
        likely_sources=[],
        formula=None,
        time_period="annual",
        intent="lookup",
        expected_answer_type="number",
        target_slot="annual revenue",
        answer_constraints=["Must be a monetary value"],
    )

    contract = search._build_loop_answer_contract(requirements)
    assert "Must be a monetary value" in contract
    assert "include its magnitude and unit" in contract

    answer = await search._resolve_answer_span(
        "业务单元 Cassin-2545 报告的年度营收是多少？",
        "445",
        "业务单元 Cassin-2545 报告了 445 百万美元的年度营收。",
        "lookup",
        requirements,
        context,
    )

    assert AgenticSearch._extract_answer_span(answer) == "445 百万美元"


@pytest.mark.asyncio
async def test_computation_verifier_corrects_grounded_trace_from_telemetry() -> None:
    """A grounded computation trace drives deterministic correction.

    The operands are corpus-agnostic (arbitrary entity/columns); correctness
    comes from re-summing the model-disclosed, evidence-grounded operands, not
    from any hardcoded row/column regex.
    """
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    context = SearchContext()
    context.telemetry = {
        "computation_trace": json.dumps(
            {"operation": "sum", "operands": [4996, 3850, 2046], "result": 30316}
        )
    }
    evidence = "\n".join([
        "region,north,4996,usd",
        "region,south,3850,usd",
        "region,east,2046,usd",
    ])

    answer, corrected = await search._verify_computation(
        "What is the total revenue across all regions?",
        "30316",
        evidence=evidence,
        context=context,
    )

    assert corrected is True
    assert AgenticSearch._extract_answer_span(answer) == "10892"
    assert context.telemetry["computation_deterministic_corrected"] is True


@pytest.mark.asyncio
async def test_computation_verifier_supports_mean_operation() -> None:
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    context = SearchContext()
    context.telemetry = {
        "computation_trace": json.dumps(
            {"operation": "average", "operands": [10, 20, 30], "result": 999}
        )
    }
    evidence = "scores: 10, 20, 30"

    answer, corrected = await search._verify_computation(
        "What is the average score?", "999",
        evidence=evidence, context=context,
    )

    assert corrected is True
    assert AgenticSearch._extract_answer_span(answer) == "20"


@pytest.mark.asyncio
async def test_computation_verifier_keeps_correct_answer() -> None:
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    context = SearchContext()
    context.telemetry = {
        "computation_trace": json.dumps(
            {"operation": "sum", "operands": [100, 200], "result": 300}
        )
    }
    answer, corrected = await search._verify_computation(
        "total", "300",
        evidence="a 100 b 200", context=context,
    )
    assert corrected is False
    assert AgenticSearch._extract_answer_span(answer) == "300"
    assert context.telemetry["computation_deterministic_verified"] is True


@pytest.mark.asyncio
async def test_computation_verifier_rejects_ungrounded_operands() -> None:
    """Hallucinated operands (absent from evidence) must not trigger correction."""
    search = object.__new__(AgenticSearch)
    search._logger = _FakeLogger()
    context = SearchContext()
    context.telemetry = {
        "computation_trace": json.dumps(
            {"operation": "sum", "operands": [4996, 9999], "result": 14995}
        )
    }
    evidence = "only one relevant value: 4996"

    answer, corrected = await search._verify_computation(
        "total", "14995",
        evidence=evidence, context=context,
    )
    # 9999 is not in evidence -> trace is not trusted -> no deterministic
    # correction, and no inline expression to fall back on.
    assert corrected is False
    assert AgenticSearch._extract_answer_span(answer) == "14995"


def test_hard_token_budget_rejects_oversized_prompt(monkeypatch) -> None:
    monkeypatch.setenv("LENS_HARD_TOKEN_BUDGET", "true")
    token = OpenAIChat.bind_token_budget(100)
    try:
        with pytest.raises(LLMTokenBudgetExceeded):
            OpenAIChat._reserve_budget(
                [{"role": "user", "content": "x" * 1000}], {},
            )
    finally:
        OpenAIChat.reset_token_budget(token)


def test_multi_arm_anchor_initialization_keeps_global_coverage() -> None:
    content = "x" * 20_000
    navigator = MultiArmNavigator(
        config=LensConfig(k_arms=4, probe_window=500),
        doc_content=content,
        doc_len=len(content),
    )

    samples = navigator.initialize_with_anchors(
        len(content),
        [(5_000, 1_500, 0.9), (15_000, 3_000, 0.7)],
    )

    assert samples
    assert any(arm.center == 5_000 for arm in navigator.arms)
    assert any(arm.arm_id.startswith("global_") for arm in navigator.arms)
