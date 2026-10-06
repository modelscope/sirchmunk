import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sirchmunk.agentic.react_agent import ReActSearchAgent, _compact_history, _duplicate_tool_query
from sirchmunk.agentic.tools import ToolRegistry
from sirchmunk.llm.openai_chat import OpenAIChat, LLMTokenBudgetExceeded
from sirchmunk.scan.dir_scanner import extract_identifiers, identifier_matches_filename
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.search import AgenticSearch, _PathScope, _json_safe, _resolve_budget


@pytest.mark.parametrize("identifier", ["TCA9555", "74LVC1G08", "Artix-7", "AMS1117-3.3", "XY9021", "9AB321"])
def test_identifier_across_language_boundaries(identifier):
    assert identifier in extract_identifiers(f"请看{identifier}的手册; compare {identifier}")
    assert identifier_matches_filename(identifier, f"vendor_{identifier}_manual.pdf")


@pytest.mark.parametrize("identifier,filename", [
    ("X123", "X1234.pdf"), ("Artix-7", "Artix-70.pdf"),
    ("Artix-7", "Artix-8.pdf"), ("manual", "manual-data.pdf"),
    ("AMS1117-3.3", "AMS1117-5.0.pdf"), ("X123", "AX123.pdf"),
])
def test_identifier_negative(identifier, filename):
    assert not identifier_matches_filename(identifier, filename)


def test_alphabetic_package_suffix_is_only_hint():
    assert identifier_matches_filename("TCA9555", "tca9555pwr_ti.pdf")
    assert not identifier_matches_filename("TCA9555", "tca95550pwr_ti.pdf")
    assert extract_identifiers("普通中文说明 123456 documentation") == []


def test_cjk_tokenization(monkeypatch):
    tokens = AgenticSearch._tokenize_for_matching("TCA9555的INT引脚 Artix-7")
    assert {"tca9555", "int", "引脚"} <= tokens
    assert "55的" not in tokens
    monkeypatch.setenv("LENS_CJK_TOKEN_BOUNDARIES", "false")
    assert "tca9555的int引脚" in AgenticSearch._tokenize_for_matching("TCA9555的INT引脚")


@pytest.mark.asyncio
async def test_probe_respects_scope_and_summary_boundaries(tmp_path, monkeypatch):
    inside = tmp_path / "scope"
    inside.mkdir()
    paths = [inside / "TCA9555pwr.pdf", inside / "noise.pdf", tmp_path / "TCA9555.pdf"]
    for path in paths:
        path.touch()
    search = object.__new__(AgenticSearch)
    search._logger = SimpleNamespace(info=AsyncMock())
    artifacts = SimpleNamespace(catalog=[
        {"path": str(paths[0]), "name": paths[0].name},
        {"path": str(paths[1]), "name": paths[1].name, "summary": "TCA95550 only"},
        {"path": str(paths[2]), "name": paths[2].name},
    ])
    hits = await search._probe_part_numbers("TCA9555的INT", artifacts, scope=_PathScope([str(inside)]))
    assert [p for p, score in hits] == [str(paths[0])]
    monkeypatch.setenv("LENS_IDENTIFIER_ROUTING", "false")
    assert await search._probe_part_numbers("TCA9555", artifacts) == []


def test_budget_explicit_defaults_win(monkeypatch):
    monkeypatch.setenv("LENS_DEFAULT_MAX_LOOPS", "25")
    monkeypatch.setenv("LENS_DEFAULT_TOKEN_BUDGET", "999999")
    assert _resolve_budget(10, "LENS_DEFAULT_MAX_LOOPS", 10) == 10
    assert _resolve_budget(128000, "LENS_DEFAULT_TOKEN_BUDGET", 128000) == 128000
    assert _resolve_budget(None, "LENS_DEFAULT_MAX_LOOPS", 10) == 25
    assert inspect.signature(AgenticSearch.search).parameters["max_loops"].default is None
    for value in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            _resolve_budget(value, "LENS_DEFAULT_MAX_LOOPS", 10)
    monkeypatch.setenv("LENS_DEFAULT_MAX_LOOPS", "bad")
    with pytest.raises(ValueError):
        _resolve_budget(None, "LENS_DEFAULT_MAX_LOOPS", 10)
    assert _resolve_budget(10, "LENS_DEFAULT_MAX_LOOPS", 10) == 10


def test_compression_is_bounded_and_preserves_pairs():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "query"}]
    for i in range(100):
        messages.extend([{"role": "assistant", "content": f"call-{i}"},
                         {"role": "user", "content": f"**Tool result**: file-{i}: " + "x" * 8000}])
        messages = _compact_history(messages, 3)
        assert len(messages) <= 9
    assert messages[-2]["content"] == "call-99"
    assert messages[3]["content"] == "call-97"
    assert len(messages[2]["content"]) < 2600
    assert "navigation hints only" in messages[2]["content"]
    assert _compact_history(messages, 0) is messages


def test_duplicate_queries_include_scope_and_normalize_strings():
    recent = []
    a = {"keywords": "INT", "paths": ["a.pdf"]}
    assert _duplicate_tool_query("keyword_search", a, recent) is None
    assert _duplicate_tool_query("keyword_search", {**a, "paths": ["b.pdf"]}, recent) is None
    assert _duplicate_tool_query("keyword_search", {**a, "keywords": ["int"]}, recent)
    assert _duplicate_tool_query("keyword_search", {**a, "include": ["*.md"]}, recent) is None
    assert len(recent) <= 3


@pytest.mark.parametrize("timeout", ["nan", "inf", "-1", "0", "bad"])
def test_timeout_is_validated(monkeypatch, timeout):
    monkeypatch.setenv("LLM_TIMEOUT", timeout)
    with pytest.raises(ValueError):
        OpenAIChat(api_key="test", base_url="http://localhost", model="test")


def test_timeout_has_no_nested_sdk_retries(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT", "7.5")
    client = OpenAIChat(api_key="test", base_url="http://localhost", model="test")
    assert client._client.timeout == 7.5
    assert client._client.max_retries == client._async_client.max_retries == 0
    client._client.close()


@pytest.mark.asyncio
async def test_budget_exit_is_partial_without_second_llm_call():
    agent = ReActSearchAgent(llm=None, tool_registry=ToolRegistry(), max_loops=3, max_token_budget=500)
    agent._logger = SimpleNamespace(**{k: AsyncMock() for k in ["info", "warning", "success"]})
    agent._call_llm = AsyncMock(side_effect=LLMTokenBudgetExceeded("test limit"))
    answer, context = await agent.run("query")
    assert agent._call_llm.await_count == 1
    assert context.telemetry["answer_status"] == "partial"
    assert context.telemetry["hard_budget_exhausted"] is True
    assert answer.startswith("Incomplete result:")
    search = object.__new__(AgenticSearch)
    outer = SearchContext()
    search._merge_loop_telemetry(outer, context)
    assert not search._fast_track_requires_fallback("No results found.", outer)


def test_cache_only_serializes_known_types():
    assert json.loads(json.dumps({"bytes": b"hello", "set": {"b", "a"}}, default=_json_safe)) == {
        "bytes": "hello", "set": ["a", "b"]}
    with pytest.raises(TypeError):
        json.dumps({"unknown": object()}, default=_json_safe)
    with pytest.raises(TypeError):
        json.dumps({"unknown_set": {object()}}, default=_json_safe)
