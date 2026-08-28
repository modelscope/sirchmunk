from pathlib import Path
from types import MethodType

import pytest

from sirchmunk.agentic.file_system_tools import FileListTool
from sirchmunk.agentic.scope_planner import ScopePlanner
from sirchmunk.agentic.tools import FileReadTool
from sirchmunk.scan.dir_scanner import DirectoryScanner
from sirchmunk.schema.search_context import SearchContext
from sirchmunk.search import AgenticSearch, CompileArtifacts
from sirchmunk.retrieve.text_retriever import GrepRetriever


@pytest.mark.asyncio
async def test_tool_registry_cache_respects_dir_scan(tmp_path: Path):
    (tmp_path / "doc.txt").write_text("alpha", encoding="utf-8")
    searcher = AgenticSearch.__new__(AgenticSearch)
    searcher.grep_retriever = GrepRetriever(work_path=tmp_path)
    searcher.knowledge_storage = object()
    searcher.llm = None
    searcher._dir_scanner = None
    searcher._tool_registry = None
    searcher._tool_registry_key = None

    def _empty_artifacts(self, paths):
        return CompileArtifacts(
            catalog=None, catalog_map={}, tree_indexer=None, tree_available_paths=set()
        )

    searcher._detect_compile_artifacts = MethodType(_empty_artifacts, searcher)

    registry_without = searcher._ensure_tool_registry(
        [str(tmp_path)], enable_dir_scan=False
    )
    assert "file_list" in registry_without.tool_names
    assert "dir_scan" not in registry_without.tool_names

    registry_with = searcher._ensure_tool_registry(
        [str(tmp_path)], enable_dir_scan=True
    )
    assert registry_with is not registry_without
    assert "file_list" in registry_with.tool_names
    assert "dir_scan" in registry_with.tool_names


@pytest.mark.asyncio
async def test_directory_scanner_handles_extensionless_and_samples_after_walk(tmp_path: Path):
    first = tmp_path / "a_first"
    later_dir = tmp_path / "z_later"
    later_dir.mkdir()
    first.write_text('{"title": "first"}\n', encoding="utf-8")
    for i in range(8):
        (later_dir / f"wiki_{i:02d}").write_text(
            f'{{"title": "later {i}"}}\n', encoding="utf-8"
        )

    scanner = DirectoryScanner(llm=None, max_files=3, max_discovered_files=100)
    result = await scanner.scan(tmp_path)

    assert result.total_files == 9
    assert len(result.candidates) == 3
    assert any(candidate.extension == "" for candidate in result.candidates)


@pytest.mark.asyncio
async def test_file_list_tool_profile_tree_and_paged_files(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.txt").write_text("alpha", encoding="utf-8")
    (docs / "beta.md").write_text("beta", encoding="utf-8")

    tool = FileListTool([tmp_path])
    ctx = SearchContext()
    _, profile = await tool.execute(ctx, view="profile")
    assert profile["file_count"] == 2
    assert profile["extension_counts"][".txt"] == 1

    _, tree = await tool.execute(ctx, view="tree")
    assert any("docs/" in line for line in tree["tree"])

    _, files = await tool.execute(ctx, view="files", limit=1)
    assert files["has_more"] is True
    assert len(files["files"]) == 1


@pytest.mark.asyncio
async def test_file_read_modes_are_bounded_and_query_aware(tmp_path: Path):
    file_path = tmp_path / "notes.md"
    file_path.write_text(
        "# Intro\n"
        "opening\n"
        "# Target\n"
        "important moon sample evidence\n"
        "detail line\n"
        "# Other\n"
        "tail\n",
        encoding="utf-8",
    )
    tool = FileReadTool(allowed_roots=[tmp_path], max_chars_per_file=2000)
    ctx = SearchContext()

    text, meta = await tool.execute(ctx, file_paths=[str(file_path)], mode="section", section="Target")
    assert "important moon sample evidence" in text
    assert meta["read_modes"][str(file_path)] == "section"

    text, meta = await tool.execute(
        SearchContext(), file_paths=[str(file_path)], mode="window", query="moon", context_lines=3
    )
    assert "important moon sample evidence" in text
    assert meta["read_modes"][str(file_path)] == "window"

    text, meta = await tool.execute(
        SearchContext(), file_paths=[str(file_path)], mode="range", start_line=3, end_line=4
    )
    assert "# Target" in text
    assert "detail line" not in text


@pytest.mark.asyncio
async def test_agentic_search_scope_planner_applies_conservative_narrowing(tmp_path: Path):
    pdf_dir = tmp_path / "pdf_files"
    text_dir = tmp_path / "text_files"
    pdf_dir.mkdir()
    text_dir.mkdir()
    (pdf_dir / "moon_report.pdf").write_text("fake pdf text", encoding="utf-8")
    (text_dir / "other.txt").write_text("misc", encoding="utf-8")

    searcher = AgenticSearch.__new__(AgenticSearch)

    class _NoopLogger:
        async def info(self, *args, **kwargs):
            return None

        async def warning(self, *args, **kwargs):
            return None

    searcher._logger = _NoopLogger()
    narrowed, plan = await searcher._plan_search_scope(
        "summarize pdf technical report", [str(tmp_path)], max_depth=2
    )
    assert plan and plan["narrowed"] is True
    assert any(Path(path).name == "pdf_files" for path in narrowed)

    preserved, plan = await searcher._plan_search_scope(
        "Who founded Methodism along with George Whitefield?", [str(tmp_path)], max_depth=2
    )
    assert preserved == [str(tmp_path)]


@pytest.mark.asyncio
async def test_scope_planner_narrows_high_confidence_directory(tmp_path: Path):
    pdf_dir = tmp_path / "pdf_files"
    text_dir = tmp_path / "text_files"
    pdf_dir.mkdir()
    text_dir.mkdir()
    (pdf_dir / "moon_report.pdf").write_text("fake pdf text", encoding="utf-8")
    (text_dir / "other.txt").write_text("misc", encoding="utf-8")

    planner = ScopePlanner([tmp_path], confidence_threshold=0.5)
    plan = planner.plan("summarize the pdf moon report", max_depth=2)

    assert plan.narrowed is True
    assert any(Path(path).name == "pdf_files" for path in plan.effective_paths)
    assert "file_read" in plan.recommended_tools
