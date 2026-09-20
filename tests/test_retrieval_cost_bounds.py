"""Cost-bound guards for the retrieval hot path.

These verify the corpus-agnostic bounds added to keep large/among-many-files
corpora from stalling rga: a type-agnostic file-size cap and a capability-based
rga adapter whitelist (see AGENTS.md 9.5). The bounds must reach the built
argument list, and the rga-only flag must not leak into the native rg fallback.
"""
from __future__ import annotations

import pytest

from sirchmunk.retrieve.text_retriever import GrepRetriever


def test_rg_compatible_args_strips_rga_only_but_keeps_max_filesize() -> None:
    args = [
        "-i",
        "--max-filesize=64M",
        "--rga-adapters=poppler,pandoc,postprocpagebreaks",
        "--rga-cache-max-blob-len=10000000",
        "pattern",
        "/some/path",
    ]
    out = GrepRetriever._rg_compatible_args(args)
    # rga-only flags are dropped for the native rg fallback ...
    assert all(not a.startswith("--rga-") for a in out)
    # ... but --max-filesize is shared with ripgrep and must survive.
    assert "--max-filesize=64M" in out
    assert "pattern" in out and "/some/path" in out


@pytest.mark.asyncio
async def test_retrieve_single_injects_cost_bounds(monkeypatch) -> None:
    captured: dict = {}

    async def fake_run_rga_async(args, json_output=True, timeout=None, allow_rg_fallback=True):
        captured["args"] = list(args)
        return {"returncode": 1, "stdout": [], "stderr": "", "search_backend": "rga"}

    async def fake_run_search_process(command, args, **kwargs):
        return {"returncode": 1, "stdout": [], "stderr": "", "search_backend": command}

    monkeypatch.setattr(GrepRetriever, "_run_rga_async", staticmethod(fake_run_rga_async))
    monkeypatch.setattr(GrepRetriever, "_run_search_process", staticmethod(fake_run_search_process))

    await GrepRetriever._retrieve_single(pattern="foo", path="/tmp")

    # The rga (rich) pass carries the cost bounds.
    args = captured["args"]
    # Default cap is 64 MB and the bounded-adapter whitelist is applied.
    assert "--max-filesize=64M" in args
    assert any(a.startswith("--rga-adapters=") for a in args)
    adapters = next(a for a in args if a.startswith("--rga-adapters="))
    # Only bounded document extractors; no unbounded recursive/streaming ones.
    for unbounded in ("decompress", "zip", "tar", "sqlite", "ffmpeg"):
        assert unbounded not in adapters


@pytest.mark.asyncio
async def test_tiered_scan_splits_text_and_rich_passes(monkeypatch) -> None:
    """Tiered scan issues a rg text pass (all files) + an rga rich pass (docs).

    The text pass must carry no rga-only flags and no rich globs; the rich pass
    must restrict to the binary document extensions and keep the rga adapter
    whitelist. Union of both passes preserves recall.
    """
    from sirchmunk.utils import constants

    rg_calls: dict = {}
    rga_calls: dict = {}

    async def fake_run_search_process(command, args, **kwargs):
        rg_calls["command"] = command
        rg_calls["args"] = list(args)
        return {"returncode": 1, "stdout": [], "stderr": "", "search_backend": command}

    async def fake_run_rga_async(args, json_output=True, timeout=None, allow_rg_fallback=True):
        rga_calls["args"] = list(args)
        return {"returncode": 1, "stdout": [], "stderr": "", "search_backend": "rga"}

    monkeypatch.setattr(GrepRetriever, "_run_search_process", staticmethod(fake_run_search_process))
    monkeypatch.setattr(GrepRetriever, "_run_rga_async", staticmethod(fake_run_rga_async))

    await GrepRetriever._retrieve_single(pattern="foo", path="/tmp")

    # Text pass = native rg, no rga-only flags, no rich include globs.
    assert rg_calls["command"] == "rg"
    assert all(not a.startswith("--rga-") for a in rg_calls["args"])
    assert "--max-filesize=64M" in rg_calls["args"]
    text_globs = {
        rg_calls["args"][i + 1]
        for i, a in enumerate(rg_calls["args"][:-1]) if a == "-g"
    }
    assert not any(g.startswith("*.") for g in text_globs)

    # Rich pass = rga limited to binary document formats, adapters preserved.
    rich_globs = {
        rga_calls["args"][i + 1]
        for i, a in enumerate(rga_calls["args"][:-1]) if a == "-g"
    }
    for ext in constants.GREP_RICH_EXTENSIONS:
        assert f"*.{ext}" in rich_globs
    assert any(a.startswith("--rga-adapters=") for a in rga_calls["args"])
