import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_upstream_sse_disconnect_cancels_search(monkeypatch, tmp_path):
    # API search module is lazy: this test neither initializes a real searcher
    # nor contacts the configured provider or modifies the user's library.
    monkeypatch.setenv("SIRCHMUNK_WORK_PATH", str(tmp_path))
    from sirchmunk.api import search as api

    started = asyncio.Event()
    canceled = asyncio.Event()

    async def search(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            canceled.set()
            raise

    monkeypatch.setattr(api, "_search_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "_create_search_instance", lambda **kw: SimpleNamespace(search=search))
    http_request = SimpleNamespace(is_disconnected=AsyncMock(side_effect=[False, True]))
    response = await api.execute_search_stream(api.SearchRequest(query="test"), http_request)
    async with asyncio.timeout(3):
        frames = [frame async for frame in response.body_iterator]
    assert started.is_set()
    assert canceled.is_set()
    assert frames == [": heartbeat\n\n"]
    assert api._search_semaphore._value == 1
