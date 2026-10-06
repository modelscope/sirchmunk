import hashlib
from unittest.mock import AsyncMock

import duckdb
import pytest

from sirchmunk.utils.embedding_util import EmbeddingUtil, get_embedding_dim
from sirchmunk.storage.knowledge_storage import KnowledgeStorage, EmbeddingCompatibilityError


@pytest.mark.parametrize("value", ["0", "-1", "nan", "384.0", "65537", ""])
def test_invalid_dimension(monkeypatch, value):
    monkeypatch.setenv("EMBEDDING_DIM", value)
    with pytest.raises(ValueError):
        get_embedding_dim()


def test_default_and_custom_dimension(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DIM", raising=False)
    assert get_embedding_dim() == 384
    monkeypatch.setenv("EMBEDDING_DIM", "1024")
    assert get_embedding_dim() == 1024


@pytest.mark.asyncio
async def test_query_instruction_does_not_format_other_braces(monkeypatch):
    client = object.__new__(EmbeddingUtil)
    client.embed = AsyncMock(return_value=[[1.0]])
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", r"{literal}\nQuery: {query}")
    assert await client.embed_query("Q{braces}") == [1.0]
    client.embed.assert_awaited_once_with(["{literal}\nQuery: Q{braces}"])
    await client.embed(["document"])
    client.embed.assert_awaited_with(["document"])
    monkeypatch.delenv("EMBEDDING_QUERY_INSTRUCTION")
    await client.embed_query("plain")
    client.embed.assert_awaited_with(["plain"])


@pytest.mark.parametrize("dimension,model", [(3, "old-model"), (2, "different-model")])
def test_incompatible_parquet_is_not_replaced(tmp_path, monkeypatch, dimension, model):
    monkeypatch.setenv("EMBEDDING_DIM", str(dimension))
    monkeypatch.setenv("EMBEDDING_MODEL_ID", model)
    pq = tmp_path / ".cache/knowledge/knowledge_clusters.parquet"
    pq.parent.mkdir(parents=True)
    with duckdb.connect() as db:
        db.execute("COPY (SELECT [1.0,2.0]::FLOAT[] embedding_vector, "
                   "'old-model' embedding_model) TO ? (FORMAT PARQUET)", [str(pq)])
    before = hashlib.sha256(pq.read_bytes()).digest()
    with pytest.raises(EmbeddingCompatibilityError):
        KnowledgeStorage(str(tmp_path))
    assert hashlib.sha256(pq.read_bytes()).digest() == before


def test_dimension_is_stable_for_storage_lifetime(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    with KnowledgeStorage(str(tmp_path)) as store:
        monkeypatch.setenv("EMBEDDING_DIM", "16")
        assert store.embedding_dim == 8
        assert any(row[1] == "FLOAT[8]" for row in store.db.fetch_all(
            "DESCRIBE knowledge_clusters"))
    store.close()
    assert not (tmp_path / ".cache/knowledge/knowledge_clusters.parquet").exists()


def test_parquet_column_discovery_ignores_nested_list_metadata(tmp_path):
    with KnowledgeStorage(str(tmp_path)) as store:
        pq = tmp_path / "quoted'filename.parquet"
        store.db.execute("COPY (SELECT [1.0,2.0]::FLOAT[] embedding_vector) TO ? (FORMAT PARQUET)", [str(pq)])
        assert store._get_parquet_columns(str(pq)) == ["embedding_vector"]
