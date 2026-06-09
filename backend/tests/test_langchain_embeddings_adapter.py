from __future__ import annotations

import asyncio

import pytest

from app.langchain_rag.adapters.embeddings import ExploreRAGEmbeddings
from app.services.work_scheduler import WorkPriority


class _EmbeddingService:
    dimension = 1024

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[float(index)] * self.dimension for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0] * self.dimension


class _ImmediateScheduler:
    def __init__(self) -> None:
        self.calls = []

    async def run(self, resource, priority, factory, **kwargs):
        self.calls.append((resource, priority, kwargs))
        return await factory()


def test_embeddings_adapter_reuses_service_and_validates_empty_values() -> None:
    service = _EmbeddingService()
    adapter = ExploreRAGEmbeddings(service, scheduler=_ImmediateScheduler(), workspace_id=7)

    vectors = adapter.embed_documents(["first", "second"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 1024
    assert adapter.embed_query("question") == [1.0] * 1024
    assert service.document_calls == [["first", "second"]]
    assert service.query_calls == ["question"]

    with pytest.raises(ValueError, match="empty"):
        adapter.embed_documents(["valid", "  "])


def test_embeddings_adapter_schedules_async_calls() -> None:
    async def scenario() -> None:
        scheduler = _ImmediateScheduler()
        adapter = ExploreRAGEmbeddings(_EmbeddingService(), scheduler=scheduler, workspace_id=17)
        assert len(await adapter.aembed_documents(["one"])) == 1
        await adapter.aembed_query("query")
        assert [call[0] for call in scheduler.calls] == ["embedding", "embedding"]
        assert all(call[1] == WorkPriority.CHAT for call in scheduler.calls)
        assert all(call[2]["workspace_id"] == 17 for call in scheduler.calls)

    asyncio.run(scenario())
