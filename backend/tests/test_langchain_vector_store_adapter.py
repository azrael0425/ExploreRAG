from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from app.langchain_rag.adapters.vector_store import ExploreRAGVectorStoreAdapter


class _Embeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.5]


class _Collection:
    def __init__(self) -> None:
        self.deleted_ids = []

    def delete(self, *, ids: list[str]) -> None:
        self.deleted_ids.extend(ids)


class _Store:
    def __init__(self) -> None:
        self.added = None
        self.query_args = None
        self.collection = _Collection()

    def add_documents(self, **kwargs) -> None:
        self.added = kwargs

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["near", "far"],
            "metadatas": [{"source": "a"}, {"source": "b"}],
            "distances": [0.1, 0.8],
        }


def test_vector_store_adapter_keeps_pending_writes_and_ready_reads() -> None:
    store = _Store()
    adapter = ExploreRAGVectorStoreAdapter(store=store, embeddings=_Embeddings())
    ids = adapter.add_texts(
        ["first"],
        metadatas=[{"document_id": 3, "visibility": "ready"}],
        ids=["doc_3_chunk_0"],
        embeddings=[[0.2]],
    )

    assert ids == ["doc_3_chunk_0"]
    assert store.added["metadatas"][0]["visibility"] == "pending"
    results = adapter.similarity_search_with_score(
        "question", k=2, filter={"source": "a"}, document_ids=[3]
    )
    assert [document.page_content for document, _ in results] == ["near", "far"]
    assert [score for _, score in results] == [0.1, 0.8]
    assert results[0][0].metadata["chunk_id"] == "chunk-1"
    assert store.query_args["where"] == {
        "$and": [
            {"visibility": "ready"},
            {"source": "a"},
            {"document_id": {"$in": [3]}},
        ]
    }


def test_vector_store_adapter_rejects_pending_reads_and_deletes_only_ids() -> None:
    store = _Store()
    adapter = ExploreRAGVectorStoreAdapter(store=store, embeddings=_Embeddings())

    with pytest.raises(ValueError, match="non-ready"):
        adapter.similarity_search("question", filter={"visibility": "pending"})

    assert adapter.delete(ids=["one", "two"])
    assert store.collection.deleted_ids == ["one", "two"]
    assert adapter.delete(ids=None) is False
