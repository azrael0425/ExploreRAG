"""LangChain VectorStore adapter for the existing workspace-isolated Chroma API."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as LangChainVectorStore

from app.langchain_rag.converters import vector_results_to_documents
from app.langchain_rag.adapters.embeddings import ExploreRAGEmbeddings
from app.services.vector_store import VectorStore


class ExploreRAGVectorStoreAdapter(LangChainVectorStore):
    """Preserve ExploreRAG's Chroma collections, metadata and visibility rule."""

    def __init__(self, store: VectorStore, embeddings: Embeddings) -> None:
        self.store = store
        self._embeddings = embeddings

    @property
    def embeddings(self) -> Embeddings:
        """The standard LangChain embedding accessor is read-only."""
        return self._embeddings

    @staticmethod
    def _ready_filter(
        metadata_filter: dict[str, Any] | None = None,
        document_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Build a Chroma filter that callers cannot use to expose pending data."""
        filters: list[dict[str, Any]] = [{"visibility": "ready"}]
        if metadata_filter:
            requested = dict(metadata_filter)
            if requested.get("visibility") not in (None, "ready"):
                raise ValueError("LangChain retrieval cannot query non-ready chunks")
            requested.pop("visibility", None)
            if requested:
                filters.append(requested)
        if document_ids:
            filters.append({"document_id": {"$in": list(document_ids)}})
        return filters[0] if len(filters) == 1 else {"$and": filters}

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "ExploreRAGVectorStoreAdapter":
        store = kwargs.pop("store", None)
        if not isinstance(store, VectorStore):
            raise ValueError("ExploreRAGVectorStoreAdapter.from_texts requires store=VectorStore")
        adapter = cls(store=store, embeddings=embedding)
        adapter.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return adapter

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        values = list(texts)
        if not values:
            return []
        if ids is None:
            raise ValueError("ExploreRAGVectorStoreAdapter requires stable caller-provided chunk ids")
        if len(ids) != len(values):
            raise ValueError("ids and texts must have the same length")
        if metadatas is not None and len(metadatas) != len(values):
            raise ValueError("metadatas and texts must have the same length")

        supplied_embeddings = kwargs.pop("embeddings", None)
        if kwargs:
            raise TypeError(f"Unsupported add_texts options: {', '.join(sorted(kwargs))}")
        vectors = supplied_embeddings or self.embeddings.embed_documents(values)
        if len(vectors) != len(values):
            raise ValueError("embeddings and texts must have the same length")
        safe_metadata = [
            {**(metadatas[index] if metadatas else {}), "visibility": "pending"}
            for index in range(len(values))
        ]
        self.store.add_documents(ids=ids, embeddings=vectors, documents=values, metadatas=safe_metadata)
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        metadata_filter = kwargs.pop("filter", kwargs.pop("where", None))
        document_ids = kwargs.pop("document_ids", None)
        if kwargs:
            raise TypeError(f"Unsupported similarity_search options: {', '.join(sorted(kwargs))}")
        where = self._ready_filter(metadata_filter, document_ids)
        results = self.store.query(
            query_embedding=self.embeddings.embed_query(query),
            n_results=k,
            where=where,
        )
        return vector_results_to_documents(results)

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if kwargs:
            raise TypeError(f"Unsupported delete options: {', '.join(sorted(kwargs))}")
        if not ids:
            return False
        self.store.collection.delete(ids=list(ids))
        return True
