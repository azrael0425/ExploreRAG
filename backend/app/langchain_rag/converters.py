"""Lossless conversion between existing ExploreRAG models and LangChain types."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.services.models.parsed_document import Citation, DeepRetrievalResult


def deep_result_to_documents(result: DeepRetrievalResult) -> list[Document]:
    """Convert deep retrieval chunks without losing citation/navigation fields."""
    documents: list[Document] = []
    for index, chunk in enumerate(result.chunks):
        citation: Citation | None = (
            result.citations[index] if index < len(result.citations) else None
        )
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "source": chunk.source_file,
                    "page_no": chunk.page_no,
                    "heading_path": list(chunk.heading_path),
                    "citation": citation.format() if citation else chunk.source_file,
                    "image_refs": list(chunk.image_refs),
                    "table_refs": list(chunk.table_refs),
                    "has_table": chunk.has_table,
                    "has_code": chunk.has_code,
                    "vector_distance": chunk.vector_distance,
                    "vector_score": chunk.vector_score,
                    "rerank_score": chunk.rerank_score,
                    "graph_entity_names": list(chunk.graph_entity_names),
                },
            )
        )
    return documents


def vector_results_to_documents(results: dict[str, Any]) -> list[tuple[Document, float]]:
    """Convert a flattened Chroma response while preserving distance direction."""
    output: list[tuple[Document, float]] = []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    ids = results.get("ids") or []
    distances = results.get("distances") or []
    for index, page_content in enumerate(documents):
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        if index < len(ids):
            metadata.setdefault("chunk_id", ids[index])
        distance = float(distances[index]) if index < len(distances) else 0.0
        metadata["distance"] = distance
        output.append((Document(page_content=page_content, metadata=metadata), distance))
    return output
