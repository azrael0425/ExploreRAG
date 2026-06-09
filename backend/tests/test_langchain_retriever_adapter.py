from __future__ import annotations

import asyncio

from app.langchain_rag.adapters.retriever import (
    ExploreRAGHybridRetriever,
    ExploreRAGRetrievalRunnable,
)
from app.langchain_rag.contracts import RetrievalInput
from app.services.models.parsed_document import Citation, DeepRetrievalResult, EnrichedChunk, RetrievalTimings


class _DeepRetriever:
    def __init__(self) -> None:
        self.calls = []

    async def query(self, **kwargs):
        self.calls.append(kwargs)
        chunk = EnrichedChunk(
            content="retrieved text",
            chunk_index=2,
            source_file="report.pdf",
            document_id=9,
            page_no=4,
            heading_path=["Chapter"],
            image_refs=["image-1"],
            table_refs=["table-1"],
        )
        return DeepRetrievalResult(
            chunks=[chunk],
            citations=[Citation(source_file="report.pdf", document_id=9, page_no=4, heading_path=["Chapter"])],
            context="context",
            query=kwargs["question"],
            knowledge_graph_summary="graph",
            timings=RetrievalTimings(vector_ms=12),
        )


def test_retriever_adapter_preserves_documents_and_side_channel_data() -> None:
    async def scenario() -> None:
        deep = _DeepRetriever()
        retriever = ExploreRAGHybridRetriever(deep_retriever=deep, workspace_id=3, top_k=5)
        request = RetrievalInput(workspace_id=3, question="where?", top_k=2, mode="vector_only")
        envelope = await ExploreRAGRetrievalRunnable(retriever).ainvoke(request)

        assert envelope.context == "context"
        assert envelope.knowledge_graph_summary == "graph"
        assert envelope.documents[0].metadata["chunk_id"] == "doc_9_chunk_2"
        assert envelope.documents[0].metadata["citation"] == "report.pdf | p.4 | Chapter"
        assert envelope.timings.vector_ms == 12
        assert deep.calls[0]["mode"] == "vector_only"
        assert deep.calls[0]["top_k"] == 2

        documents = await retriever.ainvoke("where?")
        assert documents[0].page_content == "retrieved text"

    asyncio.run(scenario())
