from __future__ import annotations

import asyncio

from app.langchain_rag.chains import retrieval
from app.langchain_rag.chains.retrieval import KnowledgeBaseContext
from app.langchain_rag.contracts import ChatChainInput, RetrievalEnvelope
from app.schemas.rag import ChatImageRef, ChatSourceChunk
from app.services.models.parsed_document import Citation, EnrichedChunk, RetrievalTimings


class _Runnable:
    async def ainvoke(self, _input, config=None):
        from langchain_core.documents import Document

        return RetrievalEnvelope(
            documents=[Document(page_content="passage", metadata={
                "chunk_id": "doc_8_chunk_3",
                "document_id": 8,
                "page_no": 5,
                "heading_path": ["Section"],
                "source": "manual.pdf",
                "image_refs": ["image-7"],
            })],
            citations=[Citation(source_file="manual.pdf", document_id=8, page_no=5, heading_path=["Section"])],
            context="raw context",
            knowledge_graph_summary="Entities:\n- Manual\n\nRelationships:\n- Manual -> Section",
            timings=RetrievalTimings(vector_ms=7),
        )


def test_retrieval_chain_uses_langchain_runnable_and_preserves_source_contract(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            retrieval,
            "get_explore_rag_retrieval_runnable",
            lambda *args, **kwargs: _Runnable(),
        )

        async def fake_images(_db, _workspace_id, _documents, _existing_ids):
            return [], [], []

        monkeypatch.setattr(retrieval, "_resolve_images", fake_images)
        existing_ids: set[str] = set()
        input = ChatChainInput(workspace_id=4, message="summarize the manual")
        result = await retrieval.build_retrieval_chain(object(), existing_ids).ainvoke(input)

        assert result.context.startswith("Source [KB-")
        assert result.sources[0].chunk_id == "doc_8_chunk_3"
        assert result.sources[0].source_file == "manual.pdf"
        assert result.sources[0].heading_path == ["Section"]
        assert result.timings.vector_ms == 7
        assert result.sources[0].index in existing_ids

    asyncio.run(scenario())


def test_retrieval_chain_injects_kg_evidence_only_when_enabled(monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            retrieval,
            "get_explore_rag_retrieval_runnable",
            lambda *args, **kwargs: _Runnable(),
        )
        async def fake_images(_db, _workspace_id, _documents, _existing_ids):
            return [], [], []

        monkeypatch.setattr(retrieval, "_resolve_images", fake_images)
        existing_ids: set[str] = set()
        result = await retrieval.build_retrieval_chain(object(), existing_ids).ainvoke(
            ChatChainInput(
                workspace_id=4,
                message="how is the manual organized?",
                lightrag_augmentation_enabled=True,
            )
        )

        kg_sources = [source for source in result.sources if source.source_type == "kg"]
        assert len(kg_sources) == 1
        assert str(kg_sources[0].index).startswith("KG-")
        assert "Knowledge Graph Evidence" in result.context

    asyncio.run(scenario())


def test_retrieval_routing_keeps_greetings_out_of_the_knowledge_base() -> None:
    assert not retrieval.requires_document_search("Hello!")
    assert not retrieval.requires_document_search("你好")
    assert retrieval.requires_document_search("Please summarize the document")
