from __future__ import annotations

import asyncio

from app.core.config import settings
from app.schemas.rag import ChatSourceChunk
from app.schemas.workspace import WorkspaceCreate, WorkspaceUpdate
from app.services.chat_context import append_knowledge_graph_source
from app.services.deep_retriever import DeepRetriever
from app.services.models.parsed_document import (
    Citation,
    EnrichedChunk,
    KnowledgeGraphEvidence,
    KnowledgeGraphFact,
)
from app.services.retrieval_policy import resolve_retrieval_policy


class _ImmediateScheduler:
    async def run(self, _resource, _priority, factory, **_kwargs):
        return await factory()


class _GraphService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def retrieve_evidence(self, question: str, mode: str) -> KnowledgeGraphEvidence:
        self.calls.append((question, mode))
        return KnowledgeGraphEvidence(
            content="Entities:\n- Alpha\n\nRelationships:\n- Alpha -> Beta: depends on",
            entity_names=["Alpha", "Beta"],
            entity_count=2,
            relationship_count=1,
        )


def test_workspace_lightrag_preference_defaults_to_disabled() -> None:
    assert WorkspaceCreate(name="Example").lightrag_augmentation_enabled is False
    assert WorkspaceUpdate(lightrag_augmentation_enabled=True).lightrag_augmentation_enabled is True


def test_retrieval_policy_prevents_workspace_wide_graph_leaks(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EXPLORERAG_ENABLE_KG", True)

    disabled = resolve_retrieval_policy(
        "hybrid", workspace_lightrag_enabled=False, scoped=False
    )
    scoped = resolve_retrieval_policy(
        "hybrid", workspace_lightrag_enabled=True, scoped=True
    )
    enabled = resolve_retrieval_policy(
        "hybrid", workspace_lightrag_enabled=True, scoped=False
    )

    assert (disabled.mode, disabled.reason) == ("vector_only", "disabled_by_workspace")
    assert (scoped.mode, scoped.reason) == ("vector_only", "scoped_query")
    assert (enabled.mode, enabled.lightrag_enabled) == ("hybrid", True)


def test_graph_evidence_is_added_as_an_explicit_kg_source() -> None:
    seen: set[str] = set()
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []

    source = append_knowledge_graph_source(
        knowledge_graph_summary="Entities:\n- Alpha",
        sources=sources,
        context_parts=context_parts,
        existing_ids=seen,
        source_label=lambda prefix, _existing: f"{prefix}-a3x9",
    )

    assert source is not None
    assert source.index == "KG-a3x9"
    assert source.source_type == "kg"
    assert source.document_id == 0
    assert sources == [source]
    assert "[KG-a3x9]" in context_parts[0]


def test_vector_chunk_entities_use_exact_retrieved_graph_labels() -> None:
    matches = DeepRetriever._entities_in_text(
        "RAG系统应使用访问控制元数据，但这里没有另一个实体。",
        ["RAG系统", "访问控制元数据", "模型投毒"],
    )

    assert matches == ["RAG系统", "访问控制元数据"]


def test_graph_relationship_facts_are_exposed_as_separate_citations() -> None:
    seen: set[str] = set()
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    counter = iter(["a3x9", "b2m7"])

    append_knowledge_graph_source(
        knowledge_graph_summary="aggregate evidence",
        sources=sources,
        context_parts=context_parts,
        existing_ids=seen,
        source_label=lambda prefix, _existing: f"{prefix}-{next(counter)}",
        graph_facts=[
            KnowledgeGraphFact(
                content="Alpha -> Beta: depends on",
                entity_names=["Alpha", "Beta"],
            ),
            KnowledgeGraphFact(
                content="Gamma -> Delta: protects",
                entity_names=["Gamma", "Delta"],
            ),
        ],
    )

    assert [source.index for source in sources] == ["KG-a3x9", "KG-b2m7"]
    assert sources[0].graph_entity_names == ["Alpha", "Beta"]
    assert "[KG-a3x9]" in context_parts[0]
    assert "[KG-b2m7]" in context_parts[1]


def test_deep_retriever_skips_graph_work_for_vector_only_mode(monkeypatch) -> None:
    async def scenario() -> None:
        graph = _GraphService()
        retriever = DeepRetriever(
            workspace_id=1,
            kg_service=graph,
            vector_store=object(),
            embedder=object(),
            reranker=None,
        )
        retriever.scheduler = _ImmediateScheduler()
        chunk = EnrichedChunk(
            content="vector passage",
            chunk_index=0,
            source_file="manual.pdf",
            document_id=1,
        )
        retriever._vector_query = lambda *_args, **_kwargs: (
            [chunk], [Citation(source_file="manual.pdf", document_id=1)]
        )
        monkeypatch.setattr(settings, "EXPLORERAG_ENABLE_RERANKER", False)

        result = await retriever.query("where?", mode="vector_only", include_images=False)

        assert graph.calls == []
        assert result.knowledge_graph_summary == ""
        assert result.timings.graph_ms is None

    asyncio.run(scenario())


def test_deep_retriever_uses_structured_graph_evidence(monkeypatch) -> None:
    async def scenario() -> None:
        graph = _GraphService()
        retriever = DeepRetriever(
            workspace_id=1,
            kg_service=graph,
            vector_store=object(),
            embedder=object(),
            reranker=None,
        )
        retriever.scheduler = _ImmediateScheduler()
        chunk = EnrichedChunk(
            content="vector passage",
            chunk_index=0,
            source_file="manual.pdf",
            document_id=1,
        )
        retriever._vector_query = lambda *_args, **_kwargs: (
            [chunk], [Citation(source_file="manual.pdf", document_id=1)]
        )
        monkeypatch.setattr(settings, "EXPLORERAG_ENABLE_RERANKER", False)

        result = await retriever.query("how are Alpha and Beta related?", include_images=False)

        assert graph.calls == [("how are Alpha and Beta related?", "hybrid")]
        assert result.knowledge_graph_evidence.entity_count == 2
        assert "Alpha -> Beta" in result.knowledge_graph_summary
        assert "Knowledge Graph Insights" in result.context

    asyncio.run(scenario())
