"""Execute the same production answer path without persisting chat history."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
from app.langchain_rag.contracts import ChatChainInput
from app.langchain_rag.service import complete_chat
from app.models.knowledge_base import KnowledgeBase
from app.services.retrieval_policy import resolve_retrieval_policy
from app.services.explore_rag_factory import get_explore_rag_service


async def run_production_target(
    db: AsyncSession,
    *,
    workspace_id: int,
    question: str,
    top_k: int = 4,
    retrieval_mode: str = "hybrid",
    history: list[dict[str, Any]] | None = None,
    enable_reranker: bool | None = None,
    enable_knowledge_graph: bool | None = None,
    prefetch_k: int | None = None,
) -> dict[str, Any]:
    """Return an answer snapshot from the production LangChain orchestrator.

    This purposefully calls the service layer instead of an HTTP endpoint: it
    uses the live retrieval/generation implementation while avoiding a second
    user/assistant message in ``chat_messages`` for every evaluation case.
    """
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    workspace = result.scalar_one_or_none()
    if workspace is None:
        raise ValueError(f"Knowledge base {workspace_id} does not exist")

    system_prompt = (workspace.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT
    graph_enabled = (
        workspace.lightrag_augmentation_enabled
        if enable_knowledge_graph is None
        else enable_knowledge_graph
    )
    policy = resolve_retrieval_policy(
        retrieval_mode,  # type: ignore[arg-type]
        workspace_lightrag_enabled=graph_enabled,
        scoped=False,
    )
    return await complete_chat(
        ChatChainInput(
            workspace_id=workspace_id,
            message=question,
            history=history or [],
            enable_thinking=False,
            llm_mode=workspace.llm_mode,
            lightrag_augmentation_enabled=graph_enabled,
            system_prompt=system_prompt,
            top_k=top_k,
            retrieval_mode=policy.mode,
            enable_reranker=enable_reranker,
            enable_knowledge_graph=graph_enabled,
            prefetch_k=prefetch_k,
        ),
        db,
        selected_attachments=[],
    )


async def run_retrieval_target(
    db: AsyncSession,
    *,
    workspace_id: int,
    question: str,
    top_k: int = 4,
    retrieval_mode: str = "hybrid",
    enable_reranker: bool | None = None,
    enable_knowledge_graph: bool | None = None,
    prefetch_k: int | None = None,
) -> dict[str, Any]:
    """Run the production retriever without generation for cheap ablations."""
    workspace = (await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )).scalar_one_or_none()
    if workspace is None:
        raise ValueError(f"Knowledge base {workspace_id} does not exist")
    service = get_explore_rag_service(db, workspace_id, llm_mode=workspace.llm_mode)
    graph_enabled = (
        workspace.lightrag_augmentation_enabled
        if enable_knowledge_graph is None
        else enable_knowledge_graph
    )
    policy = resolve_retrieval_policy(
        retrieval_mode,  # type: ignore[arg-type]
        workspace_lightrag_enabled=graph_enabled,
        scoped=False,
    )
    result = await service.query_deep(
        question=question,
        top_k=top_k,
        mode=policy.mode,
        include_images=False,
        enable_reranker=enable_reranker,
        enable_knowledge_graph=graph_enabled,
        prefetch_k=prefetch_k,
    )
    sources = []
    for rank, chunk in enumerate(result.chunks, start=1):
        citation = result.citations[rank - 1] if rank - 1 < len(result.citations) else None
        sources.append({
            "index": f"KB-e{rank:03d}",
            "chunk_id": f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
            "content": chunk.content,
            "document_id": chunk.document_id,
            "source_file": citation.source_file if citation else chunk.source_file,
            "page_no": chunk.page_no,
            "heading_path": list(chunk.heading_path),
            "score": chunk.rerank_score if chunk.rerank_score is not None else chunk.vector_score,
            "source_type": "vector",
        })
    performance = asdict(result.timings)
    performance["retrieval_trace"] = result.trace
    return {
        "answer": "",
        "sources": sources,
        "performance": performance,
        "retrieval_trace": result.trace,
        "knowledge_graph_evidence": asdict(result.knowledge_graph_evidence),
    }
