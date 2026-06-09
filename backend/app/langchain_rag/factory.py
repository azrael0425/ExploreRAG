"""Request-scoped factories for LangChain adapters and LCEL chains."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.langchain_rag.adapters.retriever import (
    ExploreRAGHybridRetriever,
    ExploreRAGRetrievalRunnable,
)
from app.services.explore_rag_factory import get_explore_rag_service


def get_explore_rag_retrieval_runnable(
    db: AsyncSession,
    workspace_id: int,
    *,
    llm_mode: str,
) -> ExploreRAGRetrievalRunnable:
    """Build a request-scoped adapter for the current workspace's deep retriever.

    ``AsyncSession`` must never be retained in a process-global chain cache.
    The underlying model/vector/KG factories remain responsible for their own
    safe resource caching.
    """
    service = get_explore_rag_service(db, workspace_id, llm_mode=llm_mode)
    retriever = ExploreRAGHybridRetriever(
        deep_retriever=service.retriever,
        workspace_id=workspace_id,
    )
    return ExploreRAGRetrievalRunnable(retriever)
