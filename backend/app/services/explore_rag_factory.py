"""Request-scoped factory for the single supported ExploreRAG backend."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.explore_rag_service import ExploreRAGService


def get_explore_rag_service(
    db: AsyncSession,
    workspace_id: int,
    kg_language: str | None = None,
    kg_entity_types: list[str] | None = None,
    llm_mode: str = "cloud",
) -> ExploreRAGService:
    """Build the retrieval/indexing backend used by LangChain adapters."""

    return ExploreRAGService(
        db=db,
        workspace_id=workspace_id,
        kg_language=kg_language,
        kg_entity_types=kg_entity_types,
        llm_mode=llm_mode,
    )
