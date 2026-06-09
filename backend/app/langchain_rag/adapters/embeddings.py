"""LangChain embedding adapter that reuses ExploreRAG's shared BGE service."""
from __future__ import annotations

import asyncio
from typing import Sequence

from langchain_core.embeddings import Embeddings

from app.services.embedder import EmbeddingService
from app.services.work_scheduler import WorkPriority, WorkScheduler, get_work_scheduler


class ExploreRAGEmbeddings(Embeddings):
    """Expose ``EmbeddingService`` without bypassing its model registry.

    The adapter validates empty inputs because the existing batch service
    intentionally omits empty strings; LangChain requires one result per input.
    """

    def __init__(
        self,
        service: EmbeddingService,
        *,
        scheduler: WorkScheduler | None = None,
        workspace_id: int | None = None,
        priority: WorkPriority = WorkPriority.CHAT,
    ) -> None:
        self.service = service
        self.scheduler = scheduler or get_work_scheduler()
        self.workspace_id = workspace_id
        self.priority = priority

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        normalized = list(texts)
        if any(not text.strip() for text in normalized):
            raise ValueError("LangChain embedding inputs must not contain empty text")
        return normalized

    @property
    def dimension(self) -> int:
        """Return the underlying model's configured embedding dimension."""
        return self.service.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.service.embed_texts(self._validate_texts(texts))

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("LangChain embedding query must not be empty")
        return self.service.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        validated = self._validate_texts(texts)
        return await self.scheduler.run(
            "embedding",
            self.priority,
            lambda: asyncio.to_thread(self.service.embed_texts, validated),
            workspace_id=self.workspace_id,
        )

    async def aembed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("LangChain embedding query must not be empty")
        return await self.scheduler.run(
            "embedding",
            self.priority,
            lambda: asyncio.to_thread(self.service.embed_query, text),
            workspace_id=self.workspace_id,
        )
