"""LangChain retrieval adapters that delegate to the existing DeepRetriever."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ConfigDict

from app.langchain_rag.contracts import RetrievalEnvelope, RetrievalInput
from app.langchain_rag.converters import deep_result_to_documents


class ExploreRAGHybridRetriever(BaseRetriever):
    """Standard retriever facade for third-party LangChain composition.

    ``BaseRetriever`` only returns documents.  Product-specific side data is
    therefore exposed by :class:`ExploreRAGRetrievalRunnable` for the real chat
    pipeline rather than being discarded here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ``DeepRetriever`` is intentionally duck-typed here.  It keeps this
    # adapter importable without eagerly constructing database-bound services
    # and makes the orchestration boundary straightforward to test.
    deep_retriever: Any
    workspace_id: int
    top_k: int = 4
    mode: str = "hybrid"
    include_images: bool = True
    document_ids: list[int] | None = None
    metadata_filter: dict[str, Any] | None = None

    async def aretrieve_envelope(self, request: RetrievalInput) -> RetrievalEnvelope:
        if request.workspace_id != self.workspace_id:
            raise ValueError("Retriever workspace does not match RetrievalInput")
        result = await self.deep_retriever.query(
            question=request.question,
            mode=request.mode,
            top_k=request.top_k,
            document_ids=request.document_ids,
            include_images=request.include_images,
            metadata_filter=request.metadata_filter,
            enable_reranker=request.enable_reranker,
            enable_knowledge_graph=request.enable_knowledge_graph,
            prefetch_k=request.prefetch_k,
        )
        return RetrievalEnvelope(
            documents=deep_result_to_documents(result),
            context=result.context,
            knowledge_graph_summary=result.knowledge_graph_summary,
            knowledge_graph_entity_names=list(result.knowledge_graph_evidence.entity_names),
            knowledge_graph_document_ids=list(result.knowledge_graph_evidence.source_document_ids),
            knowledge_graph_facts=[
                asdict(fact) for fact in result.knowledge_graph_evidence.facts
            ],
            citations=list(result.citations),
            image_refs=list(result.image_refs),
            table_refs=list(result.table_refs),
            timings=result.timings,
            raw_result=result,
        )

    def _request_for_query(self, query: str) -> RetrievalInput:
        return RetrievalInput(
            workspace_id=self.workspace_id,
            question=query,
            top_k=self.top_k,
            mode=self.mode,  # type: ignore[arg-type]
            document_ids=self.document_ids,
            include_images=self.include_images,
            metadata_filter=self.metadata_filter,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aretrieve_envelope(self._request_for_query(query))).documents
        raise RuntimeError(
            "ExploreRAGHybridRetriever.invoke() cannot run inside an active event loop; use ainvoke()"
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return (await self.aretrieve_envelope(self._request_for_query(query))).documents


class ExploreRAGRetrievalRunnable(Runnable[RetrievalInput, RetrievalEnvelope]):
    """Runnable preserving retrieval side data needed by ExploreRAG chat/SSE."""

    def __init__(self, retriever: ExploreRAGHybridRetriever) -> None:
        self.retriever = retriever

    def invoke(
        self,
        input: RetrievalInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RetrievalEnvelope:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input, config=config, **kwargs))
        raise RuntimeError("ExploreRAGRetrievalRunnable.invoke() requires no active event loop; use ainvoke()")

    async def ainvoke(
        self,
        input: RetrievalInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RetrievalEnvelope:
        if kwargs:
            raise TypeError(f"Unsupported retrieval options: {', '.join(sorted(kwargs))}")
        return await self.retriever.aretrieve_envelope(input)
