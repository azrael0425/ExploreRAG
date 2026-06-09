from __future__ import annotations

import asyncio
import time

from app.services.deep_retriever import DeepRetriever
from app.services.models.parsed_document import Citation, EnrichedChunk


class _ImmediateScheduler:
    async def run(self, _resource, _priority, factory, **_kwargs):
        return await factory()


class _AvailableReranker:
    def is_available(self) -> bool:
        return True

    def record_success(self) -> None:
        pass

    def record_failure(self, _reason: str) -> None:
        pass


def test_deep_retriever_reports_server_side_stage_timings() -> None:
    async def scenario() -> None:
        retriever = DeepRetriever(
            workspace_id=1,
            kg_service=object(),
            vector_store=object(),
            embedder=object(),
            reranker=_AvailableReranker(),
        )
        retriever.scheduler = _ImmediateScheduler()

        chunk = EnrichedChunk(
            content="GPU retrieval timing test.",
            chunk_index=0,
            source_file="test.md",
            document_id=1,
        )
        citation = Citation(source_file="test.md", document_id=1)

        async def fake_kg_query(_question: str, _mode: str) -> str:
            await asyncio.sleep(0.01)
            return "graph context"

        def fake_vector_query(*_args, **_kwargs):
            time.sleep(0.01)
            return [chunk], [citation]

        def fake_rerank(*_args, **_kwargs):
            time.sleep(0.01)
            return [chunk], [citation]

        retriever._kg_query = fake_kg_query
        retriever._vector_query = fake_vector_query
        retriever._rerank_chunks = fake_rerank

        result = await retriever.query(
            "What was measured?",
            include_images=False,
            enable_reranker=True,
        )

        assert result.timings.vector_ms > 0
        assert result.timings.graph_ms is not None and result.timings.graph_ms > 0
        assert result.timings.rerank_ms > 0
        assert result.timings.context_ms >= 0
        assert result.timings.total_ms >= max(
            result.timings.vector_ms,
            result.timings.graph_ms,
            result.timings.rerank_ms,
        )

    asyncio.run(scenario())
