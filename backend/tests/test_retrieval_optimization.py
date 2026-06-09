from __future__ import annotations

import asyncio
import time

from app.core.config import settings
from app.services.deep_retriever import DeepRetriever
from app.services.knowledge_graph_service import (
    clear_knowledge_graph_service_cache_for_tests,
    evict_knowledge_graph_service,
    get_knowledge_graph_service,
)
from app.services.models.parsed_document import Citation, EnrichedChunk
from app.services.reranker import RerankerService
from app.services.sentence_transformer_registry import (
    clear_sentence_transformer_registry_for_tests,
    get_shared_sentence_transformer,
)


class _ImmediateScheduler:
    async def run(self, _resource, _priority, factory, **_kwargs):
        return await factory()


class _CircuitStub:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.successes = 0

    def is_available(self) -> bool:
        return True

    def record_failure(self, reason: str) -> None:
        self.failures.append(reason)

    def record_success(self) -> None:
        self.successes += 1


def test_sentence_transformer_registry_reuses_one_model(monkeypatch) -> None:
    import sentence_transformers

    constructed: list[tuple[str, dict]] = []

    class _FakeModel:
        def __init__(self, model_name: str, **options):
            constructed.append((model_name, options))

        @staticmethod
        def get_sentence_embedding_dimension() -> int:
            return 1024

    clear_sentence_transformer_registry_for_tests()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeModel)

    first = get_shared_sentence_transformer("fake-bge-m3", "cpu")
    second = get_shared_sentence_transformer("fake-bge-m3", "cpu")

    assert first is second
    assert len(constructed) == 1
    clear_sentence_transformer_registry_for_tests()


def test_knowledge_graph_service_is_cached_per_workspace() -> None:
    clear_knowledge_graph_service_cache_for_tests()
    first = get_knowledge_graph_service(101)
    second = get_knowledge_graph_service(101)
    other = get_knowledge_graph_service(102)

    assert first is second
    assert first is not other

    asyncio.run(evict_knowledge_graph_service(101))
    replacement = get_knowledge_graph_service(101)
    assert replacement is not first
    clear_knowledge_graph_service_cache_for_tests()


def test_reranker_circuit_breaker_recovers(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_CIRCUIT_BREAKER_FAILURES", 1)
    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_CIRCUIT_BREAKER_RECOVERY_SECONDS", 0.01)
    service = RerankerService("unused")

    assert service.is_available()
    service.record_failure("timeout")
    assert not service.is_available()
    time.sleep(0.02)
    assert service.is_available()
    assert not service.is_available()  # only one half-open probe is admitted
    service.record_success()
    assert service.is_available()


def test_reranker_timeout_falls_back_to_vector_order(monkeypatch) -> None:
    async def scenario() -> None:
        circuit = _CircuitStub()
        retriever = DeepRetriever(
            workspace_id=1,
            kg_service=None,
            vector_store=object(),
            embedder=object(),
            reranker=circuit,
        )
        retriever.scheduler = _ImmediateScheduler()
        chunks = [
            EnrichedChunk(
                content=f"chunk {index}",
                chunk_index=index,
                source_file="test.md",
                document_id=1,
            )
            for index in range(3)
        ]
        citations = [Citation(source_file="test.md", document_id=1) for _ in chunks]

        def slow_rerank(*_args, **_kwargs):
            time.sleep(0.03)
            return list(reversed(chunks)), list(reversed(citations))

        retriever._rerank_chunks = slow_rerank
        monkeypatch.setattr(settings, "EXPLORERAG_ENABLE_RERANKER", True)
        monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_TIMEOUT_SECONDS", 0.005)

        result_chunks, result_citations = await retriever._rerank_with_budget(
            "question", chunks, citations, top_k=2
        )

        assert result_chunks == chunks[:2]
        assert result_citations == citations[:2]
        assert circuit.failures == ["inference timeout"]
        assert circuit.successes == 0

    asyncio.run(scenario())
