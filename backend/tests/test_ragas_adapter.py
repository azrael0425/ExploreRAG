"""Unit tests for the optional, dynamically imported Ragas adapter."""
from __future__ import annotations

import asyncio
import sys
import types

from app.evaluation import ragas_adapter


class _Result:
    value = 0.9


class _Faithfulness:
    def __init__(self, *, llm):
        self.llm = llm

    async def ascore(self, **kwargs):
        return _Result()


class _FactualCorrectness:
    created_with: dict[str, object] | None = None
    score_args: dict[str, object] | None = None

    def __init__(self, **kwargs):
        type(self).created_with = kwargs

    async def ascore(self, **kwargs):
        type(self).score_args = kwargs
        return _Result()


class _AnswerRelevancy:
    created_with: dict[str, object] | None = None

    def __init__(self, **kwargs):
        type(self).created_with = kwargs

    async def ascore(self, **kwargs):
        return _Result()


def test_ragas_uses_deterministic_reference_coverage_mode(monkeypatch) -> None:
    captured_factory: dict[str, object] = {}
    judge = object()

    def fake_factory(**kwargs):
        captured_factory.update(kwargs)
        return judge

    ragas_module = types.ModuleType("ragas")
    llms_module = types.ModuleType("ragas.llms")
    metrics_module = types.ModuleType("ragas.metrics")
    collections_module = types.ModuleType("ragas.metrics.collections")
    llms_module.llm_factory = fake_factory
    collections_module.Faithfulness = _Faithfulness
    collections_module.FactualCorrectness = _FactualCorrectness
    collections_module.AnswerRelevancy = _AnswerRelevancy
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.llms", llms_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", collections_module)
    monkeypatch.setattr(ragas_adapter, "_ensure_ragas_langchain_compatibility", lambda: None)
    monkeypatch.setattr(ragas_adapter, "AsyncOpenAI", lambda **kwargs: object())
    embedding = object()
    monkeypatch.setattr(ragas_adapter, "_local_ragas_embeddings", lambda: embedding)

    values, statuses = asyncio.run(ragas_adapter.evaluate_with_ragas(
        question="What is the value?",
        answer="The value is 42.",
        retrieved_contexts=["The value is 42."],
        reference_answer="42",
        llm_mode="local",
    ))

    assert values == {
        "faithfulness": 0.9,
        "factual_correctness": 0.9,
        "answer_relevancy": 0.9,
    }
    assert statuses == {
        "faithfulness": "ok",
        "factual_correctness": "ok",
        "answer_relevancy": "ok",
    }
    assert captured_factory["temperature"] == 0.0
    assert captured_factory["max_tokens"] == 8192
    assert _FactualCorrectness.created_with == {"llm": judge, "mode": "recall"}
    assert _FactualCorrectness.score_args == {
        "response": "The value is 42.",
        "reference": 'For the question "What is the value?", the correct answer is: 42',
    }
    assert _AnswerRelevancy.created_with == {
        "llm": judge,
        "embeddings": embedding,
        "strictness": 3,
    }


def test_independent_evaluation_judge_identity(monkeypatch) -> None:
    monkeypatch.setattr(ragas_adapter.settings, "EVALUATION_LLM_PROVIDER", "dashscope")
    monkeypatch.setattr(ragas_adapter.settings, "EVALUATION_LLM_MODEL", "qwen-plus")
    monkeypatch.setattr(ragas_adapter.settings, "EVALUATION_LLM_MAX_TOKENS", 8192)
    monkeypatch.setattr(ragas_adapter.settings, "EVALUATION_LLM_MAX_ATTEMPTS", 3)

    assert ragas_adapter.evaluation_judge_identity("cloud") == {
        "provider": "dashscope",
        "model": "qwen-plus",
        "max_tokens": 8192,
        "max_attempts": 3,
    }


def test_structured_judge_output_is_retried(monkeypatch) -> None:
    calls = 0

    async def no_wait(_seconds: float) -> None:
        return None

    async def flaky_score() -> _Result:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("malformed structured output")
        return _Result()

    monkeypatch.setattr(ragas_adapter.settings, "EVALUATION_LLM_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(ragas_adapter.asyncio, "sleep", no_wait)

    result = asyncio.run(ragas_adapter._score_with_retry(flaky_score))

    assert result.value == 0.9
    assert calls == 3
