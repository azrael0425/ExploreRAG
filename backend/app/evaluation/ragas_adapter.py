"""Ragas 0.4 adapter isolated from the application's primary chat path."""
from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings


def evaluation_judge_identity(llm_mode: str) -> dict[str, Any]:
    """Resolve the non-secret judge identity recorded in experiment snapshots."""
    requested = settings.EVALUATION_LLM_PROVIDER
    if requested == "workspace":
        if llm_mode == "local":
            provider = "local"
            model = settings.LOCAL_LLM_MODEL
        else:
            provider = settings.LLM_PROVIDER
            model = settings.LLM_MODEL_FAST
    else:
        provider = requested
        default_model = "qwen-plus" if provider == "dashscope" else settings.LLM_MODEL_FAST
        model = settings.EVALUATION_LLM_MODEL.strip() or default_model
    return {
        "provider": provider,
        "model": model,
        "max_tokens": settings.EVALUATION_LLM_MAX_TOKENS,
        "max_attempts": settings.EVALUATION_LLM_MAX_ATTEMPTS,
    }


def _evaluation_client(llm_mode: str) -> tuple[AsyncOpenAI, str]:
    identity = evaluation_judge_identity(llm_mode)
    provider = identity["provider"]
    if provider == "local":
        api_key = settings.LOCAL_LLM_API_KEY
        base_url = settings.LOCAL_LLM_BASE_URL
    elif provider == "dashscope":
        api_key = settings.DASHSCOPE_API_KEY
        base_url = settings.DASHSCOPE_BASE_URL
    else:
        api_key = settings.DEEPSEEK_API_KEY
        base_url = settings.DEEPSEEK_BASE_URL
    return AsyncOpenAI(api_key=api_key, base_url=base_url), str(identity["model"])


async def _score_with_retry(factory: Callable[[], Awaitable[Any]]) -> Any:
    """Retry transient or malformed structured judge output with a fresh metric call."""
    last_error: Exception | None = None
    for attempt in range(settings.EVALUATION_LLM_MAX_ATTEMPTS):
        try:
            return await factory()
        except Exception as exc:  # Ragas/provider exception types vary by version.
            last_error = exc
            if attempt + 1 < settings.EVALUATION_LLM_MAX_ATTEMPTS:
                await asyncio.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _ensure_ragas_langchain_compatibility() -> None:
    """Bridge a Ragas 0.4 optional Vertex import removed by LangChain 1.x.

    Ragas imports the legacy Vertex class when building its provider registry
    even for OpenAI-compatible clients.  The placeholder is never selected by
    this app, but lets the supported OpenAI-compatible Ragas path load.
    """
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except ModuleNotFoundError:
        module_name = "langchain_community.chat_models.vertexai"
        module = types.ModuleType(module_name)
        module.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules.setdefault(module_name, module)


def _value(result: Any) -> float:
    value = getattr(result, "value", result)
    return round(float(value), 4)


def _reference_statement(question: str, reference_answer: str) -> str:
    """Give Ragas a declarative reference even when the gold answer is terse.

    FactualCorrectness decomposes the reference into factual claims.  Values
    such as ``2048`` or an email address by themselves contain no declarative
    claim, so Ragas returns a misleading zero.  Including the question turns
    that same curated answer into a self-contained gold statement.
    """
    return f'For the question "{question}", the correct answer is: {reference_answer}'


def _local_ragas_embeddings():
    """Adapt the production BGE embedding service to Ragas' modern API."""
    from ragas.embeddings import BaseRagasEmbedding

    from app.services.embedder import get_embedding_service

    service = get_embedding_service()

    class LocalRagasEmbedding(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return [float(value) for value in service.embed_query(text)]

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

    return LocalRagasEmbedding()


async def evaluate_with_ragas(
    *,
    question: str,
    answer: str,
    retrieved_contexts: list[str],
    reference_answer: str | None,
    llm_mode: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Score grounding, correctness and response relevance with Ragas."""
    values: dict[str, float | None] = {
        "faithfulness": None,
        "factual_correctness": None,
        "answer_relevancy": None,
    }
    statuses: dict[str, str] = {}
    if not retrieved_contexts:
        statuses["faithfulness"] = "skipped:no_retrieved_contexts"
    if not reference_answer:
        statuses["factual_correctness"] = "skipped:no_reference_answer"
    if not question.strip() or not answer.strip():
        statuses["answer_relevancy"] = "skipped:no_answer"
    if not retrieved_contexts and not reference_answer and "answer_relevancy" in statuses:
        return values, statuses

    try:
        _ensure_ragas_langchain_compatibility()
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness, FactualCorrectness

        client, model = _evaluation_client(llm_mode)
        # Keep judge decisions reproducible.  `max_tokens` also leaves enough
        # room for Ragas' structured claim-decomposition responses.
        judge_llm = llm_factory(
            model=model,
            provider="openai",
            client=client,
            temperature=0.0,
            max_tokens=settings.EVALUATION_LLM_MAX_TOKENS,
        )

        jobs: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
        if retrieved_contexts:
            jobs.append(("faithfulness", lambda: Faithfulness(llm=judge_llm).ascore(
                user_input=question, response=answer, retrieved_contexts=retrieved_contexts
            )))
        if reference_answer:
            # AI-generated references are intentionally concise gold answers.
            # Recall asks whether the response covers those gold facts; using
            # Ragas' default F1 would additionally penalize well-grounded
            # explanatory detail absent from the compact reference.  Context
            # hallucination is measured separately by Faithfulness above.
            jobs.append(("factual_correctness", lambda: FactualCorrectness(
                llm=judge_llm,
                mode="recall",
            ).ascore(
                response=answer,
                reference=_reference_statement(question, reference_answer),
            )))
        if "answer_relevancy" not in statuses:
            jobs.append(("answer_relevancy", lambda: AnswerRelevancy(
                llm=judge_llm,
                embeddings=_local_ragas_embeddings(),
                strictness=3,
            ).ascore(
                user_input=question,
                response=answer,
            )))
        results = await asyncio.gather(
            *(_score_with_retry(factory) for _, factory in jobs),
            return_exceptions=True,
        )
        for (name, _factory), result in zip(jobs, results):
            if isinstance(result, Exception):
                statuses[name] = f"error:{type(result).__name__}"
            else:
                values[name] = _value(result)
                statuses[name] = "ok"
    except Exception as exc:
        for name in ("faithfulness", "factual_correctness", "answer_relevancy"):
            if name not in statuses:
                statuses[name] = f"error:{type(exc).__name__}"
    return values, statuses
