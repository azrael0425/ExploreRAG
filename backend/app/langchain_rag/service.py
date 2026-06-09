"""Application service that maps LangChain output to the existing SSE contract."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_prompt import ensure_navigation_citations, response_language_for
from app.core.config import settings
from app.langchain_rag.chains.answer import AnswerInput, build_answer_chain
from app.langchain_rag.chains.retrieval import build_retrieval_chain, requires_document_search
from app.langchain_rag.callbacks import get_local_callbacks
from app.langchain_rag.contracts import ChatChainInput
from app.langchain_rag.events import DomainEvent
from app.schemas.rag import ChatImageRef, ChatSourceChunk
from app.services.llm import get_llm_provider
from app.services.llm.types import LLMImagePart
from app.services.models.parsed_document import RetrievalTimings
from app.services.citation_ids import generate_citation_id, source_label

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    return max(1, cjk + max(0, len(text) - cjk) // 4)


def _attachment_direct_budget(system_prompt: str, history: list[dict[str, Any]], llm_mode: str) -> int:
    consumed = _estimate_tokens(system_prompt) + sum(
        _estimate_tokens(str(item.get("content", ""))) for item in history
    )
    context_window = (
        settings.LOCAL_LLM_CONTEXT_WINDOW
        if llm_mode == "local" else settings.CHAT_ATTACHMENT_CONTEXT_WINDOW_TOKENS
    )
    output_tokens = (
        settings.LOCAL_LLM_MAX_OUTPUT_TOKENS
        if llm_mode == "local" else settings.LLM_MAX_OUTPUT_TOKENS
    )
    return max(
        settings.CHAT_ATTACHMENT_DIRECT_MIN_TOKENS,
        context_window - consumed - output_tokens - settings.CHAT_ATTACHMENT_CONTEXT_SAFETY_TOKENS,
    )


class LangChainChatService:
    """Run the fixed RAG workflow through LCEL while preserving domain events."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _related_entities(self, workspace_id: int, answer: str, llm_mode: str) -> list[str]:
        try:
            from app.services.knowledge_graph_service import get_knowledge_graph_service

            graph = get_knowledge_graph_service(workspace_id, llm_mode=llm_mode)
            names = {
                entity["name"].lower(): entity["name"]
                for entity in await graph.get_entities(limit=200)
            }
            answer_lower = answer.lower()
            return [
                original for lower, original in names.items()
                if len(lower) >= 2 and lower in answer_lower
            ][:30]
        except Exception as exc:
            logger.warning("Could not match answer entities to graph: %s", exc)
            return []

    async def stream_chat(
        self,
        input: ChatChainInput,
        *,
        selected_attachments: list[Any] | None = None,
    ) -> AsyncIterator[DomainEvent]:
        """Yield the pre-existing SSE event types using LangChain internally."""
        started_at = time.perf_counter()
        selected_attachments = selected_attachments or []
        sources: list[ChatSourceChunk] = []
        image_refs: list[ChatImageRef] = []
        llm_images: list[LLMImagePart] = []
        retrieval_timings: RetrievalTimings | None = None
        kb_result = None
        answer = ""
        thinking = ""
        callbacks = get_local_callbacks()
        grounded = requires_document_search(input.message) or bool(selected_attachments)

        yield DomainEvent("status", {"step": "analyzing", "detail": "Analyzing your question..."})

        if grounded:
            existing_ids: set[str] = set()
            attachment_events: list[DomainEvent] = []

            async def emit_attachment(event: str, data: dict) -> None:
                attachment_events.append(DomainEvent(event, data))

            if selected_attachments:
                yield DomainEvent("attachment_validating", {
                    "count": len(selected_attachments),
                    "detail": "Validating selected temporary attachments",
                })
                from app.services.attachment_processor import AttachmentProcessor

                selected_attachments = await AttachmentProcessor().prepare_for_chat(
                    self.db,
                    input.workspace_id,
                    selected_attachments,
                    _attachment_direct_budget(input.system_prompt, input.history, input.llm_mode),
                    emit_attachment,
                )
                for event in attachment_events:
                    yield event

            attachment_task = None
            if selected_attachments:
                from app.services.attachment_retriever import AttachmentRetriever

                yield DomainEvent("retrieving_attachments", {"detail": "Searching selected temporary attachments"})
                attachment_task = asyncio.create_task(
                    AttachmentRetriever(input.workspace_id).retrieve(input.message, selected_attachments)
                )

            yield DomainEvent("retrieving_knowledge_base", {"detail": "Searching knowledge base"})
            yield DomainEvent("status", {
                "step": "retrieving",
                "detail": "Searching knowledge base and selected attachments...",
            })
            kb_task = asyncio.create_task(
                build_retrieval_chain(self.db, existing_ids).ainvoke(input, config={
                    "run_name": "explorerag_chat_retrieval",
                    "tags": ["rag", input.llm_mode, input.retrieval_mode],
                    "metadata": {"workspace_id": input.workspace_id},
                    "callbacks": callbacks,
                })
            )
            attachment_result = None
            if attachment_task:
                attachment_result, kb_result = await asyncio.gather(attachment_task, kb_task)
            else:
                kb_result = await kb_task

            attachment_context = ""
            attachment_sources: list[ChatSourceChunk] = []
            attachment_image_refs: list[ChatImageRef] = []
            attachment_llm_images: list[LLMImagePart] = []
            if attachment_result:
                for chunk in attachment_result.chunks:
                    citation_id = source_label("ATT", existing_ids)
                    existing_ids.add(citation_id)
                    attachment_sources.append(ChatSourceChunk(
                        index=citation_id,
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        document_id=0,
                        attachment_id=chunk.attachment_id,
                        source_file=chunk.source_file,
                        page_no=chunk.page_no,
                        heading_path=chunk.heading_path,
                        score=chunk.score,
                        source_type="attachment",
                    ))
                    location = f", page {chunk.page_no}" if chunk.page_no else ""
                    heading = f", {' > '.join(chunk.heading_path)}" if chunk.heading_path else ""
                    attachment_context += (
                        f"[ATT-{citation_id.split('-', 1)[1]}] {chunk.source_file}{location}{heading}:\n"
                        f"{chunk.content}\n\n"
                    )
                for image in attachment_result.images:
                    image_ref_id = generate_citation_id(existing_ids)
                    existing_ids.add(image_ref_id)
                    attachment_image_refs.append(ChatImageRef(
                        ref_id=image_ref_id,
                        image_id=image.image_id,
                        document_id=0,
                        attachment_id=image.attachment_id,
                        page_no=image.page_no,
                        caption=image.caption,
                        url=image.url,
                    ))
                    attachment_llm_images.append(LLMImagePart(data=image.data, mime_type=image.mime_type))

            sources = attachment_sources + kb_result.sources
            image_refs = attachment_image_refs + kb_result.image_refs
            llm_images = attachment_llm_images + kb_result.llm_images
            retrieval_timings = kb_result.timings
            context = "\n\n---\n\n".join(
                part for part in (attachment_context.strip(), kb_result.context.strip()) if part
            )
            logger.info(
                "LangChain chat retrieval workspace=%s attachments=%s attachment_sources=%s kb_sources=%s injected_tokens=%s",
                input.workspace_id,
                len(selected_attachments),
                len(attachment_sources),
                len(kb_result.sources),
                _estimate_tokens(context),
            )
            if sources:
                yield DomainEvent("sources", {"sources": [source.model_dump() for source in sources]})
            if image_refs:
                yield DomainEvent("images", {"image_refs": [image.model_dump() for image in image_refs]})
            if not sources and not llm_images:
                answer = (
                    "知识库中没有找到可用于回答该问题的相关文段。"
                    if response_language_for(input.message) == "Chinese"
                    else "No relevant passage was found in the knowledge base for this question."
                )
        else:
            context = ""

        yield DomainEvent("status", {"step": "generating", "detail": "Generating answer..."})
        generation_started_at = time.perf_counter()
        first_text_at: float | None = None
        if not answer:
            chain = build_answer_chain(
                get_llm_provider(input.llm_mode),
                enable_thinking=input.enable_thinking,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            )
            async for chunk in chain.astream(AnswerInput(
                system_prompt=input.system_prompt,
                history=input.history,
                context=context,
                question=input.message,
                grounded=grounded,
                sources=sources,
                llm_images=llm_images,
            ), config={
                "run_name": "explorerag_chat_generation",
                "tags": ["rag", input.llm_mode],
                "metadata": {"workspace_id": input.workspace_id, "has_attachments": bool(selected_attachments)},
                "callbacks": callbacks,
            }):
                additional = getattr(chunk, "additional_kwargs", {}) or {}
                thinking_delta = additional.get("thinking", "")
                if thinking_delta:
                    thinking += str(thinking_delta)
                    yield DomainEvent("thinking", {"text": str(thinking_delta)})
                content = getattr(chunk, "content", "")
                if content:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    text = str(content)
                    answer += text
                    yield DomainEvent("token", {"text": text})

        generation_ms = int((time.perf_counter() - generation_started_at) * 1000)
        postprocess_started_at = time.perf_counter()
        answer = re.sub(r"<unused\d+>:?\s*", "", answer).strip()
        answer = ensure_navigation_citations(answer, input.message, (source.index for source in sources))
        related_entities = (
            await self._related_entities(input.workspace_id, answer, input.llm_mode)
            if any(source.source_type == "kg" for source in sources)
            else []
        )
        performance = {
            "vector_ms": retrieval_timings.vector_ms if retrieval_timings else None,
            "graph_ms": retrieval_timings.graph_ms if retrieval_timings else None,
            "rerank_ms": retrieval_timings.rerank_ms if retrieval_timings else None,
            "context_ms": retrieval_timings.context_ms if retrieval_timings else None,
            "generation_ms": generation_ms,
            "first_token_ms": int((first_text_at - started_at) * 1000) if first_text_at else None,
            "postprocess_ms": int((time.perf_counter() - postprocess_started_at) * 1000),
            "total_ms": int((time.perf_counter() - started_at) * 1000),
            "retrieval_trace": (
                dict(getattr(kb_result.envelope.raw_result, "trace", {}) or {})
                if kb_result and kb_result.envelope and kb_result.envelope.raw_result
                else {}
            ),
        }
        yield DomainEvent("complete", {
            "answer": answer or "Unable to generate a response.",
            "sources": [source.model_dump() for source in sources],
            "image_refs": [image.model_dump() for image in image_refs],
            "thinking": thinking or None,
            "related_entities": related_entities,
            "performance": performance,
        })


async def complete_chat(
    input: ChatChainInput,
    db: AsyncSession,
    *,
    selected_attachments: list[Any] | None = None,
) -> dict[str, Any]:
    """Consume the same LangChain event stream for the non-streaming endpoint."""
    final: dict[str, Any] = {}
    async for event in LangChainChatService(db).stream_chat(
        input, selected_attachments=selected_attachments
    ):
        if event.event == "complete":
            final = event.data
    if not final:
        raise RuntimeError("LangChain chat completed without a final response")
    return final
