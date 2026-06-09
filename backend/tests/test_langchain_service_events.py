from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessageChunk

from app.langchain_rag.chains.retrieval import KnowledgeBaseContext
from app.langchain_rag.contracts import ChatChainInput
from app.langchain_rag.service import LangChainChatService
from app.schemas.rag import ChatSourceChunk
from app.services.models.parsed_document import RetrievalTimings


class _RetrievalChain:
    async def ainvoke(self, _input, config=None):
        return KnowledgeBaseContext(
            context="Source [KB-abcd]: evidence",
            sources=[ChatSourceChunk(
                index="KB-abcd", chunk_id="doc_1_chunk_0", content="evidence", document_id=1
            )],
            image_refs=[],
            llm_images=[],
            timings=RetrievalTimings(vector_ms=3),
        )


class _AnswerChain:
    async def astream(self, _input, config=None):
        yield AIMessageChunk(content="reasoning-free", additional_kwargs={"thinking": "thought"})
        yield AIMessageChunk(content=" answer [KB-abcd]")


def test_langchain_service_keeps_sse_event_contract(monkeypatch) -> None:
    async def scenario() -> None:
        from app.langchain_rag import service

        monkeypatch.setattr(service, "build_retrieval_chain", lambda *args, **kwargs: _RetrievalChain())
        monkeypatch.setattr(service, "build_answer_chain", lambda *args, **kwargs: _AnswerChain())
        monkeypatch.setattr(service, "get_llm_provider", lambda *_args, **_kwargs: object())

        async def no_entities(*_args, **_kwargs):
            return []

        chat = LangChainChatService(object())
        monkeypatch.setattr(chat, "_related_entities", no_entities)
        events = [
            event async for event in chat.stream_chat(ChatChainInput(
                workspace_id=1,
                message="Explain the evidence",
                system_prompt="system",
            ))
        ]
        event_names = [event.event for event in events]

        assert event_names == [
            "status", "retrieving_knowledge_base", "status", "sources",
            "status", "thinking", "token", "token", "complete",
        ]
        complete = events[-1].data
        assert complete["answer"] == "reasoning-free answer [KB-abcd]"
        assert complete["thinking"] == "thought"
        assert complete["performance"]["vector_ms"] == 3

    asyncio.run(scenario())
