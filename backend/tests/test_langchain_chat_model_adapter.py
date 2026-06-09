from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from app.langchain_rag.adapters.chat_model import ExploreRAGChatModel
from app.services.llm.base import LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk


class _FakeProvider(LLMProvider):
    provider_name = "fake"
    model_name = "fake-qwen"
    is_local = False

    def __init__(self) -> None:
        self.calls: list[tuple[list[LLMMessage], dict]] = []

    def complete(self, messages: list[LLMMessage], **kwargs):
        self.calls.append((messages, kwargs))
        return LLMResult(content="answer", thinking="reasoning")

    async def acomplete(self, messages: list[LLMMessage], **kwargs):
        return self.complete(messages, **kwargs)

    async def astream(self, messages: list[LLMMessage], **kwargs) -> AsyncGenerator[StreamChunk, None]:
        self.calls.append((messages, kwargs))
        yield StreamChunk(type="thinking", text="reasoning")
        yield StreamChunk(type="text", text="hello")
        yield StreamChunk(type="text", text=" world")

    def supports_vision(self) -> bool:
        return True


def test_chat_model_converts_system_text_and_inline_image() -> None:
    provider = _FakeProvider()
    model = ExploreRAGChatModel(provider=provider, think=True)
    image = base64.b64encode(b"image-bytes").decode("ascii")

    answer = model.invoke([
        SystemMessage(content="system rules"),
        HumanMessage(content=[
            {"type": "text", "text": "What is shown?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
        ]),
    ])

    messages, options = provider.calls[0]
    assert options["system_prompt"] == "system rules"
    assert messages[0].content == "What is shown?"
    assert messages[0].images[0].data == b"image-bytes"
    assert messages[0].images[0].mime_type == "image/png"
    assert answer.content == "answer"
    assert answer.additional_kwargs["thinking"] == "reasoning"


def test_chat_model_preserves_thinking_outside_answer_stream() -> None:
    async def scenario() -> None:
        provider = _FakeProvider()
        model = ExploreRAGChatModel(provider=provider, think=True)
        chunks = [chunk async for chunk in model.astream([HumanMessage(content="hello")])]

        assert "".join(str(chunk.content) for chunk in chunks) == "hello world"
        assert chunks[0].additional_kwargs["thinking"] == "reasoning"
        assert all("reasoning" not in str(chunk.content) for chunk in chunks)

    asyncio.run(scenario())
