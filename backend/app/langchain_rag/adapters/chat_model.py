"""LangChain chat model wrapper for the existing Qwen-compatible providers."""
from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import ConfigDict

from app.services.llm.base import LLMProvider
from app.services.llm.types import LLMImagePart, LLMMessage, LLMResult


def _decode_data_url(url: str) -> LLMImagePart | None:
    """Decode an inline image data URL without logging its contents."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    header, encoded = url.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "image/png"
    try:
        return LLMImagePart(data=base64.b64decode(encoded, validate=True), mime_type=mime_type)
    except (ValueError, TypeError):
        return None


def _message_parts(message: BaseMessage) -> tuple[str, list[LLMImagePart]]:
    """Extract text and supported inline image blocks from a LangChain message."""
    if isinstance(message.content, str):
        return message.content, []
    if not isinstance(message.content, list):
        return str(message.content), []

    text_parts: list[str] = []
    images: list[LLMImagePart] = []
    for part in message.content:
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            text_parts.append(str(part.get("text", "")))
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str):
                decoded = _decode_data_url(url)
                if decoded:
                    images.append(decoded)
            continue
        if part_type in {"image", "input_image"}:
            encoded = part.get("base64") or part.get("data")
            if isinstance(encoded, str):
                try:
                    images.append(
                        LLMImagePart(
                            data=base64.b64decode(encoded, validate=True),
                            mime_type=str(part.get("mime_type", "image/png")),
                        )
                    )
                except (ValueError, TypeError):
                    pass
            continue
        text_parts.append(str(part.get("text", "")))
    return "".join(text_parts), images


def to_provider_messages(messages: list[BaseMessage]) -> tuple[list[LLMMessage], str | None]:
    """Convert LangChain messages while preserving system and native raw data."""
    system_parts: list[str] = []
    provider_messages: list[LLMMessage] = []
    for message in messages:
        text, images = _message_parts(message)
        if isinstance(message, SystemMessage):
            if text:
                system_parts.append(text)
            continue
        role = "assistant" if isinstance(message, AIMessage) else "user"
        provider_messages.append(
            LLMMessage(
                role=role,
                content=text,
                images=images,
                _raw_provider_content=message.additional_kwargs.get("raw_provider_content"),
            )
        )
    return provider_messages, "\n\n".join(system_parts) or None


class ExploreRAGChatModel(BaseChatModel):
    """Use the project Qwen provider through LangChain's chat-model contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: LLMProvider
    temperature: float = 0.1
    max_tokens: int = 8192
    think: bool = False

    @property
    def _llm_type(self) -> str:
        return "explorerag-qwen"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "provider": getattr(self.provider, "provider_name", type(self.provider).__name__),
            "model": getattr(self.provider, "model_name", ""),
            "is_local": bool(getattr(self.provider, "is_local", False)),
        }

    def _options(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "think": kwargs.get("think", self.think),
            "tools": kwargs.get("tools"),
        }

    @staticmethod
    def _result_message(result: str | LLMResult) -> AIMessage:
        if isinstance(result, LLMResult):
            return AIMessage(content=result.content, additional_kwargs={"thinking": result.thinking})
        return AIMessage(content=result)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        provider_messages, system_prompt = to_provider_messages(messages)
        options = self._options(kwargs)
        result = self.provider.complete(
            provider_messages,
            temperature=options["temperature"],
            max_tokens=options["max_tokens"],
            system_prompt=system_prompt,
            think=options["think"],
        )
        return ChatResult(generations=[ChatGeneration(message=self._result_message(result))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        provider_messages, system_prompt = to_provider_messages(messages)
        options = self._options(kwargs)
        result = await self.provider.acomplete(
            provider_messages,
            temperature=options["temperature"],
            max_tokens=options["max_tokens"],
            system_prompt=system_prompt,
            think=options["think"],
        )
        return ChatResult(generations=[ChatGeneration(message=self._result_message(result))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # The existing provider has a native async stream.  Synchronous
        # LangChain consumers still receive a correct single completion.
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        thinking = message.additional_kwargs.get("thinking", "")
        if thinking:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", additional_kwargs={"thinking": thinking}))
        if message.content:
            yield ChatGenerationChunk(message=AIMessageChunk(content=message.content))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        provider_messages, system_prompt = to_provider_messages(messages)
        options = self._options(kwargs)
        async for part in self.provider.astream(
            provider_messages,
            temperature=options["temperature"],
            max_tokens=options["max_tokens"],
            system_prompt=system_prompt,
            think=options["think"],
            tools=options["tools"],
        ):
            if part.type == "thinking" and part.text:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", additional_kwargs={"thinking": part.text})
                )
            elif part.type == "text" and part.text:
                yield ChatGenerationChunk(message=AIMessageChunk(content=part.text))
