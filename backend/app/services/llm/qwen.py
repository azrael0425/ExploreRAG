"""Qwen provider backed by the DashScope OpenAI-compatible API.

The provider deliberately uses the standard ``openai`` client instead of the
DashScope SDK.  This keeps chat streaming, native function calling, and image
inputs on one well-supported API surface while allowing a regional/custom
Model Studio compatible endpoint to be configured with ``DASHSCOPE_BASE_URL``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager, contextmanager, nullcontext
from typing import Any, AsyncGenerator, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)


class QwenLLMProvider(LLMProvider):
    """Qwen text, vision, streaming, and native function-call provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        vision_model: str = "qwen-vl-plus",
        vision_api_key: str | None = None,
        vision_base_url: str | None = None,
        enable_thinking: bool = False,
        provider_name: str = "dashscope",
        is_local: bool = False,
        strict_errors: bool = False,
        request_timeout: float = 600.0,
        healthcheck_timeout: float = 3.0,
        max_output_tokens_cap: int | None = None,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
        self._vision_client = (
            OpenAI(api_key=vision_api_key, base_url=vision_base_url, timeout=request_timeout)
            if vision_api_key and vision_base_url
            else self._client
        )
        self._vision_async_client = (
            AsyncOpenAI(api_key=vision_api_key, base_url=vision_base_url, timeout=request_timeout)
            if vision_api_key and vision_base_url
            else self._async_client
        )
        self._model = model
        self._vision_model = vision_model
        self._enable_thinking = enable_thinking
        self.provider_name = provider_name
        self.model_name = model
        self.vision_model_name = vision_model
        self.is_local = is_local
        self._strict_errors = strict_errors
        self._max_output_tokens_cap = max_output_tokens_cap
        self._healthcheck_timeout = healthcheck_timeout
        # Used by the agent loop to send a native tool result in the next turn.
        self.last_response_message: dict[str, Any] | None = None

    @staticmethod
    def _image_url(data: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _to_messages(
        self,
        messages: list[LLMMessage],
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Convert the provider-neutral message type to OpenAI chat messages."""
        result: list[dict[str, Any]] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            # Native assistant tool calls and tool responses must be replayed
            # verbatim so the API can associate a response with its call ID.
            if msg._raw_provider_content is not None:
                if not isinstance(msg._raw_provider_content, dict):
                    raise TypeError("Qwen raw provider content must be a message dict")
                result.append(msg._raw_provider_content)
                continue

            if msg.images:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                content.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_url(img.data, img.mime_type)},
                    }
                    for img in msg.images
                )
                result.append({"role": msg.role, "content": content})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    @staticmethod
    def _contains_images(messages: list[LLMMessage]) -> bool:
        return any(message.images for message in messages)

    def _select_model(self, messages: list[LLMMessage]) -> str:
        return self._vision_model if self._contains_images(messages) else self._model

    def _request_options(self, think: bool) -> dict[str, Any]:
        """Return provider-specific reasoning controls for this request.

        DeepSeek V4 enables thinking by default.  LightRAG extraction calls
        this provider with ``think=False`` and requires a terse, delimiter
        based record format, so we must explicitly disable thinking instead
        of merely omitting a flag.  Interactive chat continues to opt in on
        a per-request basis through the existing ``enable_thinking`` switch.
        """
        if self.provider_name == "deepseek":
            return {"extra_body": {"thinking": {"type": "enabled" if think else "disabled"}}}
        if think and self._enable_thinking:
            return {"extra_body": {"enable_thinking": True}}
        return {}

    def _bounded_max_tokens(self, requested: int) -> int:
        if self._max_output_tokens_cap is None:
            return requested
        return min(requested, self._max_output_tokens_cap)

    @contextmanager
    def _sync_guard(self, model: str):
        if not self.is_local:
            with nullcontext():
                yield
            return
        from app.services.model_runtime import local_llm_guard

        with local_llm_guard(model):
            yield

    @asynccontextmanager
    async def _async_guard(self, model: str):
        if not self.is_local:
            yield
            return
        from app.services.model_runtime import async_local_llm_guard

        async with async_local_llm_guard(model):
            yield

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        use_think = think and self.supports_thinking()
        selected_model = self._select_model(messages)
        client = self._vision_client if self._contains_images(messages) else self._client
        try:
            with self._sync_guard(selected_model):
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=self._to_messages(messages, system_prompt),
                    temperature=temperature,
                    max_tokens=self._bounded_max_tokens(max_tokens),
                    **self._request_options(use_think),
                )
            message = response.choices[0].message if response.choices else None
            content = (message.content if message else None) or ""
            thinking = getattr(message, "reasoning_content", None) or ""
            return LLMResult(content=content, thinking=thinking) if use_think else content
        except Exception as exc:
            logger.error("Qwen completion failed: %s", exc, exc_info=True)
            if self._strict_errors:
                raise
            return LLMResult(content="") if use_think else ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        use_think = think and self.supports_thinking()
        selected_model = self._select_model(messages)
        client = self._vision_async_client if self._contains_images(messages) else self._async_client
        try:
            async with self._async_guard(selected_model):
                response = await client.chat.completions.create(
                    model=selected_model,
                    messages=self._to_messages(messages, system_prompt),
                    temperature=temperature,
                    max_tokens=self._bounded_max_tokens(max_tokens),
                    **self._request_options(use_think),
                )
            message = response.choices[0].message if response.choices else None
            content = (message.content if message else None) or ""
            thinking = getattr(message, "reasoning_content", None) or ""
            return LLMResult(content=content, thinking=thinking) if use_think else content
        except Exception as exc:
            logger.error("Qwen async completion failed: %s", exc, exc_info=True)
            if self._strict_errors:
                raise
            return LLMResult(content="") if use_think else ""

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream text/reasoning and reconstruct split OpenAI tool-call deltas."""
        use_think = think and self.supports_thinking()
        text_parts: list[str] = []
        # OpenAI-compatible streams may send the function name, call ID, and
        # JSON arguments in separate chunks.  Keep one accumulator per index.
        tool_calls: dict[int, dict[str, str]] = {}
        self.last_response_message = None
        selected_model = self._select_model(messages)
        client = self._vision_async_client if self._contains_images(messages) else self._async_client

        try:
            async with self._async_guard(selected_model):
                stream = await client.chat.completions.create(
                    model=selected_model,
                    messages=self._to_messages(messages, system_prompt),
                    temperature=temperature,
                    max_tokens=self._bounded_max_tokens(max_tokens),
                    stream=True,
                    tools=tools or None,
                    **self._request_options(use_think),
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    reasoning = getattr(delta, "reasoning_content", None) or ""
                    if reasoning:
                        yield StreamChunk(type="thinking", text=reasoning)

                    content = delta.content or ""
                    if content:
                        text_parts.append(content)
                        yield StreamChunk(type="text", text=content)

                    for tool_delta in getattr(delta, "tool_calls", None) or []:
                        index = tool_delta.index or 0
                        entry = tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if tool_delta.id:
                            entry["id"] = tool_delta.id
                        function = tool_delta.function
                        if function:
                            if function.name:
                                entry["name"] = function.name
                            if function.arguments:
                                entry["arguments"] += function.arguments
        except Exception as exc:
            logger.error("Qwen streaming completion failed: %s", exc, exc_info=True)
            if self._strict_errors:
                raise
            return

        if tool_calls:
            raw_tool_calls: list[dict[str, Any]] = []
            for index in sorted(tool_calls):
                call = tool_calls[index]
                try:
                    args = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    logger.warning("Qwen returned invalid function arguments: %r", call["arguments"])
                    args = {}
                raw_tool_calls.append({
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"] or "{}",
                    },
                })
                yield StreamChunk(
                    type="function_call",
                    function_call={"name": call["name"], "args": args},
                )

            self.last_response_message = {
                "role": "assistant",
                "content": "".join(text_parts) or None,
                "tool_calls": raw_tool_calls,
            }
        else:
            self.last_response_message = {
                "role": "assistant",
                "content": "".join(text_parts),
            }

    def supports_vision(self) -> bool:
        return bool(self._vision_model)

    def supports_thinking(self) -> bool:
        # DeepSeek V4 supports an explicit per-request thinking toggle.
        return self.provider_name == "deepseek" or self._enable_thinking

    def supports_native_tools(self) -> bool:
        return True

    async def ahealthcheck(self) -> tuple[bool, str | None]:
        try:
            models = await asyncio.wait_for(
                self._async_client.models.list(),
                timeout=self._healthcheck_timeout,
            )
            if self.is_local:
                available_ids = {
                    str(getattr(item, "id", "")).removesuffix(":latest")
                    for item in getattr(models, "data", [])
                }
                required_ids = {
                    self._model.removesuffix(":latest"),
                    self._vision_model.removesuffix(":latest"),
                }
                missing = sorted(model for model in required_ids if model not in available_ids)
                if missing:
                    return False, f"Local model is not installed: {', '.join(missing)}"
            return True, None
        except asyncio.TimeoutError:
            return False, f"Health check timed out after {self._healthcheck_timeout:g}s"
        except Exception as exc:
            return False, str(exc)


class QwenEmbeddingProvider(EmbeddingProvider):
    """DashScope ``text-embedding-v4`` via the OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimension: int = 1024,
        batch_size: int = 10,
    ):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start:start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimension,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    f"Qwen embedding response returned {len(ordered)} vectors for {len(batch)} inputs"
                )
            vectors.extend(item.embedding for item in ordered)

        return np.asarray(vectors, dtype=np.float32)

    def get_dimension(self) -> int:
        return self._dimension
