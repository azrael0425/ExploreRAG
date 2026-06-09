"""
LLM Provider Package
=====================
Factory functions for Qwen generation and local BGE-M3 embeddings.

Usage::

    from app.services.llm import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()          # uses Qwen settings from .env
    emb = get_embedding_provider()    # uses KG_EMBEDDING_PROVIDER from .env
"""
from __future__ import annotations

from functools import lru_cache

from app.services.llm.base import EmbeddingProvider, LLMProvider


def get_cloud_llm_credentials() -> tuple[str, str, str]:
    """Return the configured cloud provider name, credential, and endpoint."""
    from app.core.config import settings

    if settings.LLM_PROVIDER == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek cloud LLM mode.")
        return "deepseek", settings.DEEPSEEK_API_KEY, settings.DEEPSEEK_BASE_URL

    if not settings.DASHSCOPE_API_KEY:
        raise ValueError("DASHSCOPE_API_KEY is required for DashScope cloud LLM mode.")
    return "dashscope", settings.DASHSCOPE_API_KEY, settings.DASHSCOPE_BASE_URL


@lru_cache(maxsize=2)
def get_llm_provider(mode: str = "cloud") -> LLMProvider:
    """Return the cached cloud or strict-local OpenAI-compatible provider."""
    from app.core.config import settings

    from app.services.llm.qwen import QwenLLMProvider

    normalized = mode.strip().lower()
    if normalized == "cloud":
        provider_name, api_key, base_url = get_cloud_llm_credentials()
        return QwenLLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=settings.LLM_MODEL_FAST,
            # Keep the existing Qwen vision route available when the text
            # model is served by DeepSeek.
            vision_model=(
                settings.QWEN_VISION_MODEL
                if provider_name == "dashscope" or settings.DASHSCOPE_API_KEY
                else ""
            ),
            vision_api_key=(settings.DASHSCOPE_API_KEY if provider_name == "deepseek" else None),
            vision_base_url=(settings.DASHSCOPE_BASE_URL if provider_name == "deepseek" else None),
            enable_thinking=settings.QWEN_ENABLE_THINKING,
            provider_name=provider_name,
        )
    if normalized == "local":
        return QwenLLMProvider(
            api_key=settings.LOCAL_LLM_API_KEY,
            base_url=settings.LOCAL_LLM_BASE_URL,
            model=settings.LOCAL_LLM_MODEL,
            vision_model=settings.LOCAL_LLM_VISION_MODEL,
            enable_thinking=settings.LOCAL_LLM_ENABLE_THINKING,
            provider_name="ollama",
            is_local=True,
            strict_errors=True,
            request_timeout=settings.LOCAL_LLM_REQUEST_TIMEOUT_SECONDS,
            healthcheck_timeout=settings.LOCAL_LLM_HEALTH_TIMEOUT_SECONDS,
            max_output_tokens_cap=settings.LOCAL_LLM_MAX_OUTPUT_TOKENS,
        )
    raise ValueError(f"Unsupported LLM mode: {mode!r}")


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Create (and cache) the BGE-M3 provider for LightRAG."""
    from app.core.config import settings

    provider = settings.KG_EMBEDDING_PROVIDER.lower()

    if provider == "sentence_transformers":
        from app.services.llm.sentence_transformer import SentenceTransformerEmbeddingProvider

        return SentenceTransformerEmbeddingProvider(
            model=settings.KG_EMBEDDING_MODEL,
        )

    raise ValueError(
        f"Unknown KG_EMBEDDING_PROVIDER: {provider!r}. "
        "Only sentence_transformers (local BGE-M3) is supported."
    )


def release_embedding_provider() -> bool:
    """Release cached KG embedding references before local VLM inference."""
    provider = get_embedding_provider()
    released = bool(getattr(provider, "unload", lambda: False)())
    get_embedding_provider.cache_clear()
    return released


__all__ = [
    "get_llm_provider",
    "get_cloud_llm_credentials",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]
