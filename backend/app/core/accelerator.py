"""CUDA device validation shared by the local retrieval models."""
from __future__ import annotations

import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


def require_cuda(device: str, component: str) -> str:
    """Validate an explicitly configured CUDA device without loading a model."""
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        raise ValueError(
            f"{component} must use a CUDA device, got {device!r}. "
            "Set the corresponding *_DEVICE setting to 'cuda'."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{component} is configured for {normalized}, but CUDA is unavailable in "
            "the backend container. Ensure Docker GPU support is enabled and start "
            "the backend with the Compose GPU configuration."
        )

    return normalized


def validate_local_model_accelerators() -> None:
    """Fail fast when a configured local retrieval model cannot access CUDA."""
    configured: list[tuple[str, str]] = []
    if settings.KG_EMBEDDING_PROVIDER.lower() == "sentence_transformers":
        configured.append(("KG BGE embedding model", settings.KG_EMBEDDING_DEVICE))
    if settings.EXPLORERAG_EMBEDDING_PROVIDER.lower() == "sentence_transformers":
        configured.append(("primary BGE embedding model", settings.EXPLORERAG_EMBEDDING_DEVICE))
    if settings.EXPLORERAG_ENABLE_RERANKER:
        configured.append(("BGE reranker model", settings.EXPLORERAG_RERANKER_DEVICE))

    for component, device in configured:
        validated = require_cuda(device, component)
        logger.info("%s configured to use %s", component, validated)
