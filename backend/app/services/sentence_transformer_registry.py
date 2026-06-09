"""Process-wide registry for local SentenceTransformer models.

The primary vector retriever and LightRAG use the same BGE-M3 weights.  Loading
the model independently in both services wastes several gigabytes of GPU
memory, so this module owns a single model and inference lock per
``(model_name, device)`` pair.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedSentenceTransformer:
    model: Any
    inference_lock: threading.RLock


_registry: dict[tuple[str, str], SharedSentenceTransformer] = {}
_registry_lock = threading.Lock()


def _model_key(model_name: str, device: str) -> tuple[str, str]:
    path = Path(model_name)
    normalized = str(path.resolve()) if path.is_absolute() else model_name
    return normalized, device


def get_shared_sentence_transformer(
    model_name: str,
    device: str,
) -> SharedSentenceTransformer:
    """Return the only in-process instance for a model/device pair.

    Model construction is intentionally performed while holding the registry
    lock.  ``functools.lru_cache`` can invoke a slow cache miss more than once
    under concurrency, which is exactly what this registry must prevent.
    """
    key = _model_key(model_name, device)
    with _registry_lock:
        existing = _registry.get(key)
        if existing is not None:
            return existing

        from sentence_transformers import SentenceTransformer

        model_path = Path(model_name)
        if model_path.is_absolute():
            if not model_path.is_dir():
                raise RuntimeError(
                    f"Local embedding model directory is unavailable: {model_path}. "
                    "Download BAAI/bge-m3 and mount it before use."
                )
            load_options: dict[str, Any] = {
                "local_files_only": True,
                "device": device,
            }
        else:
            load_options = {"device": device}

        logger.info("Loading shared sentence-transformers model on %s: %s", device, model_name)
        model = SentenceTransformer(model_name, **load_options)
        entry = SharedSentenceTransformer(model=model, inference_lock=threading.RLock())
        _registry[key] = entry
        logger.info(
            "Shared sentence-transformers model loaded on %s: %s (dim=%s)",
            device,
            model_name,
            model.get_sentence_embedding_dimension(),
        )
        return entry


def clear_sentence_transformer_registry_for_tests() -> None:
    """Drop registry references; intended for isolated unit tests only."""
    with _registry_lock:
        _registry.clear()


def release_sentence_transformer_registry() -> bool:
    """Remove shared CUDA model references and report whether any existed."""
    with _registry_lock:
        had_models = bool(_registry)
        _registry.clear()
    return had_models
