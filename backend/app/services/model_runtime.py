"""Coordinate the 8 GB GPU between retrieval models and the local VLM.

The backend owns BGE-M3/reranker while Ollama owns Qwen3-VL.  A process-wide
gate prevents both sets of models from running at once.  Models are unloaded
at phase boundaries so local mode is usable on the target laptop GPU.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Iterator, AsyncIterator

from app.core.config import settings

logger = logging.getLogger(__name__)

_gpu_gate = threading.BoundedSemaphore(value=1)
_state_lock = threading.Lock()
_loaded_local_models: set[str] = set()
_local_inventory_checked = False


def _release_retrieval_models() -> None:
    """Drop all backend-owned CUDA model references while holding the gate."""
    released = False
    try:
        from app.services.embedder import release_embedding_service

        released = release_embedding_service() or released
    except Exception as exc:
        logger.warning("Could not release primary embedding model: %s", exc)
    try:
        from app.services.llm import release_embedding_provider

        released = release_embedding_provider() or released
    except Exception as exc:
        logger.warning("Could not release KG embedding model: %s", exc)
    try:
        from app.services.reranker import release_reranker_service

        released = release_reranker_service() or released
    except Exception as exc:
        logger.warning("Could not release reranker model: %s", exc)

    if released:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Released retrieval models before local LLM inference")


def mark_local_model_loaded(model: str) -> None:
    global _local_inventory_checked
    with _state_lock:
        _loaded_local_models.add(model)
        _local_inventory_checked = True


def _unload_local_models() -> None:
    """Ask Ollama to release locally loaded models before BGE uses CUDA."""
    global _local_inventory_checked
    with _state_lock:
        models = list(_loaded_local_models)
        _loaded_local_models.clear()
        discover_running = not _local_inventory_checked
        _local_inventory_checked = True

    native_base_url = settings.LOCAL_LLM_NATIVE_BASE_URL.rstrip("/")
    if discover_running:
        # The backend may restart while Ollama keeps a model resident. Inspect
        # once so startup retrieval does not collide with stale VLM VRAM.
        try:
            import httpx

            with httpx.Client(timeout=settings.LOCAL_LLM_HEALTH_TIMEOUT_SECONDS) as client:
                response = client.get(f"{native_base_url}/api/ps")
                response.raise_for_status()
                configured = {
                    settings.LOCAL_LLM_MODEL.removesuffix(":latest"),
                    settings.LOCAL_LLM_VISION_MODEL.removesuffix(":latest"),
                }
                for item in response.json().get("models", []):
                    name = str(item.get("name") or item.get("model") or "")
                    if name.removesuffix(":latest") in configured:
                        models.append(name)
        except Exception as exc:
            logger.debug("Could not inspect Ollama running models: %s", exc)

    models = sorted(set(models))
    if not models:
        return

    try:
        import httpx

        with httpx.Client(timeout=settings.LOCAL_LLM_HEALTH_TIMEOUT_SECONDS) as client:
            for model in models:
                response = client.post(
                    f"{native_base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": "",
                        "stream": False,
                        "keep_alive": 0,
                    },
                )
                response.raise_for_status()
        logger.info("Released local LLM models before retrieval inference: %s", models)
    except Exception as exc:
        # Continue so a clear CUDA/OOM error is surfaced by the retrieval model
        # rather than hiding the underlying local-server problem.
        logger.warning("Could not request local LLM unload for %s: %s", models, exc)


@contextmanager
def retrieval_model_guard() -> Iterator[None]:
    """Serialize retrieval inference and unload any resident local VLM first."""
    _gpu_gate.acquire()
    try:
        _unload_local_models()
        yield
    finally:
        _gpu_gate.release()


@contextmanager
def local_llm_guard(model: str) -> Iterator[None]:
    """Serialize a synchronous local request and free retrieval CUDA memory."""
    _gpu_gate.acquire()
    try:
        _release_retrieval_models()
        mark_local_model_loaded(model)
        yield
    finally:
        _gpu_gate.release()


@asynccontextmanager
async def async_local_llm_guard(model: str) -> AsyncIterator[None]:
    """Async variant that never blocks the event loop while waiting for CUDA."""
    await asyncio.to_thread(_gpu_gate.acquire)
    try:
        await asyncio.to_thread(_release_retrieval_models)
        mark_local_model_loaded(model)
        yield
    finally:
        _gpu_gate.release()
