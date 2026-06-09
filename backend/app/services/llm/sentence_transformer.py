"""
Sentence-Transformers Embedding Provider
==========================================
Concrete EmbeddingProvider using sentence-transformers for fully local,
open-source KG embeddings — no external API required.

Usage::

    KG_EMBEDDING_PROVIDER=sentence_transformers
    KG_EMBEDDING_MODEL=BAAI/bge-m3
"""
from __future__ import annotations

import logging

import numpy as np

from app.core.accelerator import require_cuda
from app.core.config import settings
from app.services.llm.base import EmbeddingProvider
from app.services.sentence_transformer_registry import (
    SharedSentenceTransformer,
    get_shared_sentence_transformer,
)

logger = logging.getLogger(__name__)

# Dimension lookup so we can report dimension before loading the model.
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large-instruct": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "jinaai/jina-embeddings-v2-base-en": 768,
}


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using any sentence-transformers model."""

    _BATCH_SIZE = 64

    def __init__(self, model: str = "BAAI/bge-m3"):
        self._model_name = model
        self._model = None
        self._shared_model: SharedSentenceTransformer | None = None
        self._dimension: int | None = _KNOWN_DIMS.get(model)

    # -- lazy load to avoid importing at startup --

    @property
    def model(self):
        if self._model is None:
            device = require_cuda(
                settings.KG_EMBEDDING_DEVICE,
                "Knowledge-graph BGE embedding model",
            )
            self._shared_model = get_shared_sentence_transformer(self._model_name, device)
            self._model = self._shared_model.model
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def _inference_lock(self):
        _ = self.model
        assert self._shared_model is not None
        return self._shared_model.inference_lock

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            all_embeddings: list[np.ndarray] = []
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                with self._inference_lock:
                    emb = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=self._BATCH_SIZE,
                        show_progress_bar=False,
                    )
                all_embeddings.append(emb)
            return np.vstack(all_embeddings).astype(np.float32)

    def get_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Force model load to detect dimension
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            return self.model.get_sentence_embedding_dimension()

    def unload(self) -> bool:
        had_model = self._model is not None or self._shared_model is not None
        self._model = None
        self._shared_model = None
        return had_model
