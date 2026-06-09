"""
Reranker Service
================
Cross-encoder reranker for improving retrieval precision.

Default model: BAAI/bge-reranker-v2-m3 (multilingual, 100+ languages).
Configurable via EXPLORERAG_RERANKER_MODEL in settings.

Usage:
    reranker = get_reranker_service()
    ranked = reranker.rerank("user question", ["chunk1", "chunk2", ...], top_k=5)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from app.core.config import settings
from app.core.accelerator import require_cuda

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranked item with its original index and relevance score."""
    index: int          # Original position in the input list
    score: float        # Cross-encoder relevance score (higher = more relevant)
    text: str           # The chunk text


class RerankerService:
    """
    Cross-encoder reranker service.
    Scores (query, document) pairs jointly through a transformer,
    producing far more accurate relevance scores than bi-encoder cosine similarity.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EXPLORERAG_RERANKER_MODEL
        self._model = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._circuit_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._half_open_probe = False

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            import torch
            from sentence_transformers import CrossEncoder

            device = require_cuda(settings.EXPLORERAG_RERANKER_DEVICE, "BGE reranker model")
            dtype_name = settings.EXPLORERAG_RERANKER_DTYPE.lower()
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            if dtype_name not in dtype_map:
                raise ValueError(f"Unsupported EXPLORERAG_RERANKER_DTYPE: {dtype_name}")

            model_path = Path(self.model_name)
            logger.info(
                "Loading reranker model on %s with dtype=%s max_length=%s: %s",
                device,
                dtype_name,
                settings.EXPLORERAG_RERANKER_MAX_LENGTH,
                self.model_name,
            )
            self._model = CrossEncoder(
                self.model_name,
                device=device,
                local_files_only=model_path.is_absolute(),
                model_kwargs={"dtype": dtype_map[dtype_name]},
                max_length=settings.EXPLORERAG_RERANKER_MAX_LENGTH,
            )
            logger.info("Reranker model loaded on %s: %s", device, self.model_name)
        return self._model

    def is_available(self) -> bool:
        """Return whether the circuit permits a rerank attempt.

        Only one request is admitted after the recovery window. Other requests
        continue to use vector order until that half-open probe succeeds.
        """
        with self._circuit_lock:
            if self._circuit_open_until == 0:
                return True
            now = time.monotonic()
            if now < self._circuit_open_until:
                return False
            if self._half_open_probe:
                return False
            self._half_open_probe = True
            return True

    def record_success(self) -> None:
        with self._circuit_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
            self._half_open_probe = False

    def record_failure(self, reason: str) -> None:
        with self._circuit_lock:
            self._consecutive_failures += 1
            self._half_open_probe = False
            if self._consecutive_failures >= settings.EXPLORERAG_RERANKER_CIRCUIT_BREAKER_FAILURES:
                self._circuit_open_until = (
                    time.monotonic()
                    + settings.EXPLORERAG_RERANKER_CIRCUIT_BREAKER_RECOVERY_SECONDS
                )
                logger.warning(
                    "Reranker circuit opened for %.1fs after %s consecutive failures: %s",
                    settings.EXPLORERAG_RERANKER_CIRCUIT_BREAKER_RECOVERY_SECONDS,
                    self._consecutive_failures,
                    reason,
                )

    def warmup(self) -> None:
        """Load weights and initialize the CUDA inference path."""
        self.rerank(
            "ExploreRAG warmup",
            ["ExploreRAG warmup document"],
            top_k=1,
            min_score=None,
        )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RerankResult]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = all)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of RerankResult sorted by score (descending),
            filtered by top_k and min_score.
        """
        if not documents:
            return []

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, doc) for doc in documents]

        # A full candidate list uses the 640-token tier; smaller lists retain
        # the 768-token ceiling. The scheduler normally serializes reranking,
        # and the lock also keeps direct callers from racing this model option.
        configured_max = settings.EXPLORERAG_RERANKER_MAX_LENGTH
        effective_max = configured_max
        if len(documents) >= settings.EXPLORERAG_RERANKER_LONG_LIST_THRESHOLD:
            effective_max = min(
                configured_max,
                settings.EXPLORERAG_RERANKER_LONG_LIST_MAX_LENGTH,
            )

        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            model = self.model
            with self._inference_lock:
                previous_max = getattr(model, "max_length", configured_max)
                model.max_length = effective_max
                try:
                    scores = model.predict(
                        pairs,
                        batch_size=settings.EXPLORERAG_RERANKER_BATCH_SIZE,
                        show_progress_bar=False,
                    ).tolist()
                finally:
                    model.max_length = previous_max

        logger.debug(
            "Reranked %s candidates with max_length=%s batch_size=%s",
            len(documents),
            effective_max,
            settings.EXPLORERAG_RERANKER_BATCH_SIZE,
        )

        # Build results with original indices
        results = [
            RerankResult(index=i, score=s, text=doc)
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending (most relevant first)
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply min_score filter
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]

        return results

    def unload(self) -> bool:
        """Release cross-encoder CUDA weights before local LLM inference."""
        with self._model_lock:
            had_model = self._model is not None
            self._model = None
        return had_model


# Singleton instance
_default_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create the default reranker service."""
    global _default_service
    if _default_service is None:
        _default_service = RerankerService()
    return _default_service


def release_reranker_service() -> bool:
    return _default_service.unload() if _default_service is not None else False
