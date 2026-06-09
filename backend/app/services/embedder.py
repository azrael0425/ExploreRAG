"""
Embedding Service
=================
Generates vector embeddings using Qwen or sentence-transformers.

The Qwen development profile uses ``text-embedding-v4`` through DashScope,
avoiding local model downloads. It remains configurable for local
sentence-transformers use.
"""
from __future__ import annotations

import logging
import re
from typing import Sequence, Optional

from app.core.config import settings
from app.core.accelerator import require_cuda
from app.services.sentence_transformer_registry import (
    SharedSentenceTransformer,
    get_shared_sentence_transformer,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    """

    # Dimension lookup for common models (used before model is loaded)
    _KNOWN_DIMS = {
        "BAAI/bge-m3": 1024,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "intfloat/multilingual-e5-large-instruct": 1024,
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EXPLORERAG_EMBEDDING_MODEL
        self._model = None
        self._shared_model: SharedSentenceTransformer | None = None
        self._provider = None

    @property
    def provider(self):
        """Lazy-create the configured remote embedding provider when selected."""
        provider_name = settings.EXPLORERAG_EMBEDDING_PROVIDER.lower()
        if provider_name != "qwen":
            return None
        if self._provider is None:
            from app.services.llm.qwen import QwenEmbeddingProvider

            if not settings.DASHSCOPE_API_KEY:
                raise ValueError(
                    "DASHSCOPE_API_KEY is required when EXPLORERAG_EMBEDDING_PROVIDER=qwen"
                )
            self._provider = QwenEmbeddingProvider(
                api_key=settings.DASHSCOPE_API_KEY,
                base_url=settings.DASHSCOPE_BASE_URL,
                model=self.model_name,
                dimension=settings.KG_EMBEDDING_DIMENSION,
            )
        return self._provider

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            device = require_cuda(
                settings.EXPLORERAG_EMBEDDING_DEVICE,
                "Primary BGE embedding model",
            )
            self._shared_model = get_shared_sentence_transformer(self.model_name, device)
            self._model = self._shared_model.model
        return self._model

    @property
    def _inference_lock(self):
        # Accessing ``model`` first also initializes the shared registry entry.
        _ = self.model
        assert self._shared_model is not None
        return self._shared_model.inference_lock

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        if self.provider is not None:
            return self.provider.get_dimension()
        if self._model is not None:
            return self._model.get_sentence_embedding_dimension()
        return self._KNOWN_DIMS.get(self.model_name, 1024)

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        if self.provider is not None:
            return self.provider.embed_sync([text])[0].tolist()
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            with self._inference_lock:
                embedding = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
        return embedding.tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        if self.provider is not None:
            return self.provider.embed_sync(valid_texts).tolist()
        prepared_texts = [self._truncate_for_indexing(text) for text in valid_texts]
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            with self._inference_lock:
                embeddings = self.model.encode(
                    prepared_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=settings.EXPLORERAG_EMBEDDING_BATCH_SIZE,
                    show_progress_bar=False,
                )
        return embeddings.tolist()

    def count_tokens(self, text: str) -> int:
        """Count model tokens without applying the model's 8192-token limit."""
        if not text:
            return 0
        if self.provider is not None:
            # Remote embeddings are not used by the production profile. Keep a
            # conservative fallback so indexing safeguards still work.
            cjk = len(re.findall(r"[\u3400-\u9fff]", text))
            return max(1, cjk + max(0, len(text) - cjk) // 4)
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def split_text_for_indexing(
        self,
        text: str,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Split enriched text into lossless, embedding-safe windows.

        Consecutive Markdown table rows are handled separately and their header
        is repeated in every window. This avoids the old behavior where one
        large table made an entire batch pad to 1000+ tokens or where a hard
        truncation silently discarded later rows.
        """
        limit = max_tokens or settings.EXPLORERAG_EMBEDDING_MAX_TOKENS
        if not text.strip():
            return []
        if self.count_tokens(text) <= limit:
            return [text.strip()]

        lines = text.splitlines()
        segments: list[tuple[str, str, str]] = []
        text_buffer: list[str] = []
        index = 0

        def flush_text_buffer() -> None:
            if text_buffer:
                content = "\n".join(text_buffer).strip()
                if content:
                    segments.append(("text", "", content))
                text_buffer.clear()

        while index < len(lines):
            if lines[index].lstrip().startswith("|"):
                prefix = ""
                if text_buffer and text_buffer[-1].strip().startswith(("[Table", "> **Table")):
                    prefix = text_buffer.pop().strip()
                flush_text_buffer()
                table_lines: list[str] = []
                while index < len(lines) and lines[index].lstrip().startswith("|"):
                    table_lines.append(lines[index].strip())
                    index += 1
                segments.append(("table", prefix, "\n".join(table_lines)))
                continue
            text_buffer.append(lines[index])
            index += 1
        flush_text_buffer()

        parts: list[str] = []
        for kind, prefix, content in segments:
            if kind == "table":
                parts.extend(self._split_markdown_table(prefix, content, limit))
            else:
                parts.extend(self._token_windows(content, limit))
        return [part for part in parts if part.strip()]

    def _split_markdown_table(self, prefix: str, table: str, limit: int) -> list[str]:
        rows = [line for line in table.splitlines() if line.strip()]
        if not rows:
            return self._token_windows(prefix, limit)

        header_size = 2 if len(rows) > 1 and "---" in rows[1] else 1
        header = rows[:header_size]
        data_rows = rows[header_size:]
        fixed_lines = ([prefix] if prefix else []) + header
        fixed = "\n".join(fixed_lines).strip()

        if self.count_tokens("\n".join(fixed_lines + data_rows)) <= limit:
            return ["\n".join(fixed_lines + data_rows).strip()]

        result: list[str] = []
        current = list(fixed_lines)
        for row in data_rows:
            candidate = "\n".join(current + [row]).strip()
            if self.count_tokens(candidate) <= limit:
                current.append(row)
                continue
            if len(current) > len(fixed_lines):
                result.append("\n".join(current).strip())
                current = list(fixed_lines)
            single = "\n".join(current + [row]).strip()
            if self.count_tokens(single) <= limit:
                current.append(row)
                continue
            # Extremely wide rows are losslessly token-windowed. The table
            # header is prepended when it leaves enough room.
            remaining = max(1, limit - self.count_tokens(fixed) - (1 if fixed else 0))
            row_parts = self._token_windows(row, remaining, overlap=0)
            for row_part in row_parts:
                combined = f"{fixed}\n{row_part}".strip() if fixed else row_part
                result.extend(self._token_windows(combined, limit, overlap=0))
            current = list(fixed_lines)

        if len(current) > len(fixed_lines):
            result.append("\n".join(current).strip())
        return result or self._token_windows(table, limit)

    def _token_windows(
        self,
        text: str,
        limit: int,
        overlap: int | None = None,
    ) -> list[str]:
        if not text.strip():
            return []
        if self.count_tokens(text) <= limit:
            return [text.strip()]
        if self.provider is not None:
            # Approximate fallback for the unused remote-embedding profile.
            max_chars = max(64, limit * 3)
            return [text[start:start + max_chars].strip() for start in range(0, len(text), max_chars)]

        token_overlap = (
            settings.EXPLORERAG_EMBEDDING_TOKEN_OVERLAP
            if overlap is None else overlap
        )
        token_overlap = min(token_overlap, max(0, limit - 1))
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            tokenizer = self.model.tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=False)
            step = max(1, limit - token_overlap)
            return [
                tokenizer.decode(tokens[start:start + limit], skip_special_tokens=True).strip()
                for start in range(0, len(tokens), step)
                if tokens[start:start + limit]
            ]

    def _truncate_for_indexing(self, text: str) -> str:
        """Final safety net; normal callers split before reaching this point."""
        limit = settings.EXPLORERAG_EMBEDDING_MAX_TOKENS
        if self.count_tokens(text) <= limit or self.provider is not None:
            return text
        from app.services.model_runtime import retrieval_model_guard

        with retrieval_model_guard():
            tokenizer = self.model.tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(tokens[:limit], skip_special_tokens=True)
        logger.warning(
            "Truncating unsplit embedding input from %s to %s tokens; caller should split it first",
            len(tokens),
            limit,
        )
        return decoded

    def unload(self) -> bool:
        """Release this service's references before the local VLM uses CUDA."""
        had_model = self._model is not None or self._shared_model is not None
        self._model = None
        self._shared_model = None
        return had_model

    def warmup(self) -> None:
        """Load the shared model and initialize one small inference pass."""
        self.embed_query("ExploreRAG embedding warmup")

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed_text(query)


# Default service instance (singleton)
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the default embedding service."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def release_embedding_service() -> bool:
    """Release primary embedding and shared-registry CUDA references."""
    released = _default_service.unload() if _default_service is not None else False
    from app.services.sentence_transformer_registry import release_sentence_transformer_registry

    return release_sentence_transformer_registry() or released


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts)
