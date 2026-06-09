"""Post-enrichment chunk normalization for embedding-safe indexing."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from app.services.embedder import EmbeddingService
from app.services.models.parsed_document import EnrichedChunk

logger = logging.getLogger(__name__)


def split_enriched_chunks(
    chunks: list[EnrichedChunk],
    embedder: EmbeddingService,
) -> tuple[list[EnrichedChunk], dict[str, int]]:
    """Split enriched chunks while retaining all retrieval metadata."""
    output: list[EnrichedChunk] = []
    oversized = 0
    max_before = 0
    max_after = 0

    for chunk in chunks:
        before_tokens = embedder.count_tokens(chunk.content)
        max_before = max(max_before, before_tokens)
        parts = embedder.split_text_for_indexing(chunk.content)
        if len(parts) > 1:
            oversized += 1
        for part in parts:
            max_after = max(max_after, embedder.count_tokens(part))
            output.append(replace(chunk, content=part, chunk_index=len(output)))

    stats = {
        "input": len(chunks),
        "output": len(output),
        "oversized": oversized,
        "max_tokens_before": max_before,
        "max_tokens_after": max_after,
    }
    if oversized:
        logger.info(
            "Embedding chunk normalization: %s->%s chunks oversized=%s max_tokens=%s->%s",
            stats["input"],
            stats["output"],
            oversized,
            max_before,
            max_after,
        )
    return output, stats


def split_attachment_chunks(
    chunks: list[dict[str, Any]],
    embedder: EmbeddingService,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize serialized temporary-attachment chunks before embedding."""
    output: list[dict[str, Any]] = []
    oversized = 0
    max_before = 0
    max_after = 0

    for chunk in chunks:
        content = str(chunk.get("content", ""))
        before_tokens = embedder.count_tokens(content)
        max_before = max(max_before, before_tokens)
        parts = embedder.split_text_for_indexing(content)
        if len(parts) > 1:
            oversized += 1
        for part in parts:
            normalized = dict(chunk)
            normalized["content"] = part
            normalized["chunk_index"] = len(output)
            output.append(normalized)
            max_after = max(max_after, embedder.count_tokens(part))

    return output, {
        "input": len(chunks),
        "output": len(output),
        "oversized": oversized,
        "max_tokens_before": max_before,
        "max_tokens_after": max_after,
    }
