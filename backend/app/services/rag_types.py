"""Shared value objects returned by the ExploreRAG retrieval service."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A retrieved chunk and its relevance score."""

    content: str
    metadata: dict
    score: float
    chunk_id: str


@dataclass
class RAGQueryResult:
    """Compatibility result for direct vector queries."""

    chunks: list[RetrievedChunk]
    context: str
    query: str
