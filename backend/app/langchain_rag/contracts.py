"""Strongly typed inputs and outputs for the LangChain orchestration layer."""
from __future__ import annotations

from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class RetrievalInput(BaseModel):
    """All domain options needed to run one durable-KB retrieval."""

    workspace_id: int
    question: str
    top_k: int = Field(default=4, ge=1, le=100)
    mode: Literal["hybrid", "vector_only", "local", "global"] = "hybrid"
    document_ids: list[int] | None = None
    include_images: bool = True
    metadata_filter: dict[str, Any] | None = None
    enable_reranker: bool | None = None
    enable_knowledge_graph: bool | None = None
    prefetch_k: int | None = Field(default=None, ge=1, le=200)


class RetrievalEnvelope(BaseModel):
    """Structured retrieval output that preserves product-specific side data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: list[Document] = Field(default_factory=list)
    context: str = ""
    knowledge_graph_summary: str = ""
    knowledge_graph_entity_names: list[str] = Field(default_factory=list)
    knowledge_graph_document_ids: list[int] = Field(default_factory=list)
    knowledge_graph_facts: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Any] = Field(default_factory=list)
    image_refs: list[Any] = Field(default_factory=list)
    table_refs: list[Any] = Field(default_factory=list)
    timings: Any | None = None
    raw_result: Any | None = None


class ChatChainInput(BaseModel):
    """Request-scoped input for the streaming chat application service."""

    workspace_id: int
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    attachment_ids: list[str] = Field(default_factory=list)
    enable_thinking: bool = False
    llm_mode: Literal["cloud", "local"] = "cloud"
    lightrag_augmentation_enabled: bool = False
    system_prompt: str = ""
    top_k: int = Field(default=4, ge=1, le=100)
    retrieval_mode: Literal["hybrid", "vector_only", "local", "global"] = "hybrid"
    document_ids: list[int] | None = None
    metadata_filter: dict[str, Any] | None = None
    enable_reranker: bool | None = None
    enable_knowledge_graph: bool | None = None
    prefetch_k: int | None = Field(default=None, ge=1, le=200)


class GenerationEnvelope(BaseModel):
    """Final structured output for a chat completion."""

    answer: str = ""
    thinking: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    image_refs: list[dict[str, Any]] = Field(default_factory=list)
    related_entities: list[str] = Field(default_factory=list)
    performance: dict[str, Any] = Field(default_factory=dict)
