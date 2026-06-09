"""
RAG-related Pydantic schemas for request/response validation.
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from app.schemas.metadata import MetadataFilter


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    document_ids: list[int] | None = Field(default=None, description="Filter to specific document IDs")
    metadata_filter: MetadataFilter | None = Field(default=None, description="Optional business-metadata filter")
    mode: Literal["hybrid", "vector_only", "local", "global"] = Field(
        default="hybrid",
        description="Search mode: hybrid (default), vector_only, local, or global"
    )


class CitationResponse(BaseModel):
    """A source citation."""
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = []
    formatted: str = ""


class RetrievedChunkResponse(BaseModel):
    """Response schema for a single retrieved chunk."""
    content: str
    chunk_id: str
    score: float
    metadata: dict
    citation: CitationResponse | None = None

    model_config = {"from_attributes": True}


class DocumentImageResponse(BaseModel):
    """Response schema for a document image."""
    image_id: str
    document_id: int
    page_no: int
    caption: str = ""
    width: int = 0
    height: int = 0
    url: str = ""


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""
    query: str
    chunks: list[RetrievedChunkResponse]
    context: str
    total_chunks: int
    knowledge_graph_summary: str = ""
    citations: list[CitationResponse] = []
    image_refs: list[DocumentImageResponse] = []


class DocumentProcessRequest(BaseModel):
    """Request schema for document processing."""
    document_id: int


class DocumentProcessResponse(BaseModel):
    """Response schema for document processing."""
    document_id: int
    status: str
    chunk_count: int
    message: str


class BatchProcessRequest(BaseModel):
    """Request schema for batch document processing."""
    document_ids: list[int] = Field(..., min_length=1, description="List of document IDs to process")


class ProjectRAGStatsResponse(BaseModel):
    """Response schema for workspace RAG statistics."""
    workspace_id: int
    total_documents: int
    indexed_documents: int
    total_chunks: int
    image_count: int = 0
    explorerag_documents: int = 0


# ---------------------------------------------------------------------------
# Knowledge Graph schemas
# ---------------------------------------------------------------------------

class KGEntityResponse(BaseModel):
    """A knowledge graph entity (node)."""
    name: str
    entity_type: str = "Unknown"
    description: str = ""
    degree: int = 0  # number of relationships


class KGRelationshipResponse(BaseModel):
    """A knowledge graph relationship (edge)."""
    source: str
    target: str
    description: str = ""
    keywords: str = ""
    weight: float = 1.0


class KGGraphNodeResponse(BaseModel):
    """Node in the graph visualization payload."""
    id: str
    label: str
    entity_type: str = "Unknown"
    description: str = ""
    degree: int = 0
    source_document_ids: list[int] = []
    source_files: list[str] = []


class KGGraphEdgeResponse(BaseModel):
    """Edge in the graph visualization payload."""
    source: str
    target: str
    label: str = ""
    weight: float = 1.0
    source_document_ids: list[int] = []
    source_files: list[str] = []


class KGGraphResponse(BaseModel):
    """Full graph export for frontend visualization."""
    nodes: list[KGGraphNodeResponse] = []
    edges: list[KGGraphEdgeResponse] = []
    is_truncated: bool = False


class KGFocusRequest(BaseModel):
    """Request a small graph that always keeps citation seed entities."""
    entity_names: list[str] = Field(..., min_length=1, max_length=50)
    document_ids: list[int] = Field(default_factory=list, max_length=50)
    max_depth: int = Field(default=1, ge=0, le=3)
    max_nodes: int = Field(default=80, ge=10, le=300)


class KGFocusResponse(KGGraphResponse):
    """Citation-focused graph plus explicit entity resolution diagnostics."""
    requested_entities: list[str] = Field(default_factory=list)
    matched_entities: list[str] = Field(default_factory=list)
    missing_entities: list[str] = Field(default_factory=list)
    selection_mode: Literal["citation_focus"] = "citation_focus"


# ---------------------------------------------------------------------------
# Chat schemas
# ---------------------------------------------------------------------------

class ChatMessageSchema(BaseModel):
    """A single chat message in conversation history."""
    role: str = Field(..., description="user or assistant")
    content: str


class ChatRequest(BaseModel):
    """Request for the chat endpoint."""
    message: str = Field(..., min_length=1, max_length=5000)
    history: list[ChatMessageSchema] = []
    document_ids: list[int] | None = None
    metadata_filter: MetadataFilter | None = None
    attachment_ids: list[str] = Field(default_factory=list, max_length=8)
    enable_thinking: bool = False


class ChatAttachmentResponse(BaseModel):
    attachment_id: str
    original_filename: str
    file_type: str
    file_size: int
    state: str
    parsed_token_count: int = 0
    error_message: str | None = None
    created_at: str = ""


class ChatSourceChunk(BaseModel):
    """A source chunk referenced in the chat answer."""
    index: str  # 4-char alphanumeric ID, e.g. "a3x9" (was: int)
    chunk_id: str

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v
    content: str
    document_id: int = 0
    attachment_id: str | None = None
    source_file: str = ""
    page_no: int = 0
    heading_path: list[str] = []
    score: float = 0.0
    source_type: str = "vector"  # "vector" | "kg"
    # Present only for LightRAG evidence. They make graph navigation specific
    # to this citation instead of falling back to the whole workspace graph.
    graph_entity_names: list[str] = []
    graph_document_ids: list[int] = []


class ChatImageRef(BaseModel):
    """An image referenced in the chat answer."""
    ref_id: str | None = None  # 4-char alphanumeric ID, e.g. "p4f2"
    image_id: str
    document_id: int = 0
    attachment_id: str | None = None
    page_no: int = 0
    caption: str = ""
    url: str = ""
    width: int = 0
    height: int = 0


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    answer: str
    sources: list[ChatSourceChunk] = []
    related_entities: list[str] = []
    kg_summary: str | None = None
    image_refs: list[ChatImageRef] = []
    thinking: str | None = None


class PersistedChatMessage(BaseModel):
    """A persisted chat message from the database."""
    id: int
    message_id: str
    role: str
    content: str
    sources: list[ChatSourceChunk] | None = None
    related_entities: list[str] | None = None
    image_refs: list[ChatImageRef] | None = None
    thinking: str | None = None
    agent_steps: list | None = None
    reply_to_message_id: str | None = None
    feedback_rating: int | None = None
    feedback_comment: str | None = None
    source_ratings: dict | None = None
    attachments: list[ChatAttachmentResponse] | None = None
    created_at: str  # ISO format

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Response for GET chat history."""
    workspace_id: int
    messages: list[PersistedChatMessage]
    total: int


class LLMCapabilitiesResponse(BaseModel):
    """Response for LLM capabilities check."""
    provider: str
    model: str
    supports_thinking: bool
    supports_vision: bool
    thinking_default: bool = True


# ---------------------------------------------------------------------------
# Debug / QA schemas
# ---------------------------------------------------------------------------

class DebugRetrievedSource(BaseModel):
    """A retrieved source for debug inspection."""
    index: str  # 4-char alphanumeric ID (was: int)
    document_id: int

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v
    page_no: int
    heading_path: list[str] = []
    source_file: str = ""
    content_preview: str = ""  # first 500 chars
    score: float = 0.0
    source_type: str = "vector"


class DebugChatResponse(BaseModel):
    """Full debug response — retrieval + LLM answer for quality inspection."""
    # Query
    question: str
    workspace_id: int

    # Retrieval
    retrieved_sources: list[DebugRetrievedSource] = []
    kg_summary: str = ""
    total_sources: int = 0

    # LLM
    system_prompt: str = ""
    answer: str = ""
    thinking: str | None = None

    # Images
    image_count: int = 0

    # Meta
    provider: str = ""
    model: str = ""
