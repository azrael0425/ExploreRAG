"""
ExploreRAG Data Models
===================

Dataclasses for the ExploreRAG pipeline: document parsing, enriched chunks,
citations, and retrieval results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedImage:
    """An image extracted from a document by Docling."""
    image_id: str
    document_id: int
    page_no: int
    file_path: str
    caption: str = ""
    width: int = 0
    height: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    mime_type: str = "image/png"


@dataclass
class ExtractedTable:
    """A table extracted from a document by Docling."""
    table_id: str
    document_id: int
    page_no: int
    content_markdown: str  # table.export_to_markdown(doc)
    caption: str = ""      # LLM-generated description
    num_rows: int = 0
    num_cols: int = 0


@dataclass
class EnrichedChunk:
    """A document chunk enriched with structural metadata."""
    content: str
    chunk_index: int
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)  # image_ids nearby
    table_refs: list[str] = field(default_factory=list)  # table_ids nearby
    has_table: bool = False
    has_code: bool = False
    contextualized: str = ""  # heading_path joined for context
    # Retrieval scores are kept separate because Chroma cosine similarity and
    # cross-encoder relevance have different scales.  They are diagnostics,
    # never interchangeable ranking signals.
    vector_distance: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None
    # Exact graph labels found in this retrieved passage.  Keeping this on the
    # chunk lets every API surface expose the same citation-navigation target.
    graph_entity_names: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Result of parsing a document with Docling."""
    document_id: int
    original_filename: str
    markdown: str
    page_count: int
    chunks: list[EnrichedChunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    tables_count: int = 0


@dataclass
class Citation:
    """A source citation pointing to a specific location in a document."""
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Format citation as a human-readable string."""
        parts = [self.source_file]
        if self.page_no > 0:
            parts.append(f"p.{self.page_no}")
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        return " | ".join(parts)


@dataclass
class RetrievalTimings:
    """Server-side timings for one deep-retrieval request, in milliseconds."""
    vector_ms: int = 0
    graph_ms: int | None = None
    rerank_ms: int = 0
    context_ms: int = 0
    total_ms: int = 0


@dataclass
class KnowledgeGraphFact:
    """One independently citable entity or relationship from LightRAG."""
    content: str = ""
    entity_names: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_document_ids: list[int] = field(default_factory=list)


@dataclass
class KnowledgeGraphEvidence:
    """Structured, query-time evidence retrieved from LightRAG.

    ``content`` is deliberately evidence only: it is assembled from LightRAG's
    stored entities, relationships, and retrieved chunks, never from a
    LightRAG-generated answer.
    """
    content: str = ""
    entity_names: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    # Document ids parsed from LightRAG's ``source_id`` values.  Keep these
    # alongside the display filenames so a graph citation can be narrowed to
    # the records that actually supplied its evidence.
    source_document_ids: list[int] = field(default_factory=list)
    facts: list[KnowledgeGraphFact] = field(default_factory=list)
    entity_count: int = 0
    relationship_count: int = 0
    chunk_count: int = 0


@dataclass
class DeepRetrievalResult:
    """Result of a deep RAG query with citations and KG insights."""
    chunks: list[EnrichedChunk]
    citations: list[Citation]
    context: str  # assembled context for LLM
    query: str
    mode: str = "hybrid"
    knowledge_graph_summary: str = ""
    knowledge_graph_evidence: KnowledgeGraphEvidence = field(
        default_factory=KnowledgeGraphEvidence
    )
    image_refs: list[ExtractedImage] = field(default_factory=list)
    table_refs: list[ExtractedTable] = field(default_factory=list)
    timings: RetrievalTimings = field(default_factory=RetrievalTimings)
    # Serializable request trace used by offline evaluation.  It records both
    # requested and effective component state, pre/post rankings and fallbacks.
    trace: dict = field(default_factory=dict)
