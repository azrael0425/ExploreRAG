"""
Deep Retriever
===============

Hybrid retrieval combining Knowledge Graph (LightRAG) + Vector Search (ChromaDB)
+ Cross-encoder Reranking (bge-reranker-v2-m3).

Pipeline:
  1. KG query  (parallel) → entity/relationship summary
  2. Vector search → over-fetch top-N candidates (EXPLORERAG_KB_PREFETCH)
  3. Cross-encoder rerank → precision filter to top-K (EXPLORERAG_KB_RERANK_TOP_K)
  4. Merge with citations + optional image references
"""
from __future__ import annotations

import asyncio
import logging
import time
import unicodedata
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable, DocumentStatus
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.reranker import RerankerService, get_reranker_service
from app.services.models.parsed_document import (
    Citation,
    DeepRetrievalResult,
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
    KnowledgeGraphEvidence,
    RetrievalTimings,
)
from app.services.work_scheduler import WorkPriority, get_work_scheduler

logger = logging.getLogger(__name__)


class DeepRetriever:
    """
    Hybrid retriever: KG traversal + vector similarity + cross-encoder reranking.
    """

    def __init__(
        self,
        workspace_id: int,
        kg_service: Optional[KnowledgeGraphService],
        vector_store: VectorStore,
        embedder: EmbeddingService,
        db: Optional[AsyncSession] = None,
        reranker: Optional[RerankerService] = None,
    ):
        self.workspace_id = workspace_id
        self.kg_service = kg_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.db = db
        self.reranker = reranker
        self.scheduler = get_work_scheduler()

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        include_images: bool = True,
        metadata_filter: dict | None = None,
        enable_reranker: bool | None = None,
        enable_knowledge_graph: bool | None = None,
        prefetch_k: int | None = None,
    ) -> DeepRetrievalResult:
        """
        Execute hybrid retrieval with reranking.

        Flow:
          1. [parallel] KG query + Vector over-fetch (EXPLORERAG_KB_PREFETCH)
          2. Cross-encoder rerank vector results → final top_k
          3. Optionally find related images from chunk pages
          4. Assemble structured context for LLM

        Args:
            question: Natural language query
            mode: "hybrid" (default) or "vector_only"
            top_k: Number of final chunks to return (after reranking)
            document_ids: Optional filter to specific documents
            include_images: Whether to find related images

        Returns:
            DeepRetrievalResult with chunks, citations, context, and optional images
        """
        started_at = time.perf_counter()
        timings = RetrievalTimings()
        requested_mode = mode
        reranker_requested = (
            settings.EXPLORERAG_ENABLE_RERANKER
            if enable_reranker is None
            else enable_reranker
        )
        graph_requested = (
            mode != "vector_only"
            if enable_knowledge_graph is None
            else enable_knowledge_graph
        )
        trace: dict = {
            "requested_mode": requested_mode,
            "effective_mode": mode,
            "top_k": top_k,
            "prefetch_k": None,
            "reranker": {
                "requested": bool(reranker_requested),
                "executed": False,
                "applied": False,
                "status": "pending",
                "model": settings.EXPLORERAG_RERANKER_MODEL,
            },
            "knowledge_graph": {
                "requested": bool(graph_requested),
                "executed": False,
                "applied": False,
                "status": "pending",
                "mode": mode,
            },
            "pre_rerank_candidates": [],
            "final_candidates": [],
        }

        # ``[]`` means a caller deliberately resolved a scope with no eligible
        # documents.  Treating it as a falsy optional filter would accidentally
        # expand the search to the complete knowledge base.
        if document_ids is not None and not document_ids:
            return DeepRetrievalResult(
                chunks=[], citations=[], context="", query=question,
                mode="vector_only", timings=timings,
                trace={
                    **trace,
                    "effective_mode": "vector_only",
                    "reranker": {**trace["reranker"], "status": "no_candidates"},
                    "knowledge_graph": {**trace["knowledge_graph"], "status": "scoped_empty"},
                },
            )

        # LightRAG's graph query is workspace-wide in this integration.  Until
        # graph source provenance is filtered by document ID, scoped retrieval
        # must not include graph context from unrelated documents.
        if document_ids is not None or metadata_filter:
            mode = "vector_only"
        trace["effective_mode"] = mode
        trace["knowledge_graph"]["mode"] = mode

        # Run KG and vector search in parallel. Timing starts before a job is
        # queued so queue contention is visible to the operator as well.
        kg_task = None
        graph_enabled = bool(
            graph_requested
            and self.kg_service is not None
            and mode != "vector_only"
        )
        if graph_enabled:
            async def run_kg() -> tuple[KnowledgeGraphEvidence, int]:
                stage_started_at = time.perf_counter()
                evidence = await self._kg_query(question, mode)
                return evidence, int((time.perf_counter() - stage_started_at) * 1000)

            kg_task = asyncio.create_task(
                run_kg()
            )

        # For an unrestricted knowledge-base question, retain a small candidate
        # set from every ready document. A global top-N alone is often filled
        # by adjacent chunks from one long document, which defeats a request
        # to compare or synthesise across the knowledge base.
        diversify_documents = document_ids is None and metadata_filter is None
        diversity_document_ids = (
            await self._ready_document_ids() if diversify_documents else []
        )

        # Over-fetch from vector DB for reranking
        effective_prefetch_k = max(
            prefetch_k or settings.EXPLORERAG_KB_PREFETCH,
            top_k,
        )
        trace["prefetch_k"] = effective_prefetch_k
        async def run_vector() -> tuple[tuple[list[EnrichedChunk], list[Citation]], int]:
            stage_started_at = time.perf_counter()
            result = await self.scheduler.run(
                "embedding",
                WorkPriority.CHAT,
                lambda: asyncio.to_thread(
                    self._vector_query,
                    question,
                    effective_prefetch_k,
                    document_ids,
                    metadata_filter,
                    diversity_document_ids,
                ),
                workspace_id=self.workspace_id,
            )
            return result, int((time.perf_counter() - stage_started_at) * 1000)

        vector_task = asyncio.create_task(
            run_vector()
        )

        # Await results
        kg_evidence = KnowledgeGraphEvidence()
        if kg_task:
            try:
                raw_evidence, timings.graph_ms = await kg_task
                # Keep older test doubles and third-party callers that return
                # a string source-compatible while the service moves to the
                # structured LightRAG data contract.
                kg_evidence = (
                    raw_evidence
                    if isinstance(raw_evidence, KnowledgeGraphEvidence)
                    else KnowledgeGraphEvidence(content=str(raw_evidence or ""))
                )
                trace["knowledge_graph"].update({
                    "executed": True,
                    "applied": bool(kg_evidence.content or kg_evidence.facts),
                    "status": "ok" if kg_evidence.content or kg_evidence.facts else "empty",
                    "entity_count": kg_evidence.entity_count,
                    "relationship_count": kg_evidence.relationship_count,
                    "chunk_count": kg_evidence.chunk_count,
                    "entity_names": list(kg_evidence.entity_names),
                    "source_document_ids": list(kg_evidence.source_document_ids),
                    "fact_count": len(kg_evidence.facts),
                    "fact_entity_names": [
                        list(fact.entity_names) for fact in kg_evidence.facts
                        if fact.entity_names
                    ],
                    "traceable_fact_count": sum(
                        1 for fact in kg_evidence.facts if fact.source_document_ids
                    ),
                })
            except Exception as e:
                logger.warning(f"KG query failed, continuing with vector only: {e}")
                timings.graph_ms = int((time.perf_counter() - started_at) * 1000)
                trace["knowledge_graph"].update({
                    "executed": False,
                    "applied": False,
                    "status": f"error:{type(e).__name__}",
                })
        else:
            if not graph_requested:
                graph_status = "disabled_by_request"
            elif mode == "vector_only":
                graph_status = "disabled_by_mode"
            elif self.kg_service is None:
                graph_status = "unavailable"
            else:
                graph_status = "disabled"
            trace["knowledge_graph"].update({"executed": False, "applied": False, "status": graph_status})

        (raw_chunks, raw_citations), timings.vector_ms = await vector_task
        trace["pre_rerank_candidates"] = self._candidate_trace(raw_chunks)

        # Rerank: cross-encoder scoring for precision
        rerank_started_at = time.perf_counter()
        chunks, citations = await self._rerank_with_budget(
            question,
            raw_chunks,
            raw_citations,
            len(raw_chunks) if diversify_documents else top_k,
            enabled=bool(reranker_requested),
            trace_state=trace["reranker"],
        )
        if diversify_documents:
            chunks, citations = self._select_document_diversity(chunks, citations, top_k)
        timings.rerank_ms = int((time.perf_counter() - rerank_started_at) * 1000)
        trace["final_candidates"] = self._candidate_trace(chunks)

        # Associate vector evidence with graph entities using the exact graph
        # labels returned by this same retrieval.  This avoids downloading a
        # truncated overview in the browser and guessing against unrelated
        # workspace entities.
        if kg_evidence.entity_names:
            for chunk in chunks:
                chunk.graph_entity_names = self._entities_in_text(
                    chunk.content,
                    kg_evidence.entity_names,
                )

        # Find related images and tables
        context_started_at = time.perf_counter()
        image_refs = []
        table_refs = []
        if include_images and self.db and chunks:
            page_nos = {(c.document_id, c.page_no) for c in chunks if c.page_no > 0}
            if page_nos:
                image_refs, table_refs = await asyncio.gather(
                    self._find_related_images(page_nos),
                    self._find_related_tables(page_nos),
                )

        # Assemble context
        context = self._assemble_context(
            chunks, citations, kg_evidence.content, image_refs, table_refs
        )
        timings.context_ms = int((time.perf_counter() - context_started_at) * 1000)
        timings.total_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "RAG retrieval timing workspace=%s mode=%s vector_ms=%s graph_ms=%s "
            "rerank_ms=%s context_ms=%s total_ms=%s kg_entities=%s kg_relationships=%s kg_chunks=%s",
            self.workspace_id,
            mode,
            timings.vector_ms,
            timings.graph_ms,
            timings.rerank_ms,
            timings.context_ms,
            timings.total_ms,
            kg_evidence.entity_count,
            kg_evidence.relationship_count,
            kg_evidence.chunk_count,
        )

        return DeepRetrievalResult(
            chunks=chunks,
            citations=citations,
            context=context,
            query=question,
            mode=mode,
            knowledge_graph_summary=kg_evidence.content,
            knowledge_graph_evidence=kg_evidence,
            image_refs=image_refs,
            table_refs=table_refs,
            timings=timings,
            trace=trace,
        )

    @staticmethod
    def _candidate_trace(chunks: list[EnrichedChunk]) -> list[dict]:
        """Return a compact, stable ranking snapshot for offline metrics."""
        return [
            {
                "rank": rank,
                "chunk_id": f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "source_file": chunk.source_file,
                "page_no": chunk.page_no,
                "vector_distance": chunk.vector_distance,
                "vector_score": chunk.vector_score,
                "rerank_score": chunk.rerank_score,
            }
            for rank, chunk in enumerate(chunks, start=1)
        ]

    @staticmethod
    def _entities_in_text(content: str, entity_names: list[str], limit: int = 12) -> list[str]:
        def normalize(value: str) -> str:
            normalized = unicodedata.normalize("NFKC", value).casefold()
            return "".join(character for character in normalized if character.isalnum())

        normalized_content = normalize(content)
        matches: list[str] = []
        seen: set[str] = set()
        for entity_name in entity_names:
            label = str(entity_name or "").strip()
            normalized_label = normalize(label)
            min_length = 2 if any(ord(character) > 127 for character in label) else 4
            if (
                label
                and len(normalized_label) >= min_length
                and normalized_label in normalized_content
                and normalized_label not in seen
            ):
                seen.add(normalized_label)
                matches.append(label)
                if len(matches) >= limit:
                    break
        return matches

    async def _kg_query(self, question: str, mode: str) -> KnowledgeGraphEvidence:
        """Get structured graph evidence without a LightRAG-generated answer."""
        if not self.kg_service:
            return KnowledgeGraphEvidence()
        try:
            return await asyncio.wait_for(
                self.kg_service.retrieve_evidence(question, mode=mode),
                timeout=settings.EXPLORERAG_KG_AUGMENTATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("KG evidence retrieval timed out")
            raise
        except Exception as e:
            logger.warning(f"KG evidence retrieval failed: {e}")
            raise

    def _vector_query(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[list[int]],
        metadata_filter: dict | None = None,
        diversity_document_ids: list[int] | None = None,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Synchronous vector search via ChromaDB (over-fetch stage)."""
        query_embedding = self.embedder.embed_query(question)

        # Only fully processed KB chunks are publicly retrievable.  Chroma
        # supports $and filters, so callers can still add document/custom
        # metadata filters without weakening the visibility requirement.
        filters: list[dict] = [{"visibility": "ready"}]
        if metadata_filter:
            filters.append(metadata_filter.copy())
        if document_ids:
            filters.append({"document_id": {"$in": document_ids}})
        where = filters[0] if len(filters) == 1 else {"$and": filters}

        chunks = []
        citations = []

        seen_chunk_ids: set[tuple[int, int]] = set()

        def add_results(results: dict) -> None:
            for i, doc_text in enumerate(results.get("documents", [])):
                meta = results["metadatas"][i] if results.get("metadatas") else {}
                document_id = int(meta.get("document_id", 0))
                chunk_index = int(meta.get("chunk_index", i))
                chunk_key = (document_id, chunk_index)
                if chunk_key in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_key)

                heading_path = []
                heading_str = meta.get("heading_path", "")
                if heading_str:
                    heading_path = heading_str.split(" > ") if isinstance(heading_str, str) else []

                image_refs = []
                image_ids_str = meta.get("image_ids", "")
                if image_ids_str and isinstance(image_ids_str, str):
                    image_refs = [iid for iid in image_ids_str.split("|") if iid]

                table_refs = []
                table_ids_str = meta.get("table_ids", "")
                if table_ids_str and isinstance(table_ids_str, str):
                    table_refs = [tid for tid in table_ids_str.split("|") if tid]

                distances = results.get("distances") or []
                vector_distance = (
                    float(distances[i]) if i < len(distances) and distances[i] is not None else None
                )
                chunks.append(EnrichedChunk(
                    content=doc_text,
                    chunk_index=chunk_index,
                    source_file=meta.get("source", ""),
                    document_id=document_id,
                    page_no=meta.get("page_no", 0),
                    heading_path=heading_path,
                    image_refs=image_refs,
                    table_refs=table_refs,
                    has_table=meta.get("has_table", False),
                    has_code=meta.get("has_code", False),
                    vector_distance=vector_distance,
                    vector_score=(1.0 - vector_distance) if vector_distance is not None else None,
                ))

                citations.append(Citation(
                    source_file=meta.get("source", "Unknown"),
                    document_id=document_id,
                    page_no=meta.get("page_no", 0),
                    heading_path=heading_path,
                ))

        add_results(self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        ))
        # Add the best two chunks from each document as candidates. Reranking
        # still decides relevance; this only prevents long documents from
        # crowding every other document out before reranking can see them.
        for document_id in diversity_document_ids or []:
            document_where = {"$and": [
                {"visibility": "ready"},
                {"document_id": document_id},
            ]}
            add_results(self.vector_store.query(
                query_embedding=query_embedding,
                n_results=2,
                where=document_where,
            ))

        return chunks, citations

    async def _ready_document_ids(self) -> list[int]:
        """Return indexed documents eligible for an unscoped diverse answer."""
        if self.db is None:
            return []
        result = await self.db.execute(
            select(Document.id).where(
                Document.workspace_id == self.workspace_id,
                Document.status == DocumentStatus.INDEXED,
            )
        )
        return [document_id for document_id in result.scalars().all()]

    @staticmethod
    def _select_document_diversity(
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Select the highest-ranked chunk from distinct documents first."""
        selected_chunks: list[EnrichedChunk] = []
        selected_citations: list[Citation] = []
        used_document_ids: set[int] = set()
        for chunk, citation in zip(chunks, citations):
            if chunk.document_id in used_document_ids:
                continue
            selected_chunks.append(chunk)
            selected_citations.append(citation)
            used_document_ids.add(chunk.document_id)
            if len(selected_chunks) >= top_k:
                return selected_chunks, selected_citations
        # If fewer distinct documents exist than requested results, preserve
        # relevance ordering while filling the remaining slots.
        selected_keys = {(chunk.document_id, chunk.chunk_index) for chunk in selected_chunks}
        for chunk, citation in zip(chunks, citations):
            if (chunk.document_id, chunk.chunk_index) in selected_keys:
                continue
            selected_chunks.append(chunk)
            selected_citations.append(citation)
            if len(selected_chunks) >= top_k:
                break
        return selected_chunks, selected_citations

    async def _rerank_with_budget(
        self,
        question: str,
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
        *,
        enabled: bool | None = None,
        trace_state: dict | None = None,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Rerank within a strict latency budget and fail open to vector order."""
        fallback = (chunks[:top_k], citations[:top_k])
        should_rerank = settings.EXPLORERAG_ENABLE_RERANKER if enabled is None else enabled
        if not chunks:
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": "no_candidates"})
            return fallback

        async def scheduled_rerank() -> tuple[list[EnrichedChunk], list[Citation]]:
            return await self.scheduler.run(
                "reranker",
                WorkPriority.CHAT,
                lambda: asyncio.to_thread(
                    self._rerank_chunks, question, chunks, citations, top_k
                ),
                workspace_id=self.workspace_id,
            )

        # Keep the disabled/no-model path behavior testable while avoiding a
        # timeout around the cheap vector-order slice.
        if not should_rerank:
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": "disabled_by_request"})
            return fallback
        if self.reranker is None:
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": "unavailable"})
            return fallback

        if not self.reranker.is_available():
            logger.warning(
                "Reranker circuit is open for workspace=%s; using vector order",
                self.workspace_id,
            )
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": "circuit_open"})
            return fallback

        try:
            result = await asyncio.wait_for(
                scheduled_rerank(),
                timeout=settings.EXPLORERAG_RERANKER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            self.reranker.record_failure("inference timeout")
            logger.warning(
                "Reranker exceeded %.1fs for workspace=%s candidates=%s; using vector order",
                settings.EXPLORERAG_RERANKER_TIMEOUT_SECONDS,
                self.workspace_id,
                len(chunks),
            )
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": "timeout"})
            return fallback
        except Exception as exc:
            self.reranker.record_failure(type(exc).__name__)
            logger.exception(
                "Reranker failed for workspace=%s candidates=%s; using vector order",
                self.workspace_id,
                len(chunks),
            )
            if trace_state is not None:
                trace_state.update({"executed": False, "applied": False, "status": f"error:{type(exc).__name__}"})
            return fallback

        self.reranker.record_success()
        if trace_state is not None:
            applied = any(chunk.rerank_score is not None for chunk in result[0])
            trace_state.update({
                "executed": True,
                "applied": applied,
                "status": "ok" if applied else "fallback_all_below_threshold",
            })
        return result

    def _rerank_chunks(
        self,
        question: str,
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """
        Cross-encoder reranking: score each (query, chunk) pair jointly,
        then filter by relevance threshold and return top_k.
        """
        if not chunks:
            return [], []

        if self.reranker is None:
            return chunks[:top_k], citations[:top_k]

        # Extract texts for reranking
        doc_texts = [c.content for c in chunks]

        reranked = self.reranker.rerank(
            query=question,
            documents=doc_texts,
            top_k=top_k,
            min_score=settings.EXPLORERAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            # Fallback: if reranker filtered everything, keep top 3 by original order
            logger.warning(
                f"Reranker filtered all {len(chunks)} chunks below threshold "
                f"{settings.EXPLORERAG_MIN_RELEVANCE_SCORE}, falling back to top 3"
            )
            return chunks[:min(3, len(chunks))], citations[:min(3, len(citations))]

        # Map reranked results back to original chunks/citations
        for result in reranked:
            chunks[result.index].rerank_score = float(result.score)
        reranked_chunks = [chunks[r.index] for r in reranked]
        reranked_citations = [citations[r.index] for r in reranked]

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks "
            f"(scores: {reranked[0].score:.3f} → {reranked[-1].score:.3f})"
        )

        return reranked_chunks, reranked_citations

    async def _find_related_images(
        self,
        page_refs: set[tuple[int, int]],  # (document_id, page_no)
    ) -> list[ExtractedImage]:
        """Find images on the exact same pages as retrieved chunks."""
        if not self.db:
            return []

        images = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentImage).where(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
            )
            for img in result.scalars().all():
                images.append(ExtractedImage(
                    image_id=img.image_id,
                    document_id=img.document_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                ))

        # Deduplicate by image_id
        seen = set()
        unique = []
        for img in images:
            if img.image_id not in seen:
                seen.add(img.image_id)
                unique.append(img)

        return unique

    async def _find_related_tables(
        self,
        page_refs: set[tuple[int, int]],
    ) -> list[ExtractedTable]:
        """Find tables on the exact same pages as retrieved chunks."""
        if not self.db:
            return []

        tables = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentTable).where(
                    DocumentTable.document_id == doc_id,
                    DocumentTable.page_no == page_no,
                )
            )
            for tbl in result.scalars().all():
                tables.append(ExtractedTable(
                    table_id=tbl.table_id,
                    document_id=tbl.document_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                ))

        # Deduplicate by table_id
        seen = set()
        unique = []
        for tbl in tables:
            if tbl.table_id not in seen:
                seen.add(tbl.table_id)
                unique.append(tbl)

        return unique

    @staticmethod
    def _assemble_context(
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        kg_summary: str,
        image_refs: list[ExtractedImage],
        table_refs: list[ExtractedTable] | None = None,
    ) -> str:
        """Assemble a structured context string for the LLM."""
        parts = []

        # KG insights
        if kg_summary:
            parts.append("## Knowledge Graph Insights")
            parts.append(kg_summary)
            parts.append("")

        # Retrieved chunks with citations
        if chunks:
            parts.append("## Retrieved Document Sections")
            for i, (chunk, citation) in enumerate(zip(chunks, citations)):
                parts.append(f"### [{i + 1}] {citation.format()}")
                parts.append(chunk.content)
                parts.append("")

        # Available images
        if image_refs:
            parts.append("## Available Document Images")
            for img in image_refs:
                caption_str = f': "{img.caption}"' if img.caption else ""
                parts.append(
                    f"- Image p.{img.page_no}{caption_str} (id: {img.image_id})"
                )
            parts.append("")

        # Available tables
        if table_refs:
            parts.append("## Available Document Tables")
            for tbl in table_refs:
                caption_str = f': "{tbl.caption}"' if tbl.caption else ""
                parts.append(
                    f"- Table p.{tbl.page_no} ({tbl.num_rows}x{tbl.num_cols}){caption_str}"
                )
            parts.append("")

        if not parts:
            return "No relevant documents found for this query."

        return "\n".join(parts)
