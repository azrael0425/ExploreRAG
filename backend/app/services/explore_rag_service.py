"""
Deep RAG Service
=================

Orchestrator for the ExploreRAG pipeline:
  Document → Docling Parse → ChromaDB Index + LightRAG KG → Hybrid Retrieval

It exposes indexing and retrieval primitives consumed by LangChain adapters.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.services.document_parser import get_document_parser
from app.services.knowledge_graph_service import (
    KnowledgeGraphService,
    get_knowledge_graph_service,
)
from app.services.deep_retriever import DeepRetriever
from app.services.embedder import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store
from app.services.reranker import get_reranker_service
from app.services.rag_types import RAGQueryResult, RetrievedChunk
from app.services.models.parsed_document import DeepRetrievalResult
from app.services.chunk_dedup import deduplicate_chunks
from app.services.index_chunking import split_enriched_chunks
from app.services.work_scheduler import WorkPriority, get_work_scheduler
from app.services.scan_profile_router import ScanProfileRouter
from app.services.document_metadata import semantic_metadata_context

logger = logging.getLogger(__name__)


class ExploreRAGService:
    """
    Full ExploreRAG pipeline orchestrator.

    Phases:
      1. PARSING  — Docling parse → markdown + chunks + images
      2. INDEXING — Embed chunks → ChromaDB + ingest markdown → LightRAG KG
      3. INDEXED  — Update document metadata in DB

    Query:
      - query()       — backward-compatible sync vector-only search
      - query_deep()  — full async hybrid retrieval (KG + vector + images)
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_id: int,
        kg_language: str | None = None,
        kg_entity_types: list[str] | None = None,
        llm_mode: str = "cloud",
    ):
        self.db = db
        self.workspace_id = workspace_id
        self.llm_mode = llm_mode

        # Services
        self.parser = get_document_parser(workspace_id=workspace_id, llm_mode=llm_mode)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)

        # KG service (optional, gated by config)
        self.kg_service: Optional[KnowledgeGraphService] = None
        if settings.EXPLORERAG_ENABLE_KG:
            self.kg_service = get_knowledge_graph_service(
                workspace_id=workspace_id,
                kg_language=kg_language,
                kg_entity_types=kg_entity_types,
                llm_mode=llm_mode,
            )

        # Retriever (with cross-encoder reranker)
        self.retriever = DeepRetriever(
            workspace_id=workspace_id,
            kg_service=self.kg_service,
            vector_store=self.vector_store,
            embedder=self.embedder,
            db=db,
            # The singleton is lazy: constructing it does not load model
            # weights.  Keeping it available lets evaluation enable reranking
            # per request without mutating process-global settings.
            reranker=get_reranker_service(),
        )
        self.scheduler = get_work_scheduler()

    # ------------------------------------------------------------------
    # Document Processing
    # ------------------------------------------------------------------

    async def process_document(self, document_id: int, file_path: str) -> int:
        """
        Process a document through the full ExploreRAG pipeline.

        Returns:
            Number of chunks created
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise ValueError(f"Document {document_id} not found")
        metadata_schema = await self.db.scalar(
            select(KnowledgeBase.metadata_schema).where(KnowledgeBase.id == document.workspace_id)
        )
        semantic_context = semantic_metadata_context(document.custom_metadata, metadata_schema)

        start_time = time.time()
        parse_ms = normalize_ms = embedding_ms = vector_store_ms = kg_ms = 0

        try:
            # Phase 1: PARSING
            document.status = DocumentStatus.PARSING
            await self.db.commit()

            scan_profile = "normal_pdf"
            scan_manifest: dict | None = None
            if str(file_path).lower().endswith(".pdf"):
                scan = ScanProfileRouter().inspect(file_path)
                scan_profile = scan.profile
                scan_manifest = scan.manifest()
                logger.info("Document %s scan profile: %s", document_id, scan_profile)

            import asyncio
            parse_started_at = time.perf_counter()
            parsed = await self.scheduler.run(
                "docling",
                WorkPriority.KNOWLEDGE_BASE,
                lambda: asyncio.to_thread(
                    self.parser.parse,
                    file_path=file_path,
                    document_id=document_id,
                    original_filename=document.original_filename,
                    scan_profile=scan_profile,
                ),
                workspace_id=self.workspace_id,
            )
            parse_ms = int((time.perf_counter() - parse_started_at) * 1000)

            normalize_started_at = time.perf_counter()
            parsed.chunks, split_stats = await asyncio.to_thread(
                split_enriched_chunks,
                parsed.chunks,
                self.embedder,
            )
            normalize_ms = int((time.perf_counter() - normalize_started_at) * 1000)
            logger.info(
                "Document %s chunk guard input=%s output=%s oversized=%s max_tokens=%s->%s normalize_ms=%s",
                document_id,
                split_stats["input"],
                split_stats["output"],
                split_stats["oversized"],
                split_stats["max_tokens_before"],
                split_stats["max_tokens_after"],
                normalize_ms,
            )

            # Save markdown + images to DB
            document.markdown_content = parsed.markdown
            document.page_count = parsed.page_count
            document.table_count = parsed.tables_count
            document.parser_version = self.parser.parser_name
            if scan_manifest is not None:
                document.processing_metadata = {
                    **(document.processing_metadata or {}),
                    "scan_profile": scan_manifest,
                }
            await self.db.commit()

            # Clean up old image records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentImage).where(DocumentImage.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted images to DB
            for img in parsed.images:
                db_image = DocumentImage(
                    document_id=document_id,
                    image_id=img.image_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                )
                self.db.add(db_image)
            # Always reset the count so a lightweight re-parse cannot retain
            # image metadata from a previous Docling-based version.
            document.image_count = len(parsed.images)
            await self.db.commit()

            # Clean up old table records before saving new ones (handles re-processing)
            await self.db.execute(
                delete(DocumentTable).where(DocumentTable.document_id == document_id)
            )
            await self.db.commit()

            # Save extracted tables to DB
            for tbl in parsed.tables:
                db_table = DocumentTable(
                    document_id=document_id,
                    table_id=tbl.table_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                )
                self.db.add(db_table)
            if parsed.tables:
                await self.db.commit()

            # Phase 1.5: PRE-INGESTION DEDUP
            if parsed.chunks:
                parsed.chunks, dedup_stats = deduplicate_chunks(parsed.chunks)
                if dedup_stats["input"] != dedup_stats["output"]:
                    logger.info(
                        f"Dedup for doc {document_id}: "
                        f"{dedup_stats['input']}→{dedup_stats['output']} chunks "
                        f"(noise={dedup_stats['noise_removed']}, "
                        f"exact={dedup_stats['exact_removed']}, "
                        f"near={dedup_stats['near_removed']})"
                    )

            # Phase 2: INDEXING
            document.status = DocumentStatus.INDEXING
            await self.db.commit()

            chunk_count = 0
            if parsed.chunks:
                def _store_index_sync(embeddings: list[list[float]]):
                    # Store already-computed embeddings in ChromaDB.
                    chunk_texts = [c.content for c in parsed.chunks]

                    ids = [
                        f"doc_{document_id}_chunk_{i}"
                        for i in range(len(parsed.chunks))
                    ]
                    # Build image_id→URL lookup for metadata
                    _img_url_map = {
                        img.image_id: f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                        for img in parsed.images
                    }

                    metadatas = []
                    for c in parsed.chunks:
                        meta = {
                            "document_id": document_id,
                            "chunk_index": c.chunk_index,
                            "source": c.source_file,
                            "file_type": document.file_type,
                            "page_no": c.page_no,
                            "heading_path": " > ".join(c.heading_path) if c.heading_path else "",
                            "has_table": c.has_table,
                            "has_code": c.has_code,
                            # Image-aware metadata: pipe-separated IDs and URLs
                            "image_ids": "|".join(c.image_refs) if c.image_refs else "",
                            "table_ids": "|".join(c.table_refs) if c.table_refs else "",
                            "image_urls": "|".join(
                                _img_url_map.get(iid, "") for iid in c.image_refs
                            ) if c.image_refs else "",
                            # Chunks are intentionally hidden until every
                            # durable parsing/indexing/KG phase has completed.
                            "visibility": "pending",
                        }
                        metadatas.append(meta)

                    self.vector_store.add_documents(
                        ids=ids,
                        embeddings=embeddings,
                        documents=chunk_texts,
                        metadatas=metadatas,
                    )
                chunk_texts = [c.content for c in parsed.chunks]
                embedding_inputs = [
                    f"{semantic_context}\n\n{chunk}" if semantic_context else chunk
                    for chunk in chunk_texts
                ]
                embeddings: list[list[float]] = []
                embedding_started_at = time.perf_counter()
                batch_size = settings.EXPLORERAG_EMBEDDING_BATCH_SIZE
                for batch_start in range(0, len(embedding_inputs), batch_size):
                    batch = embedding_inputs[batch_start:batch_start + batch_size]
                    batch_embeddings = await self.scheduler.run(
                        "embedding",
                        WorkPriority.KNOWLEDGE_BASE,
                        lambda batch=batch: asyncio.to_thread(self.embedder.embed_texts, batch),
                        workspace_id=self.workspace_id,
                    )
                    embeddings.extend(batch_embeddings)
                embedding_ms = int((time.perf_counter() - embedding_started_at) * 1000)

                vector_started_at = time.perf_counter()
                await asyncio.to_thread(_store_index_sync, embeddings)
                vector_store_ms = int((time.perf_counter() - vector_started_at) * 1000)
                chunk_count = len(parsed.chunks)

            # KG ingest completes before the pending vector chunks are made
            # visible.  A KG failure is a document-processing failure, not a
            # reason to expose partial durable knowledge.
            if self.kg_service and parsed.markdown:
                kg_started_at = time.perf_counter()
                if not document.kg_document_id:
                    document.kg_document_id = f"kb:{self.workspace_id}:doc:{document_id}"
                    await self.db.commit()
                await self.scheduler.run(
                    "llm_enrichment",
                    WorkPriority.ENRICHMENT,
                    lambda: self.kg_service.ingest(
                        f"{semantic_context}\n\n{parsed.markdown}" if semantic_context else parsed.markdown,
                        kg_document_id=document.kg_document_id or f"kb:{self.workspace_id}:doc:{document_id}",
                        source_file=document.original_filename,
                    ),
                    workspace_id=self.workspace_id,
                )
                kg_ms = int((time.perf_counter() - kg_started_at) * 1000)
                document.kg_index_status = "indexed"
                document.kg_indexed_content_version = document.content_version
            elif not settings.EXPLORERAG_ENABLE_KG:
                document.kg_index_status = "disabled"

            if parsed.chunks:
                await asyncio.to_thread(self.vector_store.mark_document_ready, document_id)

            # Phase 3: INDEXED
            elapsed_ms = int((time.time() - start_time) * 1000)
            document.status = DocumentStatus.INDEXED
            document.chunk_count = chunk_count
            document.processing_time_ms = elapsed_ms
            document.metadata_requires_reindex = False
            await self.db.commit()

            logger.info(
                f"ExploreRAG processed document {document_id}: "
                f"{chunk_count} chunks, {len(parsed.images)} images, "
                f"{parsed.tables_count} tables in {elapsed_ms}ms "
                f"(parse={parse_ms} normalize={normalize_ms} embedding={embedding_ms} "
                f"vector_store={vector_store_ms} kg={kg_ms})"
            )
            return chunk_count

        except Exception as e:
            logger.error(f"ExploreRAG failed for document {document_id}: {e}")
            try:
                # Do not leave failed documents' pending chunks addressable.
                await asyncio.to_thread(self.vector_store.delete_by_document_id, document_id)
            except Exception as cleanup_error:
                logger.warning("Could not remove pending chunks for failed document %s: %s", document_id, cleanup_error)
            document.status = DocumentStatus.FAILED
            if self.kg_service:
                document.kg_index_status = "failed"
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        metadata_filter: dict | None = None,
    ) -> RAGQueryResult:
        """
        Backward-compatible sync query (vector-only).
        Returns a compact RAGQueryResult for maintenance callers.
        """
        if document_ids is not None and not document_ids:
            return RAGQueryResult(chunks=[], context="", query=question)
        query_embedding = self.embedder.embed_query(question)

        filters: list[dict] = [{"visibility": "ready"}]
        if metadata_filter:
            filters.append(metadata_filter.copy())
        if document_ids:
            filters.append({"document_id": {"$in": document_ids}})
        where = filters[0] if len(filters) == 1 else {"$and": filters}

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        chunks = []
        for i, doc in enumerate(results.get("documents", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            chunks.append(RetrievedChunk(
                content=doc,
                metadata=meta,
                score=results["distances"][i] if results.get("distances") else 0.0,
                chunk_id=results["ids"][i] if results.get("ids") else "",
            ))

        chunks.sort(key=lambda x: x.score)

        # Assemble context with citations
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            page = chunk.metadata.get("page_no", 0)
            heading = chunk.metadata.get("heading_path", "")
            citation = source
            if page:
                citation += f" | p.{page}"
            if heading:
                citation += f" | {heading}"
            context_parts.append(f"[{i + 1}] {citation}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        return RAGQueryResult(
            chunks=chunks,
            context=context,
            query=question,
        )

    async def query_deep(
        self,
        question: str,
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        mode: str = "hybrid",
        include_images: bool = True,
        metadata_filter: dict | None = None,
        enable_reranker: bool | None = None,
        enable_knowledge_graph: bool | None = None,
        prefetch_k: int | None = None,
    ) -> DeepRetrievalResult:
        """
        Full async hybrid retrieval with KG + vector + images + citations.
        """
        return await self.retriever.query(
            question=question,
            mode=mode,
            top_k=top_k,
            document_ids=document_ids,
            include_images=include_images,
            metadata_filter=metadata_filter,
            enable_reranker=enable_reranker,
            enable_knowledge_graph=enable_knowledge_graph,
            prefetch_k=prefetch_k,
        )

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def delete_document(self, document_id: int) -> None:
        """Delete a document's data from vector store and KG."""
        self.vector_store.delete_by_document_id(document_id)

        document = (await self.db.execute(
            select(Document).where(Document.id == document_id)
        )).scalar_one_or_none()
        if self.kg_service and document and document.kg_document_id:
            await self.kg_service.delete_document(document.kg_document_id)
            document.kg_index_status = "not_indexed"
            document.kg_indexed_content_version = 0

        # Delete images from DB (cascade handles it, but clean up files)
        result = await self.db.execute(
            select(DocumentImage).where(DocumentImage.document_id == document_id)
        )
        for img in result.scalars().all():
            from pathlib import Path
            img_path = Path(img.file_path)
            if img_path.exists():
                img_path.unlink()

        logger.info(f"Deleted document {document_id} from ExploreRAG stores")

    def get_chunk_count(self) -> int:
        """Return total number of chunks in the knowledge base's vector store."""
        return self.vector_store.count()
