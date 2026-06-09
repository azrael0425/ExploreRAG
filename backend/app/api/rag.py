"""
RAG API endpoints for document querying and retrieval.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.core.config import settings
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
import logging

from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunkResponse,
    CitationResponse,
    DocumentImageResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    BatchProcessRequest,
    ProjectRAGStatsResponse,
    KGEntityResponse,
    KGRelationshipResponse,
    KGGraphResponse,
    KGGraphNodeResponse,
    KGGraphEdgeResponse,
    KGFocusRequest,
    KGFocusResponse,
    ChatRequest,
    ChatResponse,
    ChatSourceChunk,
    ChatImageRef,
    PersistedChatMessage,
    ChatHistoryResponse,
    ChatAttachmentResponse,
    LLMCapabilitiesResponse,
    DebugRetrievedSource,
    DebugChatResponse,
)

logger = logging.getLogger(__name__)
from app.services.explore_rag_factory import get_explore_rag_service
from app.services.document_metadata import MetadataValidationError, resolve_document_scope
from app.services.retrieval_policy import resolve_retrieval_policy

router = APIRouter(prefix="/rag", tags=["rag"])

UPLOAD_DIR = "uploads"

# Prompt constants — see chat_prompt.py for full documentation
from app.api.chat_prompt import (
    DEFAULT_SYSTEM_PROMPT,
    HARD_SYSTEM_PROMPT,
)


async def verify_workspace_access(
    workspace_id: int,
    db: AsyncSession,
) -> KnowledgeBase:
    """Verify knowledge base exists."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    return kb


async def _resolve_retrieval_scope(
    db: AsyncSession,
    workspace_id: int,
    document_ids: list[int] | None,
    metadata_filter: object | None,
):
    """Translate public metadata filters into safe vector document scope."""
    payload = metadata_filter.model_dump(by_alias=True) if hasattr(metadata_filter, "model_dump") else metadata_filter
    try:
        return await resolve_document_scope(db, workspace_id, document_ids, payload)
    except MetadataValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.post("/query/{workspace_id}", response_model=RAGQueryResponse)
async def query_documents(
    workspace_id: int,
    request: RAGQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Query indexed documents using semantic search (+ optional KG)."""
    kb = await verify_workspace_access(workspace_id, db)
    scope = await _resolve_retrieval_scope(
        db, workspace_id, request.document_ids, request.metadata_filter
    )
    policy = resolve_retrieval_policy(
        request.mode,
        workspace_lightrag_enabled=kb.lightrag_augmentation_enabled,
        scoped=scope.scoped,
    )
    effective_mode = policy.mode

    from app.langchain_rag.callbacks import get_local_callbacks
    from app.langchain_rag.contracts import RetrievalInput
    from app.langchain_rag.factory import get_explore_rag_retrieval_runnable

    runnable = get_explore_rag_retrieval_runnable(db, workspace_id, llm_mode=kb.llm_mode)
    envelope = await runnable.ainvoke(RetrievalInput(
        workspace_id=workspace_id,
        question=request.question,
        top_k=request.top_k,
        document_ids=scope.document_ids,
        mode=effective_mode,
        include_images=True,
        metadata_filter=None,
    ), config={
        "run_name": "explorerag_query",
        "tags": ["rag", kb.llm_mode, effective_mode],
        "metadata": {"workspace_id": workspace_id},
        "callbacks": get_local_callbacks(),
    })
    chunks_response = []
    for index, document in enumerate(envelope.documents):
        metadata = document.metadata
        citation = envelope.citations[index] if index < len(envelope.citations) else None
        citation_response = CitationResponse(
            source_file=citation.source_file,
            document_id=citation.document_id,
            page_no=citation.page_no,
            heading_path=citation.heading_path,
            formatted=citation.format(),
        ) if citation else None
        chunks_response.append(RetrievedChunkResponse(
            content=document.page_content,
            chunk_id=str(metadata.get("chunk_id", "")),
            score=float(metadata.get("distance", 0.0)),
            metadata=metadata,
            citation=citation_response,
        ))
    return RAGQueryResponse(
        query=request.question,
        chunks=chunks_response,
        context=envelope.context,
        total_chunks=len(chunks_response),
        knowledge_graph_summary=envelope.knowledge_graph_summary,
        citations=[
            CitationResponse(
                source_file=citation.source_file,
                document_id=citation.document_id,
                page_no=citation.page_no,
                heading_path=citation.heading_path,
                formatted=citation.format(),
            )
            for citation in envelope.citations
        ],
        image_refs=[
            DocumentImageResponse(
                image_id=image.image_id,
                document_id=image.document_id,
                page_no=image.page_no,
                caption=image.caption,
                width=image.width,
                height=image.height,
                url=f"/static/doc-images/kb_{workspace_id}/images/{image.image_id}.png",
            )
            for image in envelope.image_refs
        ],
    )

@router.post("/process/{document_id}", response_model=DocumentProcessResponse)
async def process_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Trigger document processing (parsing + indexing) as a background task."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in (DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING):
        # Check if stale (exceeded processing timeout) — auto-recover
        from datetime import datetime, timedelta
        from app.core.config import settings
        timeout = settings.EXPLORERAG_PROCESSING_TIMEOUT_MINUTES
        cutoff = datetime.utcnow() - timedelta(minutes=timeout)
        if document.updated_at < cutoff:
            # Stale — reset to allow re-processing
            document.status = DocumentStatus.FAILED
            document.error_message = f"Processing timeout ({timeout}min). Retrying..."
            await db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is already being analyzed"
            )

    if document.status == DocumentStatus.INDEXED:
        return DocumentProcessResponse(
            document_id=document_id,
            status=document.status.value,
            chunk_count=document.chunk_count,
            message="Document is already indexed"
        )

    from pathlib import Path
    file_path = Path(UPLOAD_DIR) / document.filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    # Mark as processing immediately so UI updates
    document.status = DocumentStatus.PROCESSING
    document.error_message = None
    await db.commit()

    # Launch background task
    from app.api.documents import process_document_background
    import asyncio
    asyncio.get_event_loop().create_task(
        process_document_background(document_id, str(file_path), document.workspace_id)
    )

    return DocumentProcessResponse(
        document_id=document_id,
        status="processing",
        chunk_count=0,
        message="Processing started. Document will be parsed and indexed in the background."
    )


@router.post("/process-batch")
async def process_batch(
    request: BatchProcessRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Process multiple documents sequentially in the background.
    Marks all as PROCESSING immediately, then processes one-by-one to avoid
    resource contention (each doc uses Docling + embeddings + KG ingest).
    """
    from pathlib import Path as _P

    accepted_ids = []
    skipped_ids = []

    for doc_id in request.document_ids:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc is None:
            skipped_ids.append(doc_id)
            continue

        # Skip documents already being processed or already indexed
        if doc.status in (
            DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING,
        ):
            skipped_ids.append(doc_id)
            continue

        file_path = _P(UPLOAD_DIR) / doc.filename
        if not file_path.exists():
            skipped_ids.append(doc_id)
            continue

        # Mark as processing immediately so UI updates
        doc.status = DocumentStatus.PROCESSING
        doc.error_message = None
        accepted_ids.append((doc_id, str(file_path), doc.workspace_id))

    await db.commit()

    if accepted_ids:
        import asyncio
        asyncio.get_event_loop().create_task(
            _process_batch_background(accepted_ids)
        )

    return {
        "message": f"Processing {len(accepted_ids)} document(s)",
        "accepted": [aid[0] for aid in accepted_ids],
        "skipped": skipped_ids,
    }


async def _process_batch_background(
    items: list[tuple[int, str, int]],
):
    """Process documents sequentially to avoid resource contention."""
    from app.api.documents import process_document_background

    for doc_id, file_path, workspace_id in items:
        try:
            await process_document_background(doc_id, file_path, workspace_id)
            logger.info(f"Batch: document {doc_id} processed")
        except Exception as e:
            logger.error(f"Batch: document {doc_id} failed: {e}")


@router.post("/reindex/{document_id}", response_model=DocumentProcessResponse)
async def reindex_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Queue an existing document for re-processing through ExploreRAG."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status in (DocumentStatus.PROCESSING, DocumentStatus.PARSING, DocumentStatus.INDEXING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is currently being processed"
        )

    workspace = await verify_workspace_access(document.workspace_id, db)

    from pathlib import Path
    file_path = Path(UPLOAD_DIR) / document.filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found on disk"
        )

    # Mark immediately, then let a separate database session run the expensive
    # parsing, embedding and graph-extraction work in the background.
    document.status = DocumentStatus.PROCESSING
    document.error_message = None
    await db.commit()

    import asyncio
    asyncio.get_running_loop().create_task(
        _reindex_document_background(
            document_id=document_id,
            file_path=str(file_path),
            workspace_id=document.workspace_id,
            llm_mode=workspace.llm_mode,
        )
    )

    return DocumentProcessResponse(
        document_id=document_id,
        status=DocumentStatus.PROCESSING.value,
        chunk_count=0,
        message="Document re-processing started"
    )


async def _reindex_document_background(
    document_id: int,
    file_path: str,
    workspace_id: int,
    llm_mode: str,
) -> None:
    """Remove old index data and rebuild it without holding the HTTP request."""
    from app.core.database import async_session_maker

    async with async_session_maker() as db:
        try:
            rag_service = get_explore_rag_service(db, workspace_id, llm_mode=llm_mode)
            try:
                await rag_service.delete_document(document_id)
            except Exception as exc:
                logger.warning("Failed to delete old data for reindex of document %s: %s", document_id, exc)
                # A failed database operation can leave this session unusable
                # until rolled back. Continue with the rebuild so that a stale
                # index does not leave the document stuck in "processing".
                await db.rollback()

            result = await db.execute(select(Document).where(Document.id == document_id))
            document = result.scalar_one_or_none()
            if document is None:
                return

            document.status = DocumentStatus.PENDING
            document.chunk_count = 0
            document.markdown_content = None
            document.image_count = 0
            document.table_count = 0
            document.parser_version = None
            document.error_message = None
            await db.commit()

            await rag_service.process_document(document_id=document_id, file_path=file_path)
            logger.info("Reindexed document %s in background", document_id)
        except Exception as exc:
            logger.error("Failed to reindex document %s: %s", document_id, exc)


@router.post("/reindex-workspace/{workspace_id}")
async def reindex_workspace(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Reindex ALL documents in a workspace.
    Deletes the old vector collection (handles embedding dimension changes)
    and re-processes every document through the ExploreRAG pipeline.
    Runs in background — returns immediately with document count.
    """
    workspace = await verify_workspace_access(workspace_id, db)

    # Find all documents in this workspace
    result = await db.execute(
        select(Document).where(
            Document.workspace_id == workspace_id,
            Document.status.notin_([
                DocumentStatus.PROCESSING,
                DocumentStatus.PARSING,
                DocumentStatus.INDEXING,
            ]),
        )
    )
    documents = list(result.scalars().all())

    if not documents:
        return {"message": "No documents to reindex", "document_count": 0}

    # Delete old vector collection (required when embedding dimensions change)
    try:
        from app.services.vector_store import get_vector_store
        vs = get_vector_store(workspace_id)
        vs.delete_collection()
        logger.info(f"Deleted old vector collection for workspace {workspace_id}")
    except Exception as e:
        logger.warning(f"Failed to delete old collection: {e}")

    async def _reindex_all(doc_ids: list[int], ws_id: int, llm_mode: str):
        """Background task: reindex each document sequentially."""
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            rag_service = get_explore_rag_service(session, ws_id, llm_mode=llm_mode)
            for did in doc_ids:
                try:
                    res = await session.execute(
                        select(Document).where(Document.id == did)
                    )
                    doc = res.scalar_one_or_none()
                    if not doc:
                        continue

                    from pathlib import Path
                    file_path = Path(UPLOAD_DIR) / doc.filename
                    if not file_path.exists():
                        logger.warning(f"Skipping doc {did}: file not found")
                        continue

                    # Delete old chunk data for this document
                    try:
                        await rag_service.delete_document(did)
                    except Exception:
                        pass

                    # Reset metadata
                    doc.status = DocumentStatus.PENDING
                    doc.chunk_count = 0
                    doc.image_count = 0
                    doc.error_message = None
                    await session.commit()

                    # Re-process
                    await rag_service.process_document(
                        document_id=did, file_path=str(file_path)
                    )
                    logger.info(f"Reindexed document {did} in workspace {ws_id}")
                except Exception as e:
                    logger.error(f"Failed to reindex document {did}: {e}")

    doc_ids = [d.id for d in documents]
    background_tasks.add_task(_reindex_all, doc_ids, workspace_id, workspace.llm_mode)

    return {
        "message": f"Reindexing {len(doc_ids)} documents in background",
        "document_count": len(doc_ids),
        "document_ids": doc_ids,
    }


@router.get("/stats/{workspace_id}", response_model=ProjectRAGStatsResponse)
async def get_workspace_rag_stats(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get RAG statistics for a knowledge base."""
    workspace = await verify_workspace_access(workspace_id, db)

    total_result = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == workspace_id)
    )
    total_documents = total_result.scalar() or 0

    indexed_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.status == DocumentStatus.INDEXED
        )
    )
    indexed_documents = indexed_result.scalar() or 0

    # Count ExploreRAG documents (parser_version = 'docling')
    explorerag_result = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == workspace_id,
            Document.parser_version == "docling"
        )
    )
    explorerag_documents = explorerag_result.scalar() or 0

    # Count total images
    image_result = await db.execute(
        select(func.count(DocumentImage.id))
        .join(Document, DocumentImage.document_id == Document.id)
        .where(Document.workspace_id == workspace_id)
    )
    image_count = image_result.scalar() or 0

    rag_service = get_explore_rag_service(db, workspace_id, llm_mode=workspace.llm_mode)
    try:
        total_chunks = rag_service.get_chunk_count()
    except Exception:
        total_chunks = 0

    return ProjectRAGStatsResponse(
        workspace_id=workspace_id,
        total_documents=total_documents,
        indexed_documents=indexed_documents,
        total_chunks=total_chunks,
        image_count=image_count,
        explorerag_documents=explorerag_documents,
    )


@router.get("/chunks/{document_id}")
async def get_document_chunks(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all chunks for a specific document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status != DocumentStatus.INDEXED:
        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunks": [],
            "message": "Document is not yet indexed"
        }

    workspace = await verify_workspace_access(document.workspace_id, db)
    rag_service = get_explore_rag_service(
        db, document.workspace_id, llm_mode=workspace.llm_mode
    )

    chunk_ids = [f"doc_{document_id}_chunk_{i}" for i in range(document.chunk_count)]

    try:
        results = rag_service.vector_store.get_by_ids(chunk_ids)

        chunks = []
        for i in range(len(results.get("ids", []))):
            chunks.append({
                "chunk_id": results["ids"][i],
                "content": results["documents"][i] if results.get("documents") else None,
                "metadata": results["metadatas"][i] if results.get("metadatas") else {}
            })

        return {
            "document_id": document_id,
            "status": document.status.value,
            "chunk_count": document.chunk_count,
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Knowledge Graph exploration endpoints (Phase 9)
# ---------------------------------------------------------------------------

async def _get_kg_service(workspace_id: int, llm_mode: str | None = None):
    """Get KnowledgeGraphService for a knowledge base (if ExploreRAG is active)."""
    from app.services.knowledge_graph_service import get_knowledge_graph_service
    return get_knowledge_graph_service(workspace_id, llm_mode=llm_mode)


@router.get("/entities/{workspace_id}", response_model=list[KGEntityResponse])
async def get_kg_entities(
    workspace_id: int,
    search: str | None = None,
    entity_type: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List entities in the workspace's knowledge graph."""
    workspace = await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id, llm_mode=workspace.llm_mode)
    try:
        entities = await kg.get_entities(
            search=search, entity_type=entity_type, limit=limit, offset=offset
        )
        return [KGEntityResponse(**e) for e in entities]
    except Exception as e:
        logger.error(f"Failed to get KG entities for workspace {workspace_id}: {e}")
        return []


@router.get("/relationships/{workspace_id}", response_model=list[KGRelationshipResponse])
async def get_kg_relationships(
    workspace_id: int,
    entity: str | None = None,
    limit: int = 500,
    db: AsyncSession = Depends(get_db),
):
    """List relationships in the workspace's knowledge graph."""
    workspace = await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id, llm_mode=workspace.llm_mode)
    try:
        rels = await kg.get_relationships(entity_name=entity, limit=limit)
        return [KGRelationshipResponse(**r) for r in rels]
    except Exception as e:
        logger.error(f"Failed to get KG relationships for workspace {workspace_id}: {e}")
        return []


@router.get("/graph/{workspace_id}", response_model=KGGraphResponse)
async def get_kg_graph(
    workspace_id: int,
    center: str | None = None,
    max_depth: int = 3,
    max_nodes: int = 250,
    db: AsyncSession = Depends(get_db),
):
    """Export knowledge graph data for frontend visualization."""
    workspace = await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id, llm_mode=workspace.llm_mode)
    try:
        data = await kg.get_graph_data(
            center_entity=center, max_depth=max_depth, max_nodes=max_nodes
        )
        nodes, edges = await _add_graph_document_provenance(data, workspace_id, db)
        return KGGraphResponse(
            nodes=[KGGraphNodeResponse(**node) for node in nodes],
            edges=[KGGraphEdgeResponse(**edge) for edge in edges],
            is_truncated=data.get("is_truncated", False),
        )
    except Exception as e:
        logger.error(f"Failed to export KG graph for workspace {workspace_id}: {e}")
        return KGGraphResponse()


@router.post("/graph/{workspace_id}/focus", response_model=KGFocusResponse)
async def get_focused_kg_graph(
    workspace_id: int,
    request: KGFocusRequest,
    db: AsyncSession = Depends(get_db),
):
    """Return a citation-focused graph that force-includes resolved seeds."""
    workspace = await verify_workspace_access(workspace_id, db)
    kg = await _get_kg_service(workspace_id, llm_mode=workspace.llm_mode)
    try:
        data = await kg.get_focus_graph_data(
            request.entity_names,
            document_ids=request.document_ids,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
        nodes, edges = await _add_graph_document_provenance(data, workspace_id, db)
        return KGFocusResponse(
            nodes=[KGGraphNodeResponse(**node) for node in nodes],
            edges=[KGGraphEdgeResponse(**edge) for edge in edges],
            is_truncated=data.get("is_truncated", False),
            requested_entities=data.get("requested_entities", []),
            matched_entities=data.get("matched_entities", []),
            missing_entities=data.get("missing_entities", []),
        )
    except Exception as exc:
        logger.error(
            "Failed to export focused KG graph for workspace %s: %s",
            workspace_id,
            exc,
        )
        return KGFocusResponse(
            requested_entities=request.entity_names,
            missing_entities=request.entity_names,
        )


async def _add_graph_document_provenance(
    data: dict,
    workspace_id: int,
    db: AsyncSession,
    document_id_by_filename: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Resolve persisted source filenames to workspace document ids once."""
    if document_id_by_filename is None:
        documents = (await db.execute(
            select(Document.id, Document.original_filename)
            .where(Document.workspace_id == workspace_id)
        )).all()
        document_id_by_filename = {
            filename: document_id for document_id, filename in documents
        }

    def add_document_provenance(item: dict) -> dict:
        known_ids = item.get("source_document_ids", [])
        file_ids = [
            document_id_by_filename[file_name]
            for file_name in item.get("source_files", [])
            if file_name in document_id_by_filename
        ]
        item["source_document_ids"] = list(dict.fromkeys([*known_ids, *file_ids]))
        return item

    return (
        [add_document_provenance(node) for node in data["nodes"]],
        [add_document_provenance(edge) for edge in data["edges"]],
    )

# ---------------------------------------------------------------------------
# Chat history persistence
# ---------------------------------------------------------------------------

@router.get("/chat/{workspace_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Load persisted chat history for a workspace."""
    await verify_workspace_access(workspace_id, db)

    from app.models.chat_message import ChatMessage as ChatMessageModel
    from app.models.chat_attachment import ChatMessageAttachment
    result = await db.execute(
        select(ChatMessageModel)
        .where(ChatMessageModel.workspace_id == workspace_id)
        .options(selectinload(ChatMessageModel.attachment_links).selectinload(ChatMessageAttachment.attachment))
        .order_by(ChatMessageModel.created_at.asc())
    )
    messages = result.scalars().all()

    return ChatHistoryResponse(
        workspace_id=workspace_id,
        messages=[
            PersistedChatMessage(
                id=m.id,
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                sources=m.sources,
                related_entities=m.related_entities,
                image_refs=m.image_refs,
                thinking=m.thinking,
                agent_steps=m.agent_steps,
                reply_to_message_id=m.reply_to_message_id,
                feedback_rating=m.feedback_rating,
                feedback_comment=m.feedback_comment,
                source_ratings=m.source_ratings,
                attachments=[
                    ChatAttachmentResponse(
                        attachment_id=link.attachment.id,
                        original_filename=link.attachment.original_filename,
                        file_type=link.attachment.file_type,
                        file_size=link.attachment.file_size,
                        state=link.attachment.state.value,
                        parsed_token_count=link.attachment.parsed_token_count,
                        error_message=link.attachment.error_message,
                        created_at=link.attachment.created_at.isoformat() if link.attachment.created_at else "",
                    )
                    for link in m.attachment_links if link.attachment is not None
                ] or None,
                created_at=m.created_at.isoformat() if m.created_at else "",
            )
            for m in messages
        ],
        total=len(messages),
    )


@router.delete("/chat/{workspace_id}/history")
async def delete_chat_history(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Clear all chat history for a workspace."""
    await verify_workspace_access(workspace_id, db)

    from app.services.chat_cleanup_service import ChatCleanupService
    return await ChatCleanupService().clear_workspace(db, workspace_id)


# ---------------------------------------------------------------------------
# Chat endpoint — LLM-powered document Q&A via ExploreRAG
# ---------------------------------------------------------------------------
# SSE Streaming chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat/{workspace_id}/stream")
async def chat_stream(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """SSE streaming chat with deterministic hybrid retrieval."""
    from app.api.chat_agent import chat_stream_endpoint
    return await chat_stream_endpoint(workspace_id, request, db)


# ---------------------------------------------------------------------------

@router.post("/chat/{workspace_id}", response_model=ChatResponse)
async def chat_with_documents(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Chat with documents using ExploreRAG retrieval + LLM answer generation."""
    kb = await verify_workspace_access(workspace_id, db)

    from app.models.chat_attachment import ChatMessageAttachment
    from app.models.chat_message import ChatMessage as ChatMessageModel
    from app.services.chat_attachment_service import ChatAttachmentService
    from app.langchain_rag.contracts import ChatChainInput
    from app.langchain_rag.service import complete_chat

    selected = (
        await ChatAttachmentService().selected(db, workspace_id, request.attachment_ids)
        if request.attachment_ids else []
    )
    final = await complete_chat(
        ChatChainInput(
            workspace_id=workspace_id,
            message=request.message,
            history=[{"role": item.role, "content": item.content} for item in request.history],
            attachment_ids=list(request.attachment_ids or []),
            enable_thinking=request.enable_thinking,
            llm_mode=kb.llm_mode,
            lightrag_augmentation_enabled=kb.lightrag_augmentation_enabled,
            system_prompt=(kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT,
            document_ids=request.document_ids,
            metadata_filter=(
                request.metadata_filter.model_dump(by_alias=True)
                if request.metadata_filter else None
            ),
        ),
        db,
        selected_attachments=selected,
    )

    source_models = [ChatSourceChunk(**source) for source in final.get("sources", [])]
    image_models = [ChatImageRef(**image) for image in final.get("image_refs", [])]
    try:
        user = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
        )
        db.add(user)
        await db.flush()
        db.add_all([
            ChatMessageAttachment(message_id=user.id, attachment_id=attachment.id)
            for attachment in selected
        ])
        db.add(ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=final.get("answer", ""),
            sources=[source.model_dump() for source in source_models] or None,
            related_entities=final.get("related_entities") or None,
            image_refs=[image.model_dump() for image in image_models] or None,
            thinking=final.get("thinking"),
        ))
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return ChatResponse(
        answer=final.get("answer", ""),
        sources=source_models,
        related_entities=final.get("related_entities") or [],
        image_refs=image_models,
        thinking=final.get("thinking"),
    )


# ---------------------------------------------------------------------------
# LLM Capabilities endpoint
# ---------------------------------------------------------------------------

@router.get("/capabilities", response_model=LLMCapabilitiesResponse)
async def get_llm_capabilities(
    workspace_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Check LLM provider capabilities (thinking, vision)."""
    from app.services.llm import get_llm_provider
    mode = "cloud"
    if workspace_id is not None:
        workspace = await verify_workspace_access(workspace_id, db)
        mode = workspace.llm_mode
    provider = get_llm_provider(mode)

    return LLMCapabilitiesResponse(
        provider=getattr(provider, "provider_name", "qwen"),
        model=getattr(provider, "model_name", ""),
        supports_thinking=provider.supports_thinking(),
        supports_vision=provider.supports_vision(),
        thinking_default=provider.supports_thinking(),
    )


# ---------------------------------------------------------------------------
# Debug endpoint — inspect retrieval + LLM answer quality
# ---------------------------------------------------------------------------

@router.post("/debug-chat/{workspace_id}", response_model=DebugChatResponse)
async def debug_chat(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inspect the exact LangChain answer path and its retrieved sources."""

    kb = await verify_workspace_access(workspace_id, db)
    from app.langchain_rag.contracts import ChatChainInput
    from app.langchain_rag.service import complete_chat
    from app.services.chat_attachment_service import ChatAttachmentService
    from app.services.llm import get_llm_provider

    selected = (
        await ChatAttachmentService().selected(db, workspace_id, request.attachment_ids)
        if request.attachment_ids else []
    )
    system_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT
    final = await complete_chat(
        ChatChainInput(
            workspace_id=workspace_id,
            message=request.message,
            history=[{"role": item.role, "content": item.content} for item in request.history],
            attachment_ids=list(request.attachment_ids or []),
            enable_thinking=request.enable_thinking,
            llm_mode=kb.llm_mode,
            lightrag_augmentation_enabled=kb.lightrag_augmentation_enabled,
            system_prompt=system_prompt,
            document_ids=request.document_ids,
            metadata_filter=(
                request.metadata_filter.model_dump(by_alias=True)
                if request.metadata_filter else None
            ),
        ),
        db,
        selected_attachments=selected,
    )

    source_rows = final.get("sources", [])
    debug_sources = [
        DebugRetrievedSource(
            index=str(source.get("index", "")),
            document_id=int(source.get("document_id", 0) or 0),
            page_no=int(source.get("page_no", 0) or 0),
            heading_path=list(source.get("heading_path", []) or []),
            source_file=str(source.get("source_file", "")),
            content_preview=str(source.get("content", ""))[:500],
            score=float(source.get("score", 0.0) or 0.0),
            source_type=str(source.get("source_type", "vector")),
        )
        for source in source_rows
    ]
    graph_context = "\n\n".join(
        str(source.get("content", ""))
        for source in source_rows
        if source.get("source_type") == "kg"
    )
    provider = get_llm_provider(kb.llm_mode)
    return DebugChatResponse(
        question=request.message,
        workspace_id=workspace_id,
        retrieved_sources=debug_sources,
        kg_summary=graph_context,
        total_sources=len(debug_sources),
        system_prompt=system_prompt,
        answer=str(final.get("answer", "")),
        thinking=final.get("thinking"),
        image_count=len(final.get("image_refs", [])),
        provider=getattr(provider, "provider_name", "qwen"),
        model=getattr(provider, "model_name", settings.LLM_MODEL_FAST),
    )
