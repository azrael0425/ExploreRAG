from __future__ import annotations

import os
import re
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.metadata import DocumentMetadataUpdate
from app.schemas.rag import DocumentImageResponse
from app.services.document_metadata import (
    MetadataValidationError,
    parse_upload_metadata,
    semantic_metadata_changed,
    validate_custom_metadata,
)

logger = logging.getLogger(__name__)


def _inject_images_from_db(
    markdown: str,
    images: list[DocumentImage],
    workspace_id: int,
) -> str:
    """Replace remaining <!-- image --> placeholders with real image markdown.

    Used as a safety net when the parser didn't inject them during processing.
    Images are matched in insertion order (by primary key) which mirrors the
    order of pictures in the original Docling document.
    """
    img_iter = iter(images)

    def _replacer(match):
        try:
            img = next(img_iter)
            url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
            caption = (img.caption or "").replace("[", "").replace("]", "")
            return f"\n![{caption}]({url})\n"
        except StopIteration:
            return ""

    return re.sub(r"<!--\s*image\s*-->", _replacer, markdown)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/workspace/{workspace_id}", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge base."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    result = await db.execute(
        select(Document).where(Document.workspace_id == workspace_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def process_document_background(document_id: int, file_path: str, workspace_id: int):
    """Background task to process document for RAG indexing."""
    from app.core.database import async_session_maker
    from app.services.explore_rag_factory import get_explore_rag_service

    async with async_session_maker() as db:
        try:
            # Load workspace-level KG settings
            from sqlalchemy import select as sa_select
            from app.models.knowledge_base import KnowledgeBase
            ws_result = await db.execute(
                sa_select(
                    KnowledgeBase.kg_language,
                    KnowledgeBase.kg_entity_types,
                    KnowledgeBase.llm_mode,
                )
                .where(KnowledgeBase.id == workspace_id)
            )
            ws_row = ws_result.one_or_none()
            kg_language = ws_row.kg_language if ws_row else None
            kg_entity_types = ws_row.kg_entity_types if ws_row else None
            llm_mode = ws_row.llm_mode if ws_row else "cloud"

            rag_service = get_explore_rag_service(
                db, workspace_id,
                kg_language=kg_language,
                kg_entity_types=kg_entity_types,
                llm_mode=llm_mode,
            )
            await rag_service.process_document(document_id, file_path)
            logger.info(f"Document {document_id} processed successfully")
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            # Guarantee FAILED status even if process_document's own handler failed
            try:
                # A database error during processing leaves this session in a
                # failed transaction. Roll it back before recording the
                # user-visible failure state.
                await db.rollback()
                from sqlalchemy import select, update
                from app.models.document import Document, DocumentStatus
                result = await db.execute(
                    select(Document.status).where(Document.id == document_id)
                )
                current_status = result.scalar_one_or_none()
                if current_status and current_status != DocumentStatus.FAILED:
                    await db.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(
                            status=DocumentStatus.FAILED,
                            error_message=str(e)[:500],
                        )
                    )
                    await db.commit()
            except Exception as recovery_err:
                logger.error(f"Failed to set FAILED status for doc {document_id}: {recovery_err}")


@router.post("/upload/{workspace_id}", response_model=DocumentUploadResponse)
async def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    custom_metadata: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to a knowledge base. Processing must be triggered separately."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    try:
        parsed_metadata = validate_custom_metadata(
            parse_upload_metadata(metadata, custom_metadata),
            kb.metadata_schema,
        )
    except MetadataValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )

    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename

    import aiofiles
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    import hashlib

    document = Document(
        workspace_id=workspace_id,
        filename=filename,
        original_filename=file.filename,
        file_type=ext[1:],
        file_size=len(content),
        status=DocumentStatus.PENDING,
        custom_metadata=parsed_metadata,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status,
        message="Document uploaded. Click 'Process' to extract and index content."
    )


@router.patch("/{document_id}/metadata", response_model=DocumentResponse)
async def update_document_metadata(
    document_id: int,
    body: DocumentMetadataUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Replace business metadata without re-parsing or re-embedding a document."""
    document = (await db.execute(select(Document).where(Document.id == document_id))).scalar_one_or_none()
    if document is None:
        raise NotFoundError("Document", document_id)
    if body.metadata_revision is not None and body.metadata_revision != document.metadata_revision:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Metadata was changed by another request; refresh and retry.",
        )
    workspace = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == document.workspace_id))).scalar_one()
    try:
        new_metadata = validate_custom_metadata(body.custom_metadata, workspace.metadata_schema)
    except MetadataValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    document.metadata_requires_reindex = document.metadata_requires_reindex or semantic_metadata_changed(
        document.custom_metadata, new_metadata, workspace.metadata_schema
    )
    document.custom_metadata = new_metadata
    document.metadata_revision += 1
    await db.commit()
    await db.refresh(document)
    return document


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get document by ID"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    return document


@router.get("/{document_id}/markdown")
async def get_document_markdown(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get the full structured markdown content of a document (ExploreRAG parsed)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if not document.markdown_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No markdown content available. Document may not have been processed with ExploreRAG."
        )

    markdown = document.markdown_content

    # Safety net: if image placeholders remain, inject real references on-the-fly
    if "<!-- image" in markdown:
        img_result = await db.execute(
            select(DocumentImage)
            .where(DocumentImage.document_id == document_id)
            .order_by(DocumentImage.id)
        )
        images = img_result.scalars().all()
        if images:
            markdown = _inject_images_from_db(markdown, images, document.workspace_id)

    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
    )


@router.get("/{document_id}/images", response_model=list[DocumentImageResponse])
async def get_document_images(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """List all extracted images for a document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    result = await db.execute(
        select(DocumentImage)
        .where(DocumentImage.document_id == document_id)
        .order_by(DocumentImage.page_no)
    )
    images = result.scalars().all()

    return [
        DocumentImageResponse(
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            width=img.width,
            height=img.height,
            url=f"/static/doc-images/kb_{document.workspace_id}/images/{img.image_id}.png",
        )
        for img in images
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document and its chunks from vector store"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status == DocumentStatus.INDEXED:
        try:
            from app.services.explore_rag_factory import get_explore_rag_service
            from app.models.knowledge_base import KnowledgeBase
            ws_result = await db.execute(
                select(KnowledgeBase.llm_mode).where(
                    KnowledgeBase.id == document.workspace_id
                )
            )
            llm_mode = ws_result.scalar_one_or_none() or "cloud"
            rag_service = get_explore_rag_service(
                db, document.workspace_id, llm_mode=llm_mode
            )
            await rag_service.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete chunks from vector store: {e}")

    file_path = UPLOAD_DIR / document.filename
    try:
        if file_path.exists():
            os.remove(file_path)
    except OSError as e:
        logger.warning(f"Failed to remove uploaded file {file_path}: {e}")

    await db.delete(document)
    await db.commit()
