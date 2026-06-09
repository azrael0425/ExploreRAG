"""HTTP endpoints for workspace-scoped temporary chat attachments."""
from __future__ import annotations

import mimetypes

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.schemas.rag import ChatAttachmentResponse
from app.services.chat_attachment_service import ChatAttachmentService
from app.services.attachment_processor import enqueue_attachment_preparation

router = APIRouter(prefix="/rag/chat", tags=["chat-attachments"])
service = ChatAttachmentService()


@router.post("/{workspace_id}/attachments", response_model=ChatAttachmentResponse)
async def upload_chat_attachment(
    workspace_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    attachment = await service.upload(db, workspace_id, file)
    enqueue_attachment_preparation(workspace_id, attachment.id)
    return ChatAttachmentResponse(**service.serialize(attachment))


@router.get("/{workspace_id}/attachments", response_model=list[ChatAttachmentResponse])
async def list_chat_attachments(workspace_id: int, db: AsyncSession = Depends(get_db)):
    attachments = await service.list(db, workspace_id)
    return [ChatAttachmentResponse(**service.serialize(attachment)) for attachment in attachments]


@router.get("/{workspace_id}/attachments/{attachment_id}", response_model=ChatAttachmentResponse)
async def get_chat_attachment(
    workspace_id: int, attachment_id: str, db: AsyncSession = Depends(get_db)
):
    attachment = (await service.selected(db, workspace_id, [attachment_id]))[0]
    return ChatAttachmentResponse(**service.serialize(attachment))


@router.get("/{workspace_id}/attachments/{attachment_id}/files/{artifact_path:path}")
async def get_chat_attachment_artifact(
    workspace_id: int,
    attachment_id: str,
    artifact_path: str,
    db: AsyncSession = Depends(get_db),
):
    file_path = await service.get_file(db, workspace_id, attachment_id, artifact_path)
    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(file_path, media_type=media_type or "application/octet-stream")
