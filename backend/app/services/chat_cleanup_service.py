"""Idempotent workspace-level cleanup for chat history and attachments."""
from __future__ import annotations

import asyncio
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_attachment import ChatAttachment, ChatAttachmentState, ChatMessageAttachment
from app.models.chat_message import ChatMessage
from app.models.knowledge_base import KnowledgeBase
from app.services.temporary_attachment_vector_store import TemporaryAttachmentVectorStore
from app.services.work_scheduler import get_work_scheduler

logger = logging.getLogger(__name__)
_workspace_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_workspace_cleanup_lock(workspace_id: int) -> asyncio.Lock:
    return _workspace_locks[workspace_id]


class ChatCleanupService:
    """Coordinates DB state with Chroma/filesystem cleanup.

    Filesystem and Chroma cannot share a SQL transaction.  We therefore first
    invalidate the workspace generation, which makes attachments immediately
    non-retrievable, then retry any external deletion that failed.
    """

    async def clear_workspace(self, db: AsyncSession, workspace_id: int) -> dict:
        lock = get_workspace_cleanup_lock(workspace_id)
        async with lock:
            result = await db.execute(
                select(KnowledgeBase).where(KnowledgeBase.id == workspace_id).with_for_update()
            )
            workspace = result.scalar_one_or_none()
            if workspace is None:
                raise ValueError(f"Workspace {workspace_id} not found")

            workspace.chat_cleanup_epoch += 1
            epoch = workspace.chat_cleanup_epoch
            attachments = list((await db.execute(
                select(ChatAttachment).where(ChatAttachment.workspace_id == workspace_id)
            )).scalars().all())
            for attachment in attachments:
                attachment.state = ChatAttachmentState.CLEARING
                attachment.cleanup_pending = False
            await db.commit()

            get_work_scheduler().cancel_workspace(workspace_id)
            external_error: Exception | None = None
            try:
                await asyncio.to_thread(TemporaryAttachmentVectorStore(workspace_id).delete_collection)
                attachment_root = settings.BASE_DIR / "data" / "chat-attachments" / f"ws_{workspace_id}"
                staging_root = settings.BASE_DIR / "data" / "chat-attachments" / ".staging" / f"ws_{workspace_id}"
                await asyncio.to_thread(self._remove_tree, attachment_root)
                await asyncio.to_thread(self._remove_tree, staging_root)
            except Exception as exc:  # pragma: no cover - depends on external FS/Chroma failures
                external_error = exc
                logger.exception("External chat attachment cleanup failed for workspace %s", workspace_id)

            if external_error is not None:
                # Keep invisible tombstones so a retry can find exactly what it
                # must remove.  Chat messages are nevertheless cleared now.
                for attachment in attachments:
                    attachment.cleanup_pending = True
                    attachment.error_message = f"cleanup_pending: {external_error}"[:1000]
                await db.execute(delete(ChatMessage).where(ChatMessage.workspace_id == workspace_id))
                await db.commit()
                asyncio.create_task(self._retry_workspace_cleanup(workspace_id))
                return {
                    "status": "cleared",
                    "workspace_id": workspace_id,
                    "cleanup_epoch": epoch,
                    "cleanup_pending": True,
                }

            if attachments:
                await db.execute(delete(ChatMessageAttachment).where(
                    ChatMessageAttachment.attachment_id.in_([attachment.id for attachment in attachments])
                ))
            await db.execute(delete(ChatAttachment).where(ChatAttachment.workspace_id == workspace_id))
            await db.execute(delete(ChatMessage).where(ChatMessage.workspace_id == workspace_id))
            await db.commit()
            return {
                "status": "cleared",
                "workspace_id": workspace_id,
                "cleanup_epoch": epoch,
                "cleanup_pending": False,
            }

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    async def is_epoch_current(self, db: AsyncSession, workspace_id: int, epoch: int) -> bool:
        result = await db.execute(
            select(KnowledgeBase.chat_cleanup_epoch).where(KnowledgeBase.id == workspace_id)
        )
        return result.scalar_one_or_none() == epoch

    async def _retry_workspace_cleanup(self, workspace_id: int, attempt: int = 0) -> None:
        """Best-effort retry using a new session, without reviving attachments."""
        await asyncio.sleep(min(30, 2 ** attempt))
        from app.core.database import async_session_maker

        try:
            async with async_session_maker() as db:
                pending = list((await db.execute(
                    select(ChatAttachment).where(
                        ChatAttachment.workspace_id == workspace_id,
                        ChatAttachment.cleanup_pending.is_(True),
                    )
                )).scalars().all())
                if not pending:
                    return
                lock = get_workspace_cleanup_lock(workspace_id)
                async with lock:
                    await asyncio.to_thread(TemporaryAttachmentVectorStore(workspace_id).delete_collection)
                    await asyncio.to_thread(
                        self._remove_tree,
                        settings.BASE_DIR / "data" / "chat-attachments" / f"ws_{workspace_id}",
                    )
                    await db.execute(delete(ChatMessageAttachment).where(
                        ChatMessageAttachment.attachment_id.in_([attachment.id for attachment in pending])
                    ))
                    await db.execute(delete(ChatAttachment).where(ChatAttachment.workspace_id == workspace_id))
                    await db.commit()
        except Exception as exc:  # pragma: no cover - external retry observability
            logger.error("Attachment cleanup retry failed for workspace %s: %s", workspace_id, exc)
            if attempt < 4:
                asyncio.create_task(self._retry_workspace_cleanup(workspace_id, attempt + 1))
