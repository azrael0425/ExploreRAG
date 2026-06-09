"""Ephemeral chat attachment models.

These tables intentionally have no relationship to ``documents``.  Keeping
the attachment domain separate is the first and most important guardrail
against accidentally publishing chat uploads into a knowledge base.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat_message import ChatMessage


class ChatAttachmentState(str, enum.Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PARSING = "parsing"
    READY_DIRECT = "ready_direct"
    INDEXED_TEMP = "indexed_temp"
    FAILED = "failed"
    CLEARING = "clearing"
    DELETED = "deleted"


class ChatAttachment(Base):
    __tablename__ = "chat_attachments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    artifact_dir: Mapped[str] = mapped_column(String(1000), nullable=False)
    state: Mapped[ChatAttachmentState] = mapped_column(
        Enum(ChatAttachmentState), nullable=False, default=ChatAttachmentState.UPLOADED, index=True
    )
    parsed_token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    temp_collection: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleanup_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cleanup_pending: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    message_links: Mapped[list["ChatMessageAttachment"]] = relationship(
        back_populates="attachment", cascade="all, delete-orphan"
    )


class ChatMessageAttachment(Base):
    __tablename__ = "chat_message_attachments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chat_messages.id", ondelete="CASCADE"), index=True
    )
    attachment_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("chat_attachments.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    message: Mapped["ChatMessage"] = relationship(back_populates="attachment_links")
    attachment: Mapped["ChatAttachment"] = relationship(back_populates="message_links")
