"""
ChatMessage model — persists chat history per workspace to PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Text, Integer, DateTime, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat_attachment import ChatMessageAttachment


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
    )
    message_id: Mapped[str] = mapped_column(String(50), index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Rich metadata (JSON columns — nullable for user messages)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    related_entities: Mapped[list | None] = mapped_column(JSON, nullable=True)
    image_refs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_steps: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # V1.5 feedback is deliberately attached to the persisted assistant
    # message, keeping product feedback and the source snapshot together.
    reply_to_message_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    feedback_rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_ratings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feedback_corrected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_reference_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    feedback_failure_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    feedback_review_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    feedback_promoted_case_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    attachment_links: Mapped[list["ChatMessageAttachment"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )
