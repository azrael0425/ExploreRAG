from sqlalchemy import String, DateTime, Text, Integer, JSON, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from typing import TYPE_CHECKING

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.document import Document


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    kg_language: Mapped[str | None] = mapped_column(String(50), nullable=True, default=None)
    kg_entity_types: Mapped[list | None] = mapped_column(JSON, nullable=True, default=None)
    # Generation provider used by chat, captions, and LightRAG extraction.
    # Embeddings/reranking remain local and provider-independent.
    llm_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default="cloud", server_default="cloud"
    )
    # Controls query-time LightRAG evidence injection only.  Documents still
    # enter the graph so visual exploration remains available when disabled.
    lightrag_augmentation_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    # Monotonically increasing workspace generation used to invalidate
    # in-flight temporary attachment jobs during chat-history cleanup.
    chat_cleanup_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Schema for business metadata collected during document import.  The
    # values themselves live on Document.custom_metadata.
    metadata_schema: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=lambda: {"version": 1, "fields": []},
    )
    metadata_schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
