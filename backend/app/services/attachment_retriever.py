"""Retrieve only from temporary attachment namespaces, separately from KB."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.models.chat_attachment import ChatAttachment, ChatAttachmentState
from app.services.embedder import get_embedding_service
from app.services.llm.types import LLMImagePart
from app.services.reranker import get_reranker_service
from app.services.temporary_attachment_vector_store import TemporaryAttachmentVectorStore
from app.services.work_scheduler import WorkPriority, get_work_scheduler

logger = logging.getLogger(__name__)


@dataclass
class AttachmentChunk:
    attachment_id: str
    source_file: str
    content: str
    chunk_id: str
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)
    score: float = 0.0
    has_table: bool = False
    image_ids: list[str] = field(default_factory=list)


@dataclass
class AttachmentImage:
    attachment_id: str
    image_id: str
    page_no: int
    caption: str
    url: str
    data: bytes
    mime_type: str


@dataclass
class AttachmentRetrievalResult:
    chunks: list[AttachmentChunk] = field(default_factory=list)
    images: list[AttachmentImage] = field(default_factory=list)


class AttachmentRetriever:
    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.scheduler = get_work_scheduler()

    async def retrieve(self, question: str, attachments: list[ChatAttachment]) -> AttachmentRetrievalResult:
        direct = [attachment for attachment in attachments if attachment.state == ChatAttachmentState.READY_DIRECT]
        indexed = [attachment for attachment in attachments if attachment.state == ChatAttachmentState.INDEXED_TEMP]
        chunks: list[AttachmentChunk] = []
        images = self._source_images(direct)

        for attachment in direct:
            if attachment.file_type in {"jpg", "jpeg", "png"}:
                continue
            path = Path(attachment.artifact_dir) / "parsed" / "document.md"
            try:
                chunks.append(AttachmentChunk(
                    attachment_id=attachment.id,
                    source_file=attachment.original_filename,
                    content=path.read_text(encoding="utf-8"),
                    chunk_id=f"att_{attachment.id}_direct",
                ))
            except OSError as exc:
                logger.warning("Could not read direct temporary attachment %s: %s", attachment.id, exc)

        if indexed:
            ids = [attachment.id for attachment in indexed]
            embedder = get_embedding_service()
            embedding = await self.scheduler.run(
                "embedding",
                WorkPriority.CHAT,
                lambda: asyncio.to_thread(embedder.embed_query, question),
                workspace_id=self.workspace_id,
                cancel_on_cleanup=True,
            )
            raw = await asyncio.to_thread(
                TemporaryAttachmentVectorStore(self.workspace_id).query,
                embedding,
                ids,
                settings.CHAT_ATTACHMENT_PREFETCH,
            )
            indexed_chunks = self._chunks_from_query(raw)
            indexed_chunks = await self._rerank(question, indexed_chunks)
            chunks.extend(indexed_chunks)
            images.extend(self._matched_page_images(indexed_chunks, indexed))

        # Inputs are capped independently from KB.  A selected image counts
        # against the same vision quota whether it was an original attachment or
        # a low-confidence/table PDF page.
        result = AttachmentRetrievalResult(
            chunks=chunks[:settings.CHAT_ATTACHMENT_RERANK_TOP_K],
            images=images[:settings.CHAT_ATTACHMENT_MAX_VISION_IMAGES],
        )
        logger.info(
            "Temporary attachment retrieval workspace=%s selected=%s results=%s vision_images=%s",
            self.workspace_id, len(attachments), len(result.chunks), len(result.images),
        )
        return result

    @staticmethod
    def _chunks_from_query(raw: dict) -> list[AttachmentChunk]:
        chunks: list[AttachmentChunk] = []
        for index, content in enumerate(raw.get("documents", [])):
            metadata = raw.get("metadatas", [])[index] if raw.get("metadatas") else {}
            chunks.append(AttachmentChunk(
                attachment_id=str(metadata.get("attachment_id", "")),
                source_file=str(metadata.get("source_file", "")),
                content=content,
                chunk_id=(raw.get("ids", [])[index] if raw.get("ids") else f"attachment_chunk_{index}"),
                page_no=int(metadata.get("page_no", 0) or 0),
                heading_path=[part for part in str(metadata.get("heading_path", "")).split(" > ") if part],
                score=float((raw.get("distances", [])[index] if raw.get("distances") else 0.0) or 0.0),
                has_table=bool(metadata.get("has_table", False)),
                image_ids=[part for part in str(metadata.get("image_ids", "")).split("|") if part],
            ))
        return chunks

    async def _rerank(self, question: str, chunks: list[AttachmentChunk]) -> list[AttachmentChunk]:
        if not chunks:
            return []
        if not settings.EXPLORERAG_ENABLE_RERANKER:
            return chunks[:settings.CHAT_ATTACHMENT_RERANK_TOP_K]
        reranker = get_reranker_service()
        fallback = chunks[:settings.CHAT_ATTACHMENT_RERANK_TOP_K]
        if not reranker.is_available():
            logger.warning(
                "Reranker circuit is open for temporary attachments workspace=%s; using vector order",
                self.workspace_id,
            )
            return fallback
        try:
            ranked = await asyncio.wait_for(
                self.scheduler.run(
                    "reranker",
                    WorkPriority.CHAT,
                    lambda: asyncio.to_thread(
                        reranker.rerank,
                        question,
                        [chunk.content for chunk in chunks],
                        settings.CHAT_ATTACHMENT_RERANK_TOP_K,
                        settings.EXPLORERAG_MIN_RELEVANCE_SCORE,
                    ),
                    workspace_id=self.workspace_id,
                    cancel_on_cleanup=True,
                ),
                timeout=settings.EXPLORERAG_RERANKER_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            reranker.record_failure("attachment inference timeout")
            logger.warning(
                "Attachment reranker exceeded %.1fs for workspace=%s; using vector order",
                settings.EXPLORERAG_RERANKER_TIMEOUT_SECONDS,
                self.workspace_id,
            )
            return fallback
        except Exception as exc:
            reranker.record_failure(type(exc).__name__)
            logger.exception(
                "Attachment reranker failed for workspace=%s; using vector order",
                self.workspace_id,
            )
            return fallback

        reranker.record_success()
        return [chunks[item.index] for item in ranked] if ranked else chunks[:settings.CHAT_ATTACHMENT_RERANK_TOP_K]

    def _source_images(self, attachments: list[ChatAttachment]) -> list[AttachmentImage]:
        images: list[AttachmentImage] = []
        for attachment in attachments:
            if attachment.file_type not in {"jpg", "jpeg", "png"}:
                continue
            path = Path(attachment.storage_path)
            try:
                images.append(AttachmentImage(
                    attachment_id=attachment.id,
                    image_id=attachment.id,
                    page_no=0,
                    caption=attachment.original_filename,
                    url=f"/api/v1/rag/chat/{self.workspace_id}/attachments/{attachment.id}/files/source/{path.name}",
                    data=path.read_bytes(),
                    mime_type="image/jpeg" if attachment.file_type in {"jpg", "jpeg"} else "image/png",
                ))
            except OSError as exc:
                logger.warning("Could not read temporary image %s: %s", attachment.id, exc)
        return images

    def _matched_page_images(
        self, chunks: list[AttachmentChunk], attachments: list[ChatAttachment]
    ) -> list[AttachmentImage]:
        attachment_by_id = {attachment.id: attachment for attachment in attachments}
        result: list[AttachmentImage] = []
        seen: set[tuple[str, int]] = set()
        for chunk in chunks:
            if not (chunk.page_no and (chunk.has_table or chunk.image_ids)):
                continue
            key = (chunk.attachment_id, chunk.page_no)
            if key in seen:
                continue
            seen.add(key)
            attachment = attachment_by_id.get(chunk.attachment_id)
            if attachment is None:
                continue
            path = Path(attachment.artifact_dir) / "pages" / f"page_{chunk.page_no}.png"
            if not path.exists():
                continue
            try:
                result.append(AttachmentImage(
                    attachment_id=attachment.id,
                    image_id=f"{attachment.id}-page-{chunk.page_no}",
                    page_no=chunk.page_no,
                    caption=f"{attachment.original_filename}, page {chunk.page_no}",
                    url=f"/api/v1/rag/chat/{self.workspace_id}/attachments/{attachment.id}/files/pages/{path.name}",
                    data=path.read_bytes(),
                    mime_type="image/png",
                ))
            except OSError as exc:
                logger.warning("Could not read temporary PDF page image %s: %s", path, exc)
        return result
