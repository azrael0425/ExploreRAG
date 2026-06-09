"""Temporary attachment parsing, routing, and isolated vector indexing."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_attachment import ChatAttachment, ChatAttachmentState
from app.models.knowledge_base import KnowledgeBase
from app.services.chat_attachment_service import IMAGE_EXTENSIONS, TEXT_EXTENSIONS, ChatAttachmentService
from app.services.chat_cleanup_service import get_workspace_cleanup_lock
from app.services.document_parser.docling_parser import DoclingDocumentParser
from app.services.embedder import get_embedding_service
from app.services.index_chunking import split_attachment_chunks
from app.services.scan_profile_router import ScanProfileRouter
from app.services.temporary_attachment_vector_store import TemporaryAttachmentVectorStore
from app.services.work_scheduler import WorkPriority, get_work_scheduler

logger = logging.getLogger(__name__)
EventEmitter = Callable[[str, dict[str, Any]], Awaitable[None] | None]
_preparation_tasks: dict[tuple[int, str], asyncio.Task[None]] = {}


def enqueue_attachment_preparation(workspace_id: int, attachment_id: str) -> asyncio.Task[None]:
    """Start parsing immediately after upload and reuse the task from chat."""
    key = (workspace_id, attachment_id)
    existing = _preparation_tasks.get(key)
    if existing is not None and not existing.done():
        return existing

    async def run() -> None:
        from app.core.database import async_session_maker

        try:
            async with async_session_maker() as db:
                processor = AttachmentProcessor()
                await processor._process_attachment(db, workspace_id, attachment_id, None)
                attachment = await processor._load_current(db, workspace_id, attachment_id)
                if (
                    attachment.state == ChatAttachmentState.READY_DIRECT
                    and attachment.file_type not in {"jpg", "jpeg", "png"}
                    and attachment.parsed_token_count >= settings.CHAT_ATTACHMENT_EAGER_INDEX_TOKENS
                ):
                    await processor._index_attachment(db, workspace_id, attachment_id, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background attachment preparation failed for %s", attachment_id)
        finally:
            _preparation_tasks.pop(key, None)

    task = asyncio.create_task(run(), name=f"prepare-attachment-{attachment_id}")
    _preparation_tasks[key] = task
    return task


class AttachmentInvalidated(RuntimeError):
    """Raised internally when cleanup moved the workspace generation forward."""


class AttachmentProcessor:
    """Route attachments without ever creating a durable ``Document`` record."""

    def __init__(self) -> None:
        self.service = ChatAttachmentService()
        self.scheduler = get_work_scheduler()

    async def prepare_for_chat(
        self,
        db: AsyncSession,
        workspace_id: int,
        attachments: list[ChatAttachment],
        direct_token_budget: int,
        emit: EventEmitter | None = None,
    ) -> list[ChatAttachment]:
        """Make selected attachments ready and index only when context requires it."""
        for attachment in attachments:
            if attachment.state == ChatAttachmentState.FAILED:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Attachment '{attachment.original_filename}' failed previously: {attachment.error_message or 'unknown error'}",
                )
            task = _preparation_tasks.get((workspace_id, attachment.id))
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=settings.CHAT_ATTACHMENT_PREPARE_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Attachment '{attachment.original_filename}' is still being prepared.",
                    ) from exc
            elif attachment.state in {ChatAttachmentState.QUEUED, ChatAttachmentState.PARSING}:
                # No live task means the process restarted during preparation.
                # Safely reset the recoverable state and resume from source.
                current = await self._load_current(db, workspace_id, attachment.id)
                current.state = ChatAttachmentState.UPLOADED
                await db.commit()
                await self._process_attachment(db, workspace_id, attachment.id, emit)
            elif attachment.state == ChatAttachmentState.UPLOADED:
                await self._process_attachment(db, workspace_id, attachment.id, emit)

        ready = await self.service.selected(db, workspace_id, [attachment.id for attachment in attachments])
        direct_text = [attachment for attachment in ready if attachment.state == ChatAttachmentState.READY_DIRECT and attachment.file_type not in {"jpg", "jpeg", "png"}]
        total_direct_tokens = sum(attachment.parsed_token_count for attachment in direct_text)
        if total_direct_tokens > direct_token_budget:
            for attachment in direct_text:
                await self._index_attachment(db, workspace_id, attachment.id, emit)
            ready = await self.service.selected(db, workspace_id, [attachment.id for attachment in attachments])
        return ready

    async def _process_attachment(
        self, db: AsyncSession, workspace_id: int, attachment_id: str, emit: EventEmitter | None
    ) -> None:
        attachment = await self._load_current(db, workspace_id, attachment_id)
        if attachment.state != ChatAttachmentState.UPLOADED:
            return
        expected_epoch = attachment.cleanup_epoch
        await self._set_state(db, workspace_id, attachment_id, expected_epoch, ChatAttachmentState.QUEUED)
        await self._emit(emit, "attachment_queued", attachment, "Waiting for document parsing resources")

        suffix = f".{attachment.file_type.lower()}"
        try:
            if suffix in IMAGE_EXTENSIONS:
                await self._publish_direct_image(db, workspace_id, attachment_id, expected_epoch)
                await self._emit(emit, "attachment_ready", attachment, "Image is ready for Qwen Vision")
                return
            if suffix in TEXT_EXTENSIONS:
                await self._emit(emit, "attachment_parsing", attachment, "Reading text attachment")
                parsed = await asyncio.to_thread(self._parse_text_sync, attachment)
            else:
                await self._set_state(db, workspace_id, attachment_id, expected_epoch, ChatAttachmentState.PARSING)
                await self._emit(emit, "attachment_parsing", attachment, "Parsing with Docling")
                llm_mode = (
                    await db.execute(
                        select(KnowledgeBase.llm_mode).where(
                            KnowledgeBase.id == workspace_id
                        )
                    )
                ).scalar_one_or_none() or "cloud"
                parsed = await self.scheduler.run(
                    "docling",
                    WorkPriority.ATTACHMENT,
                    lambda: asyncio.to_thread(
                        self._parse_docling_sync, attachment, llm_mode
                    ),
                    workspace_id=workspace_id,
                    cancel_on_cleanup=True,
                )
                if parsed.get("manifest", {}).get("ocr_retry"):
                    await self._emit(emit, "attachment_ocr_retry", attachment, "Low-confidence PDF pages were retried with OCR")
            await self._publish_parse_result(db, workspace_id, attachment_id, expected_epoch, parsed)
            await self._emit(emit, "attachment_ready", attachment, "Attachment parsed and ready")
        except AttachmentInvalidated:
            logger.info("Dropped invalidated attachment result %s", attachment_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Temporary attachment parse failed for %s", attachment_id)
            await self._mark_failed(db, workspace_id, attachment_id, expected_epoch, str(exc))
            await self._emit(emit, "error", attachment, f"Attachment parse failed: {exc}")

    def _parse_text_sync(self, attachment: ChatAttachment) -> dict:
        source = Path(attachment.storage_path).read_bytes()
        markdown = self.service.decode_text(source).replace("\r\n", "\n").replace("\r", "\n")
        return {
            "markdown": markdown,
            "token_count": self._estimate_tokens(markdown),
            "chunks": self._text_chunks(markdown, attachment),
            "stage": None,
            "manifest": {"parser": "text", "scan_profile": None, "page_count": 0, "low_confidence_pages": []},
        }

    def _parse_docling_sync(
        self,
        attachment: ChatAttachment,
        llm_mode: str = "cloud",
    ) -> dict:
        artifact_root = Path(attachment.artifact_dir)
        stage = self.service.root / ".staging" / f"ws_{attachment.workspace_id}" / f"{attachment.id}-{int(time.time() * 1000)}"
        stage.mkdir(parents=True, exist_ok=True)
        source = Path(attachment.storage_path)
        suffix = source.suffix.lower()
        scan_router = ScanProfileRouter()
        profile = scan_router.inspect(source) if suffix == ".pdf" else None
        media_url_prefix = f"/api/v1/rag/chat/{attachment.workspace_id}/attachments/{attachment.id}/files"

        parser = DoclingDocumentParser(
            workspace_id=attachment.workspace_id,
            output_dir=stage,
            media_url_prefix=media_url_prefix,
            llm_mode=llm_mode,
        )
        scan_profile = profile.profile if profile else "normal_pdf"
        parsed = parser.parse(str(source), attachment.id, attachment.original_filename, scan_profile=scan_profile)
        retry_used = False
        selective_ocr_pages: list[int] = []
        # A mixed PDF no longer runs through full-page OCR twice. Only pages
        # that pypdf identified as low-confidence are copied into a compact PDF
        # and sent through the scanned profile, then merged back by page number.
        if profile and profile.profile == "mixed_pdf" and profile.low_confidence_pages:
            retry_dir = stage / "ocr-retry"
            retry_source = retry_dir / "source" / "selected-pages.pdf"
            selective_ocr_pages = scan_router.extract_pages(
                source,
                profile.low_confidence_pages,
                retry_source,
            )
            if selective_ocr_pages:
                retry_parser = DoclingDocumentParser(
                    workspace_id=attachment.workspace_id,
                    output_dir=retry_dir,
                    media_url_prefix=media_url_prefix,
                    llm_mode=llm_mode,
                )
                retry = retry_parser.parse(
                    str(retry_source),
                    attachment.id,
                    attachment.original_filename,
                    scan_profile="scanned_pdf",
                )
                self._merge_selective_ocr(parsed, retry, selective_ocr_pages, stage, retry_dir)
                retry_used = True

        rendered_pages: list[int] = []
        if profile and profile.profile in {"scanned_pdf", "mixed_pdf"}:
            pages_to_render = profile.low_confidence_pages if profile.profile == "mixed_pdf" else None
            rendered_pages = scan_router.render_pages(source, stage / "pages", pages_to_render)

        parsed_dir = stage / "parsed"
        parsed_dir.mkdir(parents=True, exist_ok=True)
        chunks = [
            {
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_no": chunk.page_no,
                "heading_path": chunk.heading_path,
                "table_ids": chunk.table_refs,
                "image_ids": chunk.image_refs,
                "has_table": chunk.has_table,
                "has_code": chunk.has_code,
            }
            for chunk in parsed.chunks
        ]
        (parsed_dir / "document.md").write_text(parsed.markdown, encoding="utf-8")
        (parsed_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        manifest = {
            **parser.last_parse_manifest,
            "scan": profile.manifest() if profile else None,
            "ocr_retry": retry_used,
            "selective_ocr_pages": selective_ocr_pages,
            "rendered_pages": rendered_pages,
            "images": len(parsed.images),
            "tables": parsed.tables_count,
        }
        return {
            "markdown": parsed.markdown,
            "token_count": self._estimate_tokens(parsed.markdown),
            "chunks": chunks,
            "stage": str(stage),
            "manifest": manifest,
        }

    @staticmethod
    def _merge_selective_ocr(parsed, retry, actual_pages: list[int], stage: Path, retry_dir: Path) -> None:
        page_map = {index + 1: page for index, page in enumerate(actual_pages)}
        recovered_chunks = []
        for chunk in retry.chunks:
            actual_page = page_map.get(chunk.page_no, actual_pages[0])
            recovered_chunks.append(replace(
                chunk,
                content=f"[OCR recovery for page {actual_page}]\n{chunk.content}",
                chunk_index=len(parsed.chunks) + len(recovered_chunks),
                page_no=actual_page,
            ))
        parsed.chunks.extend(recovered_chunks)

        for image in retry.images:
            image.page_no = page_map.get(image.page_no, actual_pages[0])
        for table in retry.tables:
            table.page_no = page_map.get(table.page_no, actual_pages[0])
        parsed.images.extend(retry.images)
        parsed.tables.extend(retry.tables)
        parsed.tables_count = len(parsed.tables)
        if retry.markdown.strip():
            page_label = ", ".join(str(page) for page in actual_pages)
            parsed.markdown += f"\n\n## OCR recovery (pages {page_label})\n\n{retry.markdown}"

        # Publishing copies the main staging directories. Merge retry images
        # into that directory while their stable image IDs prevent collisions.
        retry_images = retry_dir / "images"
        target_images = stage / "images"
        if retry_images.exists():
            target_images.mkdir(parents=True, exist_ok=True)
            for source_image in retry_images.iterdir():
                if source_image.is_file():
                    shutil.copy2(source_image, target_images / source_image.name)

    async def _publish_direct_image(self, db: AsyncSession, workspace_id: int, attachment_id: str, epoch: int) -> None:
        async with get_workspace_cleanup_lock(workspace_id):
            attachment = await self._load_current(db, workspace_id, attachment_id)
            await self._assert_epoch(db, workspace_id, attachment, epoch)
            attachment.state = ChatAttachmentState.READY_DIRECT
            attachment.parsed_token_count = 0
            await db.commit()

    async def _publish_parse_result(
        self, db: AsyncSession, workspace_id: int, attachment_id: str, epoch: int, parsed: dict
    ) -> None:
        async with get_workspace_cleanup_lock(workspace_id):
            attachment = await self._load_current(db, workspace_id, attachment_id)
            await self._assert_epoch(db, workspace_id, attachment, epoch)
            target = Path(attachment.artifact_dir)
            stage = Path(parsed["stage"]) if parsed.get("stage") else None
            if stage:
                for name in ("parsed", "images", "pages"):
                    source_dir = stage / name
                    if source_dir.exists():
                        destination = target / name
                        if destination.exists():
                            shutil.rmtree(destination)
                        shutil.copytree(source_dir, destination)
                shutil.rmtree(stage.parent if stage.name == "ocr-retry" else stage, ignore_errors=True)
            else:
                parsed_dir = target / "parsed"
                parsed_dir.mkdir(parents=True, exist_ok=True)
                (parsed_dir / "document.md").write_text(parsed["markdown"], encoding="utf-8")
                (parsed_dir / "chunks.json").write_text(json.dumps(parsed["chunks"], ensure_ascii=False), encoding="utf-8")
            manifest_path = target / "manifest.json"
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            old_manifest.update({
                "state": ChatAttachmentState.READY_DIRECT.value,
                "parsed_token_count": parsed["token_count"],
                "parse": parsed["manifest"],
            })
            manifest_path.write_text(json.dumps(old_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            attachment.parsed_token_count = parsed["token_count"]
            attachment.state = ChatAttachmentState.READY_DIRECT
            attachment.error_message = None
            await db.commit()
            parse_manifest = parsed.get("manifest", {})
            scan_info = parse_manifest.get("scan") or {}
            logger.info(
                "Parsed temporary attachment %s (tokens=%s chunks=%s scan=%s)",
                attachment.id,
                parsed["token_count"],
                len(parsed.get("chunks", [])),
                parse_manifest.get("scan_profile") or scan_info.get("profile"),
            )

    async def _index_attachment(
        self, db: AsyncSession, workspace_id: int, attachment_id: str, emit: EventEmitter | None
    ) -> None:
        attachment = await self._load_current(db, workspace_id, attachment_id)
        if attachment.state == ChatAttachmentState.INDEXED_TEMP:
            return
        if attachment.state != ChatAttachmentState.READY_DIRECT:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Attachment is not ready to index")
        epoch = attachment.cleanup_epoch
        await self._emit(emit, "attachment_indexing", attachment, "Creating temporary retrieval index")
        chunks_path = Path(attachment.artifact_dir) / "parsed" / "chunks.json"
        if not chunks_path.exists():
            raise ValueError("Attachment parser output is missing")
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embedder = get_embedding_service()
        chunks, split_stats = await asyncio.to_thread(
            split_attachment_chunks,
            chunks,
            embedder,
        )
        logger.info(
            "Attachment %s chunk guard input=%s output=%s oversized=%s max_tokens=%s->%s",
            attachment_id,
            split_stats["input"],
            split_stats["output"],
            split_stats["oversized"],
            split_stats["max_tokens_before"],
            split_stats["max_tokens_after"],
        )
        if len(chunks) > settings.CHAT_ATTACHMENT_MAX_CHUNKS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Attachment exceeds the {settings.CHAT_ATTACHMENT_MAX_CHUNKS}-chunk temporary index limit.",
            )
        if not chunks:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Attachment has no retrievable text")

        texts = [chunk["content"] for chunk in chunks]
        batch_size = settings.EXPLORERAG_EMBEDDING_BATCH_SIZE
        embedding_started_at = time.perf_counter()
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_embeddings = await self.scheduler.run(
                "embedding",
                WorkPriority.ATTACHMENT,
                lambda batch=batch: asyncio.to_thread(embedder.embed_texts, batch),
                workspace_id=workspace_id,
                cancel_on_cleanup=True,
            )
            embeddings.extend(batch_embeddings)
        embedding_ms = int((time.perf_counter() - embedding_started_at) * 1000)

        async with get_workspace_cleanup_lock(workspace_id):
            attachment = await self._load_current(db, workspace_id, attachment_id)
            await self._assert_epoch(db, workspace_id, attachment, epoch)
            store = TemporaryAttachmentVectorStore(workspace_id)
            ids = [f"att_{attachment.id}_chunk_{chunk['chunk_index']}" for chunk in chunks]
            metadatas = [
                {
                    "attachment_id": attachment.id,
                    "workspace_id": workspace_id,
                    "page_no": int(chunk.get("page_no", 0) or 0),
                    "heading_path": " > ".join(chunk.get("heading_path") or []),
                    "table_ids": "|".join(chunk.get("table_ids") or []),
                    "image_ids": "|".join(chunk.get("image_ids") or []),
                    "has_table": bool(chunk.get("has_table", False)),
                    "cleanup_epoch": epoch,
                    "source_file": attachment.original_filename,
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                }
                for chunk in chunks
            ]
            await asyncio.to_thread(store.add_documents, ids, embeddings, texts, metadatas)
            await self._assert_epoch(db, workspace_id, attachment, epoch)
            attachment.state = ChatAttachmentState.INDEXED_TEMP
            attachment.temp_collection = store.collection_name
            await db.commit()
            logger.info(
                "Indexed %s temporary chunks for attachment %s in %s (embedding_ms=%s batch_size=%s)",
                len(chunks), attachment.id, store.collection_name, embedding_ms, batch_size,
            )
        await self._emit(emit, "attachment_ready", attachment, "Temporary retrieval index is ready")

    async def _set_state(
        self, db: AsyncSession, workspace_id: int, attachment_id: str, epoch: int, state: ChatAttachmentState
    ) -> None:
        async with get_workspace_cleanup_lock(workspace_id):
            attachment = await self._load_current(db, workspace_id, attachment_id)
            await self._assert_epoch(db, workspace_id, attachment, epoch)
            attachment.state = state
            await db.commit()

    async def _mark_failed(
        self, db: AsyncSession, workspace_id: int, attachment_id: str, epoch: int, message: str
    ) -> None:
        try:
            async with get_workspace_cleanup_lock(workspace_id):
                attachment = await self._load_current(db, workspace_id, attachment_id)
                await self._assert_epoch(db, workspace_id, attachment, epoch)
                attachment.state = ChatAttachmentState.FAILED
                attachment.error_message = message[:1000]
                await db.commit()
        except AttachmentInvalidated:
            return

    async def _load_current(self, db: AsyncSession, workspace_id: int, attachment_id: str) -> ChatAttachment:
        attachment = (await db.execute(select(ChatAttachment).where(
            ChatAttachment.id == attachment_id,
            ChatAttachment.workspace_id == workspace_id,
        ))).scalar_one_or_none()
        if attachment is None:
            raise AttachmentInvalidated(f"Attachment {attachment_id} was cleared")
        return attachment

    async def _assert_epoch(
        self, db: AsyncSession, workspace_id: int, attachment: ChatAttachment, expected_epoch: int
    ) -> None:
        current_epoch = (await db.execute(
            select(KnowledgeBase.chat_cleanup_epoch).where(KnowledgeBase.id == workspace_id)
        )).scalar_one_or_none()
        if (
            current_epoch != expected_epoch
            or attachment.cleanup_epoch != expected_epoch
            or attachment.state in {ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED}
            or attachment.cleanup_pending
        ):
            raise AttachmentInvalidated(f"Attachment {attachment.id} no longer belongs to current workspace generation")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Approximation suitable for routing/limits across Chinese and Latin
        # content.  The actual provider still enforces its own token limit.
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        return max(1, cjk + max(0, len(text) - cjk) // 4)

    def _text_chunks(self, markdown: str, attachment: ChatAttachment) -> list[dict]:
        max_chars = settings.EXPLORERAG_CHUNK_MAX_TOKENS * 4
        blocks = re.split(r"\n{2,}", markdown)
        chunks: list[str] = []
        current = ""
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            # A Markdown table is indivisible evidence.  Keep it as one block
            # even if it is long rather than silently reducing it to a caption.
            if block.startswith("|") and "|" in block:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(block)
                continue
            candidate = f"{current}\n\n{block}".strip() if current else block
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                while len(block) > max_chars:
                    boundary = block.rfind("\n", 0, max_chars)
                    boundary = boundary if boundary > max_chars // 2 else max_chars
                    chunks.append(block[:boundary].strip())
                    block = block[boundary:].strip()
                current = block
        if current:
            chunks.append(current)
        return [
            {
                "content": content,
                "chunk_index": index,
                "page_no": 0,
                "heading_path": [],
                "table_ids": [],
                "image_ids": [],
                "has_table": content.startswith("|"),
                "has_code": False,
            }
            for index, content in enumerate(chunks)
        ]

    @staticmethod
    async def _emit(emit: EventEmitter | None, event: str, attachment: ChatAttachment, detail: str) -> None:
        if emit is None:
            return
        payload = {
            "attachment_id": attachment.id,
            "filename": attachment.original_filename,
            "state": attachment.state.value if hasattr(attachment.state, "value") else str(attachment.state),
            "detail": detail,
        }
        result = emit(event, payload)
        if asyncio.iscoroutine(result):
            await result
