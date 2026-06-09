"""Upload validation and metadata access for isolated chat attachments."""
from __future__ import annotations

import io
import json
import logging
import uuid
import zipfile
from pathlib import Path

import aiofiles
from fastapi import HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.chat_attachment import ChatAttachment, ChatAttachmentState
from app.models.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

ALLOWED_ATTACHMENT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".docx", ".pptx", ".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_EXTENSIONS = {".txt", ".md"}
DOCLING_ATTACHMENT_EXTENSIONS = {".pdf", ".docx", ".pptx"}


class ChatAttachmentService:
    root = settings.BASE_DIR / "data" / "chat-attachments"

    @classmethod
    def workspace_root(cls, workspace_id: int) -> Path:
        return cls.root / f"ws_{workspace_id}"

    @classmethod
    def attachment_root(cls, workspace_id: int, attachment_id: str) -> Path:
        return cls.workspace_root(workspace_id) / attachment_id

    async def upload(self, db: AsyncSession, workspace_id: int, file: UploadFile) -> ChatAttachment:
        workspace = (await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
        )).scalar_one_or_none()
        if workspace is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")

        safe_name = Path(file.filename or "attachment").name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_ATTACHMENT_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported chat attachment type. Allowed: JPG, PNG, PDF, DOCX, PPTX, TXT, MD.",
            )

        active_count = (await db.execute(
            select(func.count(ChatAttachment.id)).where(
                ChatAttachment.workspace_id == workspace_id,
                ChatAttachment.state.notin_([ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED]),
            )
        )).scalar_one()
        if active_count >= settings.CHAT_ATTACHMENT_MAX_COUNT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"At most {settings.CHAT_ATTACHMENT_MAX_COUNT} temporary attachments can be retained per workspace.",
            )

        content = await file.read()
        image_pixels = self._validate_content(safe_name, suffix, file.content_type or "", content)
        if image_pixels:
            existing_images = list((await db.execute(
                select(ChatAttachment.storage_path).where(
                    ChatAttachment.workspace_id == workspace_id,
                    ChatAttachment.file_type.in_(["jpg", "jpeg", "png"]),
                    ChatAttachment.state.notin_([ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED]),
                )
            )).scalars().all())
            total_pixels = image_pixels + sum(self._image_pixels(Path(path)) for path in existing_images)
            if total_pixels > settings.CHAT_ATTACHMENT_MAX_TOTAL_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected temporary images exceed the workspace pixel limit.",
                )

        attachment_id = str(uuid.uuid4())
        attachment_root = self.attachment_root(workspace_id, attachment_id)
        source_dir = attachment_root / "source"
        source_dir.mkdir(parents=True, exist_ok=False)
        source_path = source_dir / f"{uuid.uuid4()}{suffix}"
        try:
            async with aiofiles.open(source_path, "wb") as stream:
                await stream.write(content)
            manifest = {
                "attachment_id": attachment_id,
                "workspace_id": workspace_id,
                "original_filename": safe_name,
                "file_type": suffix[1:],
                "file_size": len(content),
                "image_pixels": image_pixels,
                "state": ChatAttachmentState.UPLOADED.value,
            }
            (attachment_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            # The directory is unique to this upload; leave no partial object
            # behind when an upload write fails.
            import shutil
            shutil.rmtree(attachment_root, ignore_errors=True)
            raise

        attachment = ChatAttachment(
            id=attachment_id,
            workspace_id=workspace_id,
            original_filename=safe_name,
            file_type=suffix[1:],
            file_size=len(content),
            storage_path=str(source_path),
            artifact_dir=str(attachment_root),
            state=ChatAttachmentState.UPLOADED,
            cleanup_epoch=workspace.chat_cleanup_epoch,
        )
        db.add(attachment)
        await db.commit()
        await db.refresh(attachment)
        logger.info(
            "Uploaded isolated chat attachment %s in workspace %s (type=%s bytes=%s retained=%s)",
            attachment.id, workspace_id, attachment.file_type, attachment.file_size, active_count + 1,
        )
        return attachment

    def _validate_content(self, filename: str, suffix: str, content_type: str, content: bytes) -> int:
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment is empty.")
        if len(content) > settings.CHAT_ATTACHMENT_MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Attachment exceeds the {settings.CHAT_ATTACHMENT_MAX_FILE_SIZE // 1024 // 1024} MB limit.",
            )

        if suffix in {".jpg", ".jpeg"} and not content.startswith(b"\xff\xd8\xff"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File content is not a valid JPEG.")
        if suffix == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File content is not a valid PNG.")
        image_pixels = self._validate_image_pixels(content) if suffix in IMAGE_EXTENSIONS else 0
        if suffix == ".pdf":
            if not content.startswith(b"%PDF-"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File content is not a valid PDF.")
            self._validate_pdf_pages(content)
        if suffix in {".docx", ".pptx"}:
            self._validate_office_zip(suffix, content)
        if suffix in TEXT_EXTENSIONS:
            if b"\x00" in content:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TXT/MD attachment contains binary data.")
            self.decode_text(content)  # validates supported text encodings

        # MIME is intentionally a second signal, not the only signal: browsers
        # commonly send application/octet-stream for valid Office documents.
        expected_prefixes = {
            ".jpg": ("image/", "application/octet-stream"),
            ".jpeg": ("image/", "application/octet-stream"),
            ".png": ("image/", "application/octet-stream"),
            ".pdf": ("application/pdf", "application/octet-stream"),
            ".docx": ("application/vnd", "application/zip", "application/octet-stream"),
            ".pptx": ("application/vnd", "application/zip", "application/octet-stream"),
            ".txt": ("text/", "application/octet-stream"),
            ".md": ("text/", "application/octet-stream"),
        }[suffix]
        if content_type and not any(content_type.startswith(prefix) for prefix in expected_prefixes):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Attachment MIME type does not match its extension.")
        return image_pixels

    @staticmethod
    def decode_text(content: bytes) -> str:
        # ``utf-16`` will decode many arbitrary even-length byte strings into
        # nonsense, so only select it when a BOM or its characteristic NUL-byte
        # pattern is present.  Otherwise prefer UTF-8 and then GB18030.
        if content.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return content.decode("utf-16")
            except UnicodeDecodeError:
                pass
        if len(content) >= 4 and (content[1::2].count(0) > len(content) // 8 or content[::2].count(0) > len(content) // 8):
            for encoding in ("utf-16-le", "utf-16-be"):
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
        for encoding in ("utf-8-sig", "gb18030"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="TXT/MD must use UTF-8, UTF-16, or GB18030 encoding.")

    @staticmethod
    def _validate_image_pixels(content: bytes) -> int:
        try:
            from PIL import Image
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                if width * height > settings.CHAT_ATTACHMENT_MAX_TOTAL_PIXELS:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Image exceeds the temporary attachment pixel limit.",
                    )
                return width * height
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Image cannot be decoded safely: {exc}")

    @staticmethod
    def _image_pixels(path: Path) -> int:
        try:
            from PIL import Image
            with Image.open(path) as image:
                width, height = image.size
                return width * height
        except Exception:
            return 0

    def _validate_pdf_pages(self, content: bytes) -> None:
        try:
            from pypdf import PdfReader
            page_count = len(PdfReader(io.BytesIO(content)).pages)
            if page_count > settings.CHAT_ATTACHMENT_MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"PDF exceeds the {settings.CHAT_ATTACHMENT_MAX_PDF_PAGES}-page limit.",
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Could not inspect PDF page count during upload: %s", exc)

    def _validate_office_zip(self, suffix: str, content: bytes) -> None:
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
            infos = archive.infolist()
        except zipfile.BadZipFile:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Office attachment is not a valid ZIP package.")
        if len(infos) > settings.CHAT_ATTACHMENT_MAX_ZIP_ENTRIES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Office attachment has too many ZIP entries.")
        total = sum(info.file_size for info in infos)
        compressed = max(1, sum(info.compress_size for info in infos))
        if total > settings.CHAT_ATTACHMENT_MAX_ZIP_UNCOMPRESSED_SIZE:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Office attachment expands beyond the allowed size.")
        if total / compressed > settings.CHAT_ATTACHMENT_MAX_ZIP_RATIO:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Office attachment has an unsafe ZIP compression ratio.")
        names = archive.namelist()
        expected_dir = "word/" if suffix == ".docx" else "ppt/"
        if "[Content_Types].xml" not in names or not any(name.startswith(expected_dir) for name in names):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Office attachment package does not match its extension.")

    async def list(self, db: AsyncSession, workspace_id: int) -> list[ChatAttachment]:
        result = await db.execute(
            select(ChatAttachment)
            .where(
                ChatAttachment.workspace_id == workspace_id,
                ChatAttachment.state.notin_([ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED]),
            )
            .order_by(ChatAttachment.created_at.asc())
        )
        return list(result.scalars().all())

    async def selected(
        self, db: AsyncSession, workspace_id: int, attachment_ids: list[str]
    ) -> list[ChatAttachment]:
        if len(attachment_ids) > settings.CHAT_ATTACHMENT_MAX_COUNT:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Too many selected attachments.")
        unique_ids = list(dict.fromkeys(attachment_ids))
        if not unique_ids:
            return []
        result = await db.execute(
            select(ChatAttachment).where(
                ChatAttachment.workspace_id == workspace_id,
                ChatAttachment.id.in_(unique_ids),
            )
        )
        found = {attachment.id: attachment for attachment in result.scalars().all()}
        missing = [attachment_id for attachment_id in unique_ids if attachment_id not in found]
        if missing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="One or more temporary attachments do not exist in this workspace.")
        unavailable = [
            attachment.original_filename for attachment in found.values()
            if attachment.state in {ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED}
            or attachment.cleanup_pending
        ]
        if unavailable:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Selected attachment is being cleared: " + ", ".join(unavailable))
        return [found[attachment_id] for attachment_id in unique_ids]

    @staticmethod
    def serialize(attachment: ChatAttachment) -> dict:
        return {
            "attachment_id": attachment.id,
            "original_filename": attachment.original_filename,
            "file_type": attachment.file_type,
            "file_size": attachment.file_size,
            "state": attachment.state.value if hasattr(attachment.state, "value") else str(attachment.state),
            "parsed_token_count": attachment.parsed_token_count,
            "error_message": attachment.error_message,
            "created_at": attachment.created_at.isoformat() if attachment.created_at else "",
        }

    async def get_file(self, db: AsyncSession, workspace_id: int, attachment_id: str, relative_path: str) -> Path:
        attachment = (await db.execute(select(ChatAttachment).where(
            ChatAttachment.id == attachment_id,
            ChatAttachment.workspace_id == workspace_id,
            ChatAttachment.state.notin_([ChatAttachmentState.CLEARING, ChatAttachmentState.DELETED]),
            ChatAttachment.cleanup_pending.is_(False),
        ))).scalar_one_or_none()
        if attachment is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Temporary attachment not found")
        root = Path(attachment.artifact_dir).resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid attachment file path")
        if not candidate.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment artifact not found")
        return candidate
