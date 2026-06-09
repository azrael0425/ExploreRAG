"""
Document Parser Base
====================

Abstract interface for document parsers. Follows the same provider pattern
as ``services/llm/base.py`` so parsers are swappable via config.
"""
from __future__ import annotations

import logging
import hashlib
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_caption_cache: OrderedDict[str, str] = OrderedDict()
_caption_cache_lock = threading.Lock()


def _caption_cache_get(key: str) -> str | None:
    if settings.EXPLORERAG_CAPTION_CACHE_SIZE <= 0:
        return None
    with _caption_cache_lock:
        value = _caption_cache.get(key)
        if value is not None:
            _caption_cache.move_to_end(key)
        return value


def _caption_cache_put(key: str, value: str) -> None:
    limit = settings.EXPLORERAG_CAPTION_CACHE_SIZE
    if limit <= 0 or not value:
        return
    with _caption_cache_lock:
        _caption_cache[key] = value
        _caption_cache.move_to_end(key)
        while len(_caption_cache) > limit:
            _caption_cache.popitem(last=False)


class BaseDocumentParser(ABC):
    """Abstract base for the Docling document parser."""

    # Subclasses set this so explore_rag_service can store the parser name.
    parser_name: str = "unknown"

    def __init__(
        self,
        workspace_id: int,
        output_dir: Optional[Path] = None,
        llm_mode: str = "cloud",
    ):
        self.workspace_id = workspace_id
        self.llm_mode = llm_mode
        self.output_dir = output_dir or (
            settings.BASE_DIR / "data" / self.parser_name / f"kb_{workspace_id}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse a document and return a unified ``ParsedDocument``."""
        ...

    @staticmethod
    @abstractmethod
    def supported_extensions() -> set[str]:
        """Return the set of file extensions this parser handles."""
        ...

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check whether *file_path* can be handled by this parser."""
        return Path(file_path).suffix.lower() in cls.supported_extensions()

    # ------------------------------------------------------------------
    # Shared helpers — used by all concrete parsers
    # ------------------------------------------------------------------

    _IMAGE_CAPTION_PROMPT = (
        "Describe ONLY what you can directly see in this image. "
        "Do NOT infer, assume, or add any information not visible.\n\n"
        "Include:\n"
        "- Type of visual (chart, table, diagram, photo, screenshot, etc.)\n"
        "- ALL specific numbers, percentages, and labels that are VISIBLE in the image\n"
        "- Axis labels, legend text, and category names exactly as shown\n"
        "- Trends or comparisons that are visually obvious\n\n"
        "RULES:\n"
        "- Write 2-4 concise sentences, max 400 characters.\n"
        "- Do NOT start with 'This image shows' or 'Here is'.\n"
        "- Do NOT add any data, context, or interpretation beyond what is visible.\n"
        "- If text in the image is not clearly readable, say so.\n"
        "- Write in the SAME LANGUAGE as any text visible in the image. "
        "If the text is in Chinese, write in Chinese. "
        "If it is in English, write in English."
    )

    _TABLE_CAPTION_PROMPT = (
        "You are a document analysis assistant. Given a markdown table, "
        "write a concise description that covers:\n"
        "- The purpose/topic of the table\n"
        "- Key column names and what they represent\n"
        "- Notable values, trends, or outliers\n\n"
        "RULES:\n"
        "- Write 2-4 sentences, max 500 characters.\n"
        "- Be factual — describe only what is in the table.\n"
        "- Write in the SAME LANGUAGE as the table content. "
        "If the table is in Chinese, write in Chinese. "
        "If it is in English, write in English.\n\n"
        "Table:\n"
    )

    def _caption_images(self, images: list[ExtractedImage]) -> None:
        """Caption images with bounded parallelism and a content-hash cache."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMImagePart, LLMMessage

        provider = get_llm_provider(self.llm_mode)
        if not provider.supports_vision():
            logger.warning("LLM provider does not support vision — skipping image captioning")
            return

        def caption_one(img: ExtractedImage) -> None:
            if img.caption:
                return
            try:
                image_path = Path(img.file_path)
                if not image_path.exists():
                    return

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                cache_key = hashlib.sha256(
                    b"image-caption-v1\0"
                    + self._IMAGE_CAPTION_PROMPT.encode("utf-8")
                    + b"\0"
                    + image_bytes
                ).hexdigest()
                cached = _caption_cache_get(cache_key)
                if cached is not None:
                    img.caption = cached
                    return

                message = LLMMessage(
                    role="user",
                    content=self._IMAGE_CAPTION_PROMPT,
                    images=[LLMImagePart(data=image_bytes, mime_type=img.mime_type)],
                )
                result = provider.complete([message])
                if result:
                    img.caption = " ".join(result.strip().split())[:500]
                    _caption_cache_put(cache_key, img.caption)

            except Exception as e:
                logger.debug(f"Failed to caption image {img.image_id}: {e}")

        pending = [image for image in images if not image.caption]
        if not pending:
            return
        started_at = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(settings.EXPLORERAG_CAPTION_MAX_WORKERS, len(pending)),
            thread_name_prefix="image-caption",
        ) as executor:
            list(executor.map(caption_one, pending))
        logger.info(
            "Captioned %s images with max_workers=%s in %sms",
            len(pending),
            min(settings.EXPLORERAG_CAPTION_MAX_WORKERS, len(pending)),
            int((time.perf_counter() - started_at) * 1000),
        )

    def _caption_tables(self, tables: list[ExtractedTable]) -> None:
        """Caption tables with bounded parallelism and a content-hash cache."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider(self.llm_mode)

        def caption_one(tbl: ExtractedTable) -> None:
            if tbl.caption:
                return
            try:
                table_md = tbl.content_markdown
                if len(table_md) > settings.EXPLORERAG_MAX_TABLE_MARKDOWN_CHARS:
                    table_md = table_md[:settings.EXPLORERAG_MAX_TABLE_MARKDOWN_CHARS] + "\n... (truncated)"

                cache_key = hashlib.sha256(
                    ("table-caption-v1\0" + self._TABLE_CAPTION_PROMPT + table_md).encode("utf-8")
                ).hexdigest()
                cached = _caption_cache_get(cache_key)
                if cached is not None:
                    tbl.caption = cached
                    return

                message = LLMMessage(
                    role="user",
                    content=self._TABLE_CAPTION_PROMPT + table_md,
                    images=[],
                )
                result = provider.complete([message])
                if result:
                    tbl.caption = " ".join(result.strip().split())[:500]
                    _caption_cache_put(cache_key, tbl.caption)
            except Exception as e:
                logger.debug(f"Failed to caption table {tbl.table_id}: {e}")

        pending = [table for table in tables if not table.caption]
        if not pending:
            return
        started_at = time.perf_counter()
        with ThreadPoolExecutor(
            max_workers=min(settings.EXPLORERAG_CAPTION_MAX_WORKERS, len(pending)),
            thread_name_prefix="table-caption",
        ) as executor:
            list(executor.map(caption_one, pending))
        logger.info(
            "Captioned %s tables with max_workers=%s in %sms",
            len(pending),
            min(settings.EXPLORERAG_CAPTION_MAX_WORKERS, len(pending)),
            int((time.perf_counter() - started_at) * 1000),
        )

    @staticmethod
    def _inject_table_captions(
        markdown: str, tables: list[ExtractedTable]
    ) -> str:
        """Inject table captions as blockquotes after matching table blocks."""
        if not tables:
            return markdown

        captioned = [t for t in tables if t.caption]
        if not captioned:
            return markdown

        lines = markdown.split("\n")
        result_lines: list[str] = []
        matched_count = 0

        table_lookup: dict[str, ExtractedTable] = {}
        for tbl in captioned:
            tbl_lines = tbl.content_markdown.strip().split("\n")
            for tl in tbl_lines:
                tl_stripped = tl.strip()
                if tl_stripped.startswith("|") and "---" not in tl_stripped:
                    cells = [c.strip() for c in tl_stripped.split("|") if c.strip()]
                    if cells:
                        key = "|".join(cells[:3])
                        table_lookup[key] = tbl
                        break

        i = 0
        while i < len(lines):
            line = lines[i]
            result_lines.append(line)

            if line.strip().startswith("|"):
                table_block_start = i
                while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    result_lines.append(lines[i])

                block_lines = lines[table_block_start:i + 1]
                for bl in block_lines:
                    bl_stripped = bl.strip()
                    if bl_stripped.startswith("|") and "---" not in bl_stripped:
                        cells = [c.strip() for c in bl_stripped.split("|") if c.strip()]
                        if cells:
                            key = "|".join(cells[:3])
                            if key in table_lookup:
                                tbl = table_lookup.pop(key)
                                result_lines.append(f"\n> **Table:** {tbl.caption}")
                                matched_count += 1
                                break

            i += 1

        logger.info(
            f"Injected {matched_count}/{len(captioned)} table captions into markdown"
        )
        return "\n".join(result_lines)

    @staticmethod
    def _enrich_chunks_with_refs(
        chunks: list[EnrichedChunk],
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Enrich chunks with image/table references based on page matching.

        Each image/table is assigned to the FIRST chunk on its page to avoid
        duplicating descriptions across many chunks.
        """
        if not images and not tables:
            return chunks

        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for chunk in chunks:
            page_no = chunk.page_no

            # Image refs
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk.image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            if chunk.image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk.image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    chunk.content = chunk.content + "\n\n" + "\n".join(desc_parts)

            # Table refs
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk.table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk.table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk.table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl:
                        table_block = f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]"
                        if tbl.caption:
                            table_block += f": {tbl.caption}"
                        table_block += f"\n{tbl.content_markdown}"
                        tbl_parts.append(table_block)
                if tbl_parts:
                    chunk.content = chunk.content + "\n\n" + "\n".join(tbl_parts)

        if images:
            logger.info(
                f"Image-aware enrichment: {len(assigned_images)}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )
        if tables:
            logger.info(
                f"Table-aware enrichment: {len(assigned_tables)}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks
