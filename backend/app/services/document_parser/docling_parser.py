"""
Docling Document Parser
=======================

Wraps the existing Docling-based parsing pipeline (formerly in
``deep_document_parser.py``) behind the ``BaseDocumentParser`` interface.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# File extensions handled by Docling vs legacy
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}
_CONVERTER_CACHE: dict[tuple[object, ...], object] = {}
_CONVERTER_CACHE_LOCK = threading.Lock()


class DoclingDocumentParser(BaseDocumentParser):
    """
    Document parser powered by Docling.

    - Converts PDF/DOCX/PPTX/HTML via Docling DocumentConverter
    - Chunks using HybridChunker (semantic + structural)
    - Extracts images and optionally captions them via LLM Vision
    - Falls back to legacy text extraction for TXT/MD
    """

    parser_name = "docling"

    def __init__(
        self,
        workspace_id: int,
        output_dir: Optional[Path] = None,
        media_url_prefix: str | None = None,
        llm_mode: str = "cloud",
    ):
        super().__init__(workspace_id, output_dir, llm_mode=llm_mode)
        self.media_url_prefix = media_url_prefix
        self.last_parse_manifest: dict = {}

    @staticmethod
    def supported_extensions() -> set[str]:
        return _DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Converter
    # ------------------------------------------------------------------

    def _get_converter(self, scan_profile: str = "normal_pdf"):
        """Return a process-wide Docling converter for this profile."""
        converter_profile = "scanned_pdf" if scan_profile == "scanned_pdf" else "normal_pdf"
        cache_key = (
            converter_profile,
            settings.EXPLORERAG_ENABLE_IMAGE_EXTRACTION,
            settings.EXPLORERAG_DOCLING_IMAGES_SCALE,
            settings.EXPLORERAG_ENABLE_FORMULA_ENRICHMENT,
            settings.EXPLORERAG_DOCLING_ARTIFACTS_PATH,
        )
        with _CONVERTER_CACHE_LOCK:
            cached = _CONVERTER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        pipeline_options = PdfPipelineOptions()
        # The GPU is reserved for BGE-M3 and the cross-encoder reranker. Keep
        # Docling's layout, table, and OCR pipeline on CPU deliberately.
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CPU
        )
        # Docling 2.118 enables torch.compile for its Transformers layout
        # engine by default.  The production runtime intentionally has no C++
        # toolchain, and eager inference is fully equivalent for indexing.
        layout_options = getattr(pipeline_options, "layout_options", None)
        layout_engine_options = getattr(layout_options, "engine_options", None)
        if layout_engine_options is not None and hasattr(
            layout_engine_options, "compile_model"
        ):
            layout_engine_options.compile_model = False
        pipeline_options.generate_picture_images = settings.EXPLORERAG_ENABLE_IMAGE_EXTRACTION
        pipeline_options.images_scale = settings.EXPLORERAG_DOCLING_IMAGES_SCALE
        pipeline_options.do_formula_enrichment = settings.EXPLORERAG_ENABLE_FORMULA_ENRICHMENT
        # PdfPipelineOptions defaults to OCR enabled.  Be explicit here: native
        # PDFs must not initialize RapidOCR (or depend on OCR artifacts), while
        # the scanned profile below opts into full-page OCR.
        pipeline_options.do_ocr = False
        # Keep native/text PDFs on Docling's normal layout path.  Scanned PDFs
        # explicitly enable full-page OCR and accurate table processing, but
        # attribute checks make this compatible with multiple Docling 2.x APIs.
        if converter_profile == "scanned_pdf":
            pipeline_options.do_ocr = True
            try:
                from docling.datamodel.pipeline_options import RapidOcrOptions

                ocr_options = RapidOcrOptions()
                for attr, value in (
                    ("force_full_page_ocr", True),
                    ("lang", ["ch", "en"]),
                    ("languages", ["ch", "en"]),
                ):
                    if hasattr(ocr_options, attr):
                        setattr(ocr_options, attr, value)

                # docling-tools stores current RapidOCR assets in a nested
                # backend/version layout.  Docling 2.118's offline resolver
                # also accepts pinned paths, which keeps scanned-PDF OCR
                # compatible with that downloaded layout.
                if settings.EXPLORERAG_DOCLING_ARTIFACTS_PATH:
                    rapidocr_root = (
                        Path(settings.EXPLORERAG_DOCLING_ARTIFACTS_PATH) / "RapidOcr"
                    )
                    pinned_paths = {
                        "det_model_path": rapidocr_root
                        / "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
                        "cls_model_path": rapidocr_root
                        / "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                        "rec_model_path": rapidocr_root
                        / "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
                        "rec_keys_path": rapidocr_root
                        / "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
                        "font_path": rapidocr_root / "resources/fonts/FZYTK.TTF",
                    }
                    missing_paths = [
                        str(path) for path in pinned_paths.values() if not path.is_file()
                    ]
                    if missing_paths:
                        logger.warning(
                            "RapidOCR nested artifacts are incomplete; Docling's "
                            "offline resolver will validate the configured model path. "
                            "Missing: %s",
                            missing_paths,
                        )
                    else:
                        for attr, path in pinned_paths.items():
                            if hasattr(ocr_options, attr):
                                setattr(ocr_options, attr, str(path))
                        logger.info("Using pinned nested RapidOCR artifacts: %s", rapidocr_root)
                pipeline_options.ocr_options = ocr_options
            except Exception as exc:
                logger.warning("RapidOCR profile could not be configured; using Docling OCR defaults: %s", exc)
            table_options = getattr(pipeline_options, "table_structure_options", None)
            if table_options is not None:
                if hasattr(table_options, "do_cell_matching"):
                    table_options.do_cell_matching = True
                if hasattr(table_options, "mode"):
                    # Newer Docling versions accept TableFormerMode.ACCURATE;
                    # the string fallback keeps older versions operational.
                    try:
                        from docling.datamodel.pipeline_options import TableFormerMode
                        table_options.mode = TableFormerMode.ACCURATE
                    except Exception:
                        table_options.mode = "accurate"
        if settings.EXPLORERAG_DOCLING_ARTIFACTS_PATH:
            artifacts_path = Path(settings.EXPLORERAG_DOCLING_ARTIFACTS_PATH)
            if not artifacts_path.is_dir():
                raise RuntimeError(
                    "EXPLORERAG_DOCLING_ARTIFACTS_PATH is configured but unavailable: "
                    f"{artifacts_path}"
                )
            pipeline_options.artifacts_path = artifacts_path
            logger.info("Using local Docling artifacts: %s", artifacts_path)

        converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        with _CONVERTER_CACHE_LOCK:
            # The scheduler serializes Docling work, but setdefault also makes
            # construction safe if another caller bypasses it during startup.
            converter = _CONVERTER_CACHE.setdefault(cache_key, converter)
        logger.info("Cached process-wide Docling converter for profile=%s", converter_profile)
        return converter

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Check if the file format is supported by Docling (not legacy)."""
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    # ------------------------------------------------------------------
    # Main parse entry
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
        scan_profile: str = "normal_pdf",
    ) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            result = self._parse_with_docling(path, document_id, original_filename, scan_profile)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[docling] Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        self.last_parse_manifest = {
            "parser": self.parser_name,
            "scan_profile": scan_profile if suffix == ".pdf" else None,
            "page_count": result.page_count,
            "image_count": len(result.images),
            "table_count": result.tables_count,
        }
        return result

    # ------------------------------------------------------------------
    # Docling pipeline
    # ------------------------------------------------------------------

    def _parse_with_docling(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
        scan_profile: str,
    ) -> ParsedDocument:
        """Parse with Docling for rich structural extraction."""
        converter = self._get_converter(scan_profile)

        logger.info(f"Docling converting: {file_path}")
        convert_started_at = time.perf_counter()
        conv_result = converter.convert(str(file_path))
        convert_ms = int((time.perf_counter() - convert_started_at) * 1000)
        doc = conv_result.document

        # Extract images and build URL mapping for markdown references
        image_started_at = time.perf_counter()
        images, pic_url_list = self._extract_images_with_urls(doc, document_id)
        image_ms = int((time.perf_counter() - image_started_at) * 1000)

        # Extract tables
        table_started_at = time.perf_counter()
        tables = self._extract_tables(doc, document_id)
        if settings.EXPLORERAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)
        table_ms = int((time.perf_counter() - table_started_at) * 1000)

        # Export to markdown
        markdown = self._export_markdown(doc)

        # Post-process: replace image placeholders with real markdown images
        markdown = self._inject_image_references(markdown, pic_url_list)

        # Post-process: inject table captions into markdown
        markdown = self._inject_table_captions(markdown, tables)

        # Get page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Chunk with HybridChunker
        chunk_started_at = time.perf_counter()
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)
        chunk_ms = int((time.perf_counter() - chunk_started_at) * 1000)
        logger.info(
            "Docling stage timing document=%s profile=%s convert_ms=%s image_ms=%s table_ms=%s chunk_ms=%s",
            document_id,
            scan_profile,
            convert_ms,
            image_ms,
            table_ms,
            chunk_ms,
        )

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=markdown,
            page_count=page_count,
            chunks=chunks,
            images=images,
            tables=tables,
            tables_count=len(tables),
        )

    # ------------------------------------------------------------------
    # Chunking (Docling HybridChunker)
    # ------------------------------------------------------------------

    def _chunk_document(
        self,
        doc,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk document using Docling's HybridChunker with image/table enrichment."""
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            max_tokens=settings.EXPLORERAG_CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        # Build page→images lookup
        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        # Build page→tables lookup
        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            # Extract page number
            page_no = 0
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "page"):
                    page_no = chunk.meta.page or 0
                elif hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no or 0
                                    break
                            if page_no > 0:
                                break

            # Extract heading path
            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            # Detect content types
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            has_table = False
            has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower():
                            has_table = True
                        if "code" in label.lower():
                            has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            # ── Image-aware enrichment ──
            chunk_image_refs: list[str] = []
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk_image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    enriched_text = chunk_text + "\n\n" + "\n".join(desc_parts)

            # ── Table-aware enrichment ──
            chunk_table_refs: list[str] = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk_table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl:
                        # A table caption is not evidence.  Preserve the real
                        # Markdown table so long-document retrieval can answer
                        # from values/rows rather than a lossy summary.
                        table_block = f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]"
                        if tbl.caption:
                            table_block += f": {tbl.caption}"
                        table_block += f"\n{tbl.content_markdown}"
                        tbl_parts.append(table_block)
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text,
                chunk_index=i,
                source_file=original_filename,
                document_id=document_id,
                page_no=page_no,
                heading_path=heading_path,
                image_refs=chunk_image_refs,
                table_refs=chunk_table_refs,
                has_table=has_table,
                has_code=has_code,
                contextualized=contextualized,
            ))

        if images:
            logger.info(
                f"Image-aware chunking: {len(assigned_images)}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )
        if tables:
            logger.info(
                f"Table-aware chunking: {len(assigned_tables)}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def _export_markdown(self, doc) -> str:
        """Export document to markdown with page break markers if supported."""
        try:
            return doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n",
            )
        except TypeError:
            return doc.export_to_markdown()

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def _extract_images_with_urls(
        self,
        doc,
        document_id: int,
    ) -> tuple[list[ExtractedImage], list[tuple[str, str]]]:
        """Extract images and build URL mapping for markdown placeholders."""
        if not settings.EXPLORERAG_ENABLE_IMAGE_EXTRACTION:
            return [], []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        pic_to_image_idx: list[int] = []
        picture_count = 0

        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []

        for pic in doc.pictures:
            if picture_count >= settings.EXPLORERAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1)
                continue

            image_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                image_path = images_dir / f"{image_id}.png"

                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1)
                        continue
                else:
                    pic_to_image_idx.append(-1)
                    continue

                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption=caption,
                    width=width,
                    height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1

            except Exception as e:
                logger.warning(f"Failed to extract image from document {document_id}: {e}")
                pic_to_image_idx.append(-1)
                continue

        logger.info(f"Extracted {len(images)} images from document {document_id}")

        if settings.EXPLORERAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        pic_url_list: list[tuple[str, str]] = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                base_url = self.media_url_prefix or f"/static/doc-images/kb_{self.workspace_id}"
                url = f"{base_url.rstrip('/')}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))

        return images, pic_url_list

    def _inject_image_references(
        self, markdown: str, pic_url_list: list[tuple[str, str]]
    ) -> str:
        """Replace <!-- image --> placeholders with ![caption](url) markdown."""
        placeholder_count = len(re.findall(r"<!--\s*image\s*-->", markdown))

        if not pic_url_list:
            if placeholder_count > 0:
                logger.warning(
                    f"Markdown has {placeholder_count} image placeholders but "
                    f"pic_url_list is empty — images will NOT be injected"
                )
            return markdown

        logger.info(
            f"Injecting {len(pic_url_list)} image URLs into "
            f"{placeholder_count} placeholders"
        )

        injected = 0
        pic_iter = iter(pic_url_list)

        def replacer(match):
            nonlocal injected
            try:
                caption, url = next(pic_iter)
                if url:
                    safe_caption = caption.replace("[", "").replace("]", "")
                    safe_caption = " ".join(safe_caption.split())
                    injected += 1
                    return f"\n![{safe_caption}]({url})\n"
                return ""
            except StopIteration:
                return ""

        result = re.sub(r'<!--\s*image\s*-->', replacer, markdown)
        logger.info(f"Injected {injected}/{placeholder_count} image references")
        return result

    # ------------------------------------------------------------------
    # Table extraction (Docling-specific)
    # ------------------------------------------------------------------

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """Extract tables from Docling document."""
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        tables: list[ExtractedTable] = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                content_md = table.export_to_markdown(doc)
            except Exception:
                content_md = ""

            if not content_md.strip():
                continue

            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            tables.append(ExtractedTable(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                content_markdown=content_md,
                num_rows=num_rows,
                num_cols=num_cols,
            ))

        logger.info(f"Extracted {len(tables)} tables from document {document_id}")
        return tables

    # ------------------------------------------------------------------
    # Legacy fallback (TXT/MD)
    # ------------------------------------------------------------------

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD with legacy loader."""
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker

        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content,
            source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )

        chunks = [
            EnrichedChunk(
                content=tc.content,
                chunk_index=tc.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for tc in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables_count=0,
        )
