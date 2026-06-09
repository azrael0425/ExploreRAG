"""PDF scan detection and page-image generation with safe fallbacks."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ScanProfile:
    profile: str
    page_count: int = 0
    page_text_chars: list[int] = field(default_factory=list)
    low_confidence_pages: list[int] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def manifest(self) -> dict:
        return asdict(self)


class ScanProfileRouter:
    """Classify PDFs using their real page/text characteristics, not suffixes."""

    def inspect(self, file_path: str | Path) -> ScanProfile:
        path = Path(file_path)
        try:
            from pypdf import PdfReader
        except Exception as exc:
            return ScanProfile("normal_pdf", diagnostics=[f"pypdf unavailable: {exc}"])

        try:
            reader = PdfReader(str(path))
            counts = [len((page.extract_text() or "").strip()) for page in reader.pages]
        except Exception as exc:
            logger.warning("Could not inspect PDF %s: %s", path, exc)
            return ScanProfile("normal_pdf", diagnostics=[f"PDF inspection failed: {exc}"])

        page_count = len(counts)
        low_pages = [
            index + 1 for index, chars in enumerate(counts)
            if chars < settings.CHAT_ATTACHMENT_SCAN_LOW_CONFIDENCE_CHARS
        ]
        if page_count and len(low_pages) / page_count >= 0.7:
            profile = "scanned_pdf"
        elif low_pages:
            profile = "mixed_pdf"
        else:
            profile = "normal_pdf"
        return ScanProfile(
            profile=profile,
            page_count=page_count,
            page_text_chars=counts,
            low_confidence_pages=low_pages,
        )

    def extract_pages(
        self,
        file_path: str | Path,
        page_numbers: list[int],
        output_path: str | Path,
    ) -> list[int]:
        """Write only selected one-based pages to a compact OCR retry PDF."""
        selected = sorted({page for page in page_numbers if page > 0})
        if not selected:
            return []
        try:
            from pypdf import PdfReader, PdfWriter

            reader = PdfReader(str(file_path))
            valid = [page for page in selected if page <= len(reader.pages)]
            if not valid:
                return []
            writer = PdfWriter()
            for page in valid:
                writer.add_page(reader.pages[page - 1])
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as stream:
                writer.write(stream)
            return valid
        except Exception as exc:
            logger.warning("Failed to extract selective OCR pages from %s: %s", file_path, exc)
            return []

    def render_pages(
        self,
        file_path: str | Path,
        output_dir: str | Path,
        page_numbers: list[int] | None = None,
    ) -> list[int]:
        """Save bounded page images for scanned/mixed-PDF vision verification."""
        try:
            import fitz  # PyMuPDF is a Docling transitive dependency in normal deployments.
        except Exception as exc:
            logger.info("Page image rendering unavailable: %s", exc)
            return []

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        rendered: list[int] = []
        try:
            pdf = fitz.open(str(file_path))
            scale = settings.CHAT_ATTACHMENT_PAGE_IMAGE_SCALE
            if page_numbers:
                selected = [
                    page for page in sorted(set(page_numbers))
                    if 1 <= page <= pdf.page_count
                ][:settings.CHAT_ATTACHMENT_MAX_PAGE_IMAGES]
            else:
                selected = list(range(1, min(pdf.page_count, settings.CHAT_ATTACHMENT_MAX_PAGE_IMAGES) + 1))
            for page_number in selected:
                page_index = page_number - 1
                page = pdf.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pix.save(str(destination / f"page_{page_number}.png"))
                rendered.append(page_number)
            pdf.close()
        except Exception as exc:
            logger.warning("Failed to render PDF page images for %s: %s", file_path, exc)
        return rendered
