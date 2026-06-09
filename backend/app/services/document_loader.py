"""
Document Loader Service
Handles loading and extracting text from various document formats.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import logging

logger = logging.getLogger(__name__)


class LoadedDocument(NamedTuple):
    """Represents a loaded document with its content and metadata."""
    content: str
    source: str
    file_type: str
    page_count: int = 1


def load_txt_file(file_path: Path) -> LoadedDocument:
    """Load a plain text file."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )
    except UnicodeDecodeError:
        # Try with different encoding
        content = file_path.read_text(encoding="latin-1")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )


def load_pdf_file(file_path: Path) -> LoadedDocument:
    """Load a PDF file and extract text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        content = "\n\n".join(pages_text)

        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="pdf",
            page_count=len(reader.pages)
        )
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise ValueError(f"Failed to load PDF: {e}")


def load_markdown_file(file_path: Path) -> LoadedDocument:
    """Load a markdown file."""
    content = file_path.read_text(encoding="utf-8")
    return LoadedDocument(
        content=content,
        source=str(file_path),
        file_type="md",
        page_count=1
    )


def load_document(file_path: str | Path) -> LoadedDocument:
    """
    Load a document based on its file type.

    Supported formats: .txt, .pdf, .md

    Args:
        file_path: Path to the document file

    Returns:
        LoadedDocument with content and metadata

    Raises:
        ValueError: If file type is not supported or file cannot be read
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    loaders = {
        ".txt": load_txt_file,
        ".pdf": load_pdf_file,
        ".md": load_markdown_file,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    return loader(path)


def get_supported_extensions() -> list[str]:
    """Return list of supported file extensions."""
    return [".txt", ".pdf", ".md"]
