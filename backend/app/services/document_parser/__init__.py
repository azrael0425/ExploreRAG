"""
Document Parser Package
========================

Factory function to create document parsers based on config.

Usage::

    from app.services.document_parser import get_document_parser

    parser = get_document_parser(workspace_id=1)
    result = parser.parse(file_path, document_id, original_filename)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.services.document_parser.base import BaseDocumentParser


def get_document_parser(
    workspace_id: int,
    output_dir: Optional[Path] = None,
    llm_mode: str = "cloud",
) -> BaseDocumentParser:
    """Create the project's single rich-document parser: Docling."""
    from app.services.document_parser.docling_parser import DoclingDocumentParser

    return DoclingDocumentParser(workspace_id, output_dir, llm_mode=llm_mode)


__all__ = [
    "get_document_parser",
    "BaseDocumentParser",
]
