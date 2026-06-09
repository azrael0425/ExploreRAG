"""
Deep Document Parser — Backward Compatibility
==============================================

This module re-exports from the new ``document_parser`` package.
Existing imports like::

    from app.services.deep_document_parser import DeepDocumentParser

continue to work without changes.
"""
from app.services.document_parser.docling_parser import DoclingDocumentParser as DeepDocumentParser

__all__ = ["DeepDocumentParser"]
