from __future__ import annotations

import asyncio
import io
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastapi import HTTPException

from app.services.citation_ids import source_label
from app.models.chat_attachment import ChatAttachmentState
from app.services.attachment_processor import AttachmentInvalidated, AttachmentProcessor
from app.services.attachment_retriever import AttachmentRetriever
from app.services.chat_attachment_service import ChatAttachmentService
from app.services.scan_profile_router import ScanProfileRouter
from app.services.vector_store import VectorStore


def test_image_attachment_keeps_original_bytes_for_vision(tmp_path: Path) -> None:
    image = tmp_path / "source.jpg"
    image.write_bytes(b"\xff\xd8\xfftemporary-image")
    attachment = SimpleNamespace(
        id="att-image", file_type="jpg", storage_path=str(image), original_filename="scan.jpg"
    )

    result = AttachmentRetriever(7)._source_images([attachment])

    assert len(result) == 1
    assert result[0].data == image.read_bytes()
    assert result[0].mime_type == "image/jpeg"
    assert "/rag/chat/7/attachments/att-image/files/source/" in result[0].url


def test_txt_md_is_decoded_and_chunked_without_bge(tmp_path: Path) -> None:
    text = "标题\n\n第一段。\n\n| 名称 | 数值 |\n| --- | --- |\n| A | 1 |"
    source = tmp_path / "source.txt"
    source.write_text(text, encoding="utf-8")
    attachment = SimpleNamespace(
        storage_path=str(source), original_filename="note.txt", file_type="txt", id="att-text"
    )

    parsed = AttachmentProcessor()._parse_text_sync(attachment)

    assert parsed["markdown"].replace("\r\n", "\n") == text
    assert parsed["token_count"] > 0
    assert any(chunk["has_table"] for chunk in parsed["chunks"])
    assert ChatAttachmentService.decode_text("中文".encode("gb18030")) == "中文"


def test_large_attachment_query_results_remain_attachment_scoped() -> None:
    raw = {
        "ids": ["att_123_chunk_0"],
        "documents": ["large temporary attachment result"],
        "metadatas": [{
            "attachment_id": "123",
            "workspace_id": 7,
            "source_file": "large.pdf",
            "page_no": 4,
            "heading_path": "Terms > Payment",
            "table_ids": "t1",
            "image_ids": "i1",
            "cleanup_epoch": 2,
            "has_table": True,
        }],
        "distances": [0.12],
    }

    chunks = AttachmentRetriever._chunks_from_query(raw)

    assert chunks[0].attachment_id == "123"
    assert chunks[0].page_no == 4
    assert chunks[0].heading_path == ["Terms", "Payment"]
    assert chunks[0].has_table is True


def test_dual_route_source_labels_are_explicit_and_non_colliding() -> None:
    seen: set[str] = set()
    attachment = source_label("ATT", seen)
    seen.add(attachment)
    knowledge_base = source_label("KB", seen)

    assert attachment.startswith("ATT-")
    assert knowledge_base.startswith("KB-")
    assert attachment != knowledge_base


def test_upload_validation_checks_magic_bytes_and_text_encoding() -> None:
    service = ChatAttachmentService()
    from PIL import Image
    payload = io.BytesIO()
    Image.new("RGB", (1, 1)).save(payload, format="PNG")
    service._validate_content("image.png", ".png", "image/png", payload.getvalue())
    with pytest.raises(HTTPException, match="valid JPEG"):
        service._validate_content("image.jpg", ".jpg", "image/jpeg", b"not-jpeg")
    with pytest.raises(HTTPException, match="binary"):
        service._validate_content("note.txt", ".txt", "text/plain", b"a\x00b")


def test_cleanup_epoch_rejects_late_parser_publish() -> None:
    class Result:
        def scalar_one_or_none(self):
            return 3

    class Db:
        async def execute(self, _statement):
            return Result()

    attachment = SimpleNamespace(
        id="att-race",
        cleanup_epoch=2,
        state=ChatAttachmentState.PARSING,
        cleanup_pending=False,
    )

    with pytest.raises(AttachmentInvalidated):
        asyncio.run(AttachmentProcessor()._assert_epoch(Db(), 7, attachment, 2))


def test_scan_router_routes_normal_scanned_and_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    pages: list[str] = []

    class Page:
        def __init__(self, text: str):
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class Reader:
        def __init__(self, _path: str):
            self.pages = [Page(text) for text in pages]

    module = types.ModuleType("pypdf")
    module.PdfReader = Reader
    monkeypatch.setitem(sys.modules, "pypdf", module)
    router = ScanProfileRouter()

    pages[:] = ["native text " * 20, "native text " * 20]
    assert router.inspect("normal.pdf").profile == "normal_pdf"
    pages[:] = ["", "", "few"]
    assert router.inspect("scan.pdf").profile == "scanned_pdf"
    pages[:] = ["native text " * 20, ""]
    mixed = router.inspect("mixed.pdf")
    assert mixed.profile == "mixed_pdf"
    assert mixed.low_confidence_pages == [2]


def test_vector_metadata_serializes_nested_document_diagnostics() -> None:
    normalized = VectorStore._normalize_metadatas([{
        "document_id": 5,
        "scan_profile": {
            "profile": "normal_pdf",
            "diagnostics": ["pypdf unavailable"],
        },
        "page_numbers": [1, 2, 3],
    }])

    assert normalized == [{
        "document_id": 5,
        "scan_profile": '{"diagnostics": ["pypdf unavailable"], "profile": "normal_pdf"}',
        "page_numbers": [1, 2, 3],
    }]
