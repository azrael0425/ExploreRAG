from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter

from app.core.config import settings
from app.services.embedder import EmbeddingService
from app.services.index_chunking import split_enriched_chunks
from app.services.models.parsed_document import EnrichedChunk
from app.services.reranker import RerankerService
from app.services.scan_profile_router import ScanProfileRouter


class _CharTokenizer:
    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    @staticmethod
    def decode(tokens: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)


class _FakeEmbeddingModel:
    tokenizer = _CharTokenizer()


def _embedder() -> EmbeddingService:
    service = EmbeddingService("fake")
    service._model = _FakeEmbeddingModel()
    return service


def test_long_markdown_table_is_split_with_repeated_header(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EXPLORERAG_EMBEDDING_TOKEN_OVERLAP", 0)
    embedder = _embedder()
    table = "\n".join([
        "[Table on page 3]",
        "| name | value |",
        "| --- | --- |",
        "| alpha | 111111 |",
        "| beta | 222222 |",
        "| gamma | 333333 |",
    ])

    parts = embedder.split_text_for_indexing(table, max_tokens=58)

    assert len(parts) > 1
    assert all(embedder.count_tokens(part) <= 58 for part in parts)
    assert all("| name | value |" in part for part in parts)
    joined = "\n".join(parts)
    assert "alpha" in joined and "beta" in joined and "gamma" in joined


def test_enriched_chunk_split_preserves_retrieval_metadata(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EXPLORERAG_EMBEDDING_MAX_TOKENS", 24)
    monkeypatch.setattr(settings, "EXPLORERAG_EMBEDDING_TOKEN_OVERLAP", 0)
    chunk = EnrichedChunk(
        content="x" * 70,
        chunk_index=8,
        source_file="report.pdf",
        document_id=3,
        page_no=7,
        heading_path=["Results"],
        table_refs=["table-1"],
        has_table=True,
    )

    parts, stats = split_enriched_chunks([chunk], _embedder())

    assert len(parts) == 3
    assert stats["oversized"] == 1
    assert [part.chunk_index for part in parts] == [0, 1, 2]
    assert all(part.page_no == 7 and part.table_refs == ["table-1"] for part in parts)


def test_reranker_uses_640_768_length_tiers(monkeypatch) -> None:
    observed: list[tuple[int, int]] = []

    class Scores(list):
        def tolist(self):
            return list(self)

    class Model:
        max_length = 768

        def predict(self, pairs, batch_size, show_progress_bar):
            del show_progress_bar
            observed.append((self.max_length, batch_size))
            return Scores(float(index) for index, _ in enumerate(pairs))

    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_MAX_LENGTH", 768)
    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_LONG_LIST_MAX_LENGTH", 640)
    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_LONG_LIST_THRESHOLD", 16)
    monkeypatch.setattr(settings, "EXPLORERAG_RERANKER_BATCH_SIZE", 4)
    service = RerankerService("fake")
    service._model = Model()

    service.rerank("q", [str(index) for index in range(20)])
    service.rerank("q", ["a", "b"])

    assert observed == [(640, 4), (768, 4)]


def test_scan_router_extracts_only_requested_pages(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=100, height=100)
    with source.open("wb") as stream:
        writer.write(stream)

    selected = tmp_path / "selected.pdf"
    pages = ScanProfileRouter().extract_pages(source, [4, 2, 2, 99], selected)

    assert pages == [2, 4]
    assert len(PdfReader(str(selected)).pages) == 2
