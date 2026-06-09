from __future__ import annotations

import asyncio

import pytest
from app.services.document_metadata import (
    MetadataValidationError,
    metadata_matches,
    semantic_metadata_changed,
    semantic_metadata_context,
    validate_custom_metadata,
    validate_metadata_filter,
)
from app.services.knowledge_graph_service import KnowledgeGraphService


SCHEMA = {
    "version": 1,
    "fields": [
        {
            "key": "department",
            "label": "Department",
            "type": "enum",
            "options": ["Sales", "Support"],
            "filterable": True,
        },
        {
            "key": "effective_date",
            "label": "Effective date",
            "type": "date",
            "filterable": True,
        },
        {
            "key": "products",
            "label": "Products",
            "type": "multi_enum",
            "options": ["Alpha", "Beta", "Gamma"],
            "filterable": True,
            "semantic": True,
        },
        {
            "key": "internal_notes",
            "label": "Internal notes",
            "type": "string",
            "filterable": False,
        },
    ],
}


def test_metadata_is_governed_and_canonicalized() -> None:
    metadata = validate_custom_metadata(
        {
            "DEPARTMENT": "Sales",
            "effective_date": "2026-08-07",
            "products": ["Gamma", "Alpha", "Gamma"],
        },
        SCHEMA,
    )

    assert metadata == {
        "department": "Sales",
        "effective_date": "2026-08-07",
        "products": ["Alpha", "Gamma"],
    }
    with pytest.raises(MetadataValidationError, match="Unknown metadata fields"):
        validate_custom_metadata({"owner": "Ada"}, SCHEMA)
    with pytest.raises(MetadataValidationError, match="reserved"):
        validate_custom_metadata({"document_id": "123"}, SCHEMA)


def test_metadata_filter_has_typed_operators_and_boolean_logic() -> None:
    metadata_filter = validate_metadata_filter(
        {
            "and": [
                {"field": "department", "op": "eq", "value": "Sales"},
                {
                    "field": "effective_date",
                    "op": "between",
                    "value": ["2026-01-01", "2026-12-31"],
                },
            ],
            "or": [
                {"field": "products", "op": "contains_any", "value": ["Alpha"]},
                {"field": "products", "op": "contains_all", "value": ["Beta", "Gamma"]},
            ],
        },
        SCHEMA,
    )

    assert metadata_filter is not None
    assert metadata_matches(
        {"department": "Sales", "effective_date": "2026-08-07", "products": ["Alpha"]},
        metadata_filter,
    )
    assert not metadata_matches(
        {"department": "Support", "effective_date": "2026-08-07", "products": ["Alpha"]},
        metadata_filter,
    )
    with pytest.raises(MetadataValidationError, match="not filterable"):
        validate_metadata_filter(
            {"and": [{"field": "internal_notes", "op": "eq", "value": "private"}]},
            SCHEMA,
        )


def test_semantic_metadata_is_explicit_and_detects_reindex_need() -> None:
    before = {"department": "Sales", "products": ["Alpha"]}
    after = {"department": "Support", "products": ["Beta"]}

    assert semantic_metadata_context(after, SCHEMA) == "[Document metadata]\nProducts: Beta"
    assert semantic_metadata_changed(before, after, SCHEMA) is True
    assert semantic_metadata_changed(before, {**before, "department": "Support"}, SCHEMA) is False


def test_lightrag_source_id_is_passed_on_ingest_and_delete() -> None:
    class FakeGraph:
        async def get_all_nodes(self):
            return ["entity"]

    class FakeRag:
        def __init__(self) -> None:
            self.chunk_entity_relation_graph = FakeGraph()
            self.inserts: list[tuple[str, dict]] = []
            self.deleted: list[str] = []

        async def ainsert(self, content: str, **kwargs) -> None:
            self.inserts.append((content, kwargs))

        async def adelete_by_doc_id(self, document_id: str) -> None:
            self.deleted.append(document_id)

    fake_rag = FakeRag()
    service = KnowledgeGraphService(workspace_id=12)

    async def get_fake_rag() -> FakeRag:
        return fake_rag

    service._get_rag = get_fake_rag  # type: ignore[method-assign]

    async def scenario() -> None:
        await service.ingest(
            "# Handbook",
            kg_document_id="kb:12:doc:9",
            source_file="handbook.pdf",
        )
        await service.delete_document("kb:12:doc:9")

    asyncio.run(scenario())

    assert fake_rag.inserts == [
        ("# Handbook", {"ids": ["kb:12:doc:9"], "file_paths": ["handbook.pdf"]})
    ]
    assert fake_rag.deleted == ["kb:12:doc:9"]
