"""Validation and filtering for user-supplied document metadata.

PostgreSQL is the source of truth for business metadata.  Chroma only receives
stable chunk metadata, so editing a tag or date takes effect immediately and
never risks replacing ``document_id`` or ``visibility`` on a vector record.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.models.knowledge_base import KnowledgeBase
from app.schemas.metadata import MetadataFieldDefinition, MetadataFilter, MetadataFilterRule, MetadataSchema


DEFAULT_METADATA_SCHEMA: dict[str, Any] = {"version": 1, "fields": []}
RESERVED_METADATA_KEYS = {
    "document_id", "workspace_id", "chunk_id", "chunk_index", "visibility", "source",
    "file_type", "page_no", "heading_path", "image_ids", "image_urls", "table_ids",
    "kg_document_id", "content_version", "metadata_revision",
}
MAX_METADATA_BYTES = 16 * 1024
MAX_STRING_LENGTH = 2000
MAX_MULTI_ENUM_VALUES = 50


class MetadataValidationError(ValueError):
    """Raised for user-correctable metadata/schema/filter validation errors."""


@dataclass(frozen=True)
class DocumentScope:
    document_ids: list[int] | None
    scoped: bool


def default_metadata_schema() -> dict[str, Any]:
    """Return a fresh JSON-serializable default schema."""
    return {"version": 1, "fields": []}


def parse_metadata_schema(value: dict[str, Any] | None) -> MetadataSchema:
    try:
        return MetadataSchema.model_validate(value or DEFAULT_METADATA_SCHEMA)
    except Exception as exc:
        raise MetadataValidationError(str(exc)) from exc


def _ensure_safe_key(key: object) -> str:
    if not isinstance(key, str):
        raise MetadataValidationError("metadata keys must be strings")
    normalized = key.strip().lower()
    if not normalized:
        raise MetadataValidationError("metadata keys must not be empty")
    if normalized in RESERVED_METADATA_KEYS or normalized.startswith("_"):
        raise MetadataValidationError(f"metadata key '{normalized}' is reserved")
    return normalized


def _normalize_scalar(field: MetadataFieldDefinition, value: Any) -> Any:
    if field.type == "string":
        if not isinstance(value, str):
            raise MetadataValidationError(f"{field.key} must be a string")
        value = value.strip()
        if len(value) > MAX_STRING_LENGTH:
            raise MetadataValidationError(f"{field.key} is too long")
        return value
    if field.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise MetadataValidationError(f"{field.key} must be an integer")
        return value
    if field.type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise MetadataValidationError(f"{field.key} must be a finite number")
        return value
    if field.type == "boolean":
        if not isinstance(value, bool):
            raise MetadataValidationError(f"{field.key} must be a boolean")
        return value
    if field.type == "date":
        if not isinstance(value, str):
            raise MetadataValidationError(f"{field.key} must be an ISO date")
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise MetadataValidationError(f"{field.key} must be an ISO date") from exc
        return value
    if field.type == "datetime":
        if not isinstance(value, str):
            raise MetadataValidationError(f"{field.key} must be an ISO datetime")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MetadataValidationError(f"{field.key} must be an ISO datetime") from exc
        return value
    if field.type == "enum":
        if not isinstance(value, str) or value not in field.options:
            raise MetadataValidationError(f"{field.key} must be one of the configured options")
        return value
    raise MetadataValidationError(f"Unsupported scalar field type: {field.type}")


def _normalize_value(field: MetadataFieldDefinition, value: Any) -> Any:
    if field.type != "multi_enum":
        return _normalize_scalar(field, value)
    if not isinstance(value, list) or len(value) > MAX_MULTI_ENUM_VALUES:
        raise MetadataValidationError(f"{field.key} must contain up to {MAX_MULTI_ENUM_VALUES} values")
    if any(not isinstance(item, str) or item not in field.options for item in value):
        raise MetadataValidationError(f"{field.key} must use configured options")
    # Preserve definition order for a deterministic DB/API representation.
    return [option for option in field.options if option in set(value)]


def validate_custom_metadata(
    raw_metadata: dict[str, Any] | None,
    schema_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and canonicalize user metadata against a workspace schema."""
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raise MetadataValidationError("metadata must be a JSON object")

    schema = parse_metadata_schema(schema_data)
    definitions = {field.key: field for field in schema.fields}
    normalized_input = {_ensure_safe_key(key): value for key, value in raw_metadata.items()}
    unknown = sorted(set(normalized_input) - set(definitions))
    if unknown:
        raise MetadataValidationError(f"Unknown metadata fields: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    for key, definition in definitions.items():
        value = normalized_input.get(key, definition.default)
        if value is None:
            if definition.required:
                raise MetadataValidationError(f"{key} is required")
            continue
        normalized[key] = _normalize_value(definition, value)

    import json

    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise MetadataValidationError("metadata payload exceeds 16KB")
    return normalized


def semantic_metadata_context(metadata: dict[str, Any] | None, schema_data: dict[str, Any] | None) -> str:
    """Return a small, controlled prefix for fields declared semantic.

    The raw business JSON is never embedded.  Only explicitly opted-in fields
    are added to embeddings/LightRAG so ordinary labels and dates remain cheap
    to edit while domain-defining fields can improve semantic recall.
    """
    values = metadata or {}
    lines: list[str] = []
    for definition in parse_metadata_schema(schema_data).fields:
        if not definition.semantic or definition.key not in values:
            continue
        value = values[definition.key]
        rendered = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        lines.append(f"{definition.label}: {rendered}")
    return "[Document metadata]\n" + "\n".join(lines) if lines else ""


def semantic_metadata_changed(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    schema_data: dict[str, Any] | None,
) -> bool:
    definitions = parse_metadata_schema(schema_data).fields
    old_values, new_values = before or {}, after or {}
    return any(
        field.semantic and old_values.get(field.key) != new_values.get(field.key)
        for field in definitions
    )


def parse_upload_metadata(raw: str | None, legacy_raw: str | None = None) -> dict[str, Any] | None:
    """Decode the new object format and the temporary legacy key/value list."""
    import json

    payload = raw if raw is not None else legacy_raw
    if not payload:
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MetadataValidationError("metadata must be valid JSON") from exc
    if isinstance(decoded, dict):
        return decoded
    if isinstance(decoded, list):
        result: dict[str, Any] = {}
        for item in decoded:
            if not isinstance(item, dict) or set(item) - {"key", "value"} or "key" not in item:
                raise MetadataValidationError("legacy metadata items require key and value")
            key = _ensure_safe_key(item["key"])
            if key in result:
                raise MetadataValidationError(f"Duplicate metadata key: {key}")
            result[key] = item.get("value")
        return result
    raise MetadataValidationError("metadata must be a JSON object")


def _validate_filter_rule(rule: MetadataFilterRule, definitions: dict[str, MetadataFieldDefinition]) -> None:
    definition = definitions.get(rule.field)
    if definition is None:
        raise MetadataValidationError(f"Unknown metadata field: {rule.field}")
    if not definition.filterable:
        raise MetadataValidationError(f"Metadata field is not filterable: {rule.field}")
    allowed = {"eq", "neq", "exists"}
    if definition.type in {"integer", "number", "date", "datetime"}:
        allowed |= {"gt", "gte", "lt", "lte", "between", "in", "not_in"}
    elif definition.type in {"string", "enum"}:
        allowed |= {"in", "not_in"}
    elif definition.type == "multi_enum":
        allowed |= {"contains_any", "contains_all"}
    if rule.op not in allowed:
        raise MetadataValidationError(f"Operator {rule.op} is not valid for {rule.field}")
    if rule.op == "exists":
        if rule.value not in (None, True, False):
            raise MetadataValidationError("exists accepts true, false, or no value")
        return
    if rule.op in {"in", "not_in", "contains_any", "contains_all"}:
        if not isinstance(rule.value, list) or not rule.value:
            raise MetadataValidationError(f"{rule.op} requires a non-empty array")
        value_field = definition.model_copy(update={"type": "enum"}) if definition.type == "multi_enum" else definition
        for item in rule.value:
            _normalize_scalar(value_field, item)
        return
    if rule.op == "between":
        if not isinstance(rule.value, list) or len(rule.value) != 2:
            raise MetadataValidationError("between requires exactly two values")
        for item in rule.value:
            _normalize_scalar(definition, item)
        return
    _normalize_value(definition, rule.value)


def validate_metadata_filter(payload: dict[str, Any] | None, schema_data: dict[str, Any] | None) -> MetadataFilter | None:
    if payload is None:
        return None
    try:
        metadata_filter = MetadataFilter.model_validate(payload)
    except Exception as exc:
        raise MetadataValidationError(str(exc)) from exc
    definitions = {field.key: field for field in parse_metadata_schema(schema_data).fields}
    for rule in [*metadata_filter.and_rules, *metadata_filter.or_rules]:
        _validate_filter_rule(rule, definitions)
    return metadata_filter


def _matches_rule(metadata: dict[str, Any], rule: MetadataFilterRule) -> bool:
    actual = metadata.get(rule.field)
    if rule.op == "exists":
        expected = True if rule.value is None else bool(rule.value)
        return (rule.field in metadata) is expected
    if actual is None:
        return rule.op == "neq"
    if rule.op == "eq":
        return actual == rule.value
    if rule.op == "neq":
        return actual != rule.value
    if rule.op == "in":
        return actual in rule.value
    if rule.op == "not_in":
        return actual not in rule.value
    if rule.op == "contains_any":
        return isinstance(actual, list) and bool(set(actual) & set(rule.value))
    if rule.op == "contains_all":
        return isinstance(actual, list) and set(rule.value).issubset(set(actual))
    if rule.op == "between":
        lower, upper = rule.value
        return lower <= actual <= upper
    if rule.op == "gt":
        return actual > rule.value
    if rule.op == "gte":
        return actual >= rule.value
    if rule.op == "lt":
        return actual < rule.value
    if rule.op == "lte":
        return actual <= rule.value
    return False


def metadata_matches(metadata: dict[str, Any] | None, metadata_filter: MetadataFilter | None) -> bool:
    if metadata_filter is None:
        return True
    values = metadata or {}
    return (
        all(_matches_rule(values, rule) for rule in metadata_filter.and_rules)
        and (not metadata_filter.or_rules or any(_matches_rule(values, rule) for rule in metadata_filter.or_rules))
    )


async def resolve_document_scope(
    db: AsyncSession,
    workspace_id: int,
    requested_document_ids: list[int] | None,
    metadata_filter_payload: dict[str, Any] | None,
) -> DocumentScope:
    """Resolve user scope to indexed document IDs before vector/KG retrieval.

    The first implementation intentionally evaluates a validated filter against
    DB records rather than passing provider-specific syntax to Chroma.  It is
    deterministic, supports multi-value fields, and means metadata edits apply
    without a re-index.  A future SQL compiler can replace this internals-only
    scan without changing the public filter DSL.
    """
    # The overwhelmingly common unscoped route must remain independent from
    # the metadata database query.  Besides avoiding an unnecessary query in
    # production, this keeps the LangChain retrieval contract usable with its
    # lightweight test/adapter DB placeholder.
    if requested_document_ids is None and metadata_filter_payload is None:
        return DocumentScope(document_ids=None, scoped=False)
    workspace = await db.scalar(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    if workspace is None:
        raise MetadataValidationError("Knowledge base not found")
    metadata_filter = validate_metadata_filter(metadata_filter_payload, workspace.metadata_schema)
    is_scoped = True

    query = select(Document.id, Document.custom_metadata).where(
        Document.workspace_id == workspace_id,
        Document.status == DocumentStatus.INDEXED,
    )
    if requested_document_ids is not None:
        if not requested_document_ids:
            return DocumentScope(document_ids=[], scoped=True)
        query = query.where(Document.id.in_(list(set(requested_document_ids))))
    rows = (await db.execute(query)).all()
    return DocumentScope(
        document_ids=[document_id for document_id, metadata in rows if metadata_matches(metadata, metadata_filter)],
        scoped=True,
    )
