"""Schemas for knowledge-base document metadata.

Custom metadata is deliberately defined per knowledge base instead of accepting
an unbounded JSON blob from the upload form.  The schema is also shared by the
query filter compiler, which keeps UI labels and server-side semantics aligned.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MetadataFieldType = Literal[
    "string", "integer", "number", "boolean", "date", "datetime", "enum", "multi_enum"
]
MetadataOperator = Literal[
    "eq", "neq", "exists", "gt", "gte", "lt", "lte", "between", "in", "not_in",
    "contains_any", "contains_all",
]


class MetadataFieldDefinition(BaseModel):
    key: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=64)
    type: MetadataFieldType = "string"
    required: bool = False
    filterable: bool = True
    semantic: bool = False
    options: list[str] = Field(default_factory=list, max_length=100)
    default: Any | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        import re

        normalized = value.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", normalized):
            raise ValueError("key must use lowercase letters, numbers, and underscores")
        return normalized

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        return value.strip()

    @field_validator("options")
    @classmethod
    def normalize_options(cls, values: list[str]) -> list[str]:
        result = [value.strip() for value in values if value and value.strip()]
        if len(result) != len(set(result)):
            raise ValueError("options must not contain duplicates")
        return result

    @model_validator(mode="after")
    def validate_type_options(self) -> "MetadataFieldDefinition":
        if self.type in {"enum", "multi_enum"} and not self.options:
            raise ValueError("enum and multi_enum fields require options")
        if self.type not in {"enum", "multi_enum"} and self.options:
            raise ValueError("options are only allowed for enum fields")
        return self


class MetadataSchema(BaseModel):
    version: int = Field(default=1, ge=1)
    fields: list[MetadataFieldDefinition] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def ensure_unique_keys(self) -> "MetadataSchema":
        keys = [field.key for field in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError("metadata field keys must be unique")
        return self


class MetadataSchemaUpdate(MetadataSchema):
    """Replacement schema for a workspace.

    Existing document values are retained.  A subsequent document edit is
    validated against the newest schema, so clients can migrate values without
    a destructive workspace-wide rewrite.
    """


class DocumentMetadataUpdate(BaseModel):
    custom_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_revision: int | None = Field(default=None, ge=1)


class MetadataFilterRule(BaseModel):
    field: str = Field(..., min_length=1, max_length=64)
    op: MetadataOperator
    value: Any | None = None

    @field_validator("field")
    @classmethod
    def normalize_field(cls, value: str) -> str:
        return value.strip().lower()


class MetadataFilter(BaseModel):
    """A portable filter DSL; this is intentionally not a Chroma ``where``.

    ``and`` and ``or`` use aliases to keep the wire format natural while
    avoiding Python keyword names in code.
    """

    model_config = ConfigDict(populate_by_name=True)

    and_rules: list[MetadataFilterRule] = Field(default_factory=list, alias="and", max_length=32)
    or_rules: list[MetadataFilterRule] = Field(default_factory=list, alias="or", max_length=32)

    @model_validator(mode="after")
    def require_a_rule(self) -> "MetadataFilter":
        if not self.and_rules and not self.or_rules:
            raise ValueError("metadata_filter must contain at least one rule")
        return self
