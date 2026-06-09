"""
Knowledge Base (Workspace) schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from typing import Any


LLMMode = Literal["cloud", "local"]


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    llm_mode: LLMMode = "cloud"
    lightrag_augmentation_enabled: bool = False


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    llm_mode: LLMMode | None = None
    lightrag_augmentation_enabled: bool | None = None


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    description: str | None
    system_prompt: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    llm_mode: LLMMode = "cloud"
    lightrag_augmentation_enabled: bool = False
    # Runtime capability, distinct from the persisted user preference above.
    lightrag_available: bool = True
    metadata_schema: dict[str, Any] = Field(default_factory=lambda: {"version": 1, "fields": []})
    metadata_schema_version: int = 1
    document_count: int = 0
    indexed_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceSummary(BaseModel):
    """Compact summary for dropdown selectors."""
    id: int
    name: str
    document_count: int = 0
    llm_mode: LLMMode = "cloud"

    model_config = {"from_attributes": True}


class LLMRuntimeStatus(BaseModel):
    mode: LLMMode
    provider: str
    model: str
    available: bool
    detail: str | None = None
