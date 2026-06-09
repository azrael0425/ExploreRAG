from pydantic import BaseModel, Field
from datetime import datetime
from app.models.document import DocumentStatus


class DocumentBase(BaseModel):
    filename: str
    original_filename: str
    file_type: str
    file_size: int


class DocumentCreate(DocumentBase):
    workspace_id: int


class DocumentResponse(DocumentBase):
    id: int
    workspace_id: int
    status: DocumentStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    # ExploreRAG fields
    page_count: int = 0
    image_count: int = 0
    table_count: int = 0
    parser_version: str | None = None
    processing_time_ms: int = 0
    custom_metadata: dict = Field(default_factory=dict)
    processing_metadata: dict = Field(default_factory=dict)
    metadata_revision: int = 1
    content_version: int = 1
    kg_document_id: str | None = None
    kg_index_status: str = "not_indexed"
    kg_indexed_content_version: int = 0
    metadata_requires_reindex: bool = False

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    status: DocumentStatus
    message: str
