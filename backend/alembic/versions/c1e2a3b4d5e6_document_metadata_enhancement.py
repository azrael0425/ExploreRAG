"""Add governed document metadata and LightRAG lifecycle fields.

Revision ID: c1e2a3b4d5e6
Revises: b7d5f0a9e3c1
Create Date: 2026-08-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c1e2a3b4d5e6"
down_revision: Union[str, Sequence[str], None] = "b7d5f0a9e3c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "metadata_schema",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{\"version\": 1, \"fields\": []}'::jsonb"),
        ),
    )
    op.add_column(
        "knowledge_bases",
        sa.Column("metadata_schema_version", sa.Integer(), nullable=False, server_default="1"),
    )
    op.alter_column(
        "documents",
        "custom_metadata",
        existing_type=sa.JSON(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        postgresql_using="COALESCE(custom_metadata, '{}'::json)::jsonb",
        nullable=False,
        server_default=sa.text("'{}'::jsonb"),
    )
    op.add_column(
        "documents",
        sa.Column("processing_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    # Earlier ExploreRAG builds put parser diagnostics in custom_metadata.
    # Keep business metadata clean while retaining those diagnostics for the
    # document-detail screen and troubleshooting.
    op.execute(
        """
        UPDATE documents
        SET processing_metadata = jsonb_build_object('scan_profile', custom_metadata -> 'scan_profile'),
            custom_metadata = custom_metadata - 'scan_profile'
        WHERE custom_metadata ? 'scan_profile'
        """
    )
    op.add_column("documents", sa.Column("metadata_revision", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("documents", sa.Column("content_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("documents", sa.Column("content_sha256", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("metadata_requires_reindex", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("documents", sa.Column("kg_document_id", sa.String(length=128), nullable=True))
    op.add_column("documents", sa.Column("kg_index_status", sa.String(length=20), nullable=False, server_default="not_indexed"))
    op.add_column("documents", sa.Column("kg_indexed_content_version", sa.Integer(), nullable=False, server_default="0"))
    op.create_unique_constraint("uq_documents_kg_document_id", "documents", ["kg_document_id"])
    op.create_index(
        "ix_documents_custom_metadata_gin",
        "documents",
        ["custom_metadata"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_documents_custom_metadata_gin", table_name="documents")
    op.drop_constraint("uq_documents_kg_document_id", "documents", type_="unique")
    op.drop_column("documents", "kg_indexed_content_version")
    op.drop_column("documents", "kg_index_status")
    op.drop_column("documents", "kg_document_id")
    op.drop_column("documents", "content_sha256")
    op.drop_column("documents", "metadata_requires_reindex")
    op.drop_column("documents", "content_version")
    op.drop_column("documents", "metadata_revision")
    op.drop_column("documents", "processing_metadata")
    op.alter_column(
        "documents",
        "custom_metadata",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.JSON(),
        postgresql_using="custom_metadata::json",
        nullable=True,
        server_default=None,
    )
    op.drop_column("knowledge_bases", "metadata_schema_version")
    op.drop_column("knowledge_bases", "metadata_schema")
