"""Add isolated temporary chat attachments and cleanup generations.

Revision ID: 7c1b5f4e91aa
Revises: 2047460692d0
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "7c1b5f4e91aa"
down_revision: Union[str, Sequence[str], None] = "2047460692d0"
branch_labels = None
depends_on = None


attachment_state = postgresql.ENUM(
    "UPLOADED", "QUEUED", "PARSING", "READY_DIRECT", "INDEXED_TEMP",
    "FAILED", "CLEARING", "DELETED", name="chatattachmentstate",
    create_type=False,
)


def upgrade() -> None:
    attachment_state.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "knowledge_bases",
        sa.Column("chat_cleanup_epoch", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "chat_attachments",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_type", sa.String(length=20), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(length=1000), nullable=False),
        sa.Column("artifact_dir", sa.String(length=1000), nullable=False),
        sa.Column("state", attachment_state, nullable=False),
        sa.Column("parsed_token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("temp_collection", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("cleanup_epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cleanup_pending", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_attachments_workspace_id", "chat_attachments", ["workspace_id"])
    op.create_index("ix_chat_attachments_state", "chat_attachments", ["state"])
    op.create_table(
        "chat_message_attachments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attachment_id", sa.String(length=36), sa.ForeignKey("chat_attachments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_message_attachments_message_id", "chat_message_attachments", ["message_id"])
    op.create_index("ix_chat_message_attachments_attachment_id", "chat_message_attachments", ["attachment_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_message_attachments_attachment_id", table_name="chat_message_attachments")
    op.drop_index("ix_chat_message_attachments_message_id", table_name="chat_message_attachments")
    op.drop_table("chat_message_attachments")
    op.drop_index("ix_chat_attachments_state", table_name="chat_attachments")
    op.drop_index("ix_chat_attachments_workspace_id", table_name="chat_attachments")
    op.drop_table("chat_attachments")
    op.drop_column("knowledge_bases", "chat_cleanup_epoch")
    attachment_state.drop(op.get_bind(), checkfirst=True)
