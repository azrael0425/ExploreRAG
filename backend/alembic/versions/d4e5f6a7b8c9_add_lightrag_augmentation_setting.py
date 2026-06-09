"""Add workspace-level LightRAG query augmentation preference.

Revision ID: d4e5f6a7b8c9
Revises: c1e2a3b4d5e6
Create Date: 2026-08-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c1e2a3b4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing workspaces retain their current vector-only answer behaviour
    # until an administrator explicitly enables the new setting.
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "lightrag_augmentation_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column("knowledge_bases", "lightrag_augmentation_enabled")
