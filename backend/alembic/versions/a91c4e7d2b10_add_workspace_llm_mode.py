"""Add per-workspace local/cloud LLM selection.

Revision ID: a91c4e7d2b10
Revises: 7c1b5f4e91aa
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a91c4e7d2b10"
down_revision: Union[str, Sequence[str], None] = "7c1b5f4e91aa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledge_bases",
        sa.Column(
            "llm_mode",
            sa.String(length=16),
            nullable=False,
            server_default="cloud",
        ),
    )
    op.create_check_constraint(
        "ck_knowledge_bases_llm_mode",
        "knowledge_bases",
        "llm_mode IN ('cloud', 'local')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_knowledge_bases_llm_mode",
        "knowledge_bases",
        type_="check",
    )
    op.drop_column("knowledge_bases", "llm_mode")
