"""Add built-in RAG evaluation records and lightweight chat feedback.

Revision ID: b7d5f0a9e3c1
Revises: a91c4e7d2b10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7d5f0a9e3c1"
down_revision: Union[str, Sequence[str], None] = "a91c4e7d2b10"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eval_cases",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="draft"),
        sa.Column("source", sa.String(length=20), nullable=False, server_default="manual"),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=True),
        sa.Column("reference_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("reference_contexts", sa.JSON(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_eval_cases_workspace_id", "eval_cases", ["workspace_id"])
    op.create_index("ix_eval_cases_status", "eval_cases", ["status"])
    op.create_index("ix_eval_cases_source", "eval_cases", ["source"])
    op.create_index("ix_eval_cases_input_hash", "eval_cases", ["input_hash"])
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("workspace_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="queued"),
        sa.Column("run_type", sa.String(length=20), nullable=False, server_default="fast"),
        sa.Column("case_ids", sa.JSON(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("target_config", sa.JSON(), nullable=True),
        sa.Column("metrics_summary", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("baseline_run_id", sa.Integer(), sa.ForeignKey("eval_runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("is_baseline", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_eval_runs_workspace_id", "eval_runs", ["workspace_id"])
    op.create_index("ix_eval_runs_status", "eval_runs", ["status"])
    op.create_table(
        "eval_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.Integer(), sa.ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("eval_cases.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=True),
        sa.Column("reference_chunk_ids", sa.JSON(), nullable=True),
        sa.Column("retrieved_contexts", sa.JSON(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("sources", sa.JSON(), nullable=True),
        sa.Column("performance", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("metric_status", sa.JSON(), nullable=True),
        sa.Column("verdict", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_eval_results_run_id", "eval_results", ["run_id"])
    op.create_index("ix_eval_results_case_id", "eval_results", ["case_id"])
    op.add_column("chat_messages", sa.Column("reply_to_message_id", sa.String(length=50), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_rating", sa.Integer(), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_comment", sa.Text(), nullable=True))
    op.add_column("chat_messages", sa.Column("source_ratings", sa.JSON(), nullable=True))
    op.create_index("ix_chat_messages_reply_to_message_id", "chat_messages", ["reply_to_message_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_reply_to_message_id", table_name="chat_messages")
    op.drop_column("chat_messages", "source_ratings")
    op.drop_column("chat_messages", "feedback_comment")
    op.drop_column("chat_messages", "feedback_rating")
    op.drop_column("chat_messages", "reply_to_message_id")
    op.drop_index("ix_eval_results_case_id", table_name="eval_results")
    op.drop_index("ix_eval_results_run_id", table_name="eval_results")
    op.drop_table("eval_results")
    op.drop_index("ix_eval_runs_status", table_name="eval_runs")
    op.drop_index("ix_eval_runs_workspace_id", table_name="eval_runs")
    op.drop_table("eval_runs")
    op.drop_index("ix_eval_cases_input_hash", table_name="eval_cases")
    op.drop_index("ix_eval_cases_source", table_name="eval_cases")
    op.drop_index("ix_eval_cases_status", table_name="eval_cases")
    op.drop_index("ix_eval_cases_workspace_id", table_name="eval_cases")
    op.drop_table("eval_cases")
