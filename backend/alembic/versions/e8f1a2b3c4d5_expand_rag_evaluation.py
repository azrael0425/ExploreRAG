"""Expand RAG evaluation for reproducible ablations and reviewed feedback.

Revision ID: e8f1a2b3c4d5
Revises: d4e5f6a7b8c9
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e8f1a2b3c4d5"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("eval_cases", sa.Column("dataset_name", sa.String(100), nullable=False, server_default="core"))
    op.add_column("eval_cases", sa.Column("dataset_version", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("eval_cases", sa.Column("split", sa.String(20), nullable=False, server_default="dev"))
    op.add_column("eval_cases", sa.Column("is_frozen", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("eval_cases", sa.Column("category", sa.String(50), nullable=False, server_default="other"))
    op.add_column("eval_cases", sa.Column("difficulty", sa.String(20), nullable=False, server_default="medium"))
    op.add_column("eval_cases", sa.Column("expected_behavior", sa.String(20), nullable=False, server_default="answer"))
    op.add_column("eval_cases", sa.Column("review_status", sa.String(20), nullable=False, server_default="draft"))
    op.add_column("eval_cases", sa.Column("reviewed_by", sa.String(100), nullable=True))
    op.add_column("eval_cases", sa.Column("reviewed_at", sa.DateTime(), nullable=True))
    op.add_column("eval_cases", sa.Column("reference_entity_names", sa.JSON(), nullable=True))
    op.add_column("eval_cases", sa.Column("reference_relationships", sa.JSON(), nullable=True))
    op.add_column("eval_cases", sa.Column("conversation_history", sa.JSON(), nullable=True))
    op.create_index("ix_eval_cases_dataset_name", "eval_cases", ["dataset_name"])
    op.create_index("ix_eval_cases_split", "eval_cases", ["split"])
    op.create_index("ix_eval_cases_category", "eval_cases", ["category"])
    op.create_index("ix_eval_cases_review_status", "eval_cases", ["review_status"])

    op.add_column("eval_runs", sa.Column("name", sa.String(200), nullable=True))
    op.add_column("eval_runs", sa.Column("experiment_id", sa.String(64), nullable=True))
    op.add_column("eval_runs", sa.Column("variant", sa.String(20), nullable=False, server_default="custom"))
    op.add_column("eval_runs", sa.Column("dataset_name", sa.String(100), nullable=True))
    op.add_column("eval_runs", sa.Column("dataset_version", sa.Integer(), nullable=True))
    op.add_column("eval_runs", sa.Column("dataset_split", sa.String(20), nullable=True))
    op.create_index("ix_eval_runs_experiment_id", "eval_runs", ["experiment_id"])
    op.create_index("ix_eval_runs_variant", "eval_runs", ["variant"])
    op.create_index("ix_eval_runs_dataset_name", "eval_runs", ["dataset_name"])

    op.add_column("eval_results", sa.Column("retrieval_trace", sa.JSON(), nullable=True))
    op.add_column("eval_results", sa.Column("metric_details", sa.JSON(), nullable=True))
    op.add_column("eval_results", sa.Column("failure_types", sa.JSON(), nullable=True))
    op.add_column("eval_results", sa.Column("baseline_delta", sa.JSON(), nullable=True))
    op.add_column("eval_results", sa.Column("review_status", sa.String(20), nullable=False, server_default="unreviewed"))
    op.add_column("eval_results", sa.Column("reviewer_verdict", sa.String(20), nullable=True))
    op.add_column("eval_results", sa.Column("reviewer_comment", sa.Text(), nullable=True))

    op.add_column("chat_messages", sa.Column("feedback_corrected_answer", sa.Text(), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_reference_chunk_ids", sa.JSON(), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_failure_types", sa.JSON(), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_review_status", sa.String(20), nullable=True))
    op.add_column("chat_messages", sa.Column("feedback_promoted_case_id", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "feedback_promoted_case_id")
    op.drop_column("chat_messages", "feedback_review_status")
    op.drop_column("chat_messages", "feedback_failure_types")
    op.drop_column("chat_messages", "feedback_reference_chunk_ids")
    op.drop_column("chat_messages", "feedback_corrected_answer")

    op.drop_column("eval_results", "reviewer_comment")
    op.drop_column("eval_results", "reviewer_verdict")
    op.drop_column("eval_results", "review_status")
    op.drop_column("eval_results", "baseline_delta")
    op.drop_column("eval_results", "failure_types")
    op.drop_column("eval_results", "metric_details")
    op.drop_column("eval_results", "retrieval_trace")

    op.drop_index("ix_eval_runs_dataset_name", table_name="eval_runs")
    op.drop_index("ix_eval_runs_variant", table_name="eval_runs")
    op.drop_index("ix_eval_runs_experiment_id", table_name="eval_runs")
    op.drop_column("eval_runs", "dataset_version")
    op.drop_column("eval_runs", "dataset_split")
    op.drop_column("eval_runs", "dataset_name")
    op.drop_column("eval_runs", "variant")
    op.drop_column("eval_runs", "experiment_id")
    op.drop_column("eval_runs", "name")

    op.drop_index("ix_eval_cases_review_status", table_name="eval_cases")
    op.drop_index("ix_eval_cases_category", table_name="eval_cases")
    op.drop_index("ix_eval_cases_dataset_name", table_name="eval_cases")
    op.drop_index("ix_eval_cases_split", table_name="eval_cases")
    op.drop_column("eval_cases", "conversation_history")
    op.drop_column("eval_cases", "reference_relationships")
    op.drop_column("eval_cases", "reference_entity_names")
    op.drop_column("eval_cases", "review_status")
    op.drop_column("eval_cases", "reviewed_at")
    op.drop_column("eval_cases", "reviewed_by")
    op.drop_column("eval_cases", "expected_behavior")
    op.drop_column("eval_cases", "difficulty")
    op.drop_column("eval_cases", "category")
    op.drop_column("eval_cases", "dataset_version")
    op.drop_column("eval_cases", "is_frozen")
    op.drop_column("eval_cases", "split")
    op.drop_column("eval_cases", "dataset_name")
