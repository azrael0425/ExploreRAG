"""Persistent records for the built-in RAG evaluation workflow.

The evaluation feature intentionally uses three durable records only: a case,
one run, and one result per case in that run.  This keeps the first release
operable on the application's existing PostgreSQL deployment without a queue
or a separate experiment-tracking service.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft", index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual", index=True)
    dataset_name: Mapped[str] = mapped_column(String(100), nullable=False, default="core", index=True)
    dataset_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    split: Mapped[str] = mapped_column(String(20), nullable=False, default="dev", index=True)
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False, default="other", index=True)
    difficulty: Mapped[str] = mapped_column(String(20), nullable=False, default="medium")
    expected_behavior: Mapped[str] = mapped_column(String(20), nullable=False, default="answer")
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft", index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reference_contexts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reference_entity_names: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reference_relationships: Mapped[list | None] = mapped_column(JSON, nullable=True)
    conversation_history: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    extra_metadata: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    workspace_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    run_type: Mapped[str] = mapped_column(String(20), nullable=False, default="fast")
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    variant: Mapped[str] = mapped_column(String(20), nullable=False, default="custom", index=True)
    dataset_name: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    dataset_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_split: Mapped[str | None] = mapped_column(String(20), nullable=True)
    case_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    target_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("eval_runs.id", ondelete="SET NULL"), nullable=True
    )
    is_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("eval_cases.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_chunk_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    retrieved_contexts: Mapped[list | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    performance: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    retrieval_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metric_status: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metric_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    failure_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    baseline_delta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    review_status: Mapped[str] = mapped_column(String(20), nullable=False, default="unreviewed")
    reviewer_verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
    reviewer_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
