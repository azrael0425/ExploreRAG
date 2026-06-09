"""Request and response models for the RAG evaluation API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


CaseStatus = Literal["draft", "active", "archived"]
CaseSource = Literal["manual", "ai", "production"]
ReviewStatus = Literal["draft", "approved", "rejected"]
RunType = Literal["retrieval", "fast", "full"]
RunVariant = Literal["A", "B", "C", "D", "custom"]
FailureType = Literal[
    "knowledge_gap",
    "retrieval_miss",
    "rerank_error",
    "graph_miss",
    "graph_noise",
    "generation_error",
    "citation_error",
    "unanswerable_error",
]


class EvalCaseWrite(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    reference_answer: str | None = Field(default=None, max_length=20000)
    reference_chunk_ids: list[str] = Field(default_factory=list, max_length=50)
    reference_contexts: list[str] = Field(default_factory=list, max_length=50)
    tags: list[str] = Field(default_factory=list, max_length=20)
    status: CaseStatus = "draft"
    source: CaseSource = "manual"
    dataset_name: str = Field(default="core", min_length=1, max_length=100)
    dataset_version: int = Field(default=1, ge=1)
    split: Literal["dev", "test"] = "dev"
    is_frozen: bool = False
    category: str = Field(default="other", min_length=1, max_length=50)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    expected_behavior: Literal["answer", "refuse"] = "answer"
    review_status: ReviewStatus = "draft"
    reference_entity_names: list[str] = Field(default_factory=list, max_length=100)
    reference_relationships: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    conversation_history: list[dict[str, str]] = Field(default_factory=list, max_length=20)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class EvalCasePatch(BaseModel):
    question: str | None = Field(default=None, min_length=1, max_length=5000)
    reference_answer: str | None = Field(default=None, max_length=20000)
    reference_chunk_ids: list[str] | None = Field(default=None, max_length=50)
    reference_contexts: list[str] | None = Field(default=None, max_length=50)
    tags: list[str] | None = Field(default=None, max_length=20)
    status: CaseStatus | None = None
    dataset_name: str | None = Field(default=None, min_length=1, max_length=100)
    dataset_version: int | None = Field(default=None, ge=1)
    split: Literal["dev", "test"] | None = None
    is_frozen: bool | None = None
    category: str | None = Field(default=None, min_length=1, max_length=50)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    expected_behavior: Literal["answer", "refuse"] | None = None
    review_status: ReviewStatus | None = None
    reference_entity_names: list[str] | None = Field(default=None, max_length=100)
    reference_relationships: list[dict[str, Any]] | None = Field(default=None, max_length=100)
    conversation_history: list[dict[str, str]] | None = Field(default=None, max_length=20)
    metadata: dict[str, Any] | None = None


class EvalCaseResponse(EvalCaseWrite):
    id: int
    workspace_id: int
    input_hash: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EvalCaseListResponse(BaseModel):
    items: list[EvalCaseResponse]
    total: int


class EvalCaseReviewWrite(BaseModel):
    case_ids: list[int] = Field(min_length=1, max_length=500)
    review_status: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=100)
    activate: bool = True
    freeze: bool = False


class EvalImportRequest(BaseModel):
    jsonl: str = Field(min_length=1, max_length=5_000_000)
    activate: bool = False
    reviewer: str | None = Field(default=None, max_length=100)


class EvalImportResponse(BaseModel):
    created: int
    skipped: int
    errors: list[str] = Field(default_factory=list)


class AIGenerateRequest(BaseModel):
    document_ids: list[int] = Field(default_factory=list, max_length=100)
    count: int = Field(default=8, ge=1, le=30)
    llm_mode: Literal["cloud", "local"] | None = None
    activate: bool = False
    dataset_name: str = Field(default="core", min_length=1, max_length=100)
    dataset_version: int = Field(default=1, ge=1)
    split: Literal["dev", "test"] = "dev"
    categories: list[str] = Field(default_factory=list, max_length=20)
    seed: int = Field(default=0, ge=0, le=1_000_000)


class EvalRunCreate(BaseModel):
    case_ids: list[int] = Field(default_factory=list, max_length=500)
    run_type: RunType = "fast"
    name: str | None = Field(default=None, max_length=200)
    experiment_id: str | None = Field(default=None, max_length=64)
    variant: RunVariant = "custom"
    dataset_name: str | None = Field(default=None, max_length=100)
    dataset_version: int | None = Field(default=None, ge=1)
    split: Literal["dev", "test"] | None = None
    baseline_run_id: int | None = None
    top_k: int = Field(default=4, ge=1, le=20)
    prefetch_k: int = Field(default=20, ge=4, le=200)
    retrieval_mode: Literal["hybrid", "vector_only", "local", "global"] = "hybrid"
    enable_reranker: bool | None = None
    enable_knowledge_graph: bool | None = None
    warmup_queries: int = Field(default=0, ge=0, le=3)
    case_order_seed: int = Field(default=0, ge=0, le=100_000_000)


class EvalAblationCreate(BaseModel):
    case_ids: list[int] = Field(default_factory=list, max_length=500)
    run_type: RunType = "retrieval"
    name: str = Field(default="2x2 ablation", min_length=1, max_length=200)
    dataset_name: str | None = Field(default=None, max_length=100)
    dataset_version: int | None = Field(default=None, ge=1)
    split: Literal["dev", "test"] | None = None
    top_k: int = Field(default=4, ge=1, le=20)
    prefetch_k: int = Field(default=20, ge=4, le=200)
    retrieval_mode: Literal["hybrid", "local", "global"] = "hybrid"
    warmup_queries: int = Field(default=1, ge=0, le=3)
    case_order_seed: int = Field(default=20260810, ge=0, le=100_000_000)
    variant_order: list[Literal["A", "B", "C", "D"]] = Field(
        default_factory=lambda: ["A", "B", "C", "D"], min_length=4, max_length=4
    )

    @field_validator("variant_order")
    @classmethod
    def validate_variant_order(cls, value: list[str]) -> list[str]:
        if sorted(value) != ["A", "B", "C", "D"]:
            raise ValueError("variant_order must contain A, B, C and D exactly once")
        return value


class EvalRunResponse(BaseModel):
    id: int
    workspace_id: int
    status: str
    run_type: str
    name: str | None = None
    experiment_id: str | None = None
    variant: str = "custom"
    dataset_name: str | None = None
    dataset_version: int | None = None
    dataset_split: str | None = None
    case_ids: list[int]
    config: dict[str, Any] | None = None
    target_config: dict[str, Any] | None = None
    metrics_summary: dict[str, Any] | None = None
    error_message: str | None = None
    baseline_run_id: int | None = None
    is_baseline: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalResultResponse(BaseModel):
    id: int
    run_id: int
    case_id: int
    question: str
    reference_answer: str | None = None
    reference_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    answer: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    performance: dict[str, Any] | None = None
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    metric_status: dict[str, str] = Field(default_factory=dict)
    metric_details: dict[str, Any] = Field(default_factory=dict)
    failure_types: list[str] = Field(default_factory=list)
    baseline_delta: dict[str, Any] = Field(default_factory=dict)
    review_status: str = "unreviewed"
    reviewer_verdict: str | None = None
    reviewer_comment: str | None = None
    verdict: str
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunDetailResponse(EvalRunResponse):
    results: list[EvalResultResponse] = Field(default_factory=list)


class EvalAblationResponse(BaseModel):
    experiment_id: str
    runs: list[EvalRunResponse]


class EvalRunComparisonResponse(BaseModel):
    baseline_run_id: int
    candidate_run_id: int
    paired_case_count: int
    metric_deltas: dict[str, Any] = Field(default_factory=dict)
    latency_deltas: dict[str, Any] = Field(default_factory=dict)
    regressions: list[dict[str, Any]] = Field(default_factory=list)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    gate: dict[str, Any] = Field(default_factory=dict)
    subgroups: dict[str, Any] = Field(default_factory=dict)


class EvalFeedbackWrite(BaseModel):
    rating: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=4000)
    source_ratings: dict[str, Literal[-1, 1]] = Field(default_factory=dict)
    corrected_answer: str | None = Field(default=None, max_length=20000)
    reference_chunk_ids: list[str] = Field(default_factory=list, max_length=50)
    failure_types: list[FailureType] = Field(default_factory=list, max_length=20)
    review_status: Literal["pending", "reviewed", "promoted", "rejected"] = "pending"


class EvalFeedbackResponse(BaseModel):
    message_id: str
    reply_to_message_id: str | None = None
    rating: int | None = None
    comment: str | None = None
    source_ratings: dict[str, int] = Field(default_factory=dict)
    corrected_answer: str | None = None
    reference_chunk_ids: list[str] = Field(default_factory=list)
    failure_types: list[str] = Field(default_factory=list)
    review_status: str | None = None
    promoted_case_id: int | None = None
    question: str | None = None
    answer: str
    created_at: datetime


class EvalOverviewResponse(BaseModel):
    active_case_count: int
    draft_case_count: int
    total_run_count: int
    latest_run: EvalRunResponse | None = None
    feedback_count: int


class EvalResultReviewWrite(BaseModel):
    review_status: Literal["unreviewed", "reviewed"] = "reviewed"
    reviewer_verdict: Literal["pass", "fail", "needs_review"]
    reviewer_comment: str | None = Field(default=None, max_length=4000)
    failure_types: list[FailureType] = Field(default_factory=list, max_length=20)
