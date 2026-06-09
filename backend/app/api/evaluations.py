"""HTTP API for test cases, evaluation runs, and V1.5 feedback capture."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
from app.core.config import settings
from app.core.deps import get_db
from app.evaluation.fingerprints import build_experiment_snapshot
from app.evaluation.generator import generate_ai_cases, input_hash
from app.evaluation.reporting import compare_run_contracts, compare_run_results
from app.evaluation.task_manager import evaluation_task_manager
from app.models.chat_message import ChatMessage
from app.models.evaluation import EvalCase, EvalResult, EvalRun
from app.models.knowledge_base import KnowledgeBase
from app.services.explore_rag_factory import get_explore_rag_service
from app.schemas.evaluation import (
    AIGenerateRequest,
    EvalAblationCreate,
    EvalAblationResponse,
    EvalCaseListResponse,
    EvalCasePatch,
    EvalCaseResponse,
    EvalCaseReviewWrite,
    EvalCaseWrite,
    EvalFeedbackResponse,
    EvalFeedbackWrite,
    EvalImportRequest,
    EvalImportResponse,
    EvalOverviewResponse,
    EvalResultResponse,
    EvalResultReviewWrite,
    EvalRunCreate,
    EvalRunDetailResponse,
    EvalRunResponse,
    EvalRunComparisonResponse,
)

router = APIRouter(prefix="/evaluations", tags=["evaluations"])

_CHUNK_ID_RE = re.compile(r"^doc_\d+_chunk_\d+$")


async def _workspace(db: AsyncSession, workspace_id: int) -> KnowledgeBase:
    workspace = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))).scalar_one_or_none()
    if workspace is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return workspace


def _case_response(item: EvalCase) -> EvalCaseResponse:
    return EvalCaseResponse(
        id=item.id, workspace_id=item.workspace_id, question=item.question,
        reference_answer=item.reference_answer, reference_chunk_ids=list(item.reference_chunk_ids or []),
        reference_contexts=list(item.reference_contexts or []), tags=list(item.tags or []),
        status=item.status, source=item.source, metadata=dict(item.extra_metadata or {}),
        dataset_name=item.dataset_name, dataset_version=item.dataset_version,
        split=item.split, is_frozen=item.is_frozen,
        category=item.category, difficulty=item.difficulty,
        expected_behavior=item.expected_behavior, review_status=item.review_status,
        reference_entity_names=list(item.reference_entity_names or []),
        reference_relationships=list(item.reference_relationships or []),
        conversation_history=list(item.conversation_history or []),
        reviewed_by=item.reviewed_by, reviewed_at=item.reviewed_at,
        input_hash=item.input_hash, created_at=item.created_at, updated_at=item.updated_at,
    )


def _run_response(item: EvalRun) -> EvalRunResponse:
    return EvalRunResponse(
        id=item.id, workspace_id=item.workspace_id, status=item.status, run_type=item.run_type,
        name=item.name, experiment_id=item.experiment_id, variant=item.variant,
        dataset_name=item.dataset_name, dataset_version=item.dataset_version,
        dataset_split=item.dataset_split,
        case_ids=[int(value) for value in item.case_ids or []], config=item.config,
        target_config=item.target_config, metrics_summary=item.metrics_summary,
        error_message=item.error_message, baseline_run_id=item.baseline_run_id,
        is_baseline=item.is_baseline,
        started_at=item.started_at, finished_at=item.finished_at, created_at=item.created_at,
    )


def _result_response(item: EvalResult) -> EvalResultResponse:
    return EvalResultResponse(
        id=item.id, run_id=item.run_id, case_id=item.case_id, question=item.question,
        reference_answer=item.reference_answer, reference_chunk_ids=list(item.reference_chunk_ids or []),
        retrieved_contexts=list(item.retrieved_contexts or []), answer=item.answer, sources=list(item.sources or []),
        performance=item.performance, retrieval_trace=dict(item.retrieval_trace or {}),
        metrics=dict(item.metrics or {}), metric_status=dict(item.metric_status or {}),
        metric_details=dict(item.metric_details or {}), failure_types=list(item.failure_types or []),
        baseline_delta=dict(item.baseline_delta or {}), review_status=item.review_status,
        reviewer_verdict=item.reviewer_verdict, reviewer_comment=item.reviewer_comment,
        verdict=item.verdict, error_message=item.error_message, created_at=item.created_at,
    )


@router.get("/workspaces/{workspace_id}/overview", response_model=EvalOverviewResponse)
async def evaluation_overview(workspace_id: int, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    count_rows = await db.execute(
        select(EvalCase.status, func.count(EvalCase.id)).where(EvalCase.workspace_id == workspace_id).group_by(EvalCase.status)
    )
    count_by_status = {key: int(value) for key, value in count_rows.all()}
    latest = (await db.execute(
        select(EvalRun).where(EvalRun.workspace_id == workspace_id).order_by(EvalRun.created_at.desc()).limit(1)
    )).scalar_one_or_none()
    total_runs = (await db.execute(select(func.count(EvalRun.id)).where(EvalRun.workspace_id == workspace_id))).scalar_one()
    feedback_count = (await db.execute(select(func.count(ChatMessage.id)).where(
        ChatMessage.workspace_id == workspace_id, ChatMessage.role == "assistant", ChatMessage.feedback_rating.is_not(None)
    ))).scalar_one()
    return EvalOverviewResponse(
        active_case_count=count_by_status.get("active", 0), draft_case_count=count_by_status.get("draft", 0),
        total_run_count=int(total_runs or 0), latest_run=_run_response(latest) if latest else None,
        feedback_count=int(feedback_count or 0),
    )


@router.get("/workspaces/{workspace_id}/cases", response_model=EvalCaseListResponse)
async def list_cases(
    workspace_id: int,
    status_filter: str | None = None,
    source: str | None = None,
    dataset_name: str | None = None,
    dataset_version: int | None = None,
    category: str | None = None,
    split: str | None = None,
    review_status: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    await _workspace(db, workspace_id)
    query = select(EvalCase).where(EvalCase.workspace_id == workspace_id)
    if status_filter:
        query = query.where(EvalCase.status == status_filter)
    if source:
        query = query.where(EvalCase.source == source)
    if dataset_name:
        query = query.where(EvalCase.dataset_name == dataset_name)
    if dataset_version:
        query = query.where(EvalCase.dataset_version == dataset_version)
    if category:
        query = query.where(EvalCase.category == category)
    if split:
        query = query.where(EvalCase.split == split)
    if review_status:
        query = query.where(EvalCase.review_status == review_status)
    items = list((await db.execute(query.order_by(EvalCase.updated_at.desc()))).scalars().all())
    return EvalCaseListResponse(items=[_case_response(item) for item in items], total=len(items))


@router.post("/workspaces/{workspace_id}/cases", response_model=EvalCaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(workspace_id: int, body: EvalCaseWrite, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    if body.review_status == "approved":
        raise HTTPException(status_code=422, detail="Use the review endpoint to approve a case")
    reference_ids = list(body.reference_chunk_ids)
    item = EvalCase(
        workspace_id=workspace_id, status=body.status, source=body.source, question=body.question,
        dataset_name=body.dataset_name, dataset_version=body.dataset_version,
        split=body.split, is_frozen=body.is_frozen,
        category=body.category, difficulty=body.difficulty,
        expected_behavior=body.expected_behavior, review_status=body.review_status,
        reference_answer=body.reference_answer, reference_chunk_ids=reference_ids,
        reference_contexts=list(body.reference_contexts),
        reference_entity_names=list(body.reference_entity_names),
        reference_relationships=list(body.reference_relationships),
        conversation_history=list(body.conversation_history),
        tags=list(body.tags), extra_metadata=dict(body.metadata),
        input_hash=input_hash(body.question, body.reference_answer, reference_ids),
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return _case_response(item)


@router.patch("/cases/{case_id}", response_model=EvalCaseResponse)
async def update_case(case_id: int, body: EvalCasePatch, db: AsyncSession = Depends(get_db)):
    item = (await db.execute(select(EvalCase).where(EvalCase.id == case_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation case not found")
    values = body.model_dump(exclude_unset=True)
    if values.get("review_status") == "approved":
        raise HTTPException(status_code=422, detail="Use the review endpoint to approve a case")
    if item.is_frozen and set(values) != {"is_frozen"}:
        raise HTTPException(
            status_code=409,
            detail="Frozen test cases must be explicitly unfrozen before editing",
        )
    if "reference_chunk_ids" in values and "reference_contexts" not in values:
        reference_ids = [str(value) for value in values["reference_chunk_ids"]]
        if reference_ids:
            workspace = await _workspace(db, item.workspace_id)
            rag_service = get_explore_rag_service(db, item.workspace_id, llm_mode=workspace.llm_mode)
            gold_rows = rag_service.vector_store.get_by_ids(reference_ids)
            context_by_id = {
                str(chunk_id): str(content)
                for chunk_id, content in zip(
                    gold_rows.get("ids", []),
                    gold_rows.get("documents", []),
                )
            }
            missing = [chunk_id for chunk_id in reference_ids if chunk_id not in context_by_id]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Gold chunks are not present in the active vector index: {missing}",
                )
            values["reference_contexts"] = [context_by_id[chunk_id] for chunk_id in reference_ids]
        else:
            values["reference_contexts"] = []
    if "metadata" in values:
        item.extra_metadata = values.pop("metadata")
    for key, value in values.items():
        setattr(item, key, value)
    item.input_hash = input_hash(item.question, item.reference_answer, list(item.reference_chunk_ids or []))
    await db.commit()
    await db.refresh(item)
    return _case_response(item)


@router.post(
    "/workspaces/{workspace_id}/cases/review",
    response_model=list[EvalCaseResponse],
)
async def review_cases(
    workspace_id: int,
    body: EvalCaseReviewWrite,
    db: AsyncSession = Depends(get_db),
):
    await _workspace(db, workspace_id)
    rows = list((await db.execute(select(EvalCase).where(
        EvalCase.workspace_id == workspace_id,
        EvalCase.id.in_(body.case_ids),
    ))).scalars().all())
    by_id = {item.id: item for item in rows}
    if len(by_id) != len(set(body.case_ids)):
        raise HTTPException(status_code=422, detail="Every reviewed case must belong to the workspace")
    now = datetime.utcnow()
    for case_id in body.case_ids:
        item = by_id[case_id]
        if item.is_frozen:
            raise HTTPException(status_code=409, detail=f"Case {case_id} is frozen")
        if body.review_status == "approved" and item.expected_behavior == "answer":
            if not item.reference_answer or not item.reference_chunk_ids:
                raise HTTPException(
                    status_code=422,
                    detail=f"Case {case_id} needs a reference answer and at least one gold chunk",
                )
            if item.category in {"multi_hop", "cross_document"} and not item.reference_entity_names:
                raise HTTPException(
                    status_code=422,
                    detail=f"Case {case_id} needs at least one Gold Entity for graph evaluation",
                )
        item.review_status = body.review_status
        item.reviewed_by = body.reviewer.strip()
        item.reviewed_at = now
        item.status = "active" if body.activate and body.review_status == "approved" else "draft"
        item.is_frozen = bool(body.freeze and item.split == "test" and body.review_status == "approved")
    await db.commit()
    for item in rows:
        await db.refresh(item)
    return [_case_response(by_id[case_id]) for case_id in body.case_ids]


@router.delete("/cases/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def archive_case(case_id: int, db: AsyncSession = Depends(get_db)):
    item = (await db.execute(select(EvalCase).where(EvalCase.id == case_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation case not found")
    item.status = "archived"
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/workspaces/{workspace_id}/cases/import", response_model=EvalImportResponse)
async def import_cases(workspace_id: int, body: EvalImportRequest, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    if body.activate and not (body.reviewer or "").strip():
        raise HTTPException(status_code=422, detail="reviewer is required when activating imported cases")
    created, skipped, errors = 0, 0, []
    hashes = set((await db.execute(select(EvalCase.input_hash).where(EvalCase.workspace_id == workspace_id))).scalars().all())
    for line_no, line in enumerate(body.jsonl.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = EvalCaseWrite.model_validate(json.loads(line))
            digest = input_hash(parsed.question, parsed.reference_answer, parsed.reference_chunk_ids)
            if digest in hashes:
                skipped += 1
                continue
            hashes.add(digest)
            db.add(EvalCase(
                workspace_id=workspace_id, status="active" if body.activate else parsed.status, source=parsed.source,
                dataset_name=parsed.dataset_name, dataset_version=parsed.dataset_version,
                split=parsed.split, is_frozen=bool(body.activate and parsed.split == "test"),
                category=parsed.category, difficulty=parsed.difficulty,
                expected_behavior=parsed.expected_behavior,
                review_status="approved" if body.activate else parsed.review_status,
                reviewed_by=body.reviewer.strip() if body.activate and body.reviewer else None,
                reviewed_at=datetime.utcnow() if body.activate else None,
                question=parsed.question, reference_answer=parsed.reference_answer,
                reference_chunk_ids=parsed.reference_chunk_ids, reference_contexts=parsed.reference_contexts,
                reference_entity_names=parsed.reference_entity_names,
                reference_relationships=parsed.reference_relationships,
                conversation_history=parsed.conversation_history,
                tags=parsed.tags, extra_metadata=parsed.metadata, input_hash=digest,
            ))
            created += 1
        except Exception as exc:
            errors.append(f"line {line_no}: {exc}")
    await db.commit()
    return EvalImportResponse(created=created, skipped=skipped, errors=errors[:50])


@router.get("/workspaces/{workspace_id}/cases/export")
async def export_cases(workspace_id: int, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    items = list((await db.execute(select(EvalCase).where(EvalCase.workspace_id == workspace_id))).scalars().all())
    lines = [json.dumps(_case_response(item).model_dump(mode="json"), ensure_ascii=False) for item in items]
    return PlainTextResponse("\n".join(lines) + ("\n" if lines else ""), media_type="application/x-ndjson")


@router.post("/workspaces/{workspace_id}/cases/generate", response_model=list[EvalCaseResponse])
async def generate_cases(workspace_id: int, body: AIGenerateRequest, db: AsyncSession = Depends(get_db)):
    workspace = await _workspace(db, workspace_id)
    try:
        cases = await generate_ai_cases(
            db, workspace_id=workspace_id, document_ids=body.document_ids, count=body.count,
            llm_mode=body.llm_mode or workspace.llm_mode, activate=body.activate,
            dataset_name=body.dataset_name, dataset_version=body.dataset_version,
            split=body.split,
            categories=body.categories,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_case_response(item) for item in cases]


@router.get("/workspaces/{workspace_id}/runs", response_model=list[EvalRunResponse])
async def list_runs(workspace_id: int, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    rows = list((await db.execute(select(EvalRun).where(
        EvalRun.workspace_id == workspace_id
    ).order_by(EvalRun.created_at.desc()))).scalars().all())
    return [_run_response(item) for item in rows]


async def _select_run_cases(
    db: AsyncSession,
    workspace_id: int,
    case_ids: list[int],
    dataset_name: str | None,
    dataset_version: int | None,
    split: str | None = None,
) -> list[EvalCase]:
    query = select(EvalCase).where(
        EvalCase.workspace_id == workspace_id,
        EvalCase.status == "active",
        EvalCase.review_status == "approved",
    )
    if case_ids:
        query = query.where(EvalCase.id.in_(case_ids))
    if dataset_name:
        query = query.where(EvalCase.dataset_name == dataset_name)
    if dataset_version:
        query = query.where(EvalCase.dataset_version == dataset_version)
    if split:
        query = query.where(EvalCase.split == split)
    rows = list((await db.execute(query)).scalars().all())
    by_id = {item.id: item for item in rows}
    return [by_id[value] for value in case_ids if value in by_id] if case_ids else rows


async def _create_run_record(
    db: AsyncSession,
    *,
    workspace: KnowledgeBase,
    body: EvalRunCreate,
    cases: list[EvalCase],
    baseline_run_id: int | None = None,
) -> EvalRun:
    enable_reranker = (
        settings.EXPLORERAG_ENABLE_RERANKER
        if body.enable_reranker is None else body.enable_reranker
    )
    enable_knowledge_graph = (
        workspace.lightrag_augmentation_enabled
        if body.enable_knowledge_graph is None else body.enable_knowledge_graph
    )
    config = {
        "top_k": body.top_k,
        "prefetch_k": body.prefetch_k,
        "retrieval_mode": body.retrieval_mode,
        "enable_reranker": enable_reranker,
        "enable_knowledge_graph": enable_knowledge_graph,
        "warmup_queries": body.warmup_queries,
        "case_order_seed": body.case_order_seed,
    }
    prompt = (workspace.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT
    snapshot = await build_experiment_snapshot(
        db, workspace=workspace, run_config=config, system_prompt=prompt, cases=cases
    )
    dataset_names = {item.dataset_name for item in cases}
    dataset_versions = {item.dataset_version for item in cases}
    item = EvalRun(
        workspace_id=workspace.id,
        status="queued",
        run_type=body.run_type,
        name=body.name,
        experiment_id=body.experiment_id,
        variant=body.variant,
        dataset_name=body.dataset_name or (next(iter(dataset_names)) if len(dataset_names) == 1 else "mixed"),
        dataset_version=body.dataset_version or (next(iter(dataset_versions)) if len(dataset_versions) == 1 else None),
        dataset_split=body.split,
        case_ids=[case.id for case in cases],
        config=config,
        target_config={
            "target": "production_retriever" if body.run_type == "retrieval" else "production_rag_service",
            "snapshot": snapshot,
        },
        baseline_run_id=baseline_run_id if baseline_run_id is not None else body.baseline_run_id,
    )
    db.add(item)
    await db.flush()
    return item


@router.post("/workspaces/{workspace_id}/runs", response_model=EvalRunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(workspace_id: int, body: EvalRunCreate, db: AsyncSession = Depends(get_db)):
    workspace = await _workspace(db, workspace_id)
    cases = await _select_run_cases(
        db, workspace_id, body.case_ids, body.dataset_name, body.dataset_version, body.split
    )
    if not cases:
        raise HTTPException(
            status_code=422,
            detail="Select at least one active, approved case (or approve the dataset first)",
        )
    if body.baseline_run_id:
        base = (await db.execute(select(EvalRun).where(EvalRun.id == body.baseline_run_id))).scalar_one_or_none()
        if base is None or base.workspace_id != workspace_id:
            raise HTTPException(status_code=422, detail="Baseline run must belong to this workspace")
    active_baseline = (await db.execute(select(EvalRun).where(
        EvalRun.workspace_id == workspace_id, EvalRun.is_baseline.is_(True)
    ).order_by(EvalRun.created_at.desc()).limit(1))).scalar_one_or_none()
    if active_baseline is not None:
        same_contract = (
            list(active_baseline.case_ids or []) == [case.id for case in cases]
            and active_baseline.run_type == body.run_type
            and all(
                (active_baseline.config or {}).get(key) == value
                for key, value in {
                    "top_k": body.top_k,
                    "prefetch_k": body.prefetch_k,
                    "retrieval_mode": body.retrieval_mode,
                }.items()
            )
        )
        if not same_contract:
            active_baseline = None
    item = await _create_run_record(
        db,
        workspace=workspace,
        body=body,
        cases=cases,
        baseline_run_id=body.baseline_run_id or (active_baseline.id if active_baseline else None),
    )
    await db.commit()
    await db.refresh(item)
    evaluation_task_manager.enqueue(item.id)
    return _run_response(item)


@router.post(
    "/workspaces/{workspace_id}/experiments/ablation",
    response_model=EvalAblationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_ablation_experiment(
    workspace_id: int,
    body: EvalAblationCreate,
    db: AsyncSession = Depends(get_db),
):
    """Queue the canonical Vector × Reranker × KG experiment matrix."""
    workspace = await _workspace(db, workspace_id)
    cases = await _select_run_cases(
        db, workspace_id, body.case_ids, body.dataset_name, body.dataset_version, body.split
    )
    if not cases:
        raise HTTPException(status_code=422, detail="Ablation requires active, approved cases")
    experiment_id = uuid.uuid4().hex
    variants_by_name = {
        "A": ("A", False, False, "Vector baseline"),
        "B": ("B", True, False, "Vector + reranker"),
        "C": ("C", False, True, "Vector + knowledge graph"),
        "D": ("D", True, True, "Full pipeline"),
    }
    variants = [variants_by_name[name] for name in body.variant_order]
    runs: list[EvalRun] = []
    baseline_id: int | None = None
    for variant, reranker, graph, label in variants:
        request = EvalRunCreate(
            case_ids=[case.id for case in cases],
            run_type=body.run_type,
            name=f"{body.name} · {variant} {label}",
            experiment_id=experiment_id,
            variant=variant,  # type: ignore[arg-type]
            dataset_name=body.dataset_name,
            dataset_version=body.dataset_version,
            split=body.split,
            top_k=body.top_k,
            prefetch_k=body.prefetch_k,
            retrieval_mode=body.retrieval_mode,
            enable_reranker=reranker,
            enable_knowledge_graph=graph,
            warmup_queries=body.warmup_queries,
            case_order_seed=body.case_order_seed,
        )
        item = await _create_run_record(
            db,
            workspace=workspace,
            body=request,
            cases=cases,
            baseline_run_id=baseline_id,
        )
        await db.flush()
        if variant == "A":
            baseline_id = item.id
        runs.append(item)
    # ``variant_order`` controls execution order only.  If A is not created
    # first, wire every treatment to the A record after all ids exist.
    baseline_id = next(item.id for item in runs if item.variant == "A")
    for item in runs:
        if item.variant != "A":
            item.baseline_run_id = baseline_id
    await db.commit()
    for item in runs:
        await db.refresh(item)
        evaluation_task_manager.enqueue(item.id)
    return EvalAblationResponse(
        experiment_id=experiment_id,
        runs=[_run_response(item) for item in runs],
    )


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
async def get_run(run_id: int, db: AsyncSession = Depends(get_db)):
    item = (await db.execute(select(EvalRun).where(EvalRun.id == run_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    results = list((await db.execute(select(EvalResult).where(
        EvalResult.run_id == run_id
    ).order_by(EvalResult.id.asc()))).scalars().all())
    return EvalRunDetailResponse(**_run_response(item).model_dump(), results=[_result_response(row) for row in results])


@router.post("/runs/{run_id}/retry", response_model=EvalRunResponse, status_code=status.HTTP_201_CREATED)
async def retry_run(run_id: int, db: AsyncSession = Depends(get_db)):
    prior = (await db.execute(select(EvalRun).where(EvalRun.id == run_id))).scalar_one_or_none()
    if prior is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    item = EvalRun(
        workspace_id=prior.workspace_id, status="queued", run_type=prior.run_type, case_ids=list(prior.case_ids or []),
        name=f"{prior.name or 'Evaluation'} · retry", experiment_id=prior.experiment_id,
        variant=prior.variant, dataset_name=prior.dataset_name, dataset_version=prior.dataset_version,
        dataset_split=prior.dataset_split,
        config=dict(prior.config or {}), target_config=dict(prior.target_config or {}), baseline_run_id=prior.baseline_run_id,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    evaluation_task_manager.enqueue(item.id)
    return _run_response(item)


@router.get(
    "/runs/{candidate_run_id}/compare/{baseline_run_id}",
    response_model=EvalRunComparisonResponse,
)
async def compare_runs(
    candidate_run_id: int,
    baseline_run_id: int,
    db: AsyncSession = Depends(get_db),
):
    candidate = (await db.execute(
        select(EvalRun).where(EvalRun.id == candidate_run_id)
    )).scalar_one_or_none()
    baseline = (await db.execute(
        select(EvalRun).where(EvalRun.id == baseline_run_id)
    )).scalar_one_or_none()
    if candidate is None or baseline is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    if candidate.workspace_id != baseline.workspace_id:
        raise HTTPException(status_code=422, detail="Runs must belong to the same workspace")
    if candidate.status != "completed" or baseline.status != "completed":
        raise HTTPException(status_code=409, detail="Both runs must be completed")
    candidate_results = list((await db.execute(
        select(EvalResult).where(EvalResult.run_id == candidate.id)
    )).scalars().all())
    baseline_results = list((await db.execute(
        select(EvalResult).where(EvalResult.run_id == baseline.id)
    )).scalars().all())
    case_ids = set(baseline.case_ids or []) & set(candidate.case_ids or [])
    case_rows = list((await db.execute(
        select(EvalCase).where(EvalCase.id.in_(case_ids))
    )).scalars().all()) if case_ids else []
    comparison = compare_run_results(
        baseline_results,
        candidate_results,
        {case.id: case for case in case_rows},
        enforce_latency_budget=candidate.variant == "custom",
        enforce_case_regressions=candidate.variant == "custom",
    )
    compatibility = compare_run_contracts(baseline, candidate)
    comparison["compatibility"] = compatibility
    if not compatibility["valid"]:
        comparison["gate"]["status"] = "fail"
        comparison["gate"]["reasons"].append({
            "type": "incompatible_experiment_contract",
            "reasons": compatibility["reasons"],
        })
    comparison.pop("per_case_deltas", None)
    return EvalRunComparisonResponse(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        **comparison,
    )


@router.patch("/results/{result_id}/review", response_model=EvalResultResponse)
async def review_result(
    result_id: int,
    body: EvalResultReviewWrite,
    db: AsyncSession = Depends(get_db),
):
    item = (await db.execute(
        select(EvalResult).where(EvalResult.id == result_id)
    )).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation result not found")
    item.review_status = body.review_status
    item.reviewer_verdict = body.reviewer_verdict
    item.reviewer_comment = body.reviewer_comment.strip() if body.reviewer_comment else None
    item.failure_types = list(body.failure_types)
    await db.commit()
    await db.refresh(item)
    return _result_response(item)


@router.post("/runs/{run_id}/baseline", response_model=EvalRunResponse)
async def set_baseline(run_id: int, db: AsyncSession = Depends(get_db)):
    item = (await db.execute(select(EvalRun).where(EvalRun.id == run_id))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    rows = list((await db.execute(select(EvalRun).where(EvalRun.workspace_id == item.workspace_id))).scalars().all())
    for row in rows:
        row.is_baseline = False
    item.is_baseline = True
    await db.commit()
    await db.refresh(item)
    return _run_response(item)


@router.post("/workspaces/{workspace_id}/messages/{message_id}/feedback", response_model=EvalFeedbackResponse)
async def save_feedback(workspace_id: int, message_id: str, body: EvalFeedbackWrite, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    item = (await db.execute(select(ChatMessage).where(
        ChatMessage.workspace_id == workspace_id, ChatMessage.message_id == message_id, ChatMessage.role == "assistant"
    ))).scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail="Assistant message not found")
    valid_sources = {str(source.get("index", "")) for source in item.sources or []}
    invalid = set(body.source_ratings) - valid_sources
    if invalid:
        raise HTTPException(status_code=422, detail=f"Unknown source IDs: {', '.join(sorted(invalid))}")
    supplied_review_chunks = "reference_chunk_ids" in body.model_fields_set
    invalid_format = {
        chunk_id for chunk_id in body.reference_chunk_ids if not _CHUNK_ID_RE.fullmatch(chunk_id)
    } if supplied_review_chunks else set()
    if invalid_format:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid reference chunk IDs: {', '.join(sorted(invalid_format))}",
        )
    existing_chunk_ids: set[str] = set()
    if supplied_review_chunks and body.reference_chunk_ids:
        workspace = await _workspace(db, workspace_id)
        rag_service = get_explore_rag_service(db, workspace_id, llm_mode=workspace.llm_mode)
        rows = rag_service.vector_store.get_by_ids(body.reference_chunk_ids)
        existing_chunk_ids = {str(value) for value in rows.get("ids", [])}
    invalid_gold = set(body.reference_chunk_ids) - existing_chunk_ids if supplied_review_chunks else set()
    if invalid_gold:
        raise HTTPException(
            status_code=422,
            detail=f"Reference chunks are not present in the active index: {', '.join(sorted(invalid_gold))}",
        )
    next_corrected_answer = (
        body.corrected_answer.strip() if body.corrected_answer else None
    ) if "corrected_answer" in body.model_fields_set else item.feedback_corrected_answer
    next_reference_ids = (
        list(body.reference_chunk_ids)
        if supplied_review_chunks else list(item.feedback_reference_chunk_ids or [])
    )
    next_failure_types = (
        list(body.failure_types)
        if "failure_types" in body.model_fields_set else list(item.feedback_failure_types or [])
    )
    if body.review_status == "reviewed":
        if not next_corrected_answer:
            raise HTTPException(status_code=422, detail="A reviewed feedback item needs a corrected answer")
        if body.rating == -1 and not next_failure_types:
            raise HTTPException(status_code=422, detail="Negative feedback needs a failure attribution")
        gold_optional = bool({"knowledge_gap", "unanswerable_error"} & set(next_failure_types))
        if not next_reference_ids and not gold_optional:
            raise HTTPException(status_code=422, detail="Reviewed feedback needs at least one Gold chunk")
    item.feedback_rating = body.rating
    item.feedback_comment = body.comment.strip() if body.comment else None
    item.source_ratings = dict(body.source_ratings)
    if "corrected_answer" in body.model_fields_set:
        item.feedback_corrected_answer = body.corrected_answer.strip() if body.corrected_answer else None
    if supplied_review_chunks:
        item.feedback_reference_chunk_ids = list(body.reference_chunk_ids)
    if "failure_types" in body.model_fields_set:
        item.feedback_failure_types = list(body.failure_types)
    if "review_status" in body.model_fields_set:
        item.feedback_review_status = body.review_status
    await db.commit()
    question = None
    if item.reply_to_message_id:
        question = (await db.execute(select(ChatMessage.content).where(
            ChatMessage.workspace_id == workspace_id, ChatMessage.message_id == item.reply_to_message_id
        ))).scalar_one_or_none()
    return EvalFeedbackResponse(
        message_id=item.message_id, reply_to_message_id=item.reply_to_message_id, rating=item.feedback_rating,
        comment=item.feedback_comment, source_ratings=item.source_ratings or {}, question=question,
        corrected_answer=item.feedback_corrected_answer,
        reference_chunk_ids=list(item.feedback_reference_chunk_ids or []),
        failure_types=list(item.feedback_failure_types or []),
        review_status=item.feedback_review_status,
        promoted_case_id=item.feedback_promoted_case_id,
        answer=item.content, created_at=item.created_at,
    )


@router.get("/workspaces/{workspace_id}/feedback", response_model=list[EvalFeedbackResponse])
async def list_feedback(workspace_id: int, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    rows = list((await db.execute(select(ChatMessage).where(
        ChatMessage.workspace_id == workspace_id, ChatMessage.role == "assistant", ChatMessage.feedback_rating.is_not(None)
    ).order_by(ChatMessage.created_at.desc()))).scalars().all())
    questions = {row.message_id: row.content for row in (await db.execute(
        select(ChatMessage).where(ChatMessage.workspace_id == workspace_id)
    )).scalars().all()}
    return [EvalFeedbackResponse(
        message_id=item.message_id, reply_to_message_id=item.reply_to_message_id, rating=item.feedback_rating,
        comment=item.feedback_comment, source_ratings=item.source_ratings or {},
        corrected_answer=item.feedback_corrected_answer,
        reference_chunk_ids=list(item.feedback_reference_chunk_ids or []),
        failure_types=list(item.feedback_failure_types or []),
        review_status=item.feedback_review_status,
        promoted_case_id=item.feedback_promoted_case_id,
        question=questions.get(item.reply_to_message_id or ""), answer=item.content, created_at=item.created_at,
    ) for item in rows]


@router.post("/workspaces/{workspace_id}/messages/{message_id}/promote", response_model=EvalCaseResponse)
async def promote_feedback(workspace_id: int, message_id: str, db: AsyncSession = Depends(get_db)):
    await _workspace(db, workspace_id)
    message = (await db.execute(select(ChatMessage).where(
        ChatMessage.workspace_id == workspace_id, ChatMessage.message_id == message_id,
        ChatMessage.role == "assistant", ChatMessage.feedback_rating.is_not(None)
    ))).scalar_one_or_none()
    if message is None or not message.reply_to_message_id:
        raise HTTPException(status_code=422, detail="Feedback needs an associated user question before promotion")
    if message.feedback_review_status != "reviewed":
        raise HTTPException(status_code=422, detail="Feedback must be manually reviewed before promotion")
    if not (message.feedback_corrected_answer or "").strip():
        raise HTTPException(status_code=422, detail="A reviewed corrected answer is required before promotion")
    question = (await db.execute(select(ChatMessage.content).where(
        ChatMessage.workspace_id == workspace_id, ChatMessage.message_id == message.reply_to_message_id
    ))).scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=422, detail="Original user question is no longer available")
    if message.feedback_promoted_case_id:
        existing_promoted = (await db.execute(select(EvalCase).where(
            EvalCase.id == message.feedback_promoted_case_id
        ))).scalar_one_or_none()
        if existing_promoted:
            return _case_response(existing_promoted)
    source_ids = list(message.feedback_reference_chunk_ids or [])
    reference_answer = message.feedback_corrected_answer
    digest = input_hash(question, reference_answer, source_ids)
    existing = (await db.execute(select(EvalCase).where(
        EvalCase.workspace_id == workspace_id, EvalCase.input_hash == digest
    ))).scalar_one_or_none()
    if existing:
        message.feedback_promoted_case_id = existing.id
        message.feedback_review_status = "promoted"
        await db.commit()
        return _case_response(existing)
    workspace = await _workspace(db, workspace_id)
    rag_service = get_explore_rag_service(db, workspace_id, llm_mode=workspace.llm_mode)
    gold_rows = rag_service.vector_store.get_by_ids(source_ids) if source_ids else {}
    gold_context_by_id = {
        str(chunk_id): str(content)
        for chunk_id, content in zip(
            gold_rows.get("ids", []),
            gold_rows.get("documents", []),
        )
    }
    item = EvalCase(
        workspace_id=workspace_id, status="draft", source="production", question=question,
        dataset_name="production_feedback", dataset_version=1,
        category=(message.feedback_failure_types or ["other"])[0], difficulty="hard",
        expected_behavior=(
            "refuse" if "unanswerable_error" in (message.feedback_failure_types or []) else "answer"
        ), review_status="draft",
        reference_answer=reference_answer, reference_chunk_ids=source_ids,
        reference_contexts=[gold_context_by_id[chunk_id] for chunk_id in source_ids if chunk_id in gold_context_by_id],
        tags=["production_feedback", "positive" if message.feedback_rating == 1 else "negative"],
        extra_metadata={
            "feedback_message_id": message.message_id,
            "comment": message.feedback_comment,
            "failure_types": list(message.feedback_failure_types or []),
        },
        input_hash=digest,
    )
    db.add(item)
    await db.flush()
    message.feedback_promoted_case_id = item.id
    message.feedback_review_status = "promoted"
    await db.commit()
    await db.refresh(item)
    return _case_response(item)
