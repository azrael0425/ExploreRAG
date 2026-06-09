"""Execute evaluation runs and persist immutable per-case experiment results."""
from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.evaluation.metrics import (
    calculate_fast_metrics,
    failure_types_for,
    metric_details_for,
    verdict_for,
)
from app.evaluation.ragas_adapter import evaluate_with_ragas
from app.evaluation.reporting import build_run_summary, compare_run_contracts, compare_run_results
from app.evaluation.target import run_production_target, run_retrieval_target
from app.models.evaluation import EvalCase, EvalResult, EvalRun
from app.models.knowledge_base import KnowledgeBase
from app.services.reranker import get_reranker_service


def _graph_metrics(
    case: EvalCase,
    trace: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, str]]:
    expected = {
        str(value).strip().casefold()
        for value in case.reference_entity_names or []
        if str(value).strip()
    }
    graph = trace.get("knowledge_graph", {})
    actual = {
        str(value).strip().casefold()
        for value in graph.get("entity_names", [])
        if str(value).strip()
    }
    metrics: dict[str, float | None] = {}
    statuses: dict[str, str] = {}
    if expected:
        metrics["graph_evidence_recall"] = round(len(expected & actual) / len(expected), 4)
        statuses["graph_evidence_recall"] = "ok"
    else:
        metrics["graph_evidence_recall"] = None
        statuses["graph_evidence_recall"] = "skipped:no_reference_entities"
    expected_relationships = []
    for relation in case.reference_relationships or []:
        if not isinstance(relation, dict):
            continue
        source = str(relation.get("source", "")).strip().casefold()
        target = str(relation.get("target", "")).strip().casefold()
        if source and target:
            expected_relationships.append(frozenset((source, target)))
    actual_fact_entities = [
        {
            str(value).strip().casefold()
            for value in fact_entities
            if str(value).strip()
        }
        for fact_entities in graph.get("fact_entity_names", [])
        if isinstance(fact_entities, list)
    ]
    if expected_relationships:
        matched = sum(
            1 for pair in expected_relationships
            if any(pair.issubset(fact_entities) for fact_entities in actual_fact_entities)
        )
        metrics["graph_relationship_recall"] = round(
            matched / len(expected_relationships), 4
        )
        statuses["graph_relationship_recall"] = "ok"
    else:
        metrics["graph_relationship_recall"] = None
        statuses["graph_relationship_recall"] = "skipped:no_reference_relationships"
    fact_count = int(graph.get("fact_count", 0) or 0)
    if fact_count:
        metrics["graph_traceability"] = round(
            int(graph.get("traceable_fact_count", 0) or 0) / fact_count,
            4,
        )
        statuses["graph_traceability"] = "ok"
    else:
        metrics["graph_traceability"] = None
        statuses["graph_traceability"] = "skipped:no_graph_facts"
    return metrics, statuses


def _required_metrics(case: EvalCase, run_type: str) -> list[str]:
    if run_type == "retrieval":
        return ["hit_at_4", "recall_at_4"] if case.reference_chunk_ids else []
    if case.expected_behavior == "refuse":
        return ["refusal_accuracy", "citation_validity"]
    required = ["hit_at_4", "citation_validity"] if case.reference_chunk_ids else ["citation_validity"]
    if run_type == "full":
        required.extend(["faithfulness", "factual_correctness"])
    return required


async def _apply_baseline_comparison(
    db: AsyncSession,
    run: EvalRun,
    completed: list[EvalResult],
) -> dict[str, Any] | None:
    if not run.baseline_run_id:
        return None
    baseline = (await db.execute(
        select(EvalRun).where(EvalRun.id == run.baseline_run_id)
    )).scalar_one_or_none()
    if baseline is None or baseline.status != "completed":
        return {"status": "skipped", "reason": "baseline_not_completed"}
    baseline_results = list((await db.execute(
        select(EvalResult).where(EvalResult.run_id == baseline.id)
    )).scalars().all())
    case_ids = set(run.case_ids or [])
    case_rows = list((await db.execute(
        select(EvalCase).where(EvalCase.id.in_(case_ids))
    )).scalars().all()) if case_ids else []
    comparison = compare_run_results(
        baseline_results,
        completed,
        {case.id: case for case in case_rows},
        # A/B/C/D intentionally add components; their latency delta is a
        # reported trade-off, not a same-contract deployment regression.
        enforce_latency_budget=run.variant == "custom",
        enforce_case_regressions=run.variant == "custom",
    )
    compatibility = compare_run_contracts(baseline, run)
    comparison["compatibility"] = compatibility
    if not compatibility["valid"]:
        comparison["gate"]["status"] = "fail"
        comparison["gate"]["reasons"].append({
            "type": "incompatible_experiment_contract",
            "reasons": compatibility["reasons"],
        })
    per_case = comparison.pop("per_case_deltas", {})
    for item in completed:
        item.baseline_delta = per_case.get(item.case_id, {})
        deltas = item.baseline_delta or {}
        failures = list(item.failure_types or [])
        config = run.config or {}
        if config.get("enable_knowledge_graph") and any(
            value <= -0.05 for value in deltas.values() if isinstance(value, (int, float))
        ):
            failures.append("graph_noise")
        if (
            config.get("enable_reranker")
            and isinstance(deltas.get("ndcg_at_4"), (int, float))
            and deltas["ndcg_at_4"] <= -0.05
        ):
            failures.append("rerank_error")
        item.failure_types = list(dict.fromkeys(failures))
    await db.commit()
    return {"status": "ok", **comparison}


async def execute_run(db: AsyncSession, run_id: int) -> None:
    run = (await db.execute(select(EvalRun).where(EvalRun.id == run_id))).scalar_one_or_none()
    if run is None:
        return
    workspace = (await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == run.workspace_id)
    )).scalar_one()
    run.status = "running"
    run.error_message = None
    run.started_at = datetime.utcnow()
    await db.commit()

    case_rows = list((await db.execute(
        select(EvalCase).where(
            EvalCase.workspace_id == run.workspace_id,
            EvalCase.id.in_(run.case_ids or []),
        )
    )).scalars().all())
    cases = {case.id: case for case in case_rows}
    ordered_cases = [cases[case_id] for case_id in run.case_ids if case_id in cases]
    config = run.config or {}
    case_order_seed = int(config.get("case_order_seed", 0) or 0)
    if case_order_seed:
        random.Random(case_order_seed).shuffle(ordered_cases)
    completed: list[EvalResult] = []
    preflight: dict[str, Any] = {}
    try:
        if config.get("enable_reranker"):
            warmup_started = time.perf_counter()
            await asyncio.to_thread(get_reranker_service().warmup)
            preflight["reranker_warmup_ms"] = int(
                (time.perf_counter() - warmup_started) * 1000
            )
        warmup_queries = int(config.get("warmup_queries", 0) or 0)
        if warmup_queries and ordered_cases:
            warmup_case = ordered_cases[0]
            warmup_observations = []
            target = run_retrieval_target if run.run_type == "retrieval" else run_production_target
            for index in range(warmup_queries):
                warmup_started = time.perf_counter()
                warmup_result = await target(
                    db,
                    workspace_id=run.workspace_id,
                    question=warmup_case.question,
                    top_k=int(config.get("top_k", 4)),
                    retrieval_mode=str(config.get("retrieval_mode", "hybrid")),
                    enable_reranker=config.get("enable_reranker"),
                    enable_knowledge_graph=config.get("enable_knowledge_graph"),
                    prefetch_k=int(config.get("prefetch_k", 20)),
                    **({"history": list(warmup_case.conversation_history or [])} if run.run_type != "retrieval" else {}),
                )
                observation = dict(warmup_result.get("performance") or {})
                observation.pop("retrieval_trace", None)
                observation["runner_wall_ms"] = int(
                    (time.perf_counter() - warmup_started) * 1000
                )
                observation["iteration"] = index + 1
                warmup_observations.append(observation)
            preflight["warmup_case_id"] = warmup_case.id
            preflight["warmup_observations"] = warmup_observations
        for case in ordered_cases:
            wall_started = time.perf_counter()
            try:
                target = run_retrieval_target if run.run_type == "retrieval" else run_production_target
                final = await target(
                    db,
                    workspace_id=run.workspace_id,
                    question=case.question,
                    top_k=int(config.get("top_k", 4)),
                    retrieval_mode=str(config.get("retrieval_mode", "hybrid")),
                    enable_reranker=config.get("enable_reranker"),
                    enable_knowledge_graph=config.get("enable_knowledge_graph"),
                    prefetch_k=int(config.get("prefetch_k", 20)),
                    **({"history": list(case.conversation_history or [])} if run.run_type != "retrieval" else {}),
                )
                sources = [dict(item) for item in final.get("sources", [])]
                contexts = [str(item.get("content", "")) for item in sources if item.get("content")]
                performance = dict(final.get("performance") or {})
                performance["runner_wall_ms"] = int((time.perf_counter() - wall_started) * 1000)
                trace = dict(
                    final.get("retrieval_trace")
                    or performance.get("retrieval_trace")
                    or {}
                )
                performance.pop("retrieval_trace", None)
                answer = str(final.get("answer", ""))
                metrics, statuses = calculate_fast_metrics(
                    answer=answer,
                    sources=sources,
                    reference_chunk_ids=list(case.reference_chunk_ids or []),
                    reference_answer=case.reference_answer,
                    retrieval_trace=trace,
                    expected_behavior=case.expected_behavior,
                    rubric=dict(case.extra_metadata or {}),
                    include_answer_metrics=run.run_type != "retrieval",
                )
                graph_values, graph_status = _graph_metrics(case, trace)
                metrics.update(graph_values)
                statuses.update(graph_status)
                if run.run_type == "full" and case.expected_behavior != "refuse":
                    ragas_values, ragas_status = await evaluate_with_ragas(
                        question=case.question,
                        answer=answer,
                        retrieved_contexts=contexts,
                        reference_answer=case.reference_answer,
                        llm_mode=workspace.llm_mode,
                    )
                    metrics.update(ragas_values)
                    statuses.update(ragas_status)
                failures = failure_types_for(metrics, statuses)
                graph_status_value = str(trace.get("knowledge_graph", {}).get("status", ""))
                if case.reference_entity_names and graph_status_value not in {"ok", "empty"}:
                    failures.append("graph_miss")
                item = EvalResult(
                    run_id=run.id,
                    case_id=case.id,
                    question=case.question,
                    reference_answer=case.reference_answer,
                    reference_chunk_ids=list(case.reference_chunk_ids or []),
                    retrieved_contexts=contexts,
                    answer=answer or None,
                    sources=sources,
                    performance=performance,
                    retrieval_trace=trace,
                    metrics=metrics,
                    metric_status=statuses,
                    metric_details=metric_details_for(metrics, statuses),
                    failure_types=list(dict.fromkeys(failures)),
                    verdict=verdict_for(
                        metrics,
                        statuses,
                        required_metrics=_required_metrics(case, run.run_type),
                    ),
                )
            except Exception as exc:
                item = EvalResult(
                    run_id=run.id,
                    case_id=case.id,
                    question=case.question,
                    reference_answer=case.reference_answer,
                    reference_chunk_ids=list(case.reference_chunk_ids or []),
                    retrieved_contexts=[],
                    sources=[],
                    performance={"runner_wall_ms": int((time.perf_counter() - wall_started) * 1000)},
                    retrieval_trace={},
                    metrics={},
                    metric_status={},
                    metric_details={},
                    failure_types=[],
                    verdict="error",
                    error_message=str(exc)[:4000],
                )
            db.add(item)
            await db.commit()
            await db.refresh(item)
            completed.append(item)

        summary = build_run_summary(completed, cases, config, run.run_type)
        summary["preflight"] = preflight
        summary["measurement_protocol"] = {
            "warmup_queries_excluded": int(config.get("warmup_queries", 0) or 0),
            "case_order_seed": case_order_seed,
            "latency_population": "steady_state" if warmup_queries else "mixed_cold_and_warm",
        }
        comparison = await _apply_baseline_comparison(db, run, completed)
        if comparison is not None:
            if not summary.get("experiment_valid", True) and comparison.get("gate"):
                comparison["gate"]["status"] = "fail"
                comparison["gate"]["reasons"].append({
                    "type": "requested_component_not_applied",
                })
            summary["baseline_comparison"] = comparison
        run.metrics_summary = summary
        run.status = "completed"
        run.finished_at = datetime.utcnow()
        await db.commit()
    except Exception as exc:
        await db.rollback()
        run = (await db.execute(select(EvalRun).where(EvalRun.id == run_id))).scalar_one_or_none()
        if run is not None:
            run.status = "failed"
            run.error_message = str(exc)[:4000]
            run.finished_at = datetime.utcnow()
            await db.commit()
        raise
