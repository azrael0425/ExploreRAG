"""Aggregation, paired comparison and deterministic bootstrap intervals."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from app.models.evaluation import EvalCase, EvalResult, EvalRun


QUALITY_REGRESSION_TOLERANCE = 0.02
LATENCY_P95_RELATIVE_BUDGET = 0.25


def compare_run_contracts(baseline: EvalRun, candidate: EvalRun) -> dict[str, Any]:
    """Verify that two runs differ only in intended treatment variables."""
    reasons: list[str] = []
    if list(baseline.case_ids or []) != list(candidate.case_ids or []):
        reasons.append("case_ids_changed")
    baseline_snapshot = (baseline.target_config or {}).get("snapshot", {})
    candidate_snapshot = (candidate.target_config or {}).get("snapshot", {})
    for label, path in (
        ("dataset_changed", ("dataset", "fingerprint")),
        ("corpus_changed", ("corpus", "fingerprint")),
        ("prompt_changed", ("prompt_sha256",)),
        ("models_changed", ("models",)),
    ):
        left: Any = baseline_snapshot
        right: Any = candidate_snapshot
        for key in path:
            left = left.get(key) if isinstance(left, dict) else None
            right = right.get(key) if isinstance(right, dict) else None
        if left != right:
            reasons.append(label)
    for key in ("top_k", "prefetch_k", "retrieval_mode", "warmup_queries", "case_order_seed"):
        if (baseline.config or {}).get(key) != (candidate.config or {}).get(key):
            reasons.append(f"{key}_changed")
    if baseline.run_type != candidate.run_type:
        reasons.append("run_type_changed")
    return {"valid": not reasons, "reasons": reasons}


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 2)


def _rate(rows: list[EvalResult]) -> dict[str, Any]:
    verdicts: dict[str, int] = defaultdict(int)
    for row in rows:
        verdicts[row.verdict] += 1
    return {
        "count": len(rows),
        "pass_rate": round(verdicts.get("pass", 0) / len(rows), 4) if rows else None,
        "verdicts": dict(verdicts),
    }


def build_run_summary(
    results: list[EvalResult],
    cases: dict[int, EvalCase] | None = None,
    run_config: dict[str, Any] | None = None,
    run_type: str | None = None,
) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    performance_values: dict[str, list[float]] = defaultdict(list)
    component_statuses: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    component_applied: dict[str, int] = defaultdict(int)
    component_executed: dict[str, int] = defaultdict(int)

    for result in results:
        for name, status in (result.metric_status or {}).items():
            status_group = "error" if str(status).startswith("error:") else str(status).split(":", 1)[0]
            metric_statuses[name][status_group] += 1
        for name, value in (result.metrics or {}).items():
            if (result.metric_status or {}).get(name) == "ok" and isinstance(value, (int, float)):
                metric_values[name].append(float(value))
        for name, value in (result.performance or {}).items():
            if name == "retrieval_trace":
                continue
            if isinstance(value, (int, float)):
                performance_values[name].append(float(value))
        for component in ("reranker", "knowledge_graph"):
            component_trace = (result.retrieval_trace or {}).get(component, {})
            status = str(component_trace.get("status", "unknown"))
            component_statuses[component][status] += 1
            if component_trace.get("applied") is True:
                component_applied[component] += 1
            if component_trace.get("executed") is True:
                component_executed[component] += 1

    metrics = {
        name: {
            "avg": _mean(metric_values.get(name, [])),
            "evaluated_count": len(metric_values.get(name, [])),
            "skipped_count": metric_statuses[name].get("skipped", 0),
            "error_count": metric_statuses[name].get("error", 0),
        }
        for name in sorted(set(metric_values) | set(metric_statuses))
    }
    performance = {
        name: {
            "avg": _mean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "count": len(values),
        }
        for name, values in sorted(performance_values.items())
    }
    summary = {
        "case_count": len(results),
        **_rate(results),
        "metrics": metrics,
        "performance": performance,
        "component_statuses": {key: dict(value) for key, value in component_statuses.items()},
    }
    judge_complete = True
    if run_type == "full" and cases is not None:
        required = [
            result for result in results
            if getattr(cases.get(result.case_id), "expected_behavior", "answer") != "refuse"
        ]
        judge_metrics: dict[str, Any] = {}
        for name in ("faithfulness", "factual_correctness", "answer_relevancy"):
            ok_count = sum(
                1 for result in required
                if (result.metric_status or {}).get(name) == "ok"
                and isinstance((result.metrics or {}).get(name), (int, float))
            )
            error_count = sum(
                1 for result in required
                if str((result.metric_status or {}).get(name, "")).startswith("error:")
            )
            judge_metrics[name] = {
                "required_count": len(required),
                "evaluated_count": ok_count,
                "error_count": error_count,
                "missing_count": len(required) - ok_count - error_count,
                "coverage": round(ok_count / len(required), 4) if required else 1.0,
            }
        judge_complete = all(
            item["evaluated_count"] == item["required_count"]
            for item in judge_metrics.values()
        )
        summary["judge_coverage"] = {
            "required_case_count": len(required),
            "metrics": judge_metrics,
            "complete": judge_complete,
        }

    if run_config is not None:
        requested = {
            "reranker": bool(run_config.get("enable_reranker")),
            "knowledge_graph": bool(run_config.get("enable_knowledge_graph")),
        }
        component_requirements = {}
        requirements_valid = True
        for component, enabled in requested.items():
            applied_count = component_applied.get(component, 0)
            executed_count = component_executed.get(component, 0)
            if enabled:
                valid = executed_count == len(results) and bool(results)
            else:
                valid = executed_count == 0
            requirements_valid = requirements_valid and valid
            component_requirements[component] = {
                "requested": enabled,
                "applied_count": applied_count,
                "executed_count": executed_count,
                "case_count": len(results),
                "valid": valid,
            }
        summary["component_requirements"] = component_requirements
        summary["experiment_valid"] = requirements_valid and judge_complete
    if cases:
        for dimension in ("category", "source", "difficulty"):
            grouped: dict[str, list[EvalResult]] = defaultdict(list)
            for result in results:
                case = cases.get(result.case_id)
                grouped[str(getattr(case, dimension, "unknown") if case else "unknown")].append(result)
            summary[f"by_{dimension}"] = {
                # Reuse the exact global aggregation rules so subgroup tables
                # expose metric means/coverage and latency percentiles, not
                # only verdict pass rates.  Omitting ``cases`` prevents
                # recursive subgroup construction.
                key: build_run_summary(rows) for key, rows in sorted(grouped.items())
            }
    return summary


def _bootstrap_ci(differences: list[float], *, samples: int = 2000) -> list[float] | None:
    if not differences:
        return None
    if len(differences) == 1:
        value = round(differences[0], 4)
        return [value, value]
    rng = random.Random(20260809)
    means = []
    for _ in range(samples):
        draw = [differences[rng.randrange(len(differences))] for _ in differences]
        means.append(sum(draw) / len(draw))
    means.sort()
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [round(lower, 4), round(upper, 4)]


def compare_run_results(
    baseline: list[EvalResult],
    candidate: list[EvalResult],
    cases: dict[int, EvalCase] | None = None,
    *,
    enforce_latency_budget: bool = True,
    enforce_case_regressions: bool = True,
) -> dict[str, Any]:
    baseline_by_case = {row.case_id: row for row in baseline}
    candidate_by_case = {row.case_id: row for row in candidate}
    case_ids = sorted(set(baseline_by_case) & set(candidate_by_case))
    metric_names = sorted({
        name
        for case_id in case_ids
        for row in (baseline_by_case[case_id], candidate_by_case[case_id])
        for name in (row.metrics or {})
    })
    metric_deltas: dict[str, Any] = {}
    for name in metric_names:
        pairs: list[tuple[float, float]] = []
        for case_id in case_ids:
            base = baseline_by_case[case_id]
            cand = candidate_by_case[case_id]
            base_value = (base.metrics or {}).get(name)
            cand_value = (cand.metrics or {}).get(name)
            if (
                (base.metric_status or {}).get(name) == "ok"
                and (cand.metric_status or {}).get(name) == "ok"
                and isinstance(base_value, (int, float))
                and isinstance(cand_value, (int, float))
            ):
                pairs.append((float(base_value), float(cand_value)))
        if not pairs:
            continue
        base_values = [pair[0] for pair in pairs]
        cand_values = [pair[1] for pair in pairs]
        differences = [cand - base for base, cand in pairs]
        base_mean = sum(base_values) / len(base_values)
        delta = sum(differences) / len(differences)
        metric_deltas[name] = {
            "baseline": round(base_mean, 4),
            "candidate": round(sum(cand_values) / len(cand_values), 4),
            "delta": round(delta, 4),
            "relative_delta": round(delta / base_mean, 4) if base_mean else None,
            "ci95": _bootstrap_ci(differences),
            "paired_count": len(pairs),
        }

    performance_names = sorted({
        name
        for case_id in case_ids
        for row in (baseline_by_case[case_id], candidate_by_case[case_id])
        for name, value in (row.performance or {}).items()
        if name != "retrieval_trace" and isinstance(value, (int, float))
    })
    latency_deltas: dict[str, Any] = {}
    for name in performance_names:
        pairs = []
        for case_id in case_ids:
            base_value = (baseline_by_case[case_id].performance or {}).get(name)
            cand_value = (candidate_by_case[case_id].performance or {}).get(name)
            if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
                pairs.append((float(base_value), float(cand_value)))
        if pairs:
            differences = [cand - base for base, cand in pairs]
            base_values = [base for base, _ in pairs]
            candidate_values = [cand for _, cand in pairs]
            baseline_p50 = _percentile(base_values, 0.50)
            candidate_p50 = _percentile(candidate_values, 0.50)
            baseline_p95 = _percentile(base_values, 0.95)
            candidate_p95 = _percentile(candidate_values, 0.95)
            latency_deltas[name] = {
                "baseline_p50": baseline_p50,
                "candidate_p50": candidate_p50,
                "delta_p50": (
                    round(candidate_p50 - baseline_p50, 2)
                    if baseline_p50 is not None and candidate_p50 is not None else None
                ),
                "baseline_p95": baseline_p95,
                "candidate_p95": candidate_p95,
                "delta_p95": (
                    round(candidate_p95 - baseline_p95, 2)
                    if baseline_p95 is not None and candidate_p95 is not None else None
                ),
                "delta_avg": _mean(differences),
                "ci95": _bootstrap_ci(differences),
                "paired_count": len(pairs),
            }

    regressions = []
    pass_to_fail_count = 0
    per_case_deltas: dict[int, dict[str, float]] = {}
    for case_id in case_ids:
        base = baseline_by_case[case_id]
        cand = candidate_by_case[case_id]
        case_delta = {}
        for name in metric_names:
            base_value = (base.metrics or {}).get(name)
            cand_value = (cand.metrics or {}).get(name)
            if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
                case_delta[name] = round(float(cand_value) - float(base_value), 4)
        per_case_deltas[case_id] = case_delta
        material_drops = {name: value for name, value in case_delta.items() if value <= -0.05}
        if (base.verdict == "pass" and cand.verdict != "pass") or material_drops:
            regressions.append({
                "case_id": case_id,
                "question": cand.question,
                "baseline_verdict": base.verdict,
                "candidate_verdict": cand.verdict,
                "metric_drops": material_drops,
            })
        if base.verdict == "pass" and cand.verdict != "pass":
            pass_to_fail_count += 1

    gate_reasons: list[dict[str, Any]] = []
    for name, delta in metric_deltas.items():
        ci95 = delta.get("ci95")
        if (
            delta["delta"] < -QUALITY_REGRESSION_TOLERANCE
            and ci95
            and ci95[1] < 0
        ):
            gate_reasons.append({
                "type": "quality_regression",
                "metric": name,
                "delta": delta["delta"],
                "ci95": ci95,
            })
    total_latency = latency_deltas.get("total_ms") or latency_deltas.get("runner_wall_ms")
    if total_latency and total_latency.get("baseline_p95"):
        relative_p95 = total_latency["delta_p95"] / total_latency["baseline_p95"]
        total_latency["relative_delta_p95"] = round(relative_p95, 4)
        total_latency["budget_exceeded"] = relative_p95 > LATENCY_P95_RELATIVE_BUDGET
        if enforce_latency_budget and total_latency["budget_exceeded"]:
            gate_reasons.append({
                "type": "latency_budget_exceeded",
                "metric": "total_ms" if "total_ms" in latency_deltas else "runner_wall_ms",
                "relative_delta_p95": round(relative_p95, 4),
                "budget": LATENCY_P95_RELATIVE_BUDGET,
            })
    if enforce_case_regressions and pass_to_fail_count:
        gate_reasons.append({
            "type": "pass_to_fail_regression",
            "count": pass_to_fail_count,
        })

    comparison = {
        "paired_case_count": len(case_ids),
        "metric_deltas": metric_deltas,
        "latency_deltas": latency_deltas,
        "regressions": regressions,
        "gate": {
            "status": "fail" if gate_reasons else "pass",
            "reasons": gate_reasons,
            "quality_tolerance": QUALITY_REGRESSION_TOLERANCE,
            "latency_p95_relative_budget": LATENCY_P95_RELATIVE_BUDGET,
            "latency_budget_enforced": enforce_latency_budget,
            "case_regressions_enforced": enforce_case_regressions,
        },
        "per_case_deltas": per_case_deltas,
    }
    if cases:
        subgroup_comparisons: dict[str, dict[str, Any]] = {}
        for dimension in ("category", "difficulty"):
            values = sorted({
                str(getattr(cases[case_id], dimension, "unknown"))
                for case_id in case_ids
                if case_id in cases
            })
            dimension_comparisons: dict[str, Any] = {}
            for value in values:
                subgroup_ids = {
                    case_id for case_id in case_ids
                    if case_id in cases
                    and str(getattr(cases[case_id], dimension, "unknown")) == value
                }
                subgroup = compare_run_results(
                    [baseline_by_case[case_id] for case_id in sorted(subgroup_ids)],
                    [candidate_by_case[case_id] for case_id in sorted(subgroup_ids)],
                    enforce_latency_budget=enforce_latency_budget,
                    enforce_case_regressions=enforce_case_regressions,
                )
                subgroup.pop("per_case_deltas", None)
                dimension_comparisons[value] = subgroup
            subgroup_comparisons[dimension] = dimension_comparisons
        comparison["subgroups"] = subgroup_comparisons
    return comparison
