from types import SimpleNamespace

from app.evaluation.reporting import build_run_summary, compare_run_results


def _result(
    case_id: int,
    *,
    ndcg: float,
    total_ms: float,
    verdict: str = "pass",
):
    return SimpleNamespace(
        case_id=case_id,
        question=f"question {case_id}",
        verdict=verdict,
        metrics={"ndcg_at_4": ndcg},
        metric_status={"ndcg_at_4": "ok"},
        performance={"total_ms": total_ms},
        retrieval_trace={
            "reranker": {"status": "ok", "executed": True, "applied": True},
            "knowledge_graph": {"status": "disabled_by_request", "executed": False, "applied": False},
        },
    )


def test_summary_reports_metric_coverage_and_latency_percentiles() -> None:
    rows = [
        _result(1, ndcg=0.5, total_ms=100),
        _result(2, ndcg=1.0, total_ms=200),
    ]

    summary = build_run_summary(
        rows,
        run_config={"enable_reranker": True, "enable_knowledge_graph": False},
    )

    assert summary["metrics"]["ndcg_at_4"]["avg"] == 0.75
    assert summary["metrics"]["ndcg_at_4"]["evaluated_count"] == 2
    assert summary["performance"]["total_ms"]["p50"] == 150.0
    assert summary["performance"]["total_ms"]["p95"] == 195.0
    assert summary["component_statuses"]["reranker"] == {"ok": 2}
    assert summary["experiment_valid"] is True


def test_summary_reports_full_category_metric_subgroups() -> None:
    rows = [
        _result(1, ndcg=0.5, total_ms=100),
        _result(2, ndcg=1.0, total_ms=200),
    ]
    cases = {
        1: SimpleNamespace(category="multi_hop", source="manual", difficulty="hard"),
        2: SimpleNamespace(category="single_hop", source="manual", difficulty="easy"),
    }

    summary = build_run_summary(rows, cases=cases)

    assert summary["by_category"]["multi_hop"]["metrics"]["ndcg_at_4"]["avg"] == 0.5
    assert summary["by_category"]["single_hop"]["performance"]["total_ms"]["p95"] == 200


def test_full_summary_requires_complete_judge_coverage() -> None:
    answer = _result(1, ndcg=0.5, total_ms=100)
    answer.metrics.update({"faithfulness": 0.9, "factual_correctness": 0.8})
    answer.metric_status.update({"faithfulness": "ok", "factual_correctness": "ok"})
    refusal = _result(2, ndcg=0.5, total_ms=100)
    cases = {
        1: SimpleNamespace(
            category="single_hop", source="manual", difficulty="easy", expected_behavior="answer"
        ),
        2: SimpleNamespace(
            category="unanswerable", source="manual", difficulty="easy", expected_behavior="refuse"
        ),
    }

    summary = build_run_summary(
        [answer, refusal],
        cases=cases,
        run_config={"enable_reranker": True, "enable_knowledge_graph": False},
        run_type="full",
    )

    assert summary["judge_coverage"]["required_case_count"] == 1
    assert summary["judge_coverage"]["metrics"]["answer_relevancy"]["missing_count"] == 1
    assert summary["judge_coverage"]["complete"] is False
    assert summary["experiment_valid"] is False


def test_paired_comparison_returns_ci_and_regression_cases() -> None:
    baseline = [
        _result(1, ndcg=0.8, total_ms=100),
        _result(2, ndcg=0.9, total_ms=110),
    ]
    candidate = [
        _result(1, ndcg=0.9, total_ms=130),
        _result(2, ndcg=0.7, total_ms=150, verdict="fail"),
    ]

    comparison = compare_run_results(baseline, candidate)

    delta = comparison["metric_deltas"]["ndcg_at_4"]
    assert delta["paired_count"] == 2
    assert delta["delta"] == -0.05
    assert len(delta["ci95"]) == 2
    assert comparison["latency_deltas"]["total_ms"]["delta_avg"] == 35.0
    assert [item["case_id"] for item in comparison["regressions"]] == [2]
    assert comparison["gate"]["status"] == "fail"
    assert {item["type"] for item in comparison["gate"]["reasons"]} == {
        "pass_to_fail_regression",
        "latency_budget_exceeded",
    }


def test_paired_comparison_reports_category_subgroups() -> None:
    baseline = [_result(1, ndcg=0.2, total_ms=100), _result(2, ndcg=0.8, total_ms=100)]
    candidate = [_result(1, ndcg=0.6, total_ms=120), _result(2, ndcg=0.8, total_ms=120)]
    cases = {
        1: SimpleNamespace(category="multi_hop", difficulty="hard"),
        2: SimpleNamespace(category="single_hop", difficulty="easy"),
    }

    comparison = compare_run_results(baseline, candidate, cases)

    assert comparison["subgroups"]["category"]["multi_hop"]["metric_deltas"]["ndcg_at_4"]["delta"] == 0.4
    assert comparison["subgroups"]["category"]["single_hop"]["metric_deltas"]["ndcg_at_4"]["delta"] == 0.0


def test_ablation_can_report_latency_tradeoff_without_failing_quality_gate() -> None:
    baseline = [_result(1, ndcg=0.5, total_ms=100)]
    candidate = [_result(1, ndcg=0.6, total_ms=500)]

    comparison = compare_run_results(
        baseline,
        candidate,
        enforce_latency_budget=False,
    )

    assert comparison["latency_deltas"]["total_ms"]["budget_exceeded"] is True
    assert comparison["gate"]["latency_budget_enforced"] is False
    assert comparison["gate"]["status"] == "pass"


def test_ablation_keeps_case_regressions_informational() -> None:
    baseline = [_result(1, ndcg=1.0, total_ms=100, verdict="pass")]
    candidate = [_result(1, ndcg=1.0, total_ms=100, verdict="fail")]

    comparison = compare_run_results(
        baseline,
        candidate,
        enforce_case_regressions=False,
    )

    assert len(comparison["regressions"]) == 1
    assert comparison["gate"]["case_regressions_enforced"] is False
    assert comparison["gate"]["status"] == "pass"
