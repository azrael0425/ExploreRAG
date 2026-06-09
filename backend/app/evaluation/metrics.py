"""Deterministic retrieval, citation and answer metrics for RAG experiments."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

_CITATION_RE = re.compile(r"\[((?:(?:KB|ATT|KG)-)?[a-z0-9]{4})\]", re.IGNORECASE)
_WORDS = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)
_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"无法.{0,12}(回答|确定|从.{0,8}(资料|文档))",
        r"没有.{0,12}(相关|足够).{0,8}(信息|资料|上下文)",
        r"知识库.{0,12}(未|没有|不包含)",
        r"(?:cannot|can't|unable to).{0,20}(answer|determine|find)",
        r"(?:not enough|no relevant).{0,12}(information|context)",
        r"the (?:provided )?(?:documents|context).{0,12}(do not|does not|don't)",
    )
]

METRIC_THRESHOLDS: dict[str, float] = {
    "hit_at_4": 1.0,
    "recall_at_4": 0.5,
    "citation_validity": 1.0,
    "citation_precision": 0.8,
    "citation_recall": 0.5,
    "refusal_accuracy": 1.0,
    "keyword_coverage": 0.8,
    "faithfulness": 0.7,
    "factual_correctness": 0.7,
    "answer_relevancy": 0.7,
}

_DETERMINISTIC_METRICS = {
    "citation_validity",
    "citation_precision",
    "citation_recall",
    "context_recall",
    "answer_token_recall",
    "hit_at_4",
    "recall_at_4",
    "refusal_accuracy",
    "keyword_coverage",
}
_RAGAS_METRICS = {"faithfulness", "factual_correctness", "answer_relevancy"}


def _citation_labels(answer: str) -> list[str]:
    return [match.group(1).lower() for match in _CITATION_RE.finditer(answer or "")]


def _citation_ids(answer: str) -> set[str]:
    return {label.split("-", 1)[-1] for label in _citation_labels(answer)}


def _source_ids(sources: Iterable[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for source in sources:
        raw = str(source.get("index", "")).lower()
        if raw:
            ids.add(raw.split("-", 1)[-1])
    return ids


def _tokens(text: str) -> set[str]:
    text = re.sub(r"(?i)(\d+(?:\.\d+)?)\s*b(?:illion)?\b", r"\1 billion", text or "")
    return {item.lower() for item in _WORDS.findall(text or "") if len(item) > 1}


def _ratio(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _ranked_chunk_ids(candidates: Iterable[dict[str, Any]], limit: int | None = None) -> list[str]:
    values = [str(item.get("chunk_id", "")) for item in candidates if item.get("chunk_id")]
    return values[:limit] if limit is not None else values


def _ranking_metrics(ranked_ids: list[str], expected_ids: set[str], k: int) -> dict[str, float | None]:
    if not expected_ids:
        return {f"hit_at_{k}": None, f"recall_at_{k}": None, f"mrr_at_{k}": None, f"ndcg_at_{k}": None}
    top = ranked_ids[:k]
    relevant = [1 if chunk_id in expected_ids else 0 for chunk_id in top]
    hits = sum(relevant)
    first_rank = next((index for index, value in enumerate(relevant, start=1) if value), None)
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevant, start=1))
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        f"hit_at_{k}": 1.0 if hits else 0.0,
        f"recall_at_{k}": round(hits / len(expected_ids), 4),
        f"mrr_at_{k}": round(1.0 / first_rank, 4) if first_rank else 0.0,
        f"ndcg_at_{k}": round(dcg / idcg, 4) if idcg else None,
    }


def calculate_retrieval_metrics(
    *,
    retrieval_trace: dict[str, Any],
    reference_chunk_ids: list[str] | None,
    final_sources: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float | None], dict[str, str]]:
    expected_ids = {str(item) for item in reference_chunk_ids or [] if str(item)}
    pre = _ranked_chunk_ids(retrieval_trace.get("pre_rerank_candidates", []), 20)
    final = _ranked_chunk_ids(retrieval_trace.get("final_candidates", []))
    if not final:
        final = [str(item.get("chunk_id", "")) for item in final_sources or [] if item.get("chunk_id")]
    metrics: dict[str, float | None] = {}
    metrics.update(_ranking_metrics(final, expected_ids, 4))
    recall20 = _ranking_metrics(pre or final, expected_ids, 20)["recall_at_20"]
    metrics["recall_at_20"] = recall20
    status = {
        name: "ok" if expected_ids else "skipped:no_reference_chunks"
        for name in metrics
    }
    return metrics, status


def _is_refusal(answer: str) -> bool:
    normalized = (answer or "").strip()
    return bool(normalized and any(pattern.search(normalized) for pattern in _REFUSAL_PATTERNS))


def calculate_fast_metrics(
    *,
    answer: str,
    sources: list[dict[str, Any]],
    reference_chunk_ids: list[str] | None,
    reference_answer: str | None,
    retrieval_trace: dict[str, Any] | None = None,
    expected_behavior: str | None = None,
    rubric: dict[str, Any] | None = None,
    include_answer_metrics: bool = True,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Return metric values plus explicit ok/skipped statuses."""
    cited = _citation_ids(answer)
    available = _source_ids(sources)
    metrics: dict[str, float | None] = {}
    statuses: dict[str, str] = {}

    if cited:
        metrics["citation_validity"] = _ratio(len(cited & available), len(cited))
        statuses["citation_validity"] = "ok"
    elif include_answer_metrics and expected_behavior == "answer" and sources:
        metrics["citation_validity"] = 0.0
        statuses["citation_validity"] = "ok"
    elif include_answer_metrics and expected_behavior == "refuse":
        metrics["citation_validity"] = 1.0 if not cited else 0.0
        statuses["citation_validity"] = "ok"
    else:
        metrics["citation_validity"] = None
        statuses["citation_validity"] = "skipped:no_citations"

    expected_ids = {str(item) for item in reference_chunk_ids or [] if str(item)}
    retrieved_ids = {str(source.get("chunk_id", "")) for source in sources}
    if expected_ids:
        metrics["context_recall"] = _ratio(len(expected_ids & retrieved_ids), len(expected_ids))
        statuses["context_recall"] = "ok"
    else:
        metrics["context_recall"] = None
        statuses["context_recall"] = "skipped:no_reference_chunks"

    if reference_answer and include_answer_metrics:
        expected_tokens = _tokens(reference_answer)
        actual_tokens = _tokens(answer)
        metrics["answer_token_recall"] = _ratio(len(expected_tokens & actual_tokens), len(expected_tokens))
        statuses["answer_token_recall"] = "ok"
    else:
        metrics["answer_token_recall"] = None
        statuses["answer_token_recall"] = "skipped:no_reference_answer"

    if retrieval_trace is not None:
        retrieval_values, retrieval_status = calculate_retrieval_metrics(
            retrieval_trace=retrieval_trace,
            reference_chunk_ids=reference_chunk_ids,
            final_sources=sources,
        )
        metrics.update(retrieval_values)
        statuses.update(retrieval_status)

    # Keep the original fast-metric contract for legacy callers.  New
    # evaluation runs always pass ``expected_behavior`` and therefore receive
    # the stricter citation-to-gold evidence metrics.
    if include_answer_metrics and expected_behavior is not None and expected_ids:
        source_by_id: dict[str, str] = {}
        for source in sources:
            # Gold Chunk ids describe vector evidence.  KG facts use stable
            # pseudo ids and are evaluated separately by graph traceability,
            # entity recall and relationship recall; counting them as vector
            # false positives would mechanically penalize every KG citation.
            if str(source.get("source_type", "vector")) == "kg":
                continue
            label = str(source.get("index", "")).lower()
            chunk_id = str(source.get("chunk_id", ""))
            if label and chunk_id:
                source_by_id[label] = chunk_id
                source_by_id[label.split("-", 1)[-1]] = chunk_id
        cited_chunks = {
            source_by_id[label]
            for label in _citation_labels(answer)
            if label in source_by_id
        }
        metrics["citation_precision"] = _ratio(len(cited_chunks & expected_ids), len(cited_chunks))
        metrics["citation_recall"] = _ratio(len(cited_chunks & expected_ids), len(expected_ids))
        statuses["citation_precision"] = "ok" if cited_chunks else "skipped:no_valid_citations"
        statuses["citation_recall"] = "ok" if cited_chunks else "skipped:no_valid_citations"

    if include_answer_metrics and expected_behavior in {"answer", "refuse"}:
        refusal = _is_refusal(answer)
        metrics["refusal_accuracy"] = float(
            refusal if expected_behavior == "refuse" else not refusal
        )
        statuses["refusal_accuracy"] = "ok"

    expected_keywords = [str(item).lower() for item in (rubric or {}).get("expected_keywords", []) if str(item)]
    if include_answer_metrics and expected_keywords:
        normalized_answer = (answer or "").lower()
        metrics["keyword_coverage"] = _ratio(
            sum(1 for keyword in expected_keywords if keyword in normalized_answer),
            len(expected_keywords),
        )
        statuses["keyword_coverage"] = "ok"

    return metrics, statuses


def metric_details_for(
    metrics: dict[str, float | None], statuses: dict[str, str]
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "value": value,
            "status": statuses.get(name, "unknown"),
            "threshold": METRIC_THRESHOLDS.get(name),
            "passed": (
                value >= METRIC_THRESHOLDS[name]
                if value is not None and statuses.get(name) == "ok" and name in METRIC_THRESHOLDS
                else None
            ),
            "evaluator": "ragas" if name in _RAGAS_METRICS else "deterministic",
        }
        for name, value in metrics.items()
    }


def failure_types_for(
    metrics: dict[str, float | None], statuses: dict[str, str]
) -> list[str]:
    failures: list[str] = []
    if statuses.get("recall_at_20") == "ok" and (metrics.get("recall_at_20") or 0.0) < 1.0:
        failures.append("retrieval_miss")
    if (
        statuses.get("recall_at_20") == "ok"
        and (metrics.get("recall_at_20") or 0.0) > 0
        and statuses.get("hit_at_4") == "ok"
        and (metrics.get("hit_at_4") or 0.0) == 0
    ):
        failures.append("rerank_error")
    if statuses.get("faithfulness") == "ok" and (metrics.get("faithfulness") or 0.0) < METRIC_THRESHOLDS["faithfulness"]:
        failures.append("generation_error")
    if statuses.get("factual_correctness") == "ok" and (metrics.get("factual_correctness") or 0.0) < METRIC_THRESHOLDS["factual_correctness"]:
        failures.append("generation_error")
    if any(
        statuses.get(name) == "ok" and (metrics.get(name) or 0.0) < METRIC_THRESHOLDS[name]
        for name in ("citation_validity", "citation_precision", "citation_recall")
        if name in metrics
    ):
        failures.append("citation_error")
    if statuses.get("refusal_accuracy") == "ok" and metrics.get("refusal_accuracy") == 0:
        failures.append("unanswerable_error")
    return list(dict.fromkeys(failures))


def verdict_for(
    metrics: dict[str, float | None],
    statuses: dict[str, str],
    *,
    required_metrics: Iterable[str] | None = None,
) -> str:
    values = {
        name: value for name, value in metrics.items()
        if statuses.get(name) == "ok" and value is not None
    }
    if not values:
        return "needs_review"

    if required_metrics is not None:
        required = list(required_metrics)
        if any(str(statuses.get(name, "")).startswith("error:") for name in required):
            return "error"
        evaluated = [name for name in required if name in values]
        if not evaluated:
            return "needs_review"
        failed = [
            name for name in evaluated
            if values[name] < METRIC_THRESHOLDS.get(name, 0.8)
        ]
        if failed:
            return "fail" if any(name in _DETERMINISTIC_METRICS for name in failed) else "needs_review"
        if len(evaluated) == len(required):
            return "pass"
        return "needs_review"

    deterministic_values = [
        value for name, value in values.items() if name in _DETERMINISTIC_METRICS
    ]
    if any(value < 0.4 for value in deterministic_values):
        return "fail"
    if any(str(statuses.get(name, "")).startswith("error:") for name in _RAGAS_METRICS):
        return "needs_review"
    if any(value < 0.8 for value in values.values()):
        return "needs_review"
    if all(value >= 0.8 for value in values.values()):
        return "pass"
    return "needs_review"
