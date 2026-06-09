"""Export one completed ablation experiment as raw JSON and Markdown."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIMARY_METRICS = (
    "hit_at_4",
    "recall_at_4",
    "recall_at_20",
    "mrr_at_4",
    "ndcg_at_4",
    "factual_correctness",
    "faithfulness",
    "answer_relevancy",
    "citation_precision",
    "citation_recall",
    "refusal_accuracy",
    "graph_evidence_recall",
    "graph_relationship_recall",
    "graph_traceability",
)


def get_json(url: str, timeout: int = 120) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed ({exc.code}): {detail}") from exc


def value(summary: dict[str, Any], metric: str) -> str:
    item = (summary.get("metrics") or {}).get(metric) or {}
    score = item.get("avg")
    count = item.get("evaluated_count", 0)
    return "-" if score is None else f"{score:.4f} (n={count})"


def value_with_delta(
    summary: dict[str, Any],
    metric: str,
    comparison: dict[str, Any] | None = None,
) -> str:
    rendered = value(summary, metric)
    delta = ((comparison or {}).get("metric_deltas") or {}).get(metric)
    if rendered == "-" or not delta or delta.get("delta") is None:
        return rendered
    return f"{rendered}; ΔA={float(delta['delta']):+.4f}"


def percentile(summary: dict[str, Any], name: str, key: str) -> str:
    result = ((summary.get("performance") or {}).get(name) or {}).get(key)
    return "-" if result is None else f"{result:.0f} ms"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact(
    output: Path,
    *,
    experiment_id: str,
    source_api: str,
    details: list[dict[str, Any]],
    artifact_zip: Path,
) -> None:
    """Create a self-checking evidence bundle without embedding credentials."""
    snapshot = (
        ((details[0].get("target_config") or {}).get("snapshot") or {})
        if details
        else {}
    )
    evidence_files = ("raw.json", "report.md")
    manifest = {
        "artifact_schema_version": 1,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_api": source_api,
        "code_snapshot": snapshot.get("git") or {},
        "dataset_fingerprint": (snapshot.get("dataset") or {}).get("fingerprint"),
        "corpus_fingerprint": (snapshot.get("corpus") or {}).get("fingerprint"),
        "experiment_fingerprint": snapshot.get("experiment_fingerprint"),
        "files": {
            name: {
                "sha256": sha256_file(output / name),
                "bytes": (output / name).stat().st_size,
            }
            for name in evidence_files
        },
        "limitations": [
            "The bundle makes recorded outputs independently auditable; rerunning retrieval also requires the exact source corpus.",
            "LLM-backed metrics may vary when provider models or APIs change.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksummed_files = (*evidence_files, "manifest.json")
    checksums_path = output / "SHA256SUMS"
    checksums_path.write_text(
        "".join(f"{sha256_file(output / name)}  {name}\n" for name in checksummed_files),
        encoding="utf-8",
    )

    artifact_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(artifact_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*checksummed_files, "SHA256SUMS"):
            archive.write(output / name, arcname=name)


def render_markdown(
    experiment_id: str,
    details: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> str:
    by_variant = {item["variant"]: item for item in details}
    ordered = [by_variant[key] for key in ("A", "B", "C", "D") if key in by_variant]
    snapshot = (((ordered[0].get("target_config") or {}).get("snapshot")) or {}) if ordered else {}
    formal_contract = bool(ordered) and len(ordered) == 4 and all(
        item.get("run_type") == "full"
        and item.get("dataset_split") == "test"
        and (item.get("metrics_summary") or {}).get("case_count", 0) >= 40
        and int((item.get("config") or {}).get("warmup_queries", 0) or 0) >= 1
        for item in ordered
    ) and (snapshot.get("git") or {}).get("dirty") is False
    eligible = formal_contract and all(
        item["status"] == "completed"
        and (item.get("metrics_summary") or {}).get("experiment_valid") is True
        for item in ordered
    ) and all((item.get("gate") or {}).get("status") == "pass" for item in comparisons.values())

    lines = [
        f"# RAG 2×2 消融报告：{experiment_id}",
        "",
        f"- 报告状态：{'可用于 README/简历' if eligible else '草稿，不可用于简历结论'}",
        f"- Git commit：`{((snapshot.get('git') or {}).get('commit') or 'unknown')}`",
        f"- Git dirty：`{((snapshot.get('git') or {}).get('dirty'))}`",
        f"- 数据集指纹：`{((snapshot.get('dataset') or {}).get('fingerprint') or 'unknown')}`",
        f"- 语料指纹：`{((snapshot.get('corpus') or {}).get('fingerprint') or 'unknown')}`",
        f"- Prompt SHA-256：`{snapshot.get('prompt_sha256') or 'unknown'}`",
        f"- 正式实验契约：{'满足' if formal_contract else '不满足（需 full/test/n≥40/预热/clean Git）'}",
        "",
        "## 实验配置",
        "",
        "| Variant | Vector | Reranker | KG | 状态 | 组件实际生效 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in ordered:
        config = item.get("config") or {}
        summary = item.get("metrics_summary") or {}
        lines.append(
            f"| {item['variant']} | ✓ | {'✓' if config.get('enable_reranker') else '✗'} | "
            f"{'✓' if config.get('enable_knowledge_graph') else '✗'} | {item['status']} | "
            f"{'有效' if summary.get('experiment_valid') else '无效/降级'} |"
        )
    if ordered:
        protocol = (ordered[0].get("metrics_summary") or {}).get("measurement_protocol") or {}
        lines.extend([
            "",
            f"- 稳态统计前排除预热查询：{protocol.get('warmup_queries_excluded', 0)} 次/variant",
            f"- 案例顺序 seed：`{protocol.get('case_order_seed', 0)}`",
            f"- 延迟总体：`{protocol.get('latency_population', 'unknown')}`",
        ])

    lines.extend([
        "", "## 质量指标", "",
        "| 指标 | A | B（相对 A） | C（相对 A） | D（相对 A） |",
        "|---|---:|---:|---:|---:|",
    ])
    for metric in PRIMARY_METRICS:
        cells = [value((by_variant.get("A", {}).get("metrics_summary") or {}), metric)]
        cells.extend(
            value_with_delta(
                (by_variant.get(variant, {}).get("metrics_summary") or {}),
                metric,
                comparisons.get(variant),
            )
            for variant in ("B", "C", "D")
        )
        lines.append("| " + metric + " | " + " | ".join(cells) + " |")

    lines.extend(["", "## 稳态延迟（预热观测不计入）", "", "| Variant | total P50 | total P95 | rerank P95 | graph P95 |", "|---|---:|---:|---:|---:|"])
    for variant in ("A", "B", "C", "D"):
        summary = by_variant.get(variant, {}).get("metrics_summary") or {}
        lines.append(
            f"| {variant} | {percentile(summary, 'total_ms', 'p50')} | "
            f"{percentile(summary, 'total_ms', 'p95')} | "
            f"{percentile(summary, 'rerank_ms', 'p95')} | "
            f"{percentile(summary, 'graph_ms', 'p95')} |"
        )

    lines.extend(["", "### 首次预热观测", "", "| Variant | warmup total | warmup graph | warmup runner wall |", "|---|---:|---:|---:|"])
    for variant in ("A", "B", "C", "D"):
        summary = by_variant.get(variant, {}).get("metrics_summary") or {}
        observations = ((summary.get("preflight") or {}).get("warmup_observations") or [])
        observation = observations[0] if observations else {}
        def warmup_value(name: str) -> str:
            result = observation.get(name)
            return "-" if result is None else f"{result:.0f} ms"
        lines.append(
            f"| {variant} | {warmup_value('total_ms')} | {warmup_value('graph_ms')} | "
            f"{warmup_value('runner_wall_ms')} |"
        )

    lines.extend(["", "## 相对 A 的配对比较与门禁", ""])
    for variant in ("B", "C", "D"):
        comparison = comparisons.get(variant) or {}
        gate = comparison.get("gate") or {}
        compatibility = comparison.get("compatibility") or {}
        lines.append(
            f"- {variant}：gate=`{gate.get('status', 'missing')}`，"
            f"contract={'valid' if compatibility.get('valid') else 'invalid'}，"
            f"paired n={comparison.get('paired_case_count', 0)}，"
            f"regressions={len(comparison.get('regressions') or [])}"
        )
        for name, metric in (comparison.get("metric_deltas") or {}).items():
            if name in PRIMARY_METRICS:
                lines.append(
                    f"  - {name}: Δ={metric.get('delta')}, 95% CI={metric.get('ci95')}, n={metric.get('paired_count')}"
                )
        total_latency = (comparison.get("latency_deltas") or {}).get("total_ms") or {}
        if total_latency:
            lines.append(
                f"  - total P95 Δ={total_latency.get('delta_p95')} ms; "
                f"relative={total_latency.get('relative_delta_p95')}; "
                f"ablation trade-off (latency gate enforced={gate.get('latency_budget_enforced')})"
            )

    lines.extend(["", "## 关系/多跳子集的配对比较", ""])
    for variant in ("C", "D"):
        categories = (((comparisons.get(variant) or {}).get("subgroups") or {}).get("category") or {})
        for category in ("multi_hop", "cross_document"):
            subgroup = categories.get(category) or {}
            lines.append(f"- {variant}/{category}: paired n={subgroup.get('paired_case_count', 0)}")
            for name in ("factual_correctness", "faithfulness", "graph_evidence_recall", "graph_relationship_recall"):
                metric = (subgroup.get("metric_deltas") or {}).get(name)
                if metric:
                    lines.append(
                        f"  - {name}: Δ={metric.get('delta')}, 95% CI={metric.get('ci95')}, n={metric.get('paired_count')}"
                    )

    lines.extend([
        "",
        "## 结论使用规则",
        "",
        "只有 full/test/n≥40、clean Git、四组均完成、实验契约一致、请求组件实际生效且质量回归门禁通过时，才可把本报告数字写入 README 或简历。消融中新增组件的延迟预算只作为代价报告，不作为质量门禁失败原因。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--workspace-id", type=int, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("evaluation/results"))
    parser.add_argument(
        "--artifact-zip",
        type=Path,
        help="Optional path for a release-ready ZIP with hashes and provenance.",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    runs = get_json(f"{base_url}/evaluations/workspaces/{args.workspace_id}/runs")
    selected = [item for item in runs if item.get("experiment_id") == args.experiment_id]
    if not selected:
        raise SystemExit(f"No runs found for experiment {args.experiment_id}")
    # Retries intentionally retain the experiment id and variant so their
    # provenance stays auditable.  Select the newest completed attempt for
    # each variant; otherwise a failed original D run could shadow its valid
    # retry depending on list ordering.
    selected_by_variant: dict[str, dict[str, Any]] = {}
    for item in selected:
        variant = str(item.get("variant"))
        current = selected_by_variant.get(variant)
        if current is None or (
            current.get("status") != "completed" and item.get("status") == "completed"
        ) or (
            current.get("status") == item.get("status")
            and int(item.get("id", 0)) > int(current.get("id", 0))
        ):
            selected_by_variant[variant] = item
    chosen = [selected_by_variant[key] for key in ("A", "B", "C", "D") if key in selected_by_variant]
    details = [get_json(f"{base_url}/evaluations/runs/{item['id']}") for item in chosen]
    by_variant = {item["variant"]: item for item in details}
    comparisons = {}
    if "A" in by_variant:
        for variant in ("B", "C", "D"):
            if variant in by_variant and by_variant[variant]["status"] == "completed" and by_variant["A"]["status"] == "completed":
                comparisons[variant] = get_json(
                    f"{base_url}/evaluations/runs/{by_variant[variant]['id']}/compare/{by_variant['A']['id']}"
                )

    output = args.output_root / args.experiment_id
    output.mkdir(parents=True, exist_ok=True)
    raw = {
        "experiment_id": args.experiment_id,
        "runs": details,
        "comparisons_to_A": comparisons,
    }
    (output / "raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "report.md").write_text(
        render_markdown(args.experiment_id, details, comparisons),
        encoding="utf-8",
    )
    if args.artifact_zip:
        build_artifact(
            output,
            experiment_id=args.experiment_id,
            source_api=base_url,
            details=details,
            artifact_zip=args.artifact_zip,
        )
        print(args.artifact_zip.resolve())
    print((output / "report.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
