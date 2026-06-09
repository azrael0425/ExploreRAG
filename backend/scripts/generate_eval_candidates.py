"""Generate the versioned 120-case review queue through the public API.

The script never approves or activates synthetic cases.  It is safe to rerun:
the API rejects duplicate input hashes and this client only fills missing quota.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


QUOTAS = {
    "dev": {
        "single_hop": 28,
        "multi_hop": 12,
        "cross_document": 10,
        "table_numeric": 10,
        "unanswerable": 7,
        "citation": 7,
        "multi_turn": 6,
    },
    "test": {
        "single_hop": 12,
        "multi_hop": 8,
        "cross_document": 5,
        "table_numeric": 5,
        "unanswerable": 3,
        "citation": 3,
        "multi_turn": 4,
    },
}

EXPORT_FIELDS = (
    "question",
    "reference_answer",
    "reference_chunk_ids",
    "reference_contexts",
    "reference_entity_names",
    "reference_relationships",
    "conversation_history",
    "tags",
    "status",
    "source",
    "dataset_name",
    "dataset_version",
    "split",
    "is_frozen",
    "category",
    "difficulty",
    "expected_behavior",
    "review_status",
    "metadata",
)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: int = 600,
) -> Any:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {detail}") from exc
    return json.loads(raw) if raw else None


def list_cases(base_url: str, workspace_id: int, dataset_name: str, version: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset_name": dataset_name, "dataset_version": version})
    response = request_json(
        "GET",
        f"{base_url}/evaluations/workspaces/{workspace_id}/cases?{query}",
    )
    return list(response.get("items", []))


def export_candidates(cases: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for case in sorted(cases, key=lambda item: (item["split"], item["category"], item["id"])):
            record = {key: case.get(key) for key in EXPORT_FIELDS}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def archive_excess_candidates(
    base_url: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep exact quotas without deleting data; archived cases remain recoverable."""
    for split, categories in QUOTAS.items():
        for category, target in categories.items():
            rows = sorted(
                (
                    item for item in cases
                    if item.get("status") != "archived"
                    and item.get("split") == split
                    and item.get("category") == category
                ),
                key=lambda item: int(item["id"]),
            )
            excess = max(0, len(rows) - target)
            removable = [
                item for item in reversed(rows)
                if item.get("source") == "ai" and item.get("review_status") == "draft"
            ]
            for item in removable[:excess]:
                request_json(
                    "PATCH",
                    f"{base_url}/evaluations/cases/{item['id']}",
                    {"status": "archived"},
                )
                item["status"] = "archived"
                print(f"archived excess candidate #{item['id']} ({split}/{category})", flush=True)
    return [item for item in cases if item.get("status") != "archived"]


def archive_structurally_invalid_candidates(
    base_url: str,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject category/behavior contradictions before human review."""
    for item in cases:
        if item.get("status") == "archived":
            continue
        category = item.get("category")
        behavior = item.get("expected_behavior")
        reference_ids = item.get("reference_chunk_ids") or []
        history = item.get("conversation_history") or []
        valid = (
            behavior == "refuse" and not reference_ids
            if category == "unanswerable"
            else behavior == "answer" and bool(reference_ids)
        )
        if category in {"multi_hop", "cross_document"}:
            valid = valid and len(reference_ids) >= 2
        if category == "multi_turn":
            valid = valid and bool(history)
        if valid:
            continue
        if item.get("source") != "ai" or item.get("review_status") != "draft":
            continue
        request_json(
            "PATCH",
            f"{base_url}/evaluations/cases/{item['id']}",
            {"status": "archived"},
        )
        item["status"] = "archived"
        print(f"archived structurally invalid candidate #{item['id']}", flush=True)
    return [item for item in cases if item.get("status") != "archived"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--workspace-id", type=int, required=True)
    parser.add_argument("--dataset-name", default="explorerag_core")
    parser.add_argument("--dataset-version", type=int, default=1)
    parser.add_argument("--document-id", type=int, action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("evaluation/datasets/explorerag_core_v1.candidates.jsonl"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    cases = archive_excess_candidates(
        base_url,
        archive_structurally_invalid_candidates(
            base_url,
            list_cases(base_url, args.workspace_id, args.dataset_name, args.dataset_version),
        ),
    )
    counts = Counter((item["split"], item["category"]) for item in cases)
    seed = max([int((item.get("metadata") or {}).get("generation_seed", 0)) for item in cases] or [0]) + 1

    for split, categories in QUOTAS.items():
        for category, target in categories.items():
            attempts = 0
            while counts[(split, category)] < target and attempts < args.max_attempts:
                missing = target - counts[(split, category)]
                # Smaller batches reduce truncated/invalid JSON and give
                # complex multi-hop constraints more output budget per case.
                requested = min(6, missing)
                payload = {
                    "document_ids": args.document_id,
                    "count": requested,
                    "activate": False,
                    "dataset_name": args.dataset_name,
                    "dataset_version": args.dataset_version,
                    "split": split,
                    "categories": [category],
                    "seed": seed,
                }
                try:
                    generated = request_json(
                        "POST",
                        f"{base_url}/evaluations/workspaces/{args.workspace_id}/cases/generate",
                        payload,
                    )
                except RuntimeError as exc:
                    if "(422)" not in str(exc):
                        raise
                    generated = []
                added = len(generated or [])
                counts[(split, category)] += added
                print(
                    f"{split}/{category}: +{added}, "
                    f"{counts[(split, category)]}/{target} (seed={seed})",
                    flush=True,
                )
                seed += 1
                attempts += 1
                if not added:
                    time.sleep(1)
            if counts[(split, category)] < target:
                print(
                    f"WARNING: quota incomplete for {split}/{category}: "
                    f"{counts[(split, category)]}/{target}",
                    flush=True,
                )

    cases = archive_excess_candidates(
        base_url,
        archive_structurally_invalid_candidates(
            base_url,
            list_cases(base_url, args.workspace_id, args.dataset_name, args.dataset_version),
        ),
    )
    counts = Counter((item["split"], item["category"]) for item in cases)
    export_candidates(cases, args.output)
    total_target = sum(sum(values.values()) for values in QUOTAS.values())
    total = sum(
        min(counts[(split, category)], target)
        for split, categories in QUOTAS.items()
        for category, target in categories.items()
    )
    print(f"Candidate quota: {total}/{total_target}")
    print(f"Review queue exported to {args.output.resolve()}")
    return 0 if total == total_target else 2


if __name__ == "__main__":
    raise SystemExit(main())
