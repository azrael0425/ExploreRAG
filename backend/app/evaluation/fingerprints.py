"""Reproducible, secret-free snapshots for evaluation runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document
from app.models.evaluation import EvalCase
from app.models.knowledge_base import KnowledgeBase


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_snapshot() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repository, text=True, stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty_text = run("status", "--porcelain") if commit else ""
    source = "git_worktree" if commit else "build_metadata"
    if not commit:
        commit = settings.EXPLORERAG_BUILD_GIT_COMMIT.strip()
        branch = settings.EXPLORERAG_BUILD_GIT_BRANCH.strip()
    dirty_setting = settings.EXPLORERAG_BUILD_GIT_DIRTY.strip().lower()
    dirty = bool(dirty_text) if source == "git_worktree" else (
        dirty_setting == "true" if dirty_setting in {"true", "false"} else None
    )
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
        "source": source,
    }


def _package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in ("fastapi", "sqlalchemy", "chromadb", "sentence-transformers", "ragas", "langchain"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


def _hardware_snapshot() -> dict[str, Any]:
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": sys.version.split()[0],
    }
    try:
        import torch

        hardware["torch"] = str(torch.__version__)
        hardware["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            hardware["cuda_device"] = torch.cuda.get_device_name(0)
            hardware["cuda_version"] = str(torch.version.cuda)
    except Exception:
        hardware["cuda_available"] = None
    return hardware


async def build_experiment_snapshot(
    db: AsyncSession,
    *,
    workspace: KnowledgeBase,
    run_config: dict[str, Any],
    system_prompt: str,
    cases: list[EvalCase] | None = None,
) -> dict[str, Any]:
    """Return an immutable snapshot without API keys or document contents."""
    rows = list((await db.execute(
        select(Document).where(Document.workspace_id == workspace.id).order_by(Document.id)
    )).scalars().all())
    corpus_manifest = [
        {
            "id": item.id,
            "original_filename": item.original_filename,
            "status": item.status.value if hasattr(item.status, "value") else str(item.status),
            "content_sha256": item.content_sha256,
            "content_version": item.content_version,
            "metadata_revision": item.metadata_revision,
            "chunk_count": item.chunk_count,
            "kg_index_status": item.kg_index_status,
            "kg_indexed_content_version": item.kg_indexed_content_version,
        }
        for item in rows
    ]
    from app.evaluation.ragas_adapter import evaluation_judge_identity

    model_config = {
        "llm_mode": workspace.llm_mode,
        "cloud_provider": settings.LLM_PROVIDER,
        "cloud_model": settings.LLM_MODEL_FAST,
        "local_model": settings.LOCAL_LLM_MODEL,
        "evaluation_judge": evaluation_judge_identity(workspace.llm_mode),
        "embedding_model": settings.EXPLORERAG_EMBEDDING_MODEL,
        "embedding_device": settings.EXPLORERAG_EMBEDDING_DEVICE,
        "reranker_model": settings.EXPLORERAG_RERANKER_MODEL,
        "reranker_device": settings.EXPLORERAG_RERANKER_DEVICE,
        "kg_embedding_model": settings.KG_EMBEDDING_MODEL,
        "kg_embedding_provider": settings.KG_EMBEDDING_PROVIDER,
        "orchestrator": "langchain",
        "document_parser": settings.EXPLORERAG_DOCUMENT_PARSER,
        "chunk_max_tokens": settings.EXPLORERAG_CHUNK_MAX_TOKENS,
        "embedding_max_tokens": settings.EXPLORERAG_EMBEDDING_MAX_TOKENS,
        "embedding_overlap": settings.EXPLORERAG_EMBEDDING_TOKEN_OVERLAP,
    }
    case_manifest = [
        {
            "id": item.id,
            "input_hash": item.input_hash,
            "dataset_name": item.dataset_name,
            "dataset_version": item.dataset_version,
            "split": item.split,
            "category": item.category,
            "difficulty": item.difficulty,
            "expected_behavior": item.expected_behavior,
            "review_status": item.review_status,
            "is_frozen": item.is_frozen,
        }
        for item in sorted(cases or [], key=lambda value: value.id)
    ]
    snapshot = {
        "schema_version": 1,
        "git": _git_snapshot(),
        "workspace": {
            "id": workspace.id,
            "llm_mode": workspace.llm_mode,
            "lightrag_augmentation_enabled": workspace.lightrag_augmentation_enabled,
            "kg_language": workspace.kg_language,
            "kg_entity_types": workspace.kg_entity_types,
        },
        "run_config": run_config,
        "models": model_config,
        "prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "corpus": {
            "fingerprint": _sha256(corpus_manifest),
            "document_count": len(corpus_manifest),
            "chunk_count": sum(int(item["chunk_count"] or 0) for item in corpus_manifest),
            "documents": corpus_manifest,
        },
        "dataset": {
            "fingerprint": _sha256(case_manifest),
            "case_count": len(case_manifest),
            "cases": case_manifest,
        },
        "hardware": _hardware_snapshot(),
        "packages": _package_versions(),
    }
    snapshot["experiment_fingerprint"] = _sha256(snapshot)
    return snapshot
