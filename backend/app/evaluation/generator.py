"""AI-assisted test-case generation from workspace-filtered document chunks."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.models.evaluation import EvalCase
from app.models.knowledge_base import KnowledgeBase
from app.services.llm import get_llm_provider
from app.services.llm.types import LLMMessage, LLMResult
from app.services.explore_rag_factory import get_explore_rag_service


def input_hash(question: str, reference_answer: str | None, reference_chunk_ids: list[str]) -> str:
    raw = "\n".join([question.strip(), (reference_answer or "").strip(), *sorted(reference_chunk_ids)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _json_array(text: str) -> list[dict[str, Any]]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    elif not candidate.startswith("["):
        match = re.search(r"\[.*\]", candidate, re.DOTALL)
        if match:
            candidate = match.group(0)
    value = json.loads(candidate)
    if not isinstance(value, list):
        raise ValueError("AI response is not a JSON array")
    return [item for item in value if isinstance(item, dict)]


async def _workspace_chunks(
    db: AsyncSession,
    workspace_id: int,
    document_ids: list[int],
    *,
    seed: int = 0,
    prefer_tables: bool = False,
    limit: int = 18,
) -> list[dict[str, Any]]:
    query = select(Document).where(
        Document.workspace_id == workspace_id,
        Document.status == DocumentStatus.INDEXED,
    )
    if document_ids:
        query = query.where(Document.id.in_(document_ids))
    documents = list((await db.execute(query.order_by(Document.id))).scalars().all())
    if not documents:
        raise ValueError("No indexed documents are available for AI case generation")

    workspace = (await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))).scalar_one()
    rag_service = get_explore_rag_service(db, workspace_id, llm_mode=workspace.llm_mode)
    candidates_by_document: list[list[dict[str, Any]]] = []
    # Sample a small, rotating window from every document.  This keeps prompts
    # within the model context window and lets repeated seeded batches cover
    # the corpus instead of regenerating questions from the first documents.
    for document in documents:
        chunk_count = int(document.chunk_count or 0)
        if not chunk_count:
            continue
        stride = max(1, chunk_count // 6)
        start = (seed * 7 + document.id * 3) % chunk_count
        indices: list[int] = []
        cursor = start
        while len(indices) < min(8, chunk_count):
            if cursor not in indices:
                indices.append(cursor)
            cursor = (cursor + stride) % chunk_count
            if cursor == start and len(indices) < min(8, chunk_count):
                cursor = (cursor + 1) % chunk_count
        chunk_ids = [f"doc_{document.id}_chunk_{index}" for index in indices]
        if not chunk_ids:
            continue
        rows = rag_service.vector_store.get_by_ids(chunk_ids)
        per_document = []
        metadatas = rows.get("metadatas", []) or []
        for position, (chunk_id, content) in enumerate(zip(rows.get("ids", []), rows.get("documents", []))):
            if content and str(content).strip():
                metadata = metadatas[position] if position < len(metadatas) else {}
                per_document.append({
                    "chunk_id": str(chunk_id),
                    "content": str(content)[:1800],
                    "document_id": document.id,
                    "source_file": document.original_filename,
                    "has_table": bool(metadata.get("has_table", False)),
                })
        if prefer_tables:
            per_document.sort(key=lambda item: not item["has_table"])
        if per_document:
            candidates_by_document.append(per_document)

    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < limit and any(position < len(rows) for rows in candidates_by_document):
        ordered = candidates_by_document[seed % len(candidates_by_document):] + candidates_by_document[:seed % len(candidates_by_document)]
        for rows in ordered:
            if position < len(rows):
                selected.append(rows[position])
                if len(selected) >= limit:
                    break
        position += 1
    if not selected:
        raise ValueError("Indexed documents did not expose any retrievable chunks")
    return selected


async def generate_ai_cases(
    db: AsyncSession,
    *,
    workspace_id: int,
    document_ids: list[int],
    count: int,
    llm_mode: str,
    activate: bool,
    dataset_name: str = "core",
    dataset_version: int = 1,
    split: str = "dev",
    categories: list[str] | None = None,
    seed: int = 0,
) -> list[EvalCase]:
    chunks = await _workspace_chunks(
        db,
        workspace_id,
        document_ids,
        seed=seed,
        prefer_tables="table_numeric" in (categories or []),
    )
    evidence = "\n\n".join(
        f"CHUNK_ID: {item['chunk_id']}\nSOURCE_FILE: {item['source_file']}\n"
        f"HAS_TABLE: {item['has_table']}\nCONTENT:\n{item['content']}"
        for item in chunks
    )
    category_hint = ", ".join(categories or []) or (
        "single_hop, multi_hop, cross_document, table_numeric, citation, unanswerable"
    )
    prompt = f"""Create {count} diverse RAG evaluation cases from the evidence below.
Return JSON only: an array of objects with exactly question, reference_answer,
reference_chunk_ids (an array containing one or more supplied IDs), category,
difficulty, expected_behavior, conversation_history, and tags. Use these requested categories where
the supplied evidence supports them: {category_hint}.

Answerable questions must be grounded in the cited chunks. For unanswerable
cases, set expected_behavior to "refuse", reference_answer to a short statement
that the evidence is insufficient, and reference_chunk_ids to an empty array.
Do not invent answerable facts, use vague questions, or include markdown fences.
For multi_hop use at least two chunks. For cross_document use chunks from at
least two SOURCE_FILE values. For table_numeric cite a HAS_TABLE chunk. For
multi_turn include 1-3 prior user/assistant messages in conversation_history
and make the final question depend on that history. Otherwise use an empty list.
If the requested category is unanswerable, every generated item MUST ask for
information absent from all supplied evidence, use expected_behavior="refuse",
use reference_chunk_ids=[], and must not convert the question into an
answerable evidence question. For every other requested category, use
expected_behavior="answer" and one or more supplied reference chunk IDs.

EVIDENCE:
{evidence}
"""
    result = await get_llm_provider(llm_mode).acomplete(
        [LLMMessage(role="user", content=prompt)],
        system_prompt="You produce strict JSON test data for a RAG evaluation system.",
        temperature=0.2,
        max_tokens=min(8000, max(1200, count * 500)),
    )
    text = result.content if isinstance(result, LLMResult) else str(result)
    generated = _json_array(text)
    valid_ids = {item["chunk_id"] for item in chunks}
    requested_categories = list(categories or [])
    allowed_categories = set(requested_categories)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    existing_hashes = set((await db.execute(
        select(EvalCase.input_hash).where(EvalCase.workspace_id == workspace_id)
    )).scalars().all())
    chunk_by_id = {item["chunk_id"]: item for item in chunks}
    for item in generated:
        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer", "")).strip()
        reference_ids = [
            str(value)
            for value in item.get("reference_chunk_ids", [])
            if str(value) in valid_ids
        ]
        expected_behavior = (
            "refuse" if str(item.get("expected_behavior", "answer")) == "refuse" else "answer"
        )
        if not question or not reference_answer or (expected_behavior == "answer" and not reference_ids):
            continue
        category = str(item.get("category", "other")).strip()[:50] or "other"
        if allowed_categories and category not in allowed_categories:
            category = requested_categories[0]
        if category == "unanswerable":
            if expected_behavior != "refuse" or reference_ids:
                continue
        elif expected_behavior != "answer" or not reference_ids:
            continue
        difficulty = str(item.get("difficulty", "medium")).strip().lower()
        if difficulty not in {"easy", "medium", "hard"}:
            difficulty = "medium"
        referenced_files = {
            chunk_by_id[chunk_id]["source_file"]
            for chunk_id in reference_ids
            if chunk_id in chunk_by_id
        }
        if category == "multi_hop" and len(reference_ids) < 2:
            continue
        if category == "cross_document" and len(referenced_files) < 2:
            continue
        if category == "table_numeric" and not any(
            chunk_by_id[chunk_id]["has_table"]
            for chunk_id in reference_ids
            if chunk_id in chunk_by_id
        ):
            continue
        history = item.get("conversation_history", [])
        if not isinstance(history, list):
            history = []
        history = [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))[:4000]}
            for message in history[:3]
            if isinstance(message, dict)
            and str(message.get("role", "")) in {"user", "assistant"}
            and str(message.get("content", "")).strip()
        ]
        if category == "multi_turn" and not history:
            continue
        if category != "multi_turn":
            history = []
        digest = input_hash(question, reference_answer, reference_ids)
        if digest in seen or digest in existing_hashes:
            continue
        seen.add(digest)
        cases.append(EvalCase(
            workspace_id=workspace_id,
            status="draft",
            source="ai",
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            split=split,
            category=category,
            difficulty=difficulty,
            expected_behavior=expected_behavior,
            # Synthetic gold data is never implicitly approved.  A reviewer
            # must validate the answer and chunk IDs before it can enter a run.
            review_status="draft",
            question=question,
            reference_answer=reference_answer,
            reference_chunk_ids=reference_ids,
            reference_contexts=[item["content"] for item in chunks if item["chunk_id"] in reference_ids],
            conversation_history=history,
            tags=[str(tag)[:80] for tag in item.get("tags", [])[:10]],
            extra_metadata={
                "generator": "workspace_chunk_grounded",
                "human_review_required": True,
                "requested_activation": activate,
                "generation_seed": seed,
            },
            input_hash=digest,
        ))
        if len(cases) >= count:
            break
    if not cases:
        raise ValueError("AI generation returned no valid, chunk-grounded cases")
    db.add_all(cases)
    await db.commit()
    for case in cases:
        await db.refresh(case)
    return cases
