"""Shared formatting for graph evidence entering an answer prompt."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from app.schemas.rag import ChatSourceChunk
from app.services.models.parsed_document import KnowledgeGraphFact


def append_knowledge_graph_source(
    *,
    knowledge_graph_summary: str,
    sources: list[ChatSourceChunk],
    context_parts: list[str],
    existing_ids: set[str],
    source_label: Callable[[str, set[str]], str],
    graph_entity_names: list[str] | None = None,
    graph_document_ids: list[int] | None = None,
    graph_facts: list[KnowledgeGraphFact | dict[str, Any]] | None = None,
) -> ChatSourceChunk | None:
    """Expose structured LightRAG evidence as one explicit, citable source.

    The graph source intentionally has no document/page target: a graph fact
    can aggregate multiple chunks.  Its ``source_type`` makes the UI open the
    graph instead of pretending that it is a page-level document citation.
    """
    evidence = knowledge_graph_summary.strip()
    if not evidence:
        return None

    def fact_value(fact: KnowledgeGraphFact | dict[str, Any], key: str, default):
        if isinstance(fact, dict):
            return fact.get(key, default)
        return getattr(fact, key, default)

    emitted: list[ChatSourceChunk] = []
    for fact in (graph_facts or [])[:8]:
        fact_content = str(fact_value(fact, "content", "")).strip()
        if not fact_content:
            continue
        citation_id = source_label("KG", existing_ids)
        existing_ids.add(citation_id)
        digest = hashlib.sha256(fact_content.encode("utf-8")).hexdigest()[:16]
        fact_entities = list(dict.fromkeys(
            str(value).strip()
            for value in fact_value(fact, "entity_names", [])
            if str(value).strip()
        ))
        fact_document_ids = list(dict.fromkeys(
            int(value)
            for value in fact_value(fact, "source_document_ids", [])
            if str(value).isdigit()
        ))
        source = ChatSourceChunk(
            index=citation_id,
            chunk_id=f"kg-fact:{digest}",
            content=fact_content,
            document_id=0,
            source_file="LightRAG knowledge graph",
            page_no=0,
            heading_path=[],
            score=0.0,
            source_type="kg",
            graph_entity_names=fact_entities,
            graph_document_ids=fact_document_ids,
        )
        sources.append(source)
        emitted.append(source)
        entity_suffix = f" (entities: {', '.join(fact_entities)})" if fact_entities else ""
        context_parts.append(
            f"Knowledge Graph Evidence [{citation_id}]{entity_suffix}:\n{fact_content}"
        )

    if emitted:
        return emitted[0]

    citation_id = source_label("KG", existing_ids)
    existing_ids.add(citation_id)
    digest = hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:16]
    source = ChatSourceChunk(
        index=citation_id,
        chunk_id=f"kg:{digest}",
        content=evidence,
        document_id=0,
        source_file="LightRAG knowledge graph",
        page_no=0,
        heading_path=[],
        score=0.0,
        source_type="kg",
        graph_entity_names=graph_entity_names or [],
        graph_document_ids=graph_document_ids or [],
    )
    sources.append(source)
    context_parts.append(
        f"Knowledge Graph Evidence [{citation_id}]:\n{evidence}"
    )
    return source
