"""LCEL-compatible durable knowledge-base retrieval and context formatting."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.langchain_rag.contracts import ChatChainInput, RetrievalEnvelope, RetrievalInput
from app.langchain_rag.factory import get_explore_rag_retrieval_runnable
from app.models.document import DocumentImage
from app.schemas.rag import ChatImageRef, ChatSourceChunk
from app.services.llm.types import LLMImagePart
from app.services.models.parsed_document import RetrievalTimings
from app.services.document_metadata import MetadataValidationError, resolve_document_scope
from app.services.chat_context import append_knowledge_graph_source
from app.services.retrieval_policy import resolve_retrieval_policy
from app.services.citation_ids import generate_citation_id, source_label

logger = logging.getLogger(__name__)

MAX_VISION_IMAGES = 3
def requires_document_search(message: str) -> bool:
    """Keep short greetings conversational and ground all other messages."""
    normalized = re.sub(r"[^a-z\u4e00-\u9fff]+", "", message.lower())
    return normalized not in {
        "hi", "hello", "hey", "goodmorning", "goodafternoon", "goodnight",
        "thanks", "thankyou", "ok", "okay", "gotit", "bye", "goodbye",
        "你好", "您好", "嗨", "谢谢", "感谢", "好的", "明白了", "再见",
    }


@dataclass
class KnowledgeBaseContext:
    """Chat-ready durable-KB material with sources, images and timings."""

    context: str
    sources: list[ChatSourceChunk]
    image_refs: list[ChatImageRef]
    llm_images: list[LLMImagePart]
    timings: RetrievalTimings | None
    envelope: RetrievalEnvelope | None = None


async def _resolve_images(
    db: AsyncSession,
    workspace_id: int,
    documents: list,
    existing_ids: set[str],
) -> tuple[list[ChatImageRef], list[LLMImagePart], list[str]]:
    """Preserve the existing chunk-ID then same-page image lookup strategy."""
    image_ids: list[str] = []
    seen_image_ids: set[str] = set()
    for document in documents:
        for image_id in document.metadata.get("image_refs", []) or []:
            if image_id and image_id not in seen_image_ids:
                seen_image_ids.add(image_id)
                image_ids.append(str(image_id))

    images: list[DocumentImage] = []
    if image_ids:
        result = await db.execute(select(DocumentImage).where(DocumentImage.image_id.in_(image_ids)))
        images = list(result.scalars().all())
    if not images:
        pages = {
            (int(document.metadata.get("document_id", 0)), int(document.metadata.get("page_no", 0)))
            for document in documents
            if int(document.metadata.get("page_no", 0)) > 0
        }
        if pages:
            filters = [
                and_(DocumentImage.document_id == document_id, DocumentImage.page_no == page_no)
                for document_id, page_no in pages
            ]
            result = await db.execute(select(DocumentImage).where(or_(*filters)))
            seen: set[str] = set()
            images = [
                image for image in result.scalars().all()
                if not (image.image_id in seen or seen.add(image.image_id))
            ]

    image_refs: list[ChatImageRef] = []
    llm_images: list[LLMImagePart] = []
    image_context: list[str] = []
    for image in images[:MAX_VISION_IMAGES]:
        image_ref_id = generate_citation_id(existing_ids)
        existing_ids.add(image_ref_id)
        image_refs.append(ChatImageRef(
            ref_id=image_ref_id,
            image_id=image.image_id,
            document_id=image.document_id,
            page_no=image.page_no,
            caption=image.caption or "",
            url=f"/static/doc-images/kb_{workspace_id}/images/{image.image_id}.png",
            width=image.width,
            height=image.height,
        ))
        caption = f' "{image.caption}"' if image.caption else ""
        image_context.append(f"[IMG-{image_ref_id}] page {image.page_no}:{caption}")
        image_path = Path(image.file_path)
        if image_path.exists():
            try:
                llm_images.append(LLMImagePart(
                    data=image_path.read_bytes(),
                    mime_type=image.mime_type or "image/png",
                ))
            except OSError as exc:
                logger.warning("Could not read document image %s: %s", image.image_id, exc)
    return image_refs, llm_images, image_context


async def retrieve_knowledge_base(
    input: ChatChainInput,
    db: AsyncSession,
    existing_ids: set[str],
) -> KnowledgeBaseContext:
    """Execute the actual LangChain Retriever/Runnable path for one chat query."""
    try:
        scope = await resolve_document_scope(
            db,
            input.workspace_id,
            input.document_ids,
            input.metadata_filter,
        )
    except MetadataValidationError as exc:
        raise ValueError(str(exc)) from exc
    policy = resolve_retrieval_policy(
        input.retrieval_mode,
        workspace_lightrag_enabled=input.lightrag_augmentation_enabled,
        scoped=scope.scoped,
    )
    effective_mode = policy.mode
    runnable = get_explore_rag_retrieval_runnable(
        db,
        input.workspace_id,
        llm_mode=input.llm_mode,
    )
    envelope = await runnable.ainvoke(RetrievalInput(
        workspace_id=input.workspace_id,
        question=input.message,
        top_k=input.top_k,
        mode=effective_mode,
        document_ids=scope.document_ids,
        include_images=False,
        enable_reranker=input.enable_reranker,
        enable_knowledge_graph=input.enable_knowledge_graph,
        prefetch_k=input.prefetch_k,
    ), config={
        "run_name": "explorerag_kb_retrieval",
        "tags": ["rag", input.llm_mode, effective_mode],
        "metadata": {"workspace_id": input.workspace_id},
    })

    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    for index, document in enumerate(envelope.documents):
        metadata = document.metadata
        citation = envelope.citations[index] if index < len(envelope.citations) else None
        citation_id = source_label("KB", existing_ids)
        existing_ids.add(citation_id)
        source_file = citation.source_file if citation else str(metadata.get("source", ""))
        page_no = int(metadata.get("page_no", 0))
        heading_path = list(metadata.get("heading_path", []))
        sources.append(ChatSourceChunk(
            index=citation_id,
            chunk_id=str(metadata.get("chunk_id", "")),
            content=document.page_content,
            document_id=int(metadata.get("document_id", 0)),
            source_file=source_file,
            page_no=page_no,
            heading_path=heading_path,
            score=0.0,
            source_type="vector",
            graph_entity_names=list(metadata.get("graph_entity_names", [])),
        ))
        label_parts = [source_file] if source_file else []
        if page_no:
            label_parts.append(f"page {page_no}")
        if heading_path:
            label_parts.append(" > ".join(heading_path))
        suffix = f" ({', '.join(label_parts)})" if label_parts else ""
        context_parts.append(f"Source [{citation_id}]{suffix}:\n{document.page_content}")

    if policy.lightrag_enabled:
        append_knowledge_graph_source(
            knowledge_graph_summary=envelope.knowledge_graph_summary,
            sources=sources,
            context_parts=context_parts,
            existing_ids=existing_ids,
            source_label=source_label,
            graph_entity_names=envelope.knowledge_graph_entity_names,
            graph_document_ids=envelope.knowledge_graph_document_ids,
            graph_facts=envelope.knowledge_graph_facts,
        )

    image_refs, llm_images, image_context = await _resolve_images(
        db, input.workspace_id, envelope.documents, existing_ids
    )
    context = "\n\n---\n\n".join(context_parts)
    if image_context:
        context += "\n\nDocument images:\n" + "\n".join(image_context)

    return KnowledgeBaseContext(
        context=context,
        sources=sources,
        image_refs=image_refs,
        llm_images=llm_images,
        timings=envelope.timings,
        envelope=envelope,
    )


def build_retrieval_chain(
    db: AsyncSession,
    existing_ids: set[str],
) -> RunnableLambda:
    """Return the request-scoped LCEL runnable used by the chat service."""
    async def run(input: ChatChainInput) -> KnowledgeBaseContext:
        return await retrieve_knowledge_base(input, db, existing_ids)

    return RunnableLambda(run, name="explorerag_retrieval")
