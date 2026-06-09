"""HTTP/SSE transport for the single LangChain chat workflow."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
from app.langchain_rag.contracts import ChatChainInput
from app.langchain_rag.service import LangChainChatService
from app.models.chat_attachment import ChatMessageAttachment
from app.models.knowledge_base import KnowledgeBase
from app.schemas.rag import ChatRequest

logger = logging.getLogger(__name__)

SSE_HEARTBEAT_INTERVAL = 15


def format_sse_event(event: str, data: dict) -> str:
    """Serialize one domain event using the existing frontend SSE contract."""

    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


async def sse_with_heartbeat(source: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Keep slow model responses connected with standard SSE heartbeats."""

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for event in source:
                await queue.put(event)
        finally:
            await queue.put(None)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_INTERVAL)
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def chat_stream_endpoint(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession,
) -> StreamingResponse:
    """Persist messages and stream the LangChain/LCEL chat workflow."""

    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    knowledge_base = result.scalar_one_or_none()
    if knowledge_base is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found")

    from app.models.chat_message import ChatMessage

    system_prompt = (knowledge_base.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT
    history = [{"role": item.role, "content": item.content} for item in request.history]
    workspace_epoch = knowledge_base.chat_cleanup_epoch
    selected_attachments = []
    if request.attachment_ids:
        from app.services.chat_attachment_service import ChatAttachmentService

        selected_attachments = await ChatAttachmentService().selected(
            db, workspace_id, request.attachment_ids
        )

    user_message_id = str(uuid.uuid4())
    user_persisted = False
    try:
        user_row = ChatMessage(
            workspace_id=workspace_id,
            message_id=user_message_id,
            role="user",
            content=request.message,
        )
        db.add(user_row)
        await db.flush()
        db.add_all([
            ChatMessageAttachment(message_id=user_row.id, attachment_id=attachment.id)
            for attachment in selected_attachments
        ])
        await db.commit()
        user_persisted = True
    except Exception as exc:
        logger.warning("Failed to persist user message: %s", exc)
        await db.rollback()

    chain_input = ChatChainInput(
        workspace_id=workspace_id,
        message=request.message,
        history=history,
        attachment_ids=list(request.attachment_ids or []),
        enable_thinking=request.enable_thinking,
        llm_mode=knowledge_base.llm_mode,
        lightrag_augmentation_enabled=knowledge_base.lightrag_augmentation_enabled,
        system_prompt=system_prompt,
        document_ids=request.document_ids,
        metadata_filter=(
            request.metadata_filter.model_dump(by_alias=True)
            if request.metadata_filter else None
        ),
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        steps: list[dict] = []
        step_number = 0
        sources_seen: list[dict] = []
        images_seen: list[dict] = []
        try:
            async for domain_event in LangChainChatService(db).stream_chat(
                chain_input,
                selected_attachments=selected_attachments,
            ):
                event = domain_event.as_dict()
                event_type = event["event"]
                payload = event["data"]
                if event_type == "status":
                    step_number += 1
                    steps.append({
                        "id": f"step-{step_number}",
                        "step": payload.get("step", "analyzing"),
                        "detail": payload.get("detail", ""),
                        "status": "completed",
                        "timestamp": 0,
                    })
                elif event_type == "sources":
                    sources_seen.extend(payload.get("sources", []))
                elif event_type == "images":
                    images_seen.extend(payload.get("image_refs", []))
                elif event_type == "complete":
                    assistant_row: ChatMessage | None = None
                    if user_persisted:
                        try:
                            from app.services.chat_cleanup_service import ChatCleanupService

                            epoch_is_current = await ChatCleanupService().is_epoch_current(
                                db, workspace_id, workspace_epoch
                            )
                            if epoch_is_current:
                                assistant_row = ChatMessage(
                                    workspace_id=workspace_id,
                                    message_id=str(uuid.uuid4()),
                                    role="assistant",
                                    content=payload.get("answer", ""),
                                    sources=payload.get("sources") or None,
                                    related_entities=payload.get("related_entities") or None,
                                    image_refs=payload.get("image_refs") or None,
                                    thinking=payload.get("thinking"),
                                    reply_to_message_id=user_message_id,
                                )
                                db.add(assistant_row)
                                await db.flush()
                                payload["message_id"] = assistant_row.message_id
                                await db.commit()
                            else:
                                logger.info(
                                    "Skipped chat persistence after workspace %s was cleared",
                                    workspace_id,
                                )
                        except Exception as exc:
                            logger.warning("Failed to persist assistant message: %s", exc)
                            await db.rollback()
                    if sources_seen:
                        step_number += 1
                        steps.append({
                            "id": f"step-{step_number}",
                            "step": "sources_found",
                            "detail": f"Found {len(sources_seen)} sources",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(sources_seen),
                            "imageCount": len(images_seen),
                            "sourceBadges": list(dict.fromkeys(
                                source.get("index", "") for source in sources_seen[:6]
                            )),
                        })
                    step_number += 1
                    steps.append({
                        "id": f"step-{step_number}",
                        "step": "done",
                        "detail": "Done",
                        "status": "completed",
                        "timestamp": 0,
                        "performance": payload.get("performance"),
                    })
                    if assistant_row is not None:
                        assistant_row.agent_steps = steps or None
                        await db.commit()
                yield format_sse_event(event_type, payload)
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            yield format_sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        sse_with_heartbeat(event_generator()),
        media_type="text/event-stream",
    )
