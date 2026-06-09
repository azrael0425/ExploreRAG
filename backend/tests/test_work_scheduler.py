from __future__ import annotations

import asyncio

from app.services.work_scheduler import WorkPriority, WorkScheduler


def test_attachment_priority_overtakes_queued_knowledge_base_work() -> None:
    async def scenario() -> list[str]:
        scheduler = WorkScheduler()
        order: list[str] = []
        gate = asyncio.Event()

        async def blocking() -> None:
            await gate.wait()
            order.append("running")

        async def job(name: str) -> None:
            order.append(name)

        running = asyncio.create_task(scheduler.run("docling", WorkPriority.CHAT, blocking, workspace_id=1))
        await asyncio.sleep(0)
        kb = asyncio.create_task(scheduler.run("docling", WorkPriority.KNOWLEDGE_BASE, lambda: job("kb"), workspace_id=1))
        attachment = asyncio.create_task(scheduler.run("docling", WorkPriority.ATTACHMENT, lambda: job("attachment"), workspace_id=1))
        await asyncio.sleep(0)
        gate.set()
        await asyncio.gather(running, kb, attachment)
        return order

    assert asyncio.run(scenario()) == ["running", "attachment", "kb"]


def test_cancel_workspace_cancels_not_started_jobs() -> None:
    async def scenario() -> bool:
        scheduler = WorkScheduler()
        gate = asyncio.Event()

        async def blocking() -> None:
            await gate.wait()

        running = asyncio.create_task(scheduler.run("embedding", WorkPriority.CHAT, blocking, workspace_id=5))
        await asyncio.sleep(0)
        queued = asyncio.create_task(scheduler.run(
            "embedding", WorkPriority.ATTACHMENT, lambda: blocking(), workspace_id=5, cancel_on_cleanup=True
        ))
        await asyncio.sleep(0)
        assert scheduler.cancel_workspace(5) >= 1
        gate.set()
        await running
        with pytest.raises(asyncio.CancelledError):
            await queued
        return True

    import pytest
    assert asyncio.run(scenario())
