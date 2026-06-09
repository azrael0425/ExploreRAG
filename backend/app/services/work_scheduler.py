"""Small in-process priority scheduler for local Docling/BGE resources.

The deployment currently runs one API worker, so an asyncio-backed scheduler is
enough.  The public ``run`` interface deliberately does not expose the queue
implementation: a future multi-worker deployment can replace it with Redis
without changing callers.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class WorkPriority(IntEnum):
    CHAT = 0
    ATTACHMENT = 1
    KNOWLEDGE_BASE = 2
    ENRICHMENT = 3


@dataclass(eq=False)
class _QueuedJob:
    workspace_id: int | None
    future: asyncio.Future[Any]
    factory: Callable[[], Awaitable[Any]]
    cancelled: bool = False
    started: bool = False
    cancel_on_cleanup: bool = False
    queued_at: float = 0.0


class WorkScheduler:
    """One priority queue per contended resource.

    A resource worker always selects the highest priority *not yet started*
    job.  This means a P1 chat attachment overtakes queued P2 KB jobs, while a
    Docling conversion already running is allowed to finish safely.
    """

    RESOURCES = ("docling", "embedding", "reranker", "llm_enrichment")

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.PriorityQueue] = {
            name: asyncio.PriorityQueue() for name in self.RESOURCES
        }
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._sequence = itertools.count()
        self._workspace_jobs: dict[int, set[_QueuedJob]] = {}

    async def run(
        self,
        resource: str,
        priority: WorkPriority,
        factory: Callable[[], Awaitable[Any]],
        *,
        workspace_id: int | None = None,
        cancel_on_cleanup: bool = False,
    ) -> Any:
        if resource not in self._queues:
            raise ValueError(f"Unknown scheduled resource: {resource}")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        job = _QueuedJob(
            workspace_id=workspace_id,
            future=future,
            factory=factory,
            cancel_on_cleanup=cancel_on_cleanup,
            queued_at=time.perf_counter(),
        )
        if workspace_id is not None:
            self._workspace_jobs.setdefault(workspace_id, set()).add(job)
        await self._queues[resource].put((int(priority), next(self._sequence), job))
        self._ensure_worker(resource)
        try:
            return await future
        finally:
            if workspace_id is not None:
                jobs = self._workspace_jobs.get(workspace_id)
                if jobs:
                    jobs.discard(job)
                    if not jobs:
                        self._workspace_jobs.pop(workspace_id, None)

    def cancel_workspace(self, workspace_id: int) -> int:
        """Invalidate queued (not started) attachment work for a workspace."""
        jobs = self._workspace_jobs.get(workspace_id, set())
        cancelled = 0
        for job in tuple(jobs):
            if job.cancel_on_cleanup and not job.started and not job.future.done():
                job.cancelled = True
                job.future.cancel()
                cancelled += 1
        if cancelled:
            logger.info("Cancelled %s queued attachment jobs for workspace %s", cancelled, workspace_id)
        return cancelled

    def _ensure_worker(self, resource: str) -> None:
        worker = self._workers.get(resource)
        if worker is None or worker.done():
            self._workers[resource] = asyncio.create_task(self._worker(resource))

    async def _worker(self, resource: str) -> None:
        queue = self._queues[resource]
        while True:
            _, _, job = await queue.get()
            try:
                if job.cancelled or job.future.cancelled():
                    continue
                job.started = True
                wait_ms = int((time.perf_counter() - job.queued_at) * 1000)
                run_started_at = time.perf_counter()
                try:
                    result = await job.factory()
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    raise
                except Exception as exc:
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
                finally:
                    run_ms = int((time.perf_counter() - run_started_at) * 1000)
                    log = logger.info if wait_ms >= 1000 or run_ms >= 5000 else logger.debug
                    log(
                        "Work scheduler resource=%s workspace=%s wait_ms=%s run_ms=%s",
                        resource,
                        job.workspace_id,
                        wait_ms,
                        run_ms,
                    )
            finally:
                queue.task_done()


_scheduler: WorkScheduler | None = None


def get_work_scheduler() -> WorkScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = WorkScheduler()
    return _scheduler
