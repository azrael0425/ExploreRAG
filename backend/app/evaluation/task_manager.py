"""Small in-process task manager for V1 evaluation runs.

One process and one concurrent run are intentional V1 constraints.  Queued or
running jobs are marked failed on restart instead of pretending they can resume.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from sqlalchemy import update

from app.core.database import AsyncSessionLocal
from app.evaluation.runner import execute_run
from app.models.evaluation import EvalRun

logger = logging.getLogger(__name__)


class EvaluationTaskManager:
    def __init__(self) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._tasks: dict[int, asyncio.Task[None]] = {}

    def enqueue(self, run_id: int) -> None:
        existing = self._tasks.get(run_id)
        if existing and not existing.done():
            return
        task = asyncio.create_task(self._run(run_id), name=f"evaluation-run-{run_id}")
        self._tasks[run_id] = task

    async def _run(self, run_id: int) -> None:
        async with self._semaphore:
            try:
                async with AsyncSessionLocal() as db:
                    await execute_run(db, run_id)
            except Exception:
                logger.exception("Evaluation run %s failed", run_id)
            finally:
                self._tasks.pop(run_id, None)


evaluation_task_manager = EvaluationTaskManager()


async def recover_interrupted_runs() -> int:
    async with AsyncSessionLocal() as db:
        statement = (
            update(EvalRun)
            .where(EvalRun.status.in_(["queued", "running"]))
            .values(
                status="failed",
                error_message="Application restarted before this evaluation run completed.",
                finished_at=datetime.utcnow(),
            )
        )
        result = await db.execute(statement)
        await db.commit()
        return int(result.rowcount or 0)
