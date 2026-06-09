"""Privacy-preserving local callbacks for LangChain orchestration timing."""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.core.config import settings

logger = logging.getLogger(__name__)


class LocalTimingCallback(BaseCallbackHandler):
    """Log run timing and error type without recording prompts or document data."""

    def __init__(self) -> None:
        self._started_at: dict[Any, float] = {}

    def _start(self, run_id: Any) -> None:
        self._started_at[run_id] = time.perf_counter()

    def _finish(self, run_id: Any, event: str, **kwargs: Any) -> None:
        started_at = self._started_at.pop(run_id, None)
        if started_at is None:
            return
        logger.debug(
            "LangChain local callback event=%s run_id=%s elapsed_ms=%s",
            event,
            run_id,
            int((time.perf_counter() - started_at) * 1000),
        )

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], *, run_id: Any, **kwargs: Any) -> None:
        self._start(run_id)

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, "chain_end")

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, f"chain_error:{type(error).__name__}")

    def on_retriever_start(self, serialized: dict[str, Any], query: str, *, run_id: Any, **kwargs: Any) -> None:
        self._start(run_id)

    def on_retriever_end(self, documents: list[Any], *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, "retriever_end")

    def on_retriever_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, f"retriever_error:{type(error).__name__}")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: Any, **kwargs: Any) -> None:
        self._start(run_id)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, "llm_end")

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._finish(run_id, f"llm_error:{type(error).__name__}")


def get_local_callbacks() -> list[LocalTimingCallback]:
    """Return callbacks only when explicitly enabled in local configuration."""
    return [LocalTimingCallback()] if settings.EXPLORERAG_LANGCHAIN_LOCAL_CALLBACKS else []
