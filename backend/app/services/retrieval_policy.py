"""One policy for deciding whether a request may use workspace-wide KG data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.core.config import settings


RetrievalMode = Literal["hybrid", "vector_only", "local", "global"]


@dataclass(frozen=True)
class RetrievalPolicy:
    mode: RetrievalMode
    lightrag_enabled: bool
    reason: str


def resolve_retrieval_policy(
    requested_mode: RetrievalMode,
    *,
    workspace_lightrag_enabled: bool,
    scoped: bool,
) -> RetrievalPolicy:
    """Return the safe effective retrieval mode for a workspace request.

    LightRAG storage is currently scoped to a complete workspace.  It must not
    leak graph facts into a request narrowed to selected documents or metadata.
    This helper is used by HTTP, streaming, LangChain and evaluation paths so
    they cannot silently drift into different behaviour.
    """
    if requested_mode == "vector_only":
        return RetrievalPolicy("vector_only", False, "requested_vector_only")
    if not settings.EXPLORERAG_ENABLE_KG:
        return RetrievalPolicy("vector_only", False, "disabled_by_server")
    if not workspace_lightrag_enabled:
        return RetrievalPolicy("vector_only", False, "disabled_by_workspace")
    if scoped:
        return RetrievalPolicy("vector_only", False, "scoped_query")
    return RetrievalPolicy(requested_mode, True, "enabled")
