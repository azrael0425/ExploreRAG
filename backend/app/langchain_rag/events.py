"""Domain events emitted by the LangChain chat service before SSE formatting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
    """A transport-neutral counterpart to one existing client SSE event."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"event": self.event, "data": self.data}
