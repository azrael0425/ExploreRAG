"""Citation identifiers shared by LangChain retrieval and attachments."""
from __future__ import annotations

import random
import string

_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def generate_citation_id(existing: set[str]) -> str:
    """Return a unique four-character ID containing at least one letter."""

    while True:
        citation_id = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(char.isalpha() for char in citation_id) and citation_id not in existing:
            return citation_id


def source_label(prefix: str, existing: set[str]) -> str:
    """Return a route-qualified source label such as ``KB-a3x9``."""

    return f"{prefix}-{generate_citation_id(existing)}"
