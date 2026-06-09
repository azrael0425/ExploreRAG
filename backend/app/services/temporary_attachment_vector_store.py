"""Chroma namespace for ephemeral chat attachments only.

This intentionally does not subclass or instantiate ``VectorStore``.  The
different collection prefix prevents an implementation mistake from silently
placing temporary chunks in a durable ``kb_*`` collection.
"""
from __future__ import annotations

import logging
from typing import Sequence

from app.services.vector_store import get_chroma_client

logger = logging.getLogger(__name__)


class TemporaryAttachmentVectorStore:
    COLLECTION_PREFIX = "tmp_chat_ws_"

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.collection_name = f"{self.COLLECTION_PREFIX}{workspace_id}"
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = get_chroma_client().get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "explorerag_scope": "chat_attachment"},
            )
        return self._collection

    def add_documents(
        self,
        ids: Sequence[str],
        embeddings: Sequence[list[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        self.collection.upsert(
            ids=list(ids), embeddings=list(embeddings), documents=list(documents), metadatas=list(metadatas)
        )

    def query(self, query_embedding: list[float], attachment_ids: Sequence[str], n_results: int) -> dict:
        if not attachment_ids:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
        # Retrieval must never call get_or_create: a late in-flight request
        # after cleanup is allowed to return no data, but must not resurrect a
        # deleted temporary collection.
        try:
            collection = self._collection or get_chroma_client().get_collection(self.collection_name)
            self._collection = collection
        except Exception:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"attachment_id": {"$in": list(attachment_ids)}},
            include=["documents", "metadatas", "distances"],
        )
        return {
            "ids": results["ids"][0] if results.get("ids") else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
        }

    def delete_collection(self) -> None:
        try:
            get_chroma_client().delete_collection(self.collection_name)
        except Exception as exc:
            # Chroma returns an error when there was never an indexed attachment;
            # deletion remains idempotent from the caller's perspective.
            logger.info("Temporary collection cleanup for %s: %s", self.collection_name, exc)
        finally:
            self._collection = None

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0
