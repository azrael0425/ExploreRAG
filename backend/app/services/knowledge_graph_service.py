"""
Knowledge Graph Service
========================

Per-workspace Knowledge Graph using LightRAG with configurable LLM + embeddings.
File-based storage (NetworkX graph + NanoVectorDB) — no extra Docker services.

Usage:
    kg = KnowledgeGraphService(workspace_id=1)
    await kg.ingest("markdown text from document...")
    result = await kg.query("What are the key themes?", mode="hybrid")
    await kg.cleanup()
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.llm import get_embedding_provider, get_llm_provider
from app.services.llm.types import LLMMessage
from app.services.models.parsed_document import KnowledgeGraphEvidence, KnowledgeGraphFact

logger = logging.getLogger(__name__)

_SOURCE_DOCUMENT_ID_RE = re.compile(r"kb:\d+:doc:(\d+)(?:-chunk-\d+)?")


def _normalize_entity_name(value: object) -> str:
    """Normalize display labels for conservative citation-to-node matching."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _graph_provenance(properties: dict | None) -> tuple[list[int], list[str]]:
    """Read LightRAG's merged source fields without exposing storage ids."""
    props = properties or {}
    raw_ids = str(props.get("source_id", ""))
    document_ids = list(dict.fromkeys(
        int(match) for match in _SOURCE_DOCUMENT_ID_RE.findall(raw_ids)
    ))
    raw_files = str(props.get("file_path", ""))
    source_files = list(dict.fromkeys(
        value.strip() for value in raw_files.split("<SEP>") if value.strip()
    ))
    return document_ids, source_files


# ---------------------------------------------------------------------------
# Provider-based adapters for LightRAG
# ---------------------------------------------------------------------------

async def _kg_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list] = None,
    keyword_extraction: bool = False,
    llm_mode: str = "cloud",
    **kwargs,
) -> str:
    """LightRAG-compatible LLM function using the configured provider."""
    provider = get_llm_provider(llm_mode)

    messages: list[LLMMessage] = []

    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(LLMMessage(role=role, content=content))

    messages.append(LLMMessage(role="user", content=prompt))

    return await provider.acomplete(
        messages, temperature=0.0, max_tokens=4096,
    )


async def _kg_embed(texts: list[str]) -> np.ndarray:
    """LightRAG-compatible embedding function using the configured provider."""
    provider = get_embedding_provider()
    return await provider.embed(texts)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class KnowledgeGraphService:
    """
    Per-workspace Knowledge Graph service backed by LightRAG.

    Storage: file-based (NetworkX for graph, NanoVectorDB for vectors).
    Each knowledge base gets its own working directory.
    """

    def __init__(
        self,
        workspace_id: int,
        kg_language: str | None = None,
        kg_entity_types: list[str] | None = None,
        llm_mode: str = "cloud",
    ):
        self.workspace_id = workspace_id
        self.working_dir = str(
            settings.BASE_DIR / "data" / "lightrag" / f"kb_{workspace_id}"
        )
        # Per-workspace overrides (fallback to global settings)
        self.kg_language = kg_language or settings.EXPLORERAG_KG_LANGUAGE
        self.kg_entity_types = kg_entity_types or settings.EXPLORERAG_KG_ENTITY_TYPES
        self.llm_mode = llm_mode
        self._rag = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def _get_rag(self):
        """Lazy-initialize LightRAG instance."""
        if self._rag is not None and self._initialized:
            return self._rag

        async with self._initialization_lock:
            if self._rag is not None and self._initialized:
                return self._rag
            return await self._initialize_rag()

    async def _initialize_rag(self):
        """Build and initialize one LightRAG instance for this service."""
        if self._rag is not None and self._initialized:
            return self._rag

        from lightrag import LightRAG
        from lightrag.utils import wrap_embedding_func_with_attrs
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self.working_dir, exist_ok=True)

        # Dynamic embedding dimension from the configured provider
        emb_provider = get_embedding_provider()
        embedding_dim = emb_provider.get_dimension()

        # Detect dimension mismatch when switching providers
        dim_marker = Path(self.working_dir) / ".embedding_dim"
        if dim_marker.exists():
            prev_dim = int(dim_marker.read_text().strip())
            if prev_dim != embedding_dim:
                logger.warning(
                    f"Embedding dimension changed ({prev_dim} → {embedding_dim}) "
                    f"for workspace {self.workspace_id}. Clearing KG data for rebuild."
                )
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
        dim_marker.write_text(str(embedding_dim))

        @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await _kg_embed(texts)

        async def llm_complete(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: Optional[list] = None,
            keyword_extraction: bool = False,
            **kwargs,
        ) -> str:
            return await _kg_llm_complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                keyword_extraction=keyword_extraction,
                llm_mode=self.llm_mode,
                **kwargs,
            )

        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_complete,
            embedding_func=embedding_func,
            chunk_token_size=settings.EXPLORERAG_KG_CHUNK_TOKEN_SIZE,
            # Keep each graph vector batch modest and serialize calls even on
            # CUDA so ingestion cannot monopolize the shared BGE-M3 instance.
            embedding_batch_num=settings.EXPLORERAG_KG_EMBEDDING_BATCH_SIZE,
            embedding_func_max_async=settings.EXPLORERAG_KG_EMBEDDING_MAX_ASYNC,
            default_embedding_timeout=settings.EXPLORERAG_KG_EMBEDDING_TIMEOUT,
            enable_llm_cache=True,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            addon_params={
                "language": self.kg_language,
                "entity_types": self.kg_entity_types,
            },
        )

        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        self._initialized = True

        logger.info(
            f"LightRAG initialized for workspace {self.workspace_id} "
            f"(embedding_dim={embedding_dim})"
        )
        return self._rag

    async def ingest(
        self,
        markdown_content: str,
        *,
        kg_document_id: str | None = None,
        source_file: str | None = None,
    ) -> None:
        """
        Ingest markdown content into the knowledge graph.
        LightRAG extracts entities and relationships automatically.
        """
        rag = await self._get_rag()

        if not markdown_content.strip():
            logger.warning(f"Empty content for workspace {self.workspace_id}, skipping KG ingest")
            return

        try:
            insert_kwargs: dict[str, list[str]] = {}
            if kg_document_id:
                insert_kwargs["ids"] = [kg_document_id]
            if source_file:
                insert_kwargs["file_paths"] = [source_file]
            await rag.ainsert(markdown_content, **insert_kwargs)
            logger.info(
                "KG ingested %s chars for workspace %s document=%s",
                len(markdown_content), self.workspace_id, kg_document_id or "auto",
            )

            # Check if entities were actually extracted
            try:
                all_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                if not all_nodes:
                    from app.core.config import settings
                    provider = get_llm_provider(self.llm_mode)
                    model = getattr(provider, "model_name", settings.LLM_MODEL_FAST)
                    logger.warning(
                        f"KG extraction produced 0 entities for workspace {self.workspace_id}. "
                        f"Model '{model}' may not support LightRAG's entity extraction format. "
                        f"Consider using a larger model (e.g. qwen3:14b, gemma3:12b) for KG."
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"KG ingest failed for workspace {self.workspace_id}: {e}")
            raise

    async def delete_document(self, kg_document_id: str) -> None:
        """Delete one source document while preserving shared graph facts."""
        rag = await self._get_rag()
        try:
            await rag.adelete_by_doc_id(kg_document_id)
            logger.info("KG deleted document %s for workspace %s", kg_document_id, self.workspace_id)
        except Exception as exc:
            logger.error("KG deletion failed for document %s: %s", kg_document_id, exc)
            raise

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> str:
        """
        Query the knowledge graph.

        Args:
            question: Natural language question
            mode: LightRAG mode: hybrid, local, or global
            top_k: Number of results

        Returns:
            LightRAG response text with KG-augmented answer
        """
        from lightrag import QueryParam

        if mode not in {"hybrid", "local", "global"}:
            raise ValueError(f"Unsupported LightRAG query mode: {mode}")

        rag = await self._get_rag()

        try:
            result = await asyncio.wait_for(
                rag.aquery(
                    question,
                    param=QueryParam(mode=mode, top_k=top_k),
                ),
                timeout=settings.EXPLORERAG_KG_QUERY_TIMEOUT,
            )
            return result or ""
        except asyncio.TimeoutError:
            logger.warning(
                f"KG query timed out after {settings.EXPLORERAG_KG_QUERY_TIMEOUT}s "
                f"for workspace {self.workspace_id}"
            )
            return ""
        except Exception as e:
            logger.error(f"KG query failed for workspace {self.workspace_id}: {e}")
            return ""

    async def retrieve_evidence(
        self,
        question: str,
        mode: str = "hybrid",
    ) -> KnowledgeGraphEvidence:
        """Retrieve structured LightRAG evidence without asking it for an answer.

        LightRAG's ``aquery`` produces a separate model-generated response,
        which is unsuitable as an uncited side channel for this application.
        ``aquery_data`` instead returns the entities, relationships and chunks
        already selected from graph storage.  The primary chat model remains
        the only component that writes the final answer.
        """
        from lightrag import QueryParam

        if mode not in {"hybrid", "local", "global"}:
            raise ValueError(f"Unsupported LightRAG evidence mode: {mode}")

        rag = await self._get_rag()
        try:
            raw = await asyncio.wait_for(
                rag.aquery_data(
                    question,
                    param=QueryParam(
                        mode=mode,
                        top_k=settings.EXPLORERAG_KG_AUGMENTATION_TOP_K,
                        chunk_top_k=settings.EXPLORERAG_KG_AUGMENTATION_CHUNK_TOP_K,
                        enable_rerank=False,
                    ),
                ),
                timeout=settings.EXPLORERAG_KG_AUGMENTATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "LightRAG evidence timed out after %ss for workspace %s",
                settings.EXPLORERAG_KG_AUGMENTATION_TIMEOUT_SECONDS,
                self.workspace_id,
            )
            return KnowledgeGraphEvidence()
        except Exception as exc:
            logger.warning(
                "LightRAG evidence retrieval failed for workspace %s: %s",
                self.workspace_id,
                exc,
            )
            return KnowledgeGraphEvidence()

        if not isinstance(raw, dict) or raw.get("status") != "success":
            logger.info(
                "LightRAG evidence returned no usable result for workspace %s: %s",
                self.workspace_id,
                raw.get("message", "unknown") if isinstance(raw, dict) else "invalid response",
            )
            return KnowledgeGraphEvidence()

        data = raw.get("data")
        if not isinstance(data, dict):
            return KnowledgeGraphEvidence()

        entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        relationships = data.get("relationships") if isinstance(data.get("relationships"), list) else []
        chunks = data.get("chunks") if isinstance(data.get("chunks"), list) else []

        parts: list[str] = []
        entity_names: list[str] = []
        source_files: list[str] = []
        source_document_ids: list[int] = []
        seen_entities: set[str] = set()
        seen_files: set[str] = set()
        seen_document_ids: set[int] = set()
        entity_facts: list[KnowledgeGraphFact] = []
        graph_facts: list[KnowledgeGraphFact] = []

        def unique_files(value: object) -> list[str]:
            if not isinstance(value, str):
                return []
            return list(dict.fromkeys(
                item.strip() for item in value.split("<SEP>") if item.strip()
            ))

        def unique_document_ids(value: object) -> list[int]:
            return list(dict.fromkeys(
                int(document_id)
                for document_id in _SOURCE_DOCUMENT_ID_RE.findall(str(value or ""))
            ))

        def add_file(value: object) -> None:
            if not isinstance(value, str):
                return
            normalized = value.strip()
            if normalized and normalized not in seen_files:
                seen_files.add(normalized)
                source_files.append(normalized)

        def add_document_ids(value: object) -> None:
            for document_id in _SOURCE_DOCUMENT_ID_RE.findall(str(value or "")):
                numeric_id = int(document_id)
                if numeric_id not in seen_document_ids:
                    seen_document_ids.add(numeric_id)
                    source_document_ids.append(numeric_id)

        # Bounded formatting prevents an unusually dense graph from consuming
        # the entire generation context.  Preserve LightRAG ordering because it
        # already reflects its graph relevance ranking.
        if entities:
            parts.append("Entities:")
            for entity in entities[:10]:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("entity_name", "")).strip()
                if not name:
                    continue
                entity_type = str(entity.get("entity_type", "")).strip()
                description = str(entity.get("description", "")).strip()[:350]
                if name not in seen_entities:
                    seen_entities.add(name)
                    entity_names.append(name)
                add_file(entity.get("file_path"))
                add_document_ids(entity.get("source_id"))
                type_suffix = f" [{entity_type}]" if entity_type else ""
                entity_line = f"{name}{type_suffix}: {description}" if description else f"{name}{type_suffix}"
                parts.append(f"- {entity_line}")
                entity_facts.append(KnowledgeGraphFact(
                    content=entity_line,
                    entity_names=[name],
                    source_files=unique_files(entity.get("file_path")),
                    source_document_ids=unique_document_ids(entity.get("source_id")),
                ))

        if relationships:
            if parts:
                parts.append("")
            parts.append("Relationships:")
            for relation in relationships[:15]:
                if not isinstance(relation, dict):
                    continue
                source = str(relation.get("src_id", "")).strip()
                target = str(relation.get("tgt_id", "")).strip()
                if not source and not target:
                    continue
                description = str(relation.get("description", "")).strip()[:300]
                add_file(relation.get("file_path"))
                add_document_ids(relation.get("source_id"))
                edge = f"{source} -> {target}".strip(" -")
                relation_line = f"{edge}: {description}" if description else edge
                parts.append(f"- {relation_line}")
                if len(graph_facts) < 8:
                    graph_facts.append(KnowledgeGraphFact(
                        content=relation_line,
                        entity_names=list(dict.fromkeys(
                            value for value in (source, target) if value
                        )),
                        source_files=unique_files(relation.get("file_path")),
                        source_document_ids=unique_document_ids(relation.get("source_id")),
                    ))

        if chunks:
            if parts:
                parts.append("")
            parts.append("Graph-retrieved document excerpts:")
            for chunk in chunks[:5]:
                if not isinstance(chunk, dict):
                    continue
                content = str(chunk.get("content", "")).strip()
                if not content:
                    continue
                add_file(chunk.get("file_path"))
                add_document_ids(chunk.get("source_id"))
                file_label = str(chunk.get("file_path", "")).strip()
                prefix = f"[{file_label}] " if file_label else ""
                parts.append(prefix + content[:1200])

        content = "\n".join(parts).strip()
        if len(content) > settings.EXPLORERAG_KG_AUGMENTATION_MAX_CHARS:
            content = content[:settings.EXPLORERAG_KG_AUGMENTATION_MAX_CHARS].rstrip() + "\n..."

        evidence = KnowledgeGraphEvidence(
            content=content,
            entity_names=entity_names,
            source_files=source_files,
            source_document_ids=source_document_ids,
            facts=graph_facts or entity_facts[:8],
            entity_count=len(entities),
            relationship_count=len(relationships),
            chunk_count=len(chunks),
        )
        logger.info(
            "LightRAG evidence workspace=%s entities=%s relationships=%s chunks=%s emitted_chars=%s",
            self.workspace_id,
            evidence.entity_count,
            evidence.relationship_count,
            evidence.chunk_count,
            len(evidence.content),
        )
        return evidence

    async def cleanup(self) -> None:
        """Finalize storages on shutdown."""
        async with self._initialization_lock:
            rag = self._rag
            self._rag = None
            self._initialized = False
        if rag:
            try:
                await rag.finalize_storages()
                logger.info(f"KG storages finalized for workspace {self.workspace_id}")
            except Exception as e:
                logger.warning(f"KG cleanup failed for workspace {self.workspace_id}: {e}")

    def delete_project_data(self) -> None:
        """Delete all KG data for this knowledge base."""
        path = Path(self.working_dir)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted KG data for workspace {self.workspace_id}")
        self._rag = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Knowledge Graph exploration (Phase 9)
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        search: str | None = None,
        entity_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all entities in the knowledge graph.

        Returns list of dicts with: name, entity_type, description, degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
        except Exception as e:
            logger.error(f"Failed to get KG nodes for workspace {self.workspace_id}: {e}")
            return []

        entities = []
        for node in all_nodes:
            node_id = node.get("id", "")
            etype = node.get("entity_type", "Unknown")
            desc = node.get("description", "")

            # Filters
            if entity_type and etype.lower() != entity_type.lower():
                continue
            if search and search.lower() not in node_id.lower():
                continue

            # Get degree (number of relationships)
            try:
                degree = await storage.node_degree(node_id)
            except Exception:
                degree = 0

            entities.append({
                "name": node_id,
                "entity_type": etype,
                "description": desc,
                "degree": degree,
            })

        # Sort by degree descending
        entities.sort(key=lambda e: e["degree"], reverse=True)

        return entities[offset:offset + limit]

    async def get_relationships(
        self,
        entity_name: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        List relationships in the knowledge graph.

        If entity_name is provided, returns only relationships involving that entity.
        Returns list of dicts with: source, target, description, keywords, weight.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG edges for workspace {self.workspace_id}: {e}")
            return []

        relationships = []
        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")

            if entity_name:
                if entity_name.lower() not in (src.lower(), tgt.lower()):
                    continue

            relationships.append({
                "source": src,
                "target": tgt,
                "description": edge.get("description", ""),
                "keywords": edge.get("keywords", ""),
                "weight": float(edge.get("weight", 1.0)),
            })

        return relationships[:limit]

    async def get_graph_data(
        self,
        center_entity: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 250,
    ) -> dict:
        """
        Export graph data for frontend visualization.

        Returns {nodes: [...], edges: [...], is_truncated: bool}.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            label = center_entity if center_entity else "*"
            kg = await storage.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Failed to get KG graph for workspace {self.workspace_id}: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

        # ``node_degree`` is a storage round-trip. Calling it once per visible
        # node makes a large overview take tens of seconds on the
        # file-backed NetworkX store. Build the degree table in one pass over
        # the graph edges instead, then fall back to the old call only when the
        # storage adapter cannot export its edges.
        degree_by_node: dict[str, int] | None = None
        try:
            all_edges = await storage.get_all_edges()
            degree_by_node = {}
            for edge in all_edges:
                source = str(edge.get("source", ""))
                target = str(edge.get("target", ""))
                if source:
                    degree_by_node[source] = degree_by_node.get(source, 0) + 1
                if target:
                    degree_by_node[target] = degree_by_node.get(target, 0) + 1
        except Exception as exc:
            logger.warning(
                "Failed to batch graph degrees for workspace %s: %s",
                self.workspace_id,
                exc,
            )

        nodes = []
        for n in kg.nodes:
            props = n.properties if hasattr(n, "properties") else {}
            if degree_by_node is not None:
                degree = degree_by_node.get(n.id, 0)
            else:
                try:
                    degree = await storage.node_degree(n.id)
                except Exception:
                    degree = 0
            document_ids, source_files = _graph_provenance(props)
            nodes.append({
                "id": n.id,
                "label": n.id,
                "entity_type": props.get("entity_type", "Unknown"),
                "description": props.get("description", ""),
                "degree": degree,
                "source_document_ids": document_ids,
                "source_files": source_files,
            })

        edges = []
        for e in kg.edges:
            props = e.properties if hasattr(e, "properties") else {}
            document_ids, source_files = _graph_provenance(props)
            edges.append({
                "source": e.source,
                "target": e.target,
                "label": props.get("description", "")[:80],
                "weight": float(props.get("weight", 1.0)),
                "source_document_ids": document_ids,
                "source_files": source_files,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "is_truncated": kg.is_truncated if hasattr(kg, "is_truncated") else False,
        }

    async def get_focus_graph_data(
        self,
        entity_names: list[str],
        *,
        document_ids: list[int] | None = None,
        max_depth: int = 1,
        max_nodes: int = 80,
    ) -> dict:
        """Build a bounded citation subgraph without ever evicting seed nodes.

        The overview export is intentionally degree/traversal bounded.  That is
        appropriate for exploration but wrong for citation navigation because
        a cited degree-one entity can disappear.  This path resolves requested
        names against the complete stored graph, inserts every resolved seed,
        then spends the remaining budget on ranked neighbours.
        """
        requested_entities: list[str] = []
        seen_requests: set[str] = set()
        for value in entity_names:
            label = str(value or "").strip()
            normalized = _normalize_entity_name(label)
            if label and normalized and normalized not in seen_requests:
                seen_requests.add(normalized)
                requested_entities.append(label)

        empty = {
            "nodes": [],
            "edges": [],
            "is_truncated": False,
            "requested_entities": requested_entities,
            "matched_entities": [],
            "missing_entities": requested_entities,
            "selection_mode": "citation_focus",
        }
        if not requested_entities:
            return empty

        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph
        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as exc:
            logger.error(
                "Failed to build focused KG graph for workspace %s: %s",
                self.workspace_id,
                exc,
            )
            return empty

        node_by_id = {
            str(node.get("id", "")): node
            for node in all_nodes
            if isinstance(node, dict) and str(node.get("id", "")).strip()
        }
        normalized_nodes: dict[str, list[str]] = defaultdict(list)
        for node_id in node_by_id:
            normalized_nodes[_normalize_entity_name(node_id)].append(node_id)

        degree_by_node: dict[str, int] = defaultdict(int)
        adjacency: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        usable_edges: list[dict] = []
        for edge in all_edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if source not in node_by_id or target not in node_by_id:
                continue
            usable_edges.append(edge)
            degree_by_node[source] += 1
            degree_by_node[target] += 1
            adjacency[source].append((target, edge))
            adjacency[target].append((source, edge))

        matched_entities: list[str] = []
        missing_entities: list[str] = []
        for requested in requested_entities:
            normalized = _normalize_entity_name(requested)
            candidates = normalized_nodes.get(normalized, [])
            if not candidates:
                # Legacy answers sometimes preserve a short alias rather than
                # the canonical node label.  Only use a conservative substring
                # fallback and prefer the closest, highest-degree candidate.
                min_length = 2 if any(ord(char) > 127 for char in requested) else 4
                if len(normalized) >= min_length:
                    candidates = [
                        node_id
                        for node_id in node_by_id
                        if normalized in _normalize_entity_name(node_id)
                        or _normalize_entity_name(node_id) in normalized
                    ]
            if candidates:
                candidates.sort(
                    key=lambda node_id: (
                        abs(len(_normalize_entity_name(node_id)) - len(normalized)),
                        -degree_by_node.get(node_id, 0),
                        node_id.casefold(),
                    )
                )
                canonical = candidates[0]
                if canonical not in matched_entities:
                    matched_entities.append(canonical)
            else:
                missing_entities.append(requested)

        selected_order = list(matched_entities)
        selected = set(selected_order)
        frontier = set(selected_order)
        requested_document_ids = set(document_ids or [])
        is_truncated = False

        # Seeds may legitimately exceed the requested budget; citation seeds
        # are the one class of node this endpoint is never allowed to drop.
        effective_budget = max(max_nodes, len(selected_order))
        for _depth in range(max_depth):
            candidate_scores: dict[str, tuple[int, float, int]] = {}
            for current in frontier:
                for neighbor, edge in adjacency.get(current, []):
                    if neighbor in selected:
                        continue
                    edge_document_ids, _ = _graph_provenance(edge)
                    provenance_overlap = len(requested_document_ids.intersection(edge_document_ids))
                    try:
                        weight = float(edge.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    score = (provenance_overlap, weight, degree_by_node.get(neighbor, 0))
                    if score > candidate_scores.get(neighbor, (-1, -1.0, -1)):
                        candidate_scores[neighbor] = score

            ranked = sorted(
                candidate_scores,
                key=lambda node_id: (
                    -candidate_scores[node_id][0],
                    -candidate_scores[node_id][1],
                    -candidate_scores[node_id][2],
                    node_id.casefold(),
                ),
            )
            remaining = effective_budget - len(selected_order)
            if remaining <= 0:
                is_truncated = is_truncated or bool(ranked)
                break
            if len(ranked) > remaining:
                is_truncated = True
            added = ranked[:remaining]
            if not added:
                break
            selected_order.extend(added)
            selected.update(added)
            frontier = set(added)

        nodes: list[dict] = []
        for node_id in selected_order:
            properties = node_by_id[node_id]
            source_document_ids, source_files = _graph_provenance(properties)
            nodes.append({
                "id": node_id,
                "label": node_id,
                "entity_type": properties.get("entity_type", "Unknown"),
                "description": properties.get("description", ""),
                "degree": degree_by_node.get(node_id, 0),
                "source_document_ids": source_document_ids,
                "source_files": source_files,
            })

        edges: list[dict] = []
        for edge in usable_edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            if source not in selected or target not in selected:
                continue
            source_document_ids, source_files = _graph_provenance(edge)
            try:
                weight = float(edge.get("weight", 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            edges.append({
                "source": source,
                "target": target,
                "label": str(edge.get("description", ""))[:80],
                "weight": weight,
                "source_document_ids": source_document_ids,
                "source_files": source_files,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "is_truncated": is_truncated,
            "requested_entities": requested_entities,
            "matched_entities": matched_entities,
            "missing_entities": missing_entities,
            "selection_mode": "citation_focus",
        }

    async def get_relevant_context(
        self,
        question: str,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
        Build RAG context from raw KG data (no LLM generation).

        Instead of calling LightRAG's aquery() which uses LLM to generate
        a narrative (and can hallucinate), this method:
          1. Tokenizes the question into keywords
          2. Finds entities whose names match any keyword
          3. Gets relationships connecting those entities
          4. Formats everything as structured factual text

        Returns:
            Structured string of entities + relationships, or "" if nothing found.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get raw KG data for workspace {self.workspace_id}: {e}")
            return ""

        if not all_nodes:
            return ""

        # -- 1. Extract keywords from question --
        # Simple but effective: split, lowercase, filter short words
        raw_tokens = question.lower().split()
        # Also handle hyphenated/versioned tokens like "deepseek-v3.2"
        keywords = set()
        for token in raw_tokens:
            # Remove punctuation at edges
            cleaned = token.strip(".,?!:;\"'()[]{}").lower()
            if len(cleaned) >= 2:
                keywords.add(cleaned)

        if not keywords:
            return ""

        # -- 2. Find matching entities --
        matched_entity_names: set[str] = set()
        entity_info: dict[str, dict] = {}  # name → {type, description}

        for node in all_nodes:
            node_id = node.get("id", "")
            node_lower = node_id.lower()

            # Check if any keyword is a substring of entity name OR vice versa
            matched = False
            for kw in keywords:
                if kw in node_lower or node_lower in kw:
                    matched = True
                    break
                # Also check multi-word keywords (e.g., "deepseek" matches "DEEPSEEK-V3.2")
                for part in node_lower.split("-"):
                    if kw in part or part in kw:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                matched_entity_names.add(node_id)
                entity_info[node_id] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names and len(all_nodes) <= 50:
            # Small graph: include top entities by default
            for node in all_nodes[:10]:
                nid = node.get("id", "")
                matched_entity_names.add(nid)
                entity_info[nid] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names:
            return ""

        # Limit entities
        matched_list = list(matched_entity_names)[:max_entities]

        # -- 3. Find relationships involving matched entities --
        relevant_rels: list[dict] = []
        matched_lower = {n.lower() for n in matched_list}

        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src.lower() in matched_lower or tgt.lower() in matched_lower:
                relevant_rels.append({
                    "source": src,
                    "target": tgt,
                    "description": edge.get("description", ""),
                    "keywords": edge.get("keywords", ""),
                })
                # Also add connected entities we might have missed
                if src not in entity_info:
                    # Find node info
                    for n in all_nodes:
                        if n.get("id", "") == src:
                            entity_info[src] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break
                if tgt not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == tgt:
                            entity_info[tgt] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break

            if len(relevant_rels) >= max_relationships:
                break

        # -- 4. Format as structured text --
        parts: list[str] = []

        # Entities section
        if matched_list:
            parts.append("Entities found in documents:")
            for name in matched_list:
                info = entity_info.get(name, {})
                etype = info.get("entity_type", "")
                desc = info.get("description", "")
                # Truncate long descriptions
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                type_str = f" [{etype}]" if etype and etype != "Unknown" else ""
                if desc:
                    parts.append(f"- {name}{type_str}: {desc}")
                else:
                    parts.append(f"- {name}{type_str}")

        # Relationships section
        if relevant_rels:
            parts.append("")
            parts.append("Relationships:")
            for rel in relevant_rels:
                desc = rel["description"]
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                if desc:
                    parts.append(f"- {rel['source']} → {rel['target']}: {desc}")
                else:
                    parts.append(f"- {rel['source']} → {rel['target']}")

        result = "\n".join(parts)
        logger.info(
            f"KG raw context: {len(matched_list)} entities, "
            f"{len(relevant_rels)} relationships for workspace {self.workspace_id}"
        )
        return result


_service_cache: dict[int, KnowledgeGraphService] = {}
_service_cache_lock = threading.Lock()


def get_knowledge_graph_service(
    workspace_id: int,
    kg_language: str | None = None,
    kg_entity_types: list[str] | None = None,
    llm_mode: str | None = None,
) -> KnowledgeGraphService:
    """Return the process-wide LightRAG service for a workspace."""
    effective_language = kg_language or settings.EXPLORERAG_KG_LANGUAGE
    effective_types = kg_entity_types or settings.EXPLORERAG_KG_ENTITY_TYPES
    effective_llm_mode = llm_mode or "cloud"
    stale_service: KnowledgeGraphService | None = None
    with _service_cache_lock:
        existing = _service_cache.get(workspace_id)
        if existing is not None:
            explicit_override = (
                kg_language is not None
                or kg_entity_types is not None
                or llm_mode is not None
            )
            if not explicit_override or (
                existing.kg_language == effective_language
                and existing.kg_entity_types == effective_types
                and existing.llm_mode == effective_llm_mode
            ):
                return existing
            # An explicit workspace configuration is authoritative. This can
            # occur when a read initialized the default service before the
            # first document ingest loaded workspace-specific settings.
            logger.info(
                "Replacing uninitialized/default KG cache entry for workspace %s after config change",
                workspace_id,
            )
            stale_service = existing

        service = KnowledgeGraphService(
            workspace_id=workspace_id,
            kg_language=effective_language,
            kg_entity_types=effective_types,
            llm_mode=effective_llm_mode,
        )
        _service_cache[workspace_id] = service

    if stale_service is not None:
        try:
            asyncio.get_running_loop().create_task(stale_service.cleanup())
        except RuntimeError:
            # Construction outside an event loop is limited to tests/startup;
            # the stale object will be released by normal garbage collection.
            pass
    return service


async def evict_knowledge_graph_service(workspace_id: int) -> None:
    """Remove and finalize one workspace service after config/lifecycle changes."""
    with _service_cache_lock:
        service = _service_cache.pop(workspace_id, None)
    if service is not None:
        await service.cleanup()


async def delete_knowledge_graph_workspace(workspace_id: int) -> None:
    """Evict cached storage handles, then delete the workspace graph files."""
    with _service_cache_lock:
        service = _service_cache.pop(workspace_id, None)
    if service is None:
        service = KnowledgeGraphService(workspace_id)
    await service.cleanup()
    service.delete_project_data()


async def cleanup_knowledge_graph_service_cache() -> None:
    """Finalize all cached LightRAG instances during application shutdown."""
    with _service_cache_lock:
        services = list(_service_cache.values())
        _service_cache.clear()
    if services:
        await asyncio.gather(*(service.cleanup() for service in services), return_exceptions=True)


def clear_knowledge_graph_service_cache_for_tests() -> None:
    """Drop uninitialized test entries without touching filesystem storage."""
    with _service_cache_lock:
        _service_cache.clear()
