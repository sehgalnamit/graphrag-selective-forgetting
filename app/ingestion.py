"""Ingestion pipeline: chunk -> embed -> extract entities/relations ->
write nodes/edges into the graph store and chunk vectors into the vector
store.
"""
from __future__ import annotations

from app.chunking import chunk_text
from app.config import Settings
from app.embeddings import Embedder
from app.entities import extract_entities_and_relations
from app.graph_store import GraphStore
from app.schemas import GraphEdge, IngestRequest, IngestResponse, MemoryNode, NodeKind
from app.time_control import clock
from app.utils import slugify
from app.vector_store import VectorStore


def ingest_document(
    request: IngestRequest,
    tenant_id: str,
    graph: GraphStore,
    vectors: VectorStore,
    embedder: Embedder,
    settings: Settings,
) -> IngestResponse:
    now = clock.now()
    chunks = chunk_text(request.text, settings.chunk_size_words, settings.chunk_overlap_words)
    if not chunks:
        return IngestResponse(document_id=request.document_id, chunk_ids=[], entity_ids=[], relation_count=0)

    vectors.ensure_collection(tenant_id, embedder.dimension)
    embedded_vectors = embedder.embed(chunks)

    chunk_ids: list[str] = []
    new_entity_ids: list[str] = []
    relation_count = 0

    for index, (chunk, vector) in enumerate(zip(chunks, embedded_vectors)):
        chunk_id = f"{request.document_id}::chunk::{index}"
        initial_decay = 1.0 if request.pinned else request.base_importance
        chunk_node = MemoryNode(
            id=chunk_id,
            tenant_id=tenant_id,
            kind=NodeKind.CHUNK,
            text=chunk,
            base_importance=request.base_importance,
            last_accessed_at=now,
            created_at=now,
            access_count=0,
            pinned=request.pinned,
            decay_score=initial_decay,
            metadata={**request.metadata, "document_id": request.document_id, "chunk_index": index},
        )
        graph.upsert_node(chunk_node)
        vectors.upsert(
            tenant_id,
            chunk_id,
            vector,
            payload={
                "text": chunk,
                "document_id": request.document_id,
                "chunk_index": index,
                "decay_score": initial_decay,
                "pinned": request.pinned,
            },
        )
        chunk_ids.append(chunk_id)

        extraction = extract_entities_and_relations(chunk)
        entity_id_by_name: dict[str, str] = {}
        for entity_name in extraction.entities:
            entity_id = f"entity::{slugify(entity_name)}"
            entity_id_by_name[entity_name] = entity_id
            if graph.get_node(tenant_id, entity_id) is None:
                entity_node = MemoryNode(
                    id=entity_id,
                    tenant_id=tenant_id,
                    kind=NodeKind.ENTITY,
                    text=entity_name,
                    base_importance=request.base_importance,
                    last_accessed_at=now,
                    created_at=now,
                    access_count=0,
                    pinned=False,
                    decay_score=request.base_importance,
                    metadata={"first_seen_document": request.document_id},
                )
                graph.upsert_node(entity_node)
                new_entity_ids.append(entity_id)
            graph.upsert_edge(
                GraphEdge(source_id=chunk_id, target_id=entity_id, relation="MENTIONS", tenant_id=tenant_id)
            )

        for relation in extraction.relations:
            source_id = entity_id_by_name.get(relation.source)
            target_id = entity_id_by_name.get(relation.target)
            if source_id and target_id:
                graph.upsert_edge(
                    GraphEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation.relation,
                        tenant_id=tenant_id,
                    )
                )
                relation_count += 1

    return IngestResponse(
        document_id=request.document_id,
        chunk_ids=chunk_ids,
        entity_ids=new_entity_ids,
        relation_count=relation_count,
    )
