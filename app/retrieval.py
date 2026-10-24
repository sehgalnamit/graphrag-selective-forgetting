"""Hybrid retrieval: Qdrant vector similarity + Neo4j multi-hop graph
traversal, merged into one context payload.

Every chunk returned by the vector search is treated as an "access", which
reinforces its memory (increments access_count / refreshes last_accessed_at)
per the decay formula in ``decay_engine/formula.py`` - frequently-retrieved
memories decay slower, mirroring spaced-repetition reinforcement.
"""
from __future__ import annotations

from app.config import Settings
from app.embeddings import Embedder
from app.generation import generate_answer
from app.graph_store import GraphStore
from app.schemas import NodeKind, QueryResponse, RelatedEntity, RetrievedChunk
from app.time_control import clock
from app.vector_store import VectorStore


def hybrid_retrieve(
    query: str,
    tenant_id: str,
    graph: GraphStore,
    vectors: VectorStore,
    embedder: Embedder,
    settings: Settings,
    top_k: int | None = None,
    graph_hops: int | None = None,
) -> QueryResponse:
    top_k = top_k or settings.retrieval_top_k
    graph_hops = settings.retrieval_graph_hops if graph_hops is None else graph_hops
    now = clock.now()

    [query_vector] = embedder.embed([query])
    scored_chunks = vectors.search(tenant_id, query_vector, top_k)

    retrieved: list[RetrievedChunk] = []
    chunk_ids: list[str] = []
    for scored in scored_chunks:
        chunk_ids.append(scored.chunk_id)
        node = graph.record_access(tenant_id, scored.chunk_id, now)
        decay_score = node.decay_score if node is not None else float(scored.payload.get("decay_score", 0.0))
        retrieved.append(
            RetrievedChunk(
                chunk_id=scored.chunk_id,
                text=str(scored.payload.get("text", "")),
                score=scored.score,
                decay_score=decay_score,
            )
        )

    hop_results = graph.multi_hop(tenant_id, chunk_ids, graph_hops)
    related_entities = [
        RelatedEntity(entity_id=hop.node.id, name=hop.node.text, hops=hop.hops, relation_path=hop.relation_path)
        for hop in hop_results
        if hop.node.kind == NodeKind.ENTITY
    ]

    answer = generate_answer(query, retrieved, related_entities)
    return QueryResponse(
        query=query,
        answer=answer,
        chunks=retrieved,
        related_entities=related_entities,
    )
