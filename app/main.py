"""FastAPI application: document ingestion + hybrid retrieval, plus a few
demo/admin endpoints (time-warp, GDPR purge, node introspection).
"""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from app.audit_log_client import AuditLogClient
from app.config import Settings
from app.dependencies import (
    get_audit_log,
    get_embedder_dep,
    get_graph_store,
    get_settings_dep,
    get_vector_store,
)
from app.embeddings import Embedder
from app.graph_store import GraphStore
from app.ingestion import ingest_document
from app.retrieval import hybrid_retrieve
from app.schemas import (
    IngestRequest,
    IngestResponse,
    MemoryNode,
    PurgeRequest,
    PurgeResponse,
    QueryRequest,
    QueryResponse,
    TimeWarpRequest,
    TimeWarpResponse,
)
from app.time_control import clock
from app.vector_store import VectorStore
from decay_engine.audit_log import AuditAction
from decay_engine.purge import purge_node
from evaluator.jury import build_relevance_jury

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.main")

app = FastAPI(
    title="GraphRAG Selective Forgetting",
    description=(
        "Enterprise knowledge retrieval featuring neuro-inspired memory decay "
        "and active graph pruning."
    ),
    version="0.1.0",
)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/tenants/{tenant_id}/ingest", response_model=IngestResponse)
def ingest(
    tenant_id: str,
    request: IngestRequest,
    graph: GraphStore = Depends(get_graph_store),
    vectors: VectorStore = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder_dep),
    settings: Settings = Depends(get_settings_dep),
) -> IngestResponse:
    return ingest_document(request, tenant_id, graph, vectors, embedder, settings)


@app.post("/tenants/{tenant_id}/query", response_model=QueryResponse)
def query(
    tenant_id: str,
    request: QueryRequest,
    graph: GraphStore = Depends(get_graph_store),
    vectors: VectorStore = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder_dep),
    settings: Settings = Depends(get_settings_dep),
    audit_log: AuditLogClient = Depends(get_audit_log),
) -> QueryResponse:
    response = hybrid_retrieve(
        request.query, tenant_id, graph, vectors, embedder, settings, request.top_k, request.graph_hops
    )
    for chunk in response.chunks:
        audit_log.record(
            tenant_id=tenant_id,
            node_id=chunk.chunk_id,
            action=AuditAction.REINFORCEMENT,
            decay_score=chunk.decay_score,
            details={"query": request.query, "vector_score": chunk.score},
        )

    # Validate retrieval relevance quality with a lightweight jury (see
    # evaluator/jury.py). This is purely advisory/observability for the
    # demo - it is attached to the response but never blocks the answer.
    if response.chunks:
        relevance_jury = build_relevance_jury(settings)
        case = {"query": request.query, "chunk_text": response.chunks[0].text}
        decision = relevance_jury.decide(case, allowed_verdicts=["relevant", "irrelevant"], case_id=response.chunks[0].chunk_id)
        response.jury_relevance_verdict = decision.final_verdict
    return response


@app.get("/tenants/{tenant_id}/nodes/{node_id}", response_model=MemoryNode)
def get_node(
    tenant_id: str,
    node_id: str,
    graph: GraphStore = Depends(get_graph_store),
) -> MemoryNode:
    node = graph.get_node(tenant_id, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="node not found")
    return node


@app.post("/admin/time-warp", response_model=TimeWarpResponse)
def time_warp(request: TimeWarpRequest) -> TimeWarpResponse:
    """Demo-only endpoint: accelerates the shared virtual clock so decay
    and pruning can be showcased without waiting real days/weeks."""
    now = clock.advance(hours=request.hours, days=request.days)
    return TimeWarpResponse(virtual_offset_seconds=clock.offset_seconds, virtual_now=now)


@app.post("/admin/time-reset", response_model=TimeWarpResponse)
def time_reset() -> TimeWarpResponse:
    clock.reset()
    return TimeWarpResponse(virtual_offset_seconds=clock.offset_seconds, virtual_now=clock.now())


@app.post("/tenants/{tenant_id}/gdpr/purge", response_model=PurgeResponse)
def gdpr_purge(
    tenant_id: str,
    request: PurgeRequest,
    graph: GraphStore = Depends(get_graph_store),
    vectors: VectorStore = Depends(get_vector_store),
    audit_log: AuditLogClient = Depends(get_audit_log),
) -> PurgeResponse:
    """Explicit, immediate right-to-be-forgotten purge. Bypasses decay,
    the prune grace period and the ``pinned`` flag entirely - see
    ``decay_engine/purge.py`` for why this is intentionally separate from
    decay-based pruning."""
    result = purge_node(
        tenant_id=tenant_id,
        node_id=request.node_id,
        reason=request.reason,
        requested_by=request.requested_by,
        graph_store=graph,
        vector_store=vectors,
        audit_log=audit_log,
    )
    return PurgeResponse(
        node_id=result.node_id,
        deleted_from_graph=result.deleted_from_graph,
        deleted_from_vector_store=result.deleted_from_vector_store,
        audit_entry_id=result.audit_entry_id,
    )
