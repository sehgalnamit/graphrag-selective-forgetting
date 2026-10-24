"""Shared Pydantic data models used across the API, graph/vector adapters,
and the decay engine.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    ACTIVE = "active"
    MARKED_FOR_DELETION = "marked_for_deletion"
    DELETED = "deleted"


class NodeKind(str, Enum):
    CHUNK = "chunk"
    ENTITY = "entity"


class MemoryNode(BaseModel):
    """A single unit of memory: either a document chunk or an extracted
    entity. Every node carries the bookkeeping fields the decay engine
    needs: base_importance, last_accessed_at, access_count, pinned and the
    derived decay_score.
    """

    id: str
    tenant_id: str
    kind: NodeKind
    text: str
    base_importance: float = Field(default=0.5, ge=0.0, le=1.0)
    last_accessed_at: datetime
    created_at: datetime
    access_count: int = 0
    pinned: bool = False
    decay_score: float = Field(default=1.0, ge=0.0, le=1.0)
    status: NodeStatus = NodeStatus.ACTIVE
    marked_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    relation: str
    tenant_id: str
    weight: float = 1.0


def utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# API request / response payloads
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    document_id: str
    text: str
    base_importance: float = Field(default=0.5, ge=0.0, le=1.0)
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunk_ids: list[str]
    entity_ids: list[str]
    relation_count: int


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    graph_hops: int | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    decay_score: float


class RelatedEntity(BaseModel):
    entity_id: str
    name: str
    hops: int
    relation_path: list[str]


class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[RetrievedChunk]
    related_entities: list[RelatedEntity]
    jury_relevance_verdict: str | None = None


class TimeWarpRequest(BaseModel):
    hours: float = 0.0
    days: float = 0.0


class TimeWarpResponse(BaseModel):
    virtual_offset_seconds: float
    virtual_now: datetime


class PurgeRequest(BaseModel):
    node_id: str
    reason: str = "gdpr_right_to_be_forgotten"
    requested_by: str = "data_subject"


class PurgeResponse(BaseModel):
    node_id: str
    deleted_from_graph: bool
    deleted_from_vector_store: bool
    audit_entry_id: int
