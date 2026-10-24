"""GDPR-style immediate purge tests: distinct from decay-based pruning in
that it bypasses ``pinned`` and the grace period entirely."""
from __future__ import annotations

from datetime import UTC, datetime

from app.graph_store import InMemoryGraphStore
from app.schemas import MemoryNode, NodeKind
from app.vector_store import InMemoryVectorStore
from decay_engine.audit_log import AuditAction, AuditLog
from decay_engine.purge import purge_node

TENANT = "acme"


def test_purge_deletes_immediately_even_if_pinned():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    now = datetime.now(UTC)
    node = MemoryNode(
        id="pinned-fact",
        tenant_id=TENANT,
        kind=NodeKind.CHUNK,
        text="A GDPR data subject's personal record.",
        base_importance=0.9,
        last_accessed_at=now,
        created_at=now,
        pinned=True,  # pinned exempts from *decay*, but not from GDPR purge
        decay_score=1.0,
    )
    graph.upsert_node(node)
    vectors.upsert(TENANT, node.id, [0.3] * 4, {"text": node.text})

    result = purge_node(
        tenant_id=TENANT,
        node_id="pinned-fact",
        reason="data_subject_erasure_request",
        requested_by="user-42",
        graph_store=graph,
        vector_store=vectors,
        audit_log=audit,
    )

    assert result.deleted_from_graph is True
    assert result.deleted_from_vector_store is True
    assert graph.get_node(TENANT, "pinned-fact") is None
    assert vectors.get(TENANT, "pinned-fact") is None

    entries = audit.list(tenant_id=TENANT, node_id="pinned-fact")
    assert len(entries) == 1
    assert entries[0].action == AuditAction.GDPR_PURGE
    assert entries[0].details["bypassed_pinned"] is True
    assert entries[0].details["requested_by"] == "user-42"


def test_purge_of_missing_node_is_recorded_but_reports_no_deletion():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")

    result = purge_node(
        tenant_id=TENANT,
        node_id="does-not-exist",
        reason="data_subject_erasure_request",
        requested_by="user-1",
        graph_store=graph,
        vector_store=vectors,
        audit_log=audit,
    )
    assert result.deleted_from_graph is False
    assert result.deleted_from_vector_store is False
    entries = audit.list(tenant_id=TENANT, node_id="does-not-exist")
    assert entries[0].action == AuditAction.GDPR_PURGE
