"""GDPR-style "right to be forgotten" purge.

Distinct from decay-based pruning in every important way:

* **Immediate** - no grace period, no two-phase mark/delete.
* **Bypasses ``pinned``** - compliance-critical facts are exempt from
  *decay*-driven pruning, but a verified data-subject erasure request must
  still be honored; pinning a node never blocks a GDPR purge.
* **Bypasses the jury gate** - the decay engine's jury review exists to
  validate *ambiguous automatic* decay decisions. An explicit purge request
  is an unambiguous, externally-authorized instruction and is executed
  directly, with its own audit trail entry (``AuditAction.GDPR_PURGE``)
  separate from any decay/prune audit entries for the same node.

This module takes the graph/vector store + audit log as plain parameters
(no FastAPI/pydantic-settings import) so it has no dependency on the `app`
package and can be invoked from the API, a CLI, or a batch compliance job.
"""
from __future__ import annotations

from dataclasses import dataclass

from decay_engine.audit_log import AuditAction, AuditLog


@dataclass
class PurgeResult:
    node_id: str
    deleted_from_graph: bool
    deleted_from_vector_store: bool
    audit_entry_id: int


def purge_node(
    *,
    tenant_id: str,
    node_id: str,
    reason: str,
    requested_by: str,
    graph_store,
    vector_store,
    audit_log: AuditLog,
) -> PurgeResult:
    deleted_from_graph = graph_store.hard_delete(tenant_id, node_id)
    deleted_from_vector_store = vector_store.delete(tenant_id, node_id)

    audit_entry_id = audit_log.record(
        tenant_id=tenant_id,
        node_id=node_id,
        action=AuditAction.GDPR_PURGE,
        decay_score=None,
        details={
            "reason": reason,
            "requested_by": requested_by,
            "deleted_from_graph": deleted_from_graph,
            "deleted_from_vector_store": deleted_from_vector_store,
            "bypassed_pinned": True,
            "bypassed_grace_period": True,
        },
    )
    return PurgeResult(
        node_id=node_id,
        deleted_from_graph=deleted_from_graph,
        deleted_from_vector_store=deleted_from_vector_store,
        audit_entry_id=audit_entry_id,
    )
