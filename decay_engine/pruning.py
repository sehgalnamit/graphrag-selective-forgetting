"""Two-phase active pruning, with an optional jury gate for borderline
forget-vs-keep decisions.

Phase 1 (mark): an ACTIVE node whose freshly recomputed ``decay_score``
falls below ``prune_threshold`` is marked ``MARKED_FOR_DELETION`` (not
deleted yet) and an audit entry is written. If the score sits within
``jury_borderline_margin`` of the threshold and the jury gate is enabled,
the jury is consulted first; it may overturn the prune ("keep"), confirm it
("forget"), or abstain, which escalates to the human-review queue and
leaves the node untouched this cycle.

Phase 2 (hard-delete): on a later cycle, once a node has sat in
``MARKED_FOR_DELETION`` for at least ``grace_period_hours``, it is deleted
from both the graph store and the vector store, and a final audit entry is
written. ``pinned`` nodes are exempt from entering phase 1 at all, and are
defensively reactivated if ever found marked (should not normally happen).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.graph_store import GraphStore
from app.schemas import MemoryNode, NodeStatus
from app.vector_store import VectorStore
from decay_engine.audit_log import AuditAction, AuditLog
from decay_engine.formula import compute_decay_score
from evaluator.human_review import HumanReviewQueue
from evaluator.jury import Jury


@dataclass
class PruneConfig:
    half_life_hours: float = 168.0
    reinforcement_factor: float = 0.15
    reinforcement_cap: int = 20
    prune_threshold: float = 0.15
    grace_period_hours: float = 24.0
    jury_borderline_margin: float = 0.05
    jury_gate_enabled: bool = True


def process_node(
    node: MemoryNode,
    now,
    *,
    graph_store: GraphStore,
    vector_store: VectorStore,
    audit_log: AuditLog,
    config: PruneConfig,
    jury: Jury | None = None,
    human_review_queue: HumanReviewQueue | None = None,
) -> str:
    """Recompute decay (or progress phase-2 deletion) for a single node.

    Returns a short outcome label, useful for tests and operator logging:
    one of ``active_above_threshold``, ``marked_for_deletion``,
    ``jury_kept``, ``jury_escalated``, ``awaiting_grace_period``,
    ``hard_deleted``, ``reactivated_pinned``.
    """
    if node.status == NodeStatus.DELETED:
        return "already_deleted"

    if node.status == NodeStatus.MARKED_FOR_DELETION:
        return _progress_phase_two(node, now, graph_store, vector_store, audit_log, config)

    return _progress_phase_one(
        node, now, graph_store=graph_store, audit_log=audit_log, config=config, jury=jury,
        human_review_queue=human_review_queue,
    )


def _progress_phase_two(node, now, graph_store, vector_store, audit_log, config: PruneConfig) -> str:
    if node.pinned:
        # Defensive: a node should never be pinned *and* marked, but if it
        # happens (e.g. pinned after being marked), un-mark it rather than
        # silently deleting compliance-critical data.
        graph_store.reactivate(node.tenant_id, node.id)
        audit_log.record(
            tenant_id=node.tenant_id,
            node_id=node.id,
            action=AuditAction.REACTIVATED,
            decay_score=node.decay_score,
            details={"reason": "pinned_safety_reactivation"},
        )
        return "reactivated_pinned"

    elapsed_hours = (now - node.marked_at).total_seconds() / 3600.0 if node.marked_at else float("inf")
    if elapsed_hours < config.grace_period_hours:
        return "awaiting_grace_period"

    deleted_from_graph = graph_store.hard_delete(node.tenant_id, node.id)
    deleted_from_vector_store = vector_store.delete(node.tenant_id, node.id)
    audit_log.record(
        tenant_id=node.tenant_id,
        node_id=node.id,
        action=AuditAction.HARD_DELETED,
        decay_score=node.decay_score,
        details={
            "deleted_from_graph": deleted_from_graph,
            "deleted_from_vector_store": deleted_from_vector_store,
            "grace_period_hours": config.grace_period_hours,
            "elapsed_hours": elapsed_hours,
        },
    )
    return "hard_deleted"


def _progress_phase_one(node, now, *, graph_store, audit_log, config: PruneConfig, jury, human_review_queue) -> str:
    time_since_hours = (now - node.last_accessed_at).total_seconds() / 3600.0
    decay_score = compute_decay_score(
        base_importance=node.base_importance,
        time_since_last_access_hours=time_since_hours,
        access_count=node.access_count,
        pinned=node.pinned,
        half_life_hours=config.half_life_hours,
        reinforcement_factor=config.reinforcement_factor,
        reinforcement_cap=config.reinforcement_cap,
    )
    graph_store.update_decay_score(node.tenant_id, node.id, decay_score)
    audit_log.record(
        tenant_id=node.tenant_id,
        node_id=node.id,
        action=AuditAction.DECAY_RECOMPUTED,
        decay_score=decay_score,
        details={
            "previous_decay_score": node.decay_score,
            "time_since_last_access_hours": time_since_hours,
            "access_count": node.access_count,
        },
    )

    if node.pinned or decay_score >= config.prune_threshold:
        return "active_above_threshold"

    is_borderline = abs(decay_score - config.prune_threshold) <= config.jury_borderline_margin
    if config.jury_gate_enabled and jury is not None and is_borderline:
        outcome = _consult_jury(node, decay_score, config, jury, audit_log, human_review_queue)
        if outcome is not None:
            return outcome
        # outcome None => jury confirmed "forget" => fall through to mark

    graph_store.mark_for_deletion(node.tenant_id, node.id, now)
    audit_log.record(
        tenant_id=node.tenant_id,
        node_id=node.id,
        action=AuditAction.DECAY_MARKED,
        decay_score=decay_score,
        details={"prune_threshold": config.prune_threshold, "grace_period_hours": config.grace_period_hours},
    )
    return "marked_for_deletion"


def _consult_jury(node, decay_score, config: PruneConfig, jury: Jury, audit_log, human_review_queue) -> str | None:
    case = {
        "decay_score": decay_score,
        "prune_threshold": config.prune_threshold,
        "base_importance": node.base_importance,
        "access_count": node.access_count,
        "pinned": node.pinned,
        "text": node.text,
    }
    decision = jury.decide(case, allowed_verdicts=["keep", "forget"], case_id=node.id)
    vote_payload = [v.model_dump() for v in decision.votes]
    audit_log.record(
        tenant_id=node.tenant_id,
        node_id=node.id,
        action=AuditAction.JURY_REVIEW_REQUESTED,
        decay_score=decay_score,
        details={"votes": vote_payload, "method": decision.method},
    )

    if decision.escalated:
        if human_review_queue is not None:
            human_review_queue.enqueue(
                tenant_id=node.tenant_id,
                node_id=node.id,
                case_type="decay_forget_vs_keep",
                case_payload=case,
                votes=vote_payload,
            )
        audit_log.record(
            tenant_id=node.tenant_id,
            node_id=node.id,
            action=AuditAction.JURY_ESCALATED,
            decay_score=decay_score,
            details={"aggregated_confidence": decision.aggregated_confidence},
        )
        return "jury_escalated"

    if decision.final_verdict == "keep":
        audit_log.record(
            tenant_id=node.tenant_id,
            node_id=node.id,
            action=AuditAction.JURY_KEPT,
            decay_score=decay_score,
            details={"aggregated_confidence": decision.aggregated_confidence},
        )
        return "jury_kept"

    audit_log.record(
        tenant_id=node.tenant_id,
        node_id=node.id,
        action=AuditAction.JURY_CONFIRMED_FORGET,
        decay_score=decay_score,
        details={"aggregated_confidence": decision.aggregated_confidence},
    )
    return None
