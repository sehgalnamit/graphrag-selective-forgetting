"""Two-phase pruning logic tests, using the in-memory fake graph/vector
store adapters (no docker required).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.graph_store import InMemoryGraphStore
from app.schemas import MemoryNode, NodeKind, NodeStatus
from app.vector_store import InMemoryVectorStore
from decay_engine.audit_log import AuditAction, AuditLog
from decay_engine.pruning import PruneConfig
from decay_engine.worker import DecayWorker
from evaluator.human_review import HumanReviewQueue
from evaluator.jury import Jury
from evaluator.schemas import Verdict

TENANT = "acme"


def _make_node(node_id, *, base_importance, last_accessed_at, access_count=0, pinned=False, now=None):
    now = now or last_accessed_at
    return MemoryNode(
        id=node_id,
        tenant_id=TENANT,
        kind=NodeKind.CHUNK,
        text=f"text for {node_id}",
        base_importance=base_importance,
        last_accessed_at=last_accessed_at,
        created_at=last_accessed_at,
        access_count=access_count,
        pinned=pinned,
        decay_score=base_importance,
    )


def _worker(graph, vectors, audit, config, jury=None, human_review_queue=None, now=None):
    return DecayWorker(
        graph_store=graph,
        vector_store=vectors,
        audit_log=audit,
        config=config,
        jury=jury,
        human_review_queue=human_review_queue,
        now_fn=(lambda: now),
    )


def test_pinned_node_is_never_marked_even_with_near_zero_importance():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    now = datetime.now(UTC)
    node = _make_node(
        "pinned-1", base_importance=0.01, last_accessed_at=now - timedelta(days=365), pinned=True
    )
    graph.upsert_node(node)
    vectors.upsert(TENANT, node.id, [0.1] * 4, {"text": node.text})

    config = PruneConfig(prune_threshold=0.15, grace_period_hours=1, jury_gate_enabled=False)
    worker = _worker(graph, vectors, audit, config, now=now)
    outcomes = worker.run_once([TENANT])

    assert outcomes[TENANT]["active_above_threshold"] == 1
    stored = graph.get_node(TENANT, "pinned-1")
    assert stored.status == NodeStatus.ACTIVE
    assert stored.decay_score == 1.0


def test_low_decay_node_is_marked_then_hard_deleted_after_grace_period():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    now = datetime.now(UTC)
    node = _make_node("stale-1", base_importance=0.5, last_accessed_at=now - timedelta(days=60))
    graph.upsert_node(node)
    vectors.upsert(TENANT, node.id, [0.2] * 4, {"text": node.text})

    config = PruneConfig(
        half_life_hours=168, prune_threshold=0.2, grace_period_hours=24, jury_gate_enabled=False
    )

    # Cycle 1: should drop below threshold and be marked (not yet deleted).
    worker = _worker(graph, vectors, audit, config, now=now)
    outcomes = worker.run_once([TENANT])
    assert outcomes[TENANT]["marked_for_deletion"] == 1
    marked = graph.get_node(TENANT, "stale-1")
    assert marked.status == NodeStatus.MARKED_FOR_DELETION
    assert marked.marked_at is not None
    # still present in the vector store (grace period, not yet hard-deleted)
    assert vectors.get(TENANT, "stale-1") is not None

    # Cycle 2 (before grace period elapses): no-op.
    worker_mid = _worker(graph, vectors, audit, config, now=now + timedelta(hours=1))
    outcomes_mid = worker_mid.run_once([TENANT])
    assert outcomes_mid[TENANT]["awaiting_grace_period"] == 1

    # Cycle 3 (after grace period elapses): hard-deleted from both stores.
    worker_late = _worker(graph, vectors, audit, config, now=now + timedelta(hours=25))
    outcomes_late = worker_late.run_once([TENANT])
    assert outcomes_late[TENANT]["hard_deleted"] == 1
    assert graph.get_node(TENANT, "stale-1") is None
    assert vectors.get(TENANT, "stale-1") is None

    actions = [e.action for e in audit.list(tenant_id=TENANT, node_id="stale-1")]
    assert actions == [
        AuditAction.DECAY_RECOMPUTED,
        AuditAction.DECAY_MARKED,
        AuditAction.HARD_DELETED,
    ]


def test_jury_gate_can_overturn_a_borderline_prune_decision():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    now = datetime.now(UTC)
    # Craft base_importance/elapsed time so decay lands just under threshold,
    # inside the borderline jury margin.
    node = _make_node("borderline-1", base_importance=0.5, last_accessed_at=now - timedelta(hours=250))
    graph.upsert_node(node)

    config = PruneConfig(
        half_life_hours=168, prune_threshold=0.2, jury_borderline_margin=0.2, jury_gate_enabled=True
    )

    class AlwaysKeepJudge:
        name = "always_keep"

        def evaluate(self, case, allowed_verdicts):
            return Verdict(judge_name=self.name, verdict="keep", confidence=0.9, rationale="test stub")

    jury = Jury(judges=[AlwaysKeepJudge(), AlwaysKeepJudge()], abstain_margin=0.15)
    worker = _worker(graph, vectors, audit, config, jury=jury, now=now)
    outcomes = worker.run_once([TENANT])

    assert outcomes[TENANT]["jury_kept"] == 1
    stored = graph.get_node(TENANT, "borderline-1")
    assert stored.status == NodeStatus.ACTIVE
    actions = [e.action for e in audit.list(tenant_id=TENANT, node_id="borderline-1")]
    assert AuditAction.JURY_REVIEW_REQUESTED in actions
    assert AuditAction.JURY_KEPT in actions
    assert AuditAction.DECAY_MARKED not in actions


def test_jury_gate_confirms_forget_and_proceeds_to_mark():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    now = datetime.now(UTC)
    node = _make_node("borderline-2", base_importance=0.5, last_accessed_at=now - timedelta(hours=250))
    graph.upsert_node(node)

    config = PruneConfig(half_life_hours=168, prune_threshold=0.2, jury_borderline_margin=0.2, jury_gate_enabled=True)

    class AlwaysForgetJudge:
        name = "always_forget"

        def evaluate(self, case, allowed_verdicts):
            return Verdict(judge_name=self.name, verdict="forget", confidence=0.9, rationale="test stub")

    jury = Jury(judges=[AlwaysForgetJudge(), AlwaysForgetJudge()], abstain_margin=0.15)
    worker = _worker(graph, vectors, audit, config, jury=jury, now=now)
    outcomes = worker.run_once([TENANT])

    assert outcomes[TENANT]["marked_for_deletion"] == 1
    stored = graph.get_node(TENANT, "borderline-2")
    assert stored.status == NodeStatus.MARKED_FOR_DELETION
    actions = [e.action for e in audit.list(tenant_id=TENANT, node_id="borderline-2")]
    assert AuditAction.JURY_CONFIRMED_FORGET in actions
    assert AuditAction.DECAY_MARKED in actions


def test_jury_gate_escalation_enqueues_human_review_and_leaves_node_active():
    graph = InMemoryGraphStore()
    vectors = InMemoryVectorStore()
    audit = AuditLog(":memory:")
    queue = HumanReviewQueue(":memory:")
    now = datetime.now(UTC)
    node = _make_node("borderline-3", base_importance=0.5, last_accessed_at=now - timedelta(hours=250))
    graph.upsert_node(node)

    config = PruneConfig(half_life_hours=168, prune_threshold=0.2, jury_borderline_margin=0.2, jury_gate_enabled=True)

    class SplitJudgeA:
        name = "split_a"

        def evaluate(self, case, allowed_verdicts):
            return Verdict(judge_name=self.name, verdict="keep", confidence=0.6, rationale="split a")

    class SplitJudgeB:
        name = "split_b"

        def evaluate(self, case, allowed_verdicts):
            return Verdict(judge_name=self.name, verdict="forget", confidence=0.6, rationale="split b")

    jury = Jury(judges=[SplitJudgeA(), SplitJudgeB()], abstain_margin=0.9)
    worker = _worker(graph, vectors, audit, config, jury=jury, human_review_queue=queue, now=now)
    outcomes = worker.run_once([TENANT])

    assert outcomes[TENANT]["jury_escalated"] == 1
    stored = graph.get_node(TENANT, "borderline-3")
    assert stored.status == NodeStatus.ACTIVE  # left untouched pending human review
    pending = queue.list_pending(tenant_id=TENANT)
    assert len(pending) == 1
    assert pending[0].node_id == "borderline-3"
    actions = [e.action for e in audit.list(tenant_id=TENANT, node_id="borderline-3")]
    assert AuditAction.JURY_ESCALATED in actions
