"""The decay-engine background worker.

``DecayWorker.run_once`` performs one full sweep (recompute decay / advance
pruning phase) over every node for a set of tenants. ``run_forever`` wraps
that in a simple ``asyncio`` loop with a configurable interval - no extra
scheduling dependency required. ``decay_engine/service.py`` additionally
shows how to drive the exact same ``run_once`` via APScheduler for the
docker-compose worker container, satisfying either preference.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime

from app.graph_store import GraphStore
from app.schemas import NodeStatus
from app.time_control import clock as default_clock
from app.vector_store import VectorStore
from decay_engine.audit_log import AuditLog
from decay_engine.pruning import PruneConfig, process_node
from evaluator.human_review import HumanReviewQueue
from evaluator.jury import Jury

logger = logging.getLogger(__name__)


class DecayWorker:
    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStore,
        audit_log: AuditLog,
        config: PruneConfig,
        jury: Jury | None = None,
        human_review_queue: HumanReviewQueue | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.audit_log = audit_log
        self.config = config
        self.jury = jury
        self.human_review_queue = human_review_queue
        self._now_fn = now_fn or default_clock.now

    def run_once(self, tenant_ids: list[str]) -> dict[str, Counter]:
        """Sweep every node for each tenant. Returns per-tenant outcome
        counters, handy for tests/observability (e.g. {"hard_deleted": 2})."""
        now = self._now_fn()
        results: dict[str, Counter] = {}
        for tenant_id in tenant_ids:
            counter: Counter = Counter()
            nodes = self.graph_store.list_nodes(tenant_id, status=NodeStatus.ACTIVE)
            nodes += self.graph_store.list_nodes(tenant_id, status=NodeStatus.MARKED_FOR_DELETION)
            for node in nodes:
                outcome = process_node(
                    node,
                    now,
                    graph_store=self.graph_store,
                    vector_store=self.vector_store,
                    audit_log=self.audit_log,
                    config=self.config,
                    jury=self.jury,
                    human_review_queue=self.human_review_queue,
                )
                counter[outcome] += 1
            results[tenant_id] = counter
            logger.info("decay sweep tenant=%s outcomes=%s", tenant_id, dict(counter))
        return results

    async def run_forever(self, tenant_ids: list[str], interval_seconds: float) -> None:  # pragma: no cover
        """Long-running loop for the worker process. Not exercised by unit
        tests (it never returns); use ``run_once`` directly in tests."""
        while True:
            try:
                self.run_once(tenant_ids)
            except Exception:
                logger.exception("decay sweep failed")
            await asyncio.sleep(interval_seconds)
