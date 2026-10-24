"""Entrypoint for the standalone decay-engine worker process (the
``decay-engine`` service in docker-compose).

Builds real (or fake, if ``USE_FAKE_STORES=true``) store adapters from
environment-driven settings and drives :meth:`DecayWorker.run_once` on a
fixed interval using APScheduler's ``BlockingScheduler`` - this is the
production-style scheduling option; see ``decay_engine/worker.py`` for the
lighter-weight ``asyncio``-loop alternative used in tests/scripts.

Run with: ``python -m decay_engine.service``
"""
from __future__ import annotations

import logging
import os

from apscheduler.schedulers.blocking import BlockingScheduler

from app.config import get_settings
from app.graph_store import GraphStore, InMemoryGraphStore, Neo4jGraphStore
from app.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore
from decay_engine.audit_log import AuditLog
from decay_engine.pruning import PruneConfig
from decay_engine.worker import DecayWorker
from evaluator.human_review import HumanReviewQueue
from evaluator.jury import build_default_jury

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("decay_engine.service")


def build_worker() -> tuple[DecayWorker, list[str]]:
    settings = get_settings()

    graph_store: GraphStore
    vector_store: VectorStore
    if settings.use_fake_stores:
        graph_store = InMemoryGraphStore()
        vector_store = InMemoryVectorStore()
    else:
        graph_store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        vector_store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key)

    audit_log = AuditLog(settings.audit_db_path if not settings.use_fake_stores else ":memory:")
    human_review_queue = HumanReviewQueue(settings.audit_db_path if not settings.use_fake_stores else ":memory:")
    jury = build_default_jury(settings) if settings.jury_gate_enabled else None

    config = PruneConfig(
        half_life_hours=settings.decay_half_life_hours,
        reinforcement_factor=settings.decay_reinforcement_factor,
        reinforcement_cap=settings.decay_reinforcement_cap,
        prune_threshold=settings.prune_threshold,
        grace_period_hours=settings.prune_grace_period_hours,
        jury_borderline_margin=settings.jury_borderline_margin,
        jury_gate_enabled=settings.jury_gate_enabled,
    )

    worker = DecayWorker(
        graph_store=graph_store,
        vector_store=vector_store,
        audit_log=audit_log,
        config=config,
        jury=jury,
        human_review_queue=human_review_queue,
    )
    tenant_ids = [t.strip() for t in os.environ.get("DECAY_TENANT_IDS", "demo").split(",") if t.strip()]
    return worker, tenant_ids


def main() -> None:
    settings = get_settings()
    worker, tenant_ids = build_worker()

    scheduler = BlockingScheduler()

    def _sweep() -> None:
        results = worker.run_once(tenant_ids)
        logger.info("decay sweep complete: %s", results)

    scheduler.add_job(_sweep, "interval", seconds=settings.decay_interval_seconds)
    logger.info(
        "decay-engine worker starting: tenants=%s interval=%ss threshold=%s grace_period_hours=%s",
        tenant_ids,
        settings.decay_interval_seconds,
        settings.prune_threshold,
        settings.prune_grace_period_hours,
    )
    _sweep()  # run one sweep immediately on startup
    scheduler.start()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    main()
