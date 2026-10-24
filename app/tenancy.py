"""Multi-tenant naming conventions.

Full multi-tenant *infrastructure* (per-tenant auth, quota, routing) is out
of scope for this repo - that is the focus of the sibling
``telemetry-rca-agentic-runtime`` project. What we document and implement
here is the **naming convention** every store adapter uses so a production
deployment can graduate to true per-tenant isolation without changing any
calling code:

* **Neo4j**: one database per tenant (``tenant_<id>``), requires Neo4j
  Enterprise/AuraDB multi-database support. This demo's docker-compose uses
  Neo4j Community Edition, which only has a single default database, so the
  in-repo ``Neo4jGraphStore`` partitions tenants with a ``tenant_id``
  property + index instead and only switches to a real per-tenant database
  when ``NEO4J_MULTI_DB_ENABLED=true`` against a multi-db-capable server.
* **Qdrant**: one collection per tenant (``tenant_<id>_chunks``). Qdrant
  supports unlimited collections on a single instance "for free", so the
  vector store adapter always uses true collection-per-tenant isolation,
  even in the local demo.
"""
from __future__ import annotations

import re

_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize(tenant_id: str) -> str:
    sanitized = _SAFE_RE.sub("_", tenant_id.lower()).strip("_")
    return sanitized or "default"


def neo4j_database_name(tenant_id: str) -> str:
    return f"tenant_{_sanitize(tenant_id)}"


def qdrant_collection_name(tenant_id: str) -> str:
    return f"tenant_{_sanitize(tenant_id)}_chunks"
