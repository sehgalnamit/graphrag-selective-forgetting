"""FastAPI dependency providers.

Each ``get_*`` function lazily constructs (and caches) a singleton backend
instance chosen from :class:`app.config.Settings`. Tests replace these via
``app.dependency_overrides`` so the exact same API code path runs against
in-memory fakes with no docker required.
"""
from __future__ import annotations

from functools import lru_cache

from app.audit_log_client import AuditLogClient
from app.config import Settings, get_settings
from app.embeddings import Embedder, get_embedder
from app.graph_store import GraphStore, InMemoryGraphStore, Neo4jGraphStore
from app.vector_store import InMemoryVectorStore, QdrantVectorStore, VectorStore
from evaluator.jury import Jury, build_default_jury


@lru_cache
def get_settings_dep() -> Settings:
    return get_settings()


@lru_cache
def get_graph_store() -> GraphStore:
    settings = get_settings_dep()
    if settings.use_fake_stores:
        return InMemoryGraphStore()
    return Neo4jGraphStore(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        multi_db_enabled=settings.neo4j_multi_db_enabled,
    )


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings_dep()
    if settings.use_fake_stores:
        return InMemoryVectorStore()
    return QdrantVectorStore(settings.qdrant_url, settings.qdrant_api_key)


@lru_cache
def get_embedder_dep() -> Embedder:
    return get_embedder(get_settings_dep())


@lru_cache
def get_audit_log() -> AuditLogClient:
    settings = get_settings_dep()
    return AuditLogClient(settings.audit_db_path if not settings.use_fake_stores else ":memory:")


@lru_cache
def get_jury() -> Jury:
    return build_default_jury(get_settings_dep())


def reset_caches() -> None:
    """Used by tests to force fresh singletons between test modules."""
    get_settings_dep.cache_clear()
    get_graph_store.cache_clear()
    get_vector_store.cache_clear()
    get_embedder_dep.cache_clear()
    get_audit_log.cache_clear()
    get_jury.cache_clear()
