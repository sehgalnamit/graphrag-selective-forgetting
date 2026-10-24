"""Shared pytest fixtures.

All fixtures build the **fake** in-memory graph/vector store adapters so
the full test suite runs with zero external services, zero network access,
and zero API keys - matching the "must pass with `pytest`" requirement.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("USE_FAKE_STORES", "true")
os.environ.setdefault("EMBEDDER_BACKEND", "fake")

from app import dependencies as deps  # noqa: E402
from app.config import Settings  # noqa: E402
from app.embeddings import FakeHashEmbedder  # noqa: E402
from app.graph_store import InMemoryGraphStore  # noqa: E402
from app.main import app  # noqa: E402
from app.time_control import clock  # noqa: E402
from app.vector_store import InMemoryVectorStore  # noqa: E402
from decay_engine.audit_log import AuditLog  # noqa: E402
from evaluator.jury import build_default_jury  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_virtual_clock():
    clock.reset()
    yield
    clock.reset()


@pytest.fixture
def settings() -> Settings:
    return Settings(use_fake_stores=True, embedder_backend="fake", vector_size=128)


@pytest.fixture
def graph_store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def embedder(settings: Settings) -> FakeHashEmbedder:
    return FakeHashEmbedder(dimension=settings.vector_size)


@pytest.fixture
def audit_log() -> AuditLog:
    return AuditLog(":memory:")


@pytest.fixture
def jury(settings: Settings):
    return build_default_jury(settings)


@pytest.fixture
def api_client(settings, graph_store, vector_store, embedder, audit_log) -> TestClient:
    app.dependency_overrides[deps.get_settings_dep] = lambda: settings
    app.dependency_overrides[deps.get_graph_store] = lambda: graph_store
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps.get_embedder_dep] = lambda: embedder
    app.dependency_overrides[deps.get_audit_log] = lambda: audit_log
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
