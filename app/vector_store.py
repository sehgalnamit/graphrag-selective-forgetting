"""Vector store abstraction over Qdrant, with an in-memory fake used for
tests and zero-dependency demos. Mirrors :mod:`app.graph_store`'s pattern.
"""
from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.tenancy import qdrant_collection_name


@dataclass
class VectorRecord:
    chunk_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class ScoredChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(self, tenant_id: str, dimension: int) -> None: ...

    @abstractmethod
    def upsert(self, tenant_id: str, chunk_id: str, vector: list[float], payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(self, tenant_id: str, vector: list[float], top_k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def update_payload(self, tenant_id: str, chunk_id: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def delete(self, tenant_id: str, chunk_id: str) -> bool: ...

    @abstractmethod
    def get(self, tenant_id: str, chunk_id: str) -> VectorRecord | None: ...

    def close(self) -> None:  # pragma: no cover
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(VectorStore):
    """Pure-python fake adapter with the same interface as the real Qdrant
    adapter, partitioned per tenant collection name for parity."""

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, VectorRecord]] = {}
        self._lock = threading.RLock()

    def ensure_collection(self, tenant_id: str, dimension: int) -> None:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            self._collections.setdefault(name, {})

    def upsert(self, tenant_id: str, chunk_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            self._collections.setdefault(name, {})
            self._collections[name][chunk_id] = VectorRecord(chunk_id=chunk_id, vector=vector, payload=payload)

    def search(self, tenant_id: str, vector: list[float], top_k: int) -> list[ScoredChunk]:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            records = list(self._collections.get(name, {}).values())
        scored = [
            ScoredChunk(chunk_id=r.chunk_id, score=_cosine(vector, r.vector), payload=dict(r.payload))
            for r in records
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def update_payload(self, tenant_id: str, chunk_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            record = self._collections.get(name, {}).get(chunk_id)
            if record is not None:
                record.payload.update(payload)

    def delete(self, tenant_id: str, chunk_id: str) -> bool:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            return self._collections.get(name, {}).pop(chunk_id, None) is not None

    def get(self, tenant_id: str, chunk_id: str) -> VectorRecord | None:
        with self._lock:
            name = qdrant_collection_name(tenant_id)
            record = self._collections.get(name, {}).get(chunk_id)
            return VectorRecord(record.chunk_id, list(record.vector), dict(record.payload)) if record else None


class QdrantVectorStore(VectorStore):
    """Real Qdrant-backed implementation. One collection per tenant, named
    via :func:`app.tenancy.qdrant_collection_name`.
    """

    def __init__(self, url: str, api_key: str | None = None) -> None:
        from qdrant_client import QdrantClient  # imported lazily

        self._client = QdrantClient(url=url, api_key=api_key)

    def close(self) -> None:
        self._client.close()

    def ensure_collection(self, tenant_id: str, dimension: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        name = qdrant_collection_name(tenant_id)
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def upsert(self, tenant_id: str, chunk_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct

        name = qdrant_collection_name(tenant_id)
        self._client.upsert(
            collection_name=name,
            points=[PointStruct(id=chunk_id, vector=vector, payload=payload)],
        )

    def search(self, tenant_id: str, vector: list[float], top_k: int) -> list[ScoredChunk]:
        name = qdrant_collection_name(tenant_id)
        results = self._client.search(collection_name=name, query_vector=vector, limit=top_k)
        return [ScoredChunk(chunk_id=str(r.id), score=float(r.score), payload=dict(r.payload or {})) for r in results]

    def update_payload(self, tenant_id: str, chunk_id: str, payload: dict[str, Any]) -> None:
        name = qdrant_collection_name(tenant_id)
        self._client.set_payload(collection_name=name, payload=payload, points=[chunk_id])

    def delete(self, tenant_id: str, chunk_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        name = qdrant_collection_name(tenant_id)
        self._client.delete(collection_name=name, points_selector=PointIdsList(points=[chunk_id]))
        return True

    def get(self, tenant_id: str, chunk_id: str) -> VectorRecord | None:
        name = qdrant_collection_name(tenant_id)
        points = self._client.retrieve(collection_name=name, ids=[chunk_id], with_vectors=True)
        if not points:
            return None
        point = points[0]
        return VectorRecord(chunk_id=str(point.id), vector=list(point.vector or []), payload=dict(point.payload or {}))
