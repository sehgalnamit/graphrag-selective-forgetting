"""Embedding backends.

The demo must run with **zero external API keys and zero guaranteed
internet access**. We therefore implement a small strategy:

* ``FakeHashEmbedder`` - a deterministic, dependency-free embedder built from
  stable hashing of word tokens into a fixed-size vector, L2-normalized.
  It has no semantic understanding, but it is deterministic (same text ->
  same vector) and gives stable cosine-similarity ordering for exact/partial
  token overlap, which is enough to demo the retrieval pipeline end-to-end.
* ``SentenceTransformerEmbedder`` - wraps ``sentence-transformers`` for real
  semantic embeddings when the package + model weights are available.

``get_embedder()`` auto-selects: it tries sentence-transformers first when
``embedder_backend in {"auto", "sentence-transformers"}`` and transparently
falls back to the fake embedder on any ImportError/OSError (e.g. offline
environment, model download blocked) so the app never hard-fails to start.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeHashEmbedder:
    """Deterministic hashing embedder requiring zero external downloads.

    Each token is hashed into a bucket in a fixed-size vector; bucket values
    are accumulated and the final vector is L2-normalized. This is a classic
    "feature hashing" / hashing-trick embedding, not a learned semantic
    embedding - documented here so nobody mistakes it for one.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Real semantic embeddings via the `sentence-transformers` library."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def get_embedder(settings: Settings) -> Embedder:
    backend = settings.embedder_backend
    if backend in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(settings.embedding_model_name)
        except Exception as exc:  # pragma: no cover - exercised only w/o the lib
            if backend == "sentence-transformers":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to "
                "FakeHashEmbedder for this demo run.",
                exc,
            )
    return FakeHashEmbedder(dimension=settings.vector_size)
