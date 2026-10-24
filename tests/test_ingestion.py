"""Lightweight tests for chunking, entity/relation extraction, and the
hashing embedder helpers used by the ingestion pipeline."""
from __future__ import annotations

from app.chunking import chunk_text
from app.embeddings import FakeHashEmbedder
from app.entities import extract_entities_and_relations


def test_chunk_text_splits_on_word_boundaries_with_overlap():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size_words=20, overlap_words=5)
    assert len(chunks) >= 2
    assert all(chunk for chunk in chunks)


def test_chunk_text_handles_empty_input():
    assert chunk_text("", chunk_size_words=20) == []


def test_entity_and_relation_extraction_regex_fallback():
    result = extract_entities_and_relations("Jane Doe works at Acme Corp.", use_spacy=False)
    assert "Jane Doe" in result.entities
    assert "Acme Corp" in result.entities
    assert any(r.relation == "WORKS_AT" for r in result.relations)


def test_fake_hash_embedder_is_deterministic_and_normalized():
    embedder = FakeHashEmbedder(dimension=64)
    v1 = embedder.embed(["hello world"])[0]
    v2 = embedder.embed(["hello world"])[0]
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-6 or norm == 0.0


def test_fake_hash_embedder_differs_for_different_text():
    embedder = FakeHashEmbedder(dimension=64)
    v1 = embedder.embed(["apples and oranges"])[0]
    v2 = embedder.embed(["quantum physics research"])[0]
    assert v1 != v2
