"""Pluggable generation step.

No real LLM call is required to run the demo or the tests: we build a
deterministic, clearly-labelled extractive "answer" from the retrieved
context. The function signature and the ``[MOCK_LLM]`` prefix are the swap
point for a real provider - e.g. replace the body with a call to Azure
OpenAI / OpenAI / a local Ollama chat model, keeping the same
``generate_answer(query, chunks, related_entities) -> str`` contract so
``app/retrieval.py`` does not need to change.
"""
from __future__ import annotations

from app.schemas import RelatedEntity, RetrievedChunk


def generate_answer(query: str, chunks: list[RetrievedChunk], related_entities: list[RelatedEntity]) -> str:
    if not chunks:
        return (
            "[MOCK_LLM] No supporting context was retrieved for this query - it may have been "
            "pruned by the memory decay engine, or no matching documents were ever ingested."
        )

    context_excerpt = " ".join(chunk.text for chunk in chunks)[:400]
    entity_names = ", ".join(sorted({entity.name for entity in related_entities})) or "none"
    best = chunks[0]
    return (
        f"[MOCK_LLM extractive answer] Synthesized from {len(chunks)} retrieved chunk(s); "
        f"top match similarity={best.score:.3f}, decay_score={best.decay_score:.3f}. "
        f"Related graph entities: {entity_names}. "
        f"Context excerpt: \"{context_excerpt}\""
    )
