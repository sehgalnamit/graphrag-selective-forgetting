"""Simple, dependency-free word-based chunking.

Good enough for a demo corpus; swap for a token-aware splitter (e.g. using
the target embedding model's tokenizer) in production.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size_words: int = 120, overlap_words: int = 20) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words >= chunk_size_words:
        overlap_words = max(0, chunk_size_words // 4)

    chunks: list[str] = []
    start = 0
    step = chunk_size_words - overlap_words
    while start < len(words):
        chunk_words = words[start : start + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size_words >= len(words):
            break
        start += step
    return chunks
