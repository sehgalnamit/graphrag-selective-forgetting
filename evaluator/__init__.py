"""LLM-Jury Evaluator: a reusable panel-of-judges module for validating
borderline forget-vs-keep decisions and retrieval relevance.

This module is intentionally self-contained (only depends on ``pydantic``
and the python standard library, with an optional ``httpx`` call for the
Ollama judge) so it can be copy-pasted into sibling repositories.
"""
