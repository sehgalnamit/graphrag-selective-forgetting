"""Centralized runtime configuration.

All settings are environment-variable driven (12-factor style) so the same
image can run against real Neo4j/Qdrant (docker-compose, cloud) or against
the in-memory fake adapters used for unit tests and zero-dependency demos.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- service identity ---
    app_name: str = "graphrag-selective-forgetting"
    environment: str = "local"

    # --- store backends ---
    # When true, use pure-python in-memory adapters instead of real Neo4j/Qdrant.
    # Defaults to True so `pytest` and a bare `python -m uvicorn` work with zero
    # external services. docker-compose explicitly sets this to false.
    use_fake_stores: bool = True

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "please-change-me"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    vector_size: int = 384

    # --- embeddings ---
    # "auto" tries sentence-transformers and silently falls back to the
    # deterministic hashing embedder if the library/model can't be loaded
    # (e.g. no internet access to download weights).
    embedder_backend: str = "auto"  # auto | sentence-transformers | fake
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- chunking ---
    chunk_size_words: int = 120
    chunk_overlap_words: int = 20

    # --- decay engine ---
    decay_half_life_hours: float = 168.0  # 7 days
    decay_reinforcement_factor: float = 0.15
    decay_reinforcement_cap: int = 20
    decay_interval_seconds: int = 300
    prune_threshold: float = 0.15
    prune_grace_period_hours: float = 24.0
    jury_borderline_margin: float = 0.05
    jury_gate_enabled: bool = True

    # --- jury evaluator ---
    jury_panel_size: int = 3
    jury_abstain_margin: float = 0.15
    jury_ollama_enabled: bool = False
    jury_ollama_url: str = "http://localhost:11434"
    jury_ollama_models: str = ""  # comma separated model tags

    # --- audit log / human review ---
    audit_db_path: str = "./data/audit_log.sqlite3"

    # --- retrieval ---
    retrieval_top_k: int = 5
    retrieval_graph_hops: int = 2


def get_settings() -> Settings:
    """Settings are intentionally re-read (not cached at module import time)
    so tests can monkeypatch environment variables per-test."""
    return Settings()
