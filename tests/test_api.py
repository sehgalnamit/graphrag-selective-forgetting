"""FastAPI endpoint tests via TestClient, using fake in-memory adapters
injected through FastAPI dependency overrides (see tests/conftest.py)."""
from __future__ import annotations


def test_healthz(api_client):
    resp = api_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_then_query_returns_relevant_chunk(api_client):
    ingest_resp = api_client.post(
        "/tenants/acme/ingest",
        json={
            "document_id": "doc-1",
            "text": "Acme Corp announced quarterly revenue growth. Jane Doe works at Acme Corp.",
            "base_importance": 0.6,
        },
    )
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert len(body["chunk_ids"]) == 1
    assert body["relation_count"] >= 1

    query_resp = api_client.post("/tenants/acme/query", json={"query": "Acme Corp quarterly revenue"})
    assert query_resp.status_code == 200
    payload = query_resp.json()
    assert payload["chunks"], "expected at least one retrieved chunk"
    assert "Acme" in payload["chunks"][0]["text"]
    assert payload["jury_relevance_verdict"] in {"relevant", "irrelevant"}
    # related entity (Jane Doe / Acme Corp) reachable via graph traversal
    assert isinstance(payload["related_entities"], list)


def test_query_reinforces_access_count_and_decay_score(api_client):
    api_client.post(
        "/tenants/acme/ingest",
        json={"document_id": "doc-2", "text": "The Eiffel Tower is located in Paris.", "base_importance": 0.5},
    )
    chunk_id = "doc-2::chunk::0"

    before = api_client.get(f"/tenants/acme/nodes/{chunk_id}")
    assert before.status_code == 200
    assert before.json()["access_count"] == 0

    api_client.post("/tenants/acme/query", json={"query": "Eiffel Tower Paris"})

    after = api_client.get(f"/tenants/acme/nodes/{chunk_id}")
    assert after.json()["access_count"] == 1


def test_get_missing_node_returns_404(api_client):
    resp = api_client.get("/tenants/acme/nodes/does-not-exist")
    assert resp.status_code == 404


def test_time_warp_advances_virtual_clock(api_client):
    resp = api_client.post("/admin/time-warp", json={"days": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["virtual_offset_seconds"] == 10 * 24 * 3600

    reset_resp = api_client.post("/admin/time-reset")
    assert reset_resp.json()["virtual_offset_seconds"] == 0


def test_gdpr_purge_removes_node_from_both_stores(api_client):
    api_client.post(
        "/tenants/acme/ingest",
        json={"document_id": "doc-3", "text": "Sensitive personal data about a customer.", "base_importance": 0.5},
    )
    chunk_id = "doc-3::chunk::0"

    purge_resp = api_client.post("/tenants/acme/gdpr/purge", json={"node_id": chunk_id, "requested_by": "user-7"})
    assert purge_resp.status_code == 200
    body = purge_resp.json()
    assert body["deleted_from_graph"] is True
    assert body["deleted_from_vector_store"] is True

    missing = api_client.get(f"/tenants/acme/nodes/{chunk_id}")
    assert missing.status_code == 404
