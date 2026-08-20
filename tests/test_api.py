"""API contract tests. The pipeline is faked, so no models are loaded."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from app.chunking import chunk_transcript
from app.interfaces import GenerationError
from app.prompts import REFUSAL_TEXT
from rag import RagPipeline


@pytest.fixture
def client(monkeypatch, transcript, embedder, store, chat_model) -> TestClient:
    """A test client whose app serves a pipeline backed by in-memory fakes."""
    chunks = chunk_transcript(transcript, max_tokens=20, overlap_tokens=5, count_tokens=embedder.count_tokens)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    pipeline = RagPipeline(embedder, store, chat_model, top_k=3, min_score=-1.0)

    # Startup now validates credentials and the on-disk index; neither exists in
    # a test process, and neither is what these tests are about.
    monkeypatch.setattr(main, "validate_serving_config", lambda settings: None)
    monkeypatch.setattr(main, "build_pipeline", lambda settings, read_only=True: pipeline)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_reports_the_index(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0
    assert body["indexed_episodes"] == 1


def test_ask_returns_an_answer_with_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "what did they say about burnout?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["refused"] is False
    assert body["sources"]
    source = body["sources"][0]
    assert source["episode"] == "ep-01"
    assert ":" in source["timestamp"]
    assert 0.0 <= source["score"] <= 1.0
    assert body["latency_ms"] >= 0


def test_ask_echoes_a_correlation_id(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"question": "burnout?"}, headers={"x-request-id": "abc123"}
    )

    assert response.headers["x-request-id"] == "abc123"
    assert response.json()["request_id"] == "abc123"


def test_top_k_override_limits_the_sources(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "burnout?", "top_k": 1})

    assert len(response.json()["sources"]) == 1


@pytest.mark.parametrize("payload", [{}, {"question": "hi"}, {"question": "ok", "top_k": 0}])
def test_invalid_payloads_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_generation_failure_maps_to_502(client: TestClient, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise GenerationError("groq is down")

    monkeypatch.setattr(main.app.state.pipeline, "answer", boom)

    response = client.post("/ask", json={"question": "what about burnout?"})

    assert response.status_code == 502
    assert "groq is down" in response.json()["detail"]


def test_refusal_is_reported_without_sources(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(main.app.state.pipeline, "_min_score", 0.99)

    body = client.post("/ask", json={"question": "unrelated topic entirely"}).json()

    assert body["refused"] is True
    assert body["answer"] == REFUSAL_TEXT
    assert body["sources"] == []


def test_ui_is_served_at_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ask-form" in response.text


def test_ui_falls_back_when_the_asset_is_absent(client: TestClient, monkeypatch, tmp_path) -> None:
    """A trimmed deployment image without app/static/ must still start."""
    monkeypatch.setattr(main, "_UI_PATH", tmp_path / "missing.html")

    response = client.get("/")

    assert response.status_code == 200
    assert "/docs" in response.text


def test_ui_is_hidden_from_the_openapi_schema(client: TestClient) -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]
