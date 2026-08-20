"""Tests for the HTTP embedding adapter, using a mock transport (no network)."""

from __future__ import annotations

import httpx
import pytest

from app.providers.embeddings_remote import EmbeddingError, RemoteEmbedder


def make_embedder(handler, **kwargs) -> RemoteEmbedder:
    """Build an embedder whose HTTP calls are served by ``handler``."""
    defaults = dict(
        provider="jina",
        model="jina-embeddings-v3",
        base_url="https://api.jina.ai/v1",
        api_key="test-key",
        max_retries=1,
    )
    return RemoteEmbedder(**{**defaults, **kwargs}, transport=httpx.MockTransport(handler))


def ok_response(vectors: list[list[float]]) -> httpx.Response:
    """An OpenAI-shaped embeddings response."""
    return httpx.Response(
        200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}
    )


def test_query_vector_is_l2_normalised() -> None:
    embedder = make_embedder(lambda request: ok_response([[3.0, 4.0]]))

    vector = embedder.embed_query("anything")

    assert vector == pytest.approx([0.6, 0.8])  # 3-4-5 triangle
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_request_shape_matches_the_openai_dialect() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = httpx.Response(200, content=request.content).json()
        return ok_response([[1.0, 0.0]])

    make_embedder(handler).embed_query("a question")

    assert seen["url"] == "https://api.jina.ai/v1/embeddings"
    assert seen["auth"] == "Bearer test-key"
    assert seen["body"] == {"model": "jina-embeddings-v3", "input": ["a question"]}


def test_out_of_order_responses_are_realigned_to_input_order() -> None:
    """Providers may return items out of order; `index` is authoritative."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]},
        )

    vectors = make_embedder(handler).embed_documents(["first", "second"])

    assert vectors[0] == pytest.approx([1.0, 0.0])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_documents_are_batched() -> None:
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = httpx.Response(200, content=request.content).json()["input"]
        batches.append(len(texts))
        return ok_response([[1.0, 0.0]] * len(texts))

    make_embedder(handler, batch_size=2).embed_documents(["a", "b", "c", "d", "e"])

    assert batches == [2, 2, 1]


def test_dimension_mismatch_fails_loudly() -> None:
    embedder = make_embedder(lambda request: ok_response([[1.0, 0.0, 0.0]]), dimension=384)

    with pytest.raises(EmbeddingError, match="384"):
        embedder.embed_query("x")


def test_zero_vector_is_rejected() -> None:
    embedder = make_embedder(lambda request: ok_response([[0.0, 0.0]]))

    with pytest.raises(EmbeddingError, match="zero vector"):
        embedder.embed_query("x")


def test_malformed_response_is_rejected() -> None:
    embedder = make_embedder(lambda request: httpx.Response(200, json={"oops": True}))

    with pytest.raises(EmbeddingError, match="unexpected embeddings response"):
        embedder.embed_query("x")


def test_short_response_is_rejected() -> None:
    embedder = make_embedder(lambda request: ok_response([[1.0, 0.0]]))

    with pytest.raises(EmbeddingError, match="asked for 2 embeddings, got 1"):
        embedder.embed_documents(["a", "b"])


def test_rate_limit_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
        return ok_response([[1.0, 0.0]])

    vector = make_embedder(handler).embed_query("x")

    assert calls["n"] == 2
    assert vector == pytest.approx([1.0, 0.0])


def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "bad key"})

    with pytest.raises(EmbeddingError):
        make_embedder(handler).embed_query("x")

    assert calls["n"] == 1  # a bad key will not fix itself


def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not be called")

    assert make_embedder(handler).embed_documents([]) == []


def test_identity_names_provider_and_model() -> None:
    assert make_embedder(lambda r: ok_response([[1.0]])).name == "jina:jina-embeddings-v3"
