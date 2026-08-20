"""Embedding adapter that calls an HTTP API instead of running a model locally.

The deployed service cannot afford torch: ``sentence-transformers`` pulls ~800MB
of resident memory and several hundred MB of wheels, which does not fit a free
tier. Embedding the *question* is the only inference the query path needs, so it
moves to an HTTP call — one small request per query, no model in the process.

Jina, OpenAI, Google (via its OpenAI-compatible endpoint), Voyage, DeepInfra and
Together all speak the same ``POST {base_url}/embeddings`` dialect::

    {"model": "...", "input": ["text one", "text two"]}
    -> {"data": [{"index": 0, "embedding": [...]}, ...]}

so a single adapter covers all of them; they differ only in base URL, model name
and credential.

Vectors are L2-normalised here rather than trusting the provider, so the store's
dot product is always exactly cosine similarity.
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterator, Sequence

import httpx

from app.logging_config import get_logger

log = get_logger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend cannot be reached or returns an error."""


#: base_url and default model for the providers with a usable free tier.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "jina": {
        "base_url": "https://api.jina.ai/v1",
        "model": "jina-embeddings-v3",
        "signup": "https://jina.ai/embeddings/ (free tier, no card)",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
        "signup": "https://platform.openai.com/api-keys (paid)",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "text-embedding-004",
        "signup": "https://aistudio.google.com/apikey (free tier)",
    },
}


class RemoteEmbedder:
    """An :class:`~app.interfaces.Embedder` backed by an HTTP embeddings API.

    Args:
        provider: Label used in the embedder identity and in logs.
        model: Model identifier as the provider names it.
        base_url: API root; ``/embeddings`` is appended.
        api_key: Bearer credential.
        dimension: Expected vector width, for a fail-fast sanity check. ``None``
            skips the check.
        batch_size: Texts per request during ingestion.
        timeout_s: Per-request timeout.
        max_retries: Retries for transient failures (429 / 5xx / connection).
        transport: Injected HTTP transport. Tests pass an ``httpx.MockTransport``
            so the whole adapter is exercised without a network or a key.
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        dimension: int | None = None,
        batch_size: int = 32,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._dimension = dimension
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._transport = transport
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        """Provider-qualified identity, recorded in the index manifest."""
        return f"{self._provider}:{self._model}"

    @property
    def max_input_tokens(self) -> int:
        """Generous upper bound.

        Hosted embedding models accept 8k tokens or more, so unlike the local
        MiniLM (256) chunk size is not constrained by the encoder here.
        """
        return 8192

    def count_tokens(self, text: str) -> int:
        """Approximate token count without shipping a tokenizer.

        Used only by the chunker. ~4 characters per token is the standard
        English approximation and is accurate enough to size chunks; exactness
        would mean a tokenizer dependency the runtime image does not want.
        """
        return max(1, math.ceil(len(text) / 4))

    @property
    def client(self) -> httpx.Client:
        """Lazily constructed HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_s,
                transport=self._transport,
                headers={
                    "authorization": f"Bearer {self._api_key}",
                    "content-type": "application/json",
                },
            )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages, chunked into provider-sized requests."""
        vectors: list[list[float]] = []
        for batch in _batched(list(texts), self._batch_size):
            vectors.extend(self._request(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single question."""
        return self._request([text])[0]

    def _request(self, texts: list[str]) -> list[list[float]]:
        """POST one batch, with retry and backoff.

        Raises:
            EmbeddingError: If every attempt fails or the response is malformed.
        """
        if not texts:
            return []
        payload = {"model": self._model, "input": texts}
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.post("/embeddings", json=payload)
                response.raise_for_status()
                vectors = _parse_embeddings(response.json(), expected=len(texts))
                log.debug(
                    "embedding.completed",
                    extra={
                        "event": "embedding.completed",
                        "model": self.name,
                        "texts": len(texts),
                        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
                return [_normalise(v, self._dimension, self.name) for v in vectors]
            except Exception as exc:
                last_error = exc
                retryable = _is_retryable(exc)
                log.warning(
                    "embedding.attempt_failed",
                    extra={
                        "event": "embedding.attempt_failed",
                        "model": self.name,
                        "attempt": attempt + 1,
                        "retryable": retryable,
                        "error": str(exc)[:300],
                    },
                )
                if not retryable or attempt == self._max_retries:
                    break
                time.sleep(_retry_after(exc) or 2**attempt)

        raise EmbeddingError(
            f"{self.name} failed to embed {len(texts)} text(s): {last_error}"
        ) from last_error

    def close(self) -> None:
        """Release the HTTP connection pool."""
        if self._client is not None:
            self._client.close()
            self._client = None


def _parse_embeddings(body: Any, *, expected: int) -> list[list[float]]:
    """Pull vectors out of an OpenAI-shaped embeddings response, in input order."""
    try:
        data = body["data"]
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors = [item["embedding"] for item in ordered]
    except (KeyError, TypeError) as exc:
        raise EmbeddingError(f"unexpected embeddings response shape: {str(body)[:300]}") from exc
    if len(vectors) != expected:
        raise EmbeddingError(f"asked for {expected} embeddings, got {len(vectors)}")
    return vectors


def _normalise(vector: Sequence[float], dimension: int | None, name: str) -> list[float]:
    """L2-normalise, and check the width matches what the index expects."""
    if dimension is not None and len(vector) != dimension:
        raise EmbeddingError(
            f"{name} returned {len(vector)}-dimensional vectors but the index expects {dimension}"
        )
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise EmbeddingError(f"{name} returned a zero vector")
    return [value / norm for value in vector]


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    """Yield fixed-size slices."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _is_retryable(exc: Exception) -> bool:
    """Whether an embedding failure is worth another attempt."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 429) or status >= 500
    return isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout))


def _retry_after(exc: Exception) -> float | None:
    """Honour a provider's Retry-After header when it sends one."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("retry-after")
    try:
        return min(float(raw), 60.0) if raw else None
    except (TypeError, ValueError):
        return None
