"""Embedding adapter backed by sentence-transformers (local, no API key)."""

from __future__ import annotations

import time
from typing import Any, Sequence

from app.logging_config import get_logger

log = get_logger(__name__)


class SentenceTransformerEmbedder:
    """Local :class:`~app.interfaces.Embedder` using a sentence-transformers model.

    The model is loaded lazily on first use so that importing this module (in the
    API process, in tests, in ``--help`` paths) stays cheap.

    Vectors are L2-normalised on the way out, which lets the vector store treat
    cosine distance as ``1 - dot``.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any | None = None

    @property
    def name(self) -> str:
        """The model identifier."""
        return self._model_name

    @property
    def model(self) -> Any:
        """The underlying ``SentenceTransformer``, loaded on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import

            started = time.perf_counter()
            self._model = SentenceTransformer(self._model_name, device=self._device)
            log.info(
                "embedder.loaded",
                extra={
                    "event": "embedder.loaded",
                    "model": self._model_name,
                    "device": self._device,
                    "max_input_tokens": self._model.get_max_seq_length(),
                    "load_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        return self._model

    @property
    def max_input_tokens(self) -> int:
        """Sequence length beyond which the model truncates input."""
        return int(self.model.get_max_seq_length())

    @property
    def dimension(self) -> int:
        """Output vector width."""
        return int(self.model.get_sentence_embedding_dimension())

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's own tokenizer (no special tokens)."""
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        ``all-MiniLM-L6-v2`` is a symmetric model: queries and passages use the
        same encoder with no instruction prefix.
        """
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Run the encoder and return plain Python lists."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
