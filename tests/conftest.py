"""Test doubles for every provider protocol.

Because the pipeline only depends on the protocols in :mod:`app.interfaces`,
the whole retrieve-then-generate loop can be tested with no model downloads, no
database and no network.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import pytest

from app.models import Chunk, RetrievedChunk, Transcript, TranscriptSegment

_DIM = 64


def _bag_of_words_vector(text: str) -> list[float]:
    """A deterministic, normalised hashed bag-of-words embedding.

    Not semantic, but lexically sensible: texts sharing words score higher. That
    is enough to assert on retrieval ordering.
    """
    vector = [0.0] * _DIM
    for word in text.lower().split():
        vector[hash(word) % _DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class FakeEmbedder:
    """In-memory :class:`~app.interfaces.Embedder`."""

    name = "fake-embedder"
    max_input_tokens = 256

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_bag_of_words_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _bag_of_words_vector(text)


class FakeVectorStore:
    """In-memory :class:`~app.interfaces.VectorStore` with exact cosine search."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, embeddings):
            self._items[chunk.id] = (chunk, list(vector))

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=chunk, score=sum(a * b for a, b in zip(embedding, vector)))
            for chunk, vector in self._items.values()
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def delete_episode(self, episode: str) -> None:
        self._items = {k: v for k, v in self._items.items() if v[0].episode != episode}

    def episodes(self) -> set[str]:
        return {chunk.episode for chunk, _ in self._items.values()}

    def count(self) -> int:
        return len(self._items)


class FakeChatModel:
    """Scripted :class:`~app.interfaces.ChatModel` that records its prompts."""

    name = "fake:model"

    def __init__(self, reply: str = "They discussed burnout at length [1].") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


class FakeTranscriber:
    """:class:`~app.interfaces.Transcriber` returning a fixed transcript."""

    name = "fake-transcriber"

    def __init__(self, transcript: Transcript) -> None:
        self._transcript = transcript

    def transcribe(self, audio_path: Path, *, episode: str) -> Transcript:
        return self._transcript


@pytest.fixture
def transcript() -> Transcript:
    """A short transcript with one segment every five seconds."""
    words = [
        "we talked about burnout and rest",
        "the guest built a database company",
        "advice for founders is to talk to users",
        "then we argued about tabs versus spaces",
        "and finally we covered hiring mistakes",
    ]
    segments = [
        TranscriptSegment(text=text, start=index * 5.0, end=index * 5.0 + 5.0)
        for index, text in enumerate(words)
    ]
    return Transcript(episode="ep-01", segments=segments, language="en", duration=25.0)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def chat_model() -> FakeChatModel:
    return FakeChatModel()
