"""Provider-facing contracts.

Every external dependency (speech-to-text, embeddings, vector storage, text
generation) sits behind one of these ``Protocol`` classes. Application code —
``ingest.py``, ``rag.py``, ``eval.py`` — depends only on these, so swapping Groq
for Ollama or Chroma for pgvector is a change in ``app/factory.py`` and nowhere
else.

``Protocol`` rather than ABC is deliberate: adapters do not need to import or
inherit from anything here, which keeps the dependency arrow pointing one way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from app.models import Chunk, RetrievedChunk, Transcript


@runtime_checkable
class Transcriber(Protocol):
    """Turns an audio file into timestamped transcript segments."""

    @property
    def name(self) -> str:
        """Identifier for logs and transcript-cache invalidation."""
        ...

    def transcribe(self, audio_path: Path, *, episode: str) -> Transcript:
        """Transcribe one audio file.

        Args:
            audio_path: Path to the audio file.
            episode: Human-readable episode name used in citations.

        Returns:
            A transcript whose segments carry ``start``/``end`` offsets in seconds.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors.

    Implementations must return L2-normalised vectors so that the vector store
    can treat inner product and cosine similarity interchangeably.
    """

    @property
    def name(self) -> str:
        """Model identifier, recorded in logs and collection metadata."""
        ...

    @property
    def max_input_tokens(self) -> int:
        """Longest input the model encodes without truncation."""
        ...

    def count_tokens(self, text: str) -> int:
        """Token count under this model's own tokenizer.

        Chunking uses this so that chunk sizes are expressed in the units the
        embedding model actually cares about.
        """
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent similarity search over embedded chunks."""

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks and their vectors."""
        ...

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best first."""
        ...

    def delete_episode(self, episode: str) -> None:
        """Remove every chunk belonging to an episode (used for re-ingestion)."""
        ...

    def episodes(self) -> set[str]:
        """Names of every episode currently indexed."""
        ...

    def count(self) -> int:
        """Total number of stored chunks."""
        ...


@runtime_checkable
class ChatModel(Protocol):
    """Text generation backend."""

    @property
    def name(self) -> str:
        """Provider-qualified model identifier, e.g. ``groq:llama-3.3-70b-versatile``."""
        ...

    def complete(self, *, system: str, user: str) -> str:
        """Generate a completion for a system + user message pair.

        Raises:
            GenerationError: If the backend is unreachable or returns an error.
        """
        ...


class GenerationError(RuntimeError):
    """Raised when a :class:`ChatModel` cannot produce a completion."""
