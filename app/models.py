"""Core domain models shared across ingestion, retrieval and generation.

These are deliberately plain dataclasses rather than pydantic models: they are
internal domain objects, not I/O schemas. The pydantic models used for the HTTP
boundary live in ``app.schemas``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Render a number of seconds as ``H:MM:SS`` (or ``M:SS`` under an hour).

    Args:
        seconds: Offset from the start of the episode, in seconds.

    Returns:
        A human-readable timestamp suitable for citations.

    >>> format_timestamp(75.4)
    '1:15'
    >>> format_timestamp(3725)
    '1:02:05'
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def slugify(value: str) -> str:
    """Turn an episode name into a filesystem/ID-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "episode"


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A single timestamped span of speech as produced by the transcriber."""

    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the on-disk transcript cache."""
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptSegment:
        """Rehydrate from the on-disk transcript cache."""
        return cls(text=data["text"], start=float(data["start"]), end=float(data["end"]))


@dataclass(frozen=True, slots=True)
class Transcript:
    """A full episode transcript plus the metadata needed to cite it."""

    episode: str
    segments: list[TranscriptSegment]
    language: str | None = None
    duration: float = 0.0
    source_path: str | None = None

    @property
    def text(self) -> str:
        """The whole transcript as one string (used for previews and debugging)."""
        return " ".join(segment.text for segment in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the on-disk transcript cache."""
        return {
            "episode": self.episode,
            "language": self.language,
            "duration": self.duration,
            "source_path": self.source_path,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Rehydrate from the on-disk transcript cache."""
        return cls(
            episode=data["episode"],
            segments=[TranscriptSegment.from_dict(s) for s in data["segments"]],
            language=data.get("language"),
            duration=float(data.get("duration", 0.0)),
            source_path=data.get("source_path"),
        )


@dataclass(frozen=True, slots=True)
class Chunk:
    """An embeddable slice of a transcript, anchored to a time range.

    ``id`` is deterministic (``<episode-slug>:<index>``) so that re-ingesting an
    episode overwrites its chunks instead of duplicating them.
    """

    id: str
    text: str
    episode: str
    start: float
    end: float
    index: int
    n_tokens: int = 0

    @property
    def citation(self) -> str:
        """A short human-readable source label, e.g. ``ep-01 @ 12:30``."""
        return f"{self.episode} @ {format_timestamp(self.start)}"

    def to_metadata(self) -> dict[str, Any]:
        """Metadata payload stored alongside the vector in Chroma.

        Chroma only accepts scalar metadata values (str/int/float/bool), so
        everything here is flattened.
        """
        return {
            "episode": self.episode,
            "start": float(self.start),
            "end": float(self.end),
            "index": int(self.index),
            "n_tokens": int(self.n_tokens),
            "timestamp": format_timestamp(self.start),
        }

    @classmethod
    def from_storage(cls, chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
        """Rebuild a chunk from what the vector store gave back."""
        return cls(
            id=chunk_id,
            text=text,
            episode=str(metadata.get("episode", "unknown")),
            start=float(metadata.get("start", 0.0)),
            end=float(metadata.get("end", 0.0)),
            index=int(metadata.get("index", 0)),
            n_tokens=int(metadata.get("n_tokens", 0)),
        )


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by the vector store, with its similarity score.

    ``score`` is a cosine similarity in ``[-1, 1]``; higher is more relevant.
    Distances are converted to similarities inside the store adapter so that no
    caller has to remember which direction "better" points in.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """The result of a full retrieve-then-generate cycle.

    Attributes:
        sources: Chunks the answer is grounded in — empty when the pipeline
            refused, so callers never cite a source for a non-answer.
        retrieved: Everything retrieval returned, refusal or not. This is the
            diagnostic trace the eval harness grades retrieval on.
    """

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    model: str = ""
