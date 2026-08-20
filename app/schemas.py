"""Pydantic request/response models for the HTTP boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import RagAnswer, RetrievedChunk


class AskRequest(BaseModel):
    """Body of ``POST /ask``."""

    question: str = Field(..., min_length=3, max_length=1000, examples=["What did they say about burnout?"])
    top_k: int | None = Field(None, ge=1, le=20, description="Override the configured number of chunks to retrieve.")


class Source(BaseModel):
    """One retrieved transcript chunk, as returned to the caller."""

    episode: str
    start_seconds: float
    end_seconds: float
    timestamp: str = Field(..., description="Human-readable start offset, e.g. '12:30'.")
    score: float = Field(..., description="Cosine similarity to the question; higher is better.")
    chunk_id: str
    text: str

    @classmethod
    def from_retrieved(cls, retrieved: RetrievedChunk) -> Source:
        """Project a domain object onto the wire schema."""
        chunk = retrieved.chunk
        return cls(
            episode=chunk.episode,
            start_seconds=round(chunk.start, 2),
            end_seconds=round(chunk.end, 2),
            timestamp=chunk.to_metadata()["timestamp"],
            score=round(retrieved.score, 4),
            chunk_id=chunk.id,
            text=chunk.text,
        )


class AskResponse(BaseModel):
    """Body of a successful ``POST /ask``."""

    question: str
    answer: str
    sources: list[Source]
    refused: bool = Field(..., description="True when the answer is the 'not covered in the episodes' fallback.")
    model: str
    request_id: str
    latency_ms: float
    retrieval_ms: float
    generation_ms: float

    @classmethod
    def from_answer(cls, result: RagAnswer, request_id: str) -> AskResponse:
        """Project a pipeline result onto the wire schema."""
        return cls(
            question=result.question,
            answer=result.answer,
            sources=[Source.from_retrieved(source) for source in result.sources],
            refused=result.refused,
            model=result.model,
            request_id=request_id,
            latency_ms=round(result.total_ms, 1),
            retrieval_ms=round(result.retrieval_ms, 1),
            generation_ms=round(result.generation_ms, 1),
        )


class HealthResponse(BaseModel):
    """Body of ``GET /health``."""

    status: str = Field(..., description="'ok' when the index has content, 'degraded' when it is empty.")
    version: str
    indexed_chunks: int
    indexed_episodes: int
    embedding_model: str
    llm: str
    boot_seconds: float = Field(0.0, description="How long startup took, for cold-start visibility.")


class ErrorResponse(BaseModel):
    """Body returned for handled error conditions."""

    detail: str
    request_id: str
