"""Tests for the retrieve-then-generate loop, using fake providers."""

from __future__ import annotations

from app.chunking import chunk_transcript
from app.prompts import REFUSAL_TEXT
from rag import RagPipeline
import pytest

from app.models import Chunk
from tests.conftest import FakeChatModel


def chunk_for(transcript) -> Chunk:
    """One chunk standing in for an indexed episode."""
    return Chunk(id="ep:0", text="text", episode=transcript.episode, start=0.0, end=5.0, index=0)


def index(transcript, embedder, store) -> None:
    """Chunk, embed and store a transcript through the real chunker."""
    chunks = chunk_transcript(transcript, max_tokens=20, overlap_tokens=5, count_tokens=embedder.count_tokens)
    store.upsert(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))


def test_answer_is_grounded_in_retrieved_chunks(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=2, min_score=-1.0)

    result = pipeline.answer("what did they say about burnout")

    assert result.answer == chat_model.reply
    assert result.sources
    assert not result.refused
    assert result.model == "fake:model"


def test_prompt_carries_the_context_and_the_grounding_rule(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=2, min_score=-1.0)

    pipeline.answer("burnout")

    system, user = chat_model.calls[0]
    assert REFUSAL_TEXT in system
    assert "CONTEXT:" in user
    assert "episode: ep-01" in user
    assert "QUESTION: burnout" in user


def test_low_similarity_refuses_without_calling_the_model(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=3, min_score=0.99)

    result = pipeline.answer("completely unrelated quantum chromodynamics question")

    assert result.refused
    assert result.answer == REFUSAL_TEXT
    assert result.sources == []
    assert chat_model.calls == []  # no generation call was paid for


def test_empty_index_refuses(embedder, store, chat_model) -> None:
    pipeline = RagPipeline(embedder, store, chat_model, top_k=3, min_score=-1.0)

    result = pipeline.answer("anything at all")

    assert result.refused
    assert chat_model.calls == []


def test_model_refusal_drops_the_sources(transcript, embedder, store) -> None:
    chat_model = FakeChatModel(reply=REFUSAL_TEXT)
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=3, min_score=-1.0)

    result = pipeline.answer("what about burnout")

    assert result.refused
    assert result.sources == []
    assert result.retrieved  # the trace is still available for eval/debugging


def test_retrieval_trace_survives_a_refusal(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=2, min_score=0.99)

    result = pipeline.answer("burnout")

    assert result.retrieved
    assert result.sources == []


def test_latencies_are_recorded(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model, top_k=2, min_score=-1.0)

    result = pipeline.answer("burnout")

    assert result.retrieval_ms > 0
    assert result.total_ms >= result.retrieval_ms


def test_index_stats_reports_what_is_indexed(transcript, embedder, store, chat_model) -> None:
    index(transcript, embedder, store)
    pipeline = RagPipeline(embedder, store, chat_model)

    stats = pipeline.index_stats()

    assert stats["indexed_chunks"] > 0
    assert stats["indexed_episodes"] == 1
    assert stats["llm"] == "fake:model"


def test_index_compatibility_is_verified(transcript, embedder, store, chat_model, tmp_path) -> None:
    """A pipeline must refuse an index built by a different embedder."""
    from app.providers.store_numpy import IndexMismatchError, NumpyVectorStore

    numpy_store = NumpyVectorStore(tmp_path / "index")
    numpy_store.upsert([chunk_for(transcript)], [[1.0, 0.0]])
    numpy_store.set_manifest(embedder="some-other-embedder")

    pipeline = RagPipeline(embedder, numpy_store, chat_model)

    with pytest.raises(IndexMismatchError):
        pipeline.verify_index_compatibility()


def test_index_compatibility_passes_when_embedders_match(
    transcript, embedder, chat_model, tmp_path
) -> None:
    from app.providers.store_numpy import NumpyVectorStore

    numpy_store = NumpyVectorStore(tmp_path / "index")
    numpy_store.upsert([chunk_for(transcript)], [[1.0, 0.0]])
    numpy_store.set_manifest(embedder=embedder.name)

    RagPipeline(embedder, numpy_store, chat_model).verify_index_compatibility()


def test_stores_without_provenance_skip_the_check(transcript, embedder, store, chat_model) -> None:
    """The in-memory fake has no manifest; verification must be a no-op."""
    RagPipeline(embedder, store, chat_model).verify_index_compatibility()
