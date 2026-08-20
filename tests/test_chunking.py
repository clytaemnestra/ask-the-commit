"""Tests for the transcript chunker — the part most worth getting right."""

from __future__ import annotations

import pytest

from app.chunking import chunk_transcript, whitespace_token_counter
from app.models import Transcript, TranscriptSegment


def make_transcript(n_segments: int, words_per_segment: int = 10) -> Transcript:
    """Build a synthetic transcript with predictable sizes and timestamps."""
    segments = [
        TranscriptSegment(
            text=" ".join(f"w{index}-{position}" for position in range(words_per_segment)),
            start=index * 5.0,
            end=index * 5.0 + 5.0,
        )
        for index in range(n_segments)
    ]
    return Transcript(episode="ep-01", segments=segments, duration=n_segments * 5.0)


def test_chunks_respect_the_token_budget() -> None:
    chunks = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=10)

    assert chunks
    assert all(chunk.n_tokens <= 50 for chunk in chunks)


def test_chunk_ids_are_deterministic_and_ordered() -> None:
    first = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=10)
    second = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=10)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.index for chunk in first] == list(range(len(first)))
    assert first[0].id.startswith("ep-01:")


def test_timestamps_are_monotonic_and_within_the_episode() -> None:
    chunks = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=10)

    assert all(chunk.start <= chunk.end for chunk in chunks)
    assert [chunk.start for chunk in chunks] == sorted(chunk.start for chunk in chunks)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 100.0


def test_overlap_repeats_trailing_content() -> None:
    chunks = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=20)

    # The tail of one chunk reappears at the head of the next.
    assert chunks[1].start < chunks[0].end


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_transcript(make_transcript(20), max_tokens=50, overlap_tokens=0)

    for previous, current in zip(chunks, chunks[1:]):
        assert current.start >= previous.end


def test_empty_and_blank_segments_are_dropped() -> None:
    transcript = Transcript(
        episode="ep-01",
        segments=[
            TranscriptSegment(text="  ", start=0.0, end=1.0),
            TranscriptSegment(text="real content here", start=1.0, end=2.0),
            TranscriptSegment(text="", start=2.0, end=3.0),
        ],
    )

    chunks = chunk_transcript(transcript, max_tokens=50, overlap_tokens=10)

    assert len(chunks) == 1
    assert chunks[0].text == "real content here"


def test_empty_transcript_yields_no_chunks() -> None:
    assert chunk_transcript(Transcript(episode="ep-01", segments=[]), max_tokens=50, overlap_tokens=10) == []


def test_oversized_segment_is_split_with_interpolated_timestamps() -> None:
    transcript = Transcript(
        episode="ep-01",
        segments=[TranscriptSegment(text=" ".join(f"w{i}" for i in range(120)), start=0.0, end=60.0)],
    )

    chunks = chunk_transcript(transcript, max_tokens=30, overlap_tokens=5)

    assert len(chunks) >= 4
    assert all(chunk.n_tokens <= 30 for chunk in chunks)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == pytest.approx(60.0)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_transcript(make_transcript(5), max_tokens=50, overlap_tokens=50)


def test_a_near_max_segment_still_advances_the_window() -> None:
    """A pathological case: every segment nearly fills a chunk on its own."""
    transcript = Transcript(
        episode="ep-01",
        segments=[
            TranscriptSegment(text=" ".join(f"w{i}-{j}" for j in range(9)), start=i * 5.0, end=i * 5.0 + 5.0)
            for i in range(6)
        ],
    )

    chunks = chunk_transcript(transcript, max_tokens=10, overlap_tokens=9)

    assert len(chunks) == 6  # one per segment; no infinite loop, no dropped content


def test_custom_token_counter_is_used() -> None:
    calls: list[str] = []

    def counting(text: str) -> int:
        calls.append(text)
        return whitespace_token_counter(text)

    chunk_transcript(make_transcript(3), max_tokens=50, overlap_tokens=10, count_tokens=counting)

    assert calls
