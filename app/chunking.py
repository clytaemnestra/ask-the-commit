"""Transcript chunking.

The chunker is pure: it takes segments and a token-counting function and returns
:class:`~app.models.Chunk` objects. No model loading, no I/O — which makes it the
easiest part of the system to unit-test and to tune.

Design notes (the "why", expanded in the README):

* **Segments are atomic.** faster-whisper emits sentence-ish segments with their
  own timestamps. Packing whole segments into a chunk means every chunk boundary
  falls on a natural pause, and the chunk's ``start`` is a real timestamp you can
  seek to in the audio rather than an interpolation.
* **Sizes are measured in the embedding model's own tokens**, not characters or
  words, so ``chunk_tokens`` means the same thing to the chunker and to the model.
* **Overlap is carried as trailing segments**, so an answer that straddles a
  boundary survives in at least one chunk.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

from app.models import Chunk, Transcript, TranscriptSegment, slugify

#: A function that returns the number of tokens in a string.
TokenCounter = Callable[[str], int]

#: Segment paired with its precomputed token count.
_Sized = tuple[TranscriptSegment, int]


def whitespace_token_counter(text: str) -> int:
    """Approximate token count by whitespace splitting.

    Only used as a fallback (tests, or previewing chunking without loading the
    embedding model). Real ingestion uses the embedder's tokenizer.
    """
    return len(text.split())


def chunk_transcript(
    transcript: Transcript,
    *,
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: TokenCounter = whitespace_token_counter,
) -> list[Chunk]:
    """Split a transcript into overlapping, timestamped chunks.

    Args:
        transcript: The episode transcript to chunk.
        max_tokens: Target maximum tokens per chunk.
        overlap_tokens: Tokens of trailing context repeated in the next chunk.
        count_tokens: Tokenizer-backed counting function.

    Returns:
        Chunks in reading order, each with a deterministic ID and a time range.

    Raises:
        ValueError: If ``overlap_tokens`` is not smaller than ``max_tokens``,
            which would stop the window from advancing.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    sized = _prepare(transcript.segments, max_tokens=max_tokens, count_tokens=count_tokens)
    if not sized:
        return []

    chunks: list[Chunk] = []
    window: list[_Sized] = []
    window_tokens = 0

    for segment, n_tokens in sized:
        if window and window_tokens + n_tokens > max_tokens:
            chunks.append(_build_chunk(window, transcript.episode, len(chunks)))
            window, window_tokens = _carry_overlap(window, overlap_tokens)
        window.append((segment, n_tokens))
        window_tokens += n_tokens

    if window:
        chunks.append(_build_chunk(window, transcript.episode, len(chunks)))
    return chunks


def _prepare(
    segments: Iterable[TranscriptSegment],
    *,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[_Sized]:
    """Drop empty segments, split oversized ones, and attach token counts."""
    prepared: list[_Sized] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        normalised = TranscriptSegment(text=text, start=segment.start, end=segment.end)
        n_tokens = count_tokens(text)
        if n_tokens <= max_tokens:
            prepared.append((normalised, n_tokens))
            continue
        prepared.extend(_split_oversized(normalised, max_tokens=max_tokens, count_tokens=count_tokens))
    return prepared


def _split_oversized(
    segment: TranscriptSegment,
    *,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[_Sized]:
    """Split a single over-long segment on word boundaries.

    Rare in practice — whisper segments are a few seconds each — but a segment
    longer than ``max_tokens`` would otherwise produce a chunk the embedder
    silently truncates. Timestamps for the pieces are interpolated linearly
    across the segment's duration.
    """
    words = segment.text.split()
    if len(words) <= 1:
        return [(segment, count_tokens(segment.text))]

    word_tokens = [max(1, count_tokens(word)) for word in words]
    duration = max(segment.end - segment.start, 0.0)
    pieces: list[_Sized] = []
    start_idx = 0
    running = 0

    for idx, n_tokens in enumerate(word_tokens):
        if running and running + n_tokens > max_tokens:
            pieces.append(_slice_words(segment, words, start_idx, idx, duration, running))
            start_idx, running = idx, 0
        running += n_tokens

    if start_idx < len(words):
        pieces.append(_slice_words(segment, words, start_idx, len(words), duration, running))
    return pieces


def _slice_words(
    segment: TranscriptSegment,
    words: Sequence[str],
    start_idx: int,
    end_idx: int,
    duration: float,
    n_tokens: int,
) -> _Sized:
    """Build one piece of a split segment with interpolated timestamps."""
    total = len(words)
    start = segment.start + duration * (start_idx / total)
    end = segment.start + duration * (end_idx / total)
    text = " ".join(words[start_idx:end_idx])
    return TranscriptSegment(text=text, start=start, end=end), n_tokens


def _carry_overlap(window: list[_Sized], overlap_tokens: int) -> tuple[list[_Sized], int]:
    """Return the trailing segments to repeat at the head of the next chunk.

    At least one segment from the emitted chunk is always dropped, guaranteeing
    the window advances even when a single segment is close to ``max_tokens``.
    """
    if overlap_tokens <= 0 or len(window) < 2:
        return [], 0

    carried: list[_Sized] = []
    total = 0
    for segment, n_tokens in reversed(window[1:]):
        if total + n_tokens > overlap_tokens:
            break
        carried.append((segment, n_tokens))
        total += n_tokens
    carried.reverse()
    return carried, total


def _build_chunk(window: Sequence[_Sized], episode: str, index: int) -> Chunk:
    """Materialise the current window into a :class:`Chunk`."""
    segments = [segment for segment, _ in window]
    text = " ".join(segment.text for segment in segments).strip()
    return Chunk(
        id=f"{slugify(episode)}:{index:05d}",
        text=text,
        episode=episode,
        start=segments[0].start,
        end=segments[-1].end,
        index=index,
        n_tokens=sum(n for _, n in window),
    )
