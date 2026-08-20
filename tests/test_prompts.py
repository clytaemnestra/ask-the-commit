"""Tests for prompt assembly and refusal detection."""

from __future__ import annotations

import pytest

from app.models import Chunk, RetrievedChunk, format_timestamp, slugify
from app.prompts import REFUSAL_TEXT, build_user_prompt, format_context, is_refusal


def retrieved(text: str, *, episode: str = "ep-01", start: float = 90.0) -> RetrievedChunk:
    """Build a retrieved chunk for prompt tests."""
    chunk = Chunk(id=f"{episode}:0", text=text, episode=episode, start=start, end=start + 30, index=0)
    return RetrievedChunk(chunk=chunk, score=0.5)


def test_context_is_numbered_and_attributed() -> None:
    context = format_context([retrieved("first"), retrieved("second", episode="ep-02", start=3600)])

    assert "[1] episode: ep-01 | time: 1:30-2:00" in context
    assert "[2] episode: ep-02 | time: 1:00:00-1:00:30" in context
    assert "first" in context and "second" in context


def test_user_prompt_puts_the_question_after_the_context() -> None:
    prompt = build_user_prompt("why?", [retrieved("because")])

    assert prompt.index("CONTEXT:") < prompt.index("QUESTION: why?")
    assert REFUSAL_TEXT in prompt


@pytest.mark.parametrize(
    "answer",
    [
        REFUSAL_TEXT,
        "this isn't covered in the episodes",
        "This isn’t covered in the episodes.",  # curly apostrophe
        "  This isn't covered in the episodes!  ",
    ],
)
def test_refusals_are_recognised_despite_formatting(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    ["They discussed burnout [1].", "The episodes cover this in detail.", ""],
)
def test_real_answers_are_not_treated_as_refusals(answer: str) -> None:
    assert not is_refusal(answer)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0:00"), (75.4, "1:15"), (599, "9:59"), (3600, "1:00:00"), (3725, "1:02:05")],
)
def test_timestamp_formatting(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("Ep 01 - Burnout!", "ep-01-burnout"), ("  ", "episode"), ("A/B", "a-b")],
)
def test_slugify(value: str, expected: str) -> None:
    assert slugify(value) == expected
