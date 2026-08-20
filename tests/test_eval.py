"""Tests for eval-set parsing and grading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import Chunk, RetrievedChunk
from app.prompts import REFUSAL_TEXT
from eval import EvalCase, load_eval_set, grade_answer, grade_retrieval, summarise, CaseResult


def chunk(text: str, episode: str = "ep-01") -> RetrievedChunk:
    """Build a retrieved chunk for grading tests."""
    return RetrievedChunk(
        chunk=Chunk(id=f"{episode}:0", text=text, episode=episode, start=0.0, end=10.0, index=0),
        score=0.5,
    )


def test_expectations_accept_a_bare_string() -> None:
    case = EvalCase.from_dict({"question": "q", "expected_answer_contains": "burnout"})

    assert case.expected_answer_contains == ("burnout",)


def test_load_accepts_a_list_or_a_cases_object(tmp_path: Path) -> None:
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([{"question": "a"}]))
    as_object = tmp_path / "object.json"
    as_object.write_text(json.dumps({"cases": [{"question": "a"}]}))

    assert len(load_eval_set(as_list)) == len(load_eval_set(as_object)) == 1


def test_load_rejects_an_unknown_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"questions": []}))

    assert load_eval_set(path) == []  # {"cases": ...} missing -> no cases

    path.write_text(json.dumps("nope"))
    with pytest.raises(ValueError):
        load_eval_set(path)


def test_retrieval_hit_on_substring() -> None:
    case = EvalCase(question="q", expected_answer_contains=("burnout",))

    assert grade_retrieval(case, [chunk("we talked about BURNOUT")]) is True
    assert grade_retrieval(case, [chunk("tabs versus spaces")]) is False


def test_retrieval_hit_on_expected_episode() -> None:
    case = EvalCase(question="q", expected_episode="ep-02")

    assert grade_retrieval(case, [chunk("anything", episode="ep-02")]) is True
    assert grade_retrieval(case, [chunk("anything", episode="ep-01")]) is False


def test_retrieval_is_not_graded_for_refusal_cases() -> None:
    case = EvalCase(question="q", expect_refusal=True)

    assert grade_retrieval(case, [chunk("anything")]) is None


def test_answer_grading_requires_every_substring() -> None:
    case = EvalCase(question="q", expected_answer_contains=("burnout", "sabbatical"))

    assert grade_answer(case, "They took a sabbatical after burnout [1].") is True
    assert grade_answer(case, "They discussed burnout [1].") is False


def test_a_refusal_never_passes_a_content_case() -> None:
    case = EvalCase(question="q", expected_answer_contains=("burnout",))

    assert grade_answer(case, REFUSAL_TEXT) is False


def test_refusal_cases_pass_only_when_the_model_declines() -> None:
    case = EvalCase(question="q", expect_refusal=True)

    assert grade_answer(case, REFUSAL_TEXT) is True
    assert grade_answer(case, "The capital of Mongolia is Ulaanbaatar.") is False


def test_summary_separates_retrieval_from_answer_accuracy() -> None:
    results = [
        CaseResult("a", "x", retrieval_hit=True, answer_hit=True, refused=False, top_score=0.8, latency_ms=100),
        CaseResult("b", "x", retrieval_hit=False, answer_hit=False, refused=False, top_score=0.2, latency_ms=200),
        CaseResult("c", "x", retrieval_hit=None, answer_hit=True, refused=True, top_score=0.1, latency_ms=300),
    ]

    summary = summarise(results)

    assert summary["cases"] == 3
    assert summary["retrieval_recall"] == 0.5  # only the two graded cases count
    assert summary["retrieval_graded"] == 2
    assert summary["answer_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["p50_latency_ms"] == 200
