"""Offline evaluation harness.

Separates the two failure modes a RAG system actually has:

* **Retrieval**  — did the right chunk make it into the context at all?
  (If not, no prompt tweak will save the answer.)
* **Generation** — given a context that contained the answer, did the model say it?

Usage::

    python eval.py                       # run eval_set.json, print a table
    python eval.py --markdown            # emit a Markdown table for the README
    python eval.py --json results.json   # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.config import get_settings
from app.logging_config import configure_logging, get_logger, new_request_id, request_id_var
from app.models import RetrievedChunk
from app.prompts import is_refusal
from rag import RagPipeline, build_pipeline

log = get_logger(__name__)

DEFAULT_EVAL_SET = Path("eval_set.json")

#: LLMs emit typographic punctuation that breaks naive substring matching: a
#: narrow no-break space inside "Cheese Shop", a non-breaking hyphen in
#: "open-source". Fold all of it to ASCII, and treat dashes as spaces so that
#: "open-source" matches an expectation written "open source".
_PUNCTUATION = str.maketrans(
    {
        "‐": " ", "‑": " ", "‒": " ", "–": " ", "—": " ", "-": " ",
        "‘": "'", "’": "'", "“": '"', "”": '"',
    }
)


def normalise(text: str) -> str:
    """Fold text to a comparable form: NFKC, ASCII punctuation, collapsed spaces."""
    folded = unicodedata.normalize("NFKC", text).translate(_PUNCTUATION)
    return " ".join(folded.lower().split())


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One graded question.

    Attributes:
        question: The question to ask.
        expected_answer_contains: Substrings the answer must contain (all of
            them, case-insensitive). Accepts a single string in the JSON file.
        expected_episode: Optional episode name that retrieval should surface.
        expect_refusal: True for out-of-scope questions the system should decline.
    """

    question: str
    expected_answer_contains: tuple[str, ...] = ()
    expected_episode: str | None = None
    expect_refusal: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        """Parse one case, tolerating a string or a list for the expectations."""
        expected = data.get("expected_answer_contains", [])
        if isinstance(expected, str):
            expected = [expected]
        return cls(
            question=data["question"],
            expected_answer_contains=tuple(expected),
            expected_episode=data.get("expected_episode"),
            expect_refusal=bool(data.get("expect_refusal", False)),
        )


@dataclass
class CaseResult:
    """Graded outcome for a single case."""

    question: str
    answer: str
    retrieval_hit: bool | None
    answer_hit: bool
    refused: bool
    top_score: float | None
    scores: list[float] = field(default_factory=list)
    episodes: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A case passes on its answer grade; retrieval is diagnostic."""
        return self.answer_hit and self.error is None


def load_eval_set(path: Path) -> list[EvalCase]:
    """Load cases from JSON.

    Accepts either a bare list of cases or ``{"cases": [...]}``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is not a recognised shape.
    """
    if not path.exists():
        raise FileNotFoundError(f"eval set not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of cases or {{'cases': [...]}}")
    return [EvalCase.from_dict(case) for case in payload]


def grade_retrieval(case: EvalCase, retrieved: Sequence[RetrievedChunk]) -> bool | None:
    """Did retrieval surface a chunk that plausibly contains the answer?

    A hit means the expected episode was among the retrieved chunks, or one of
    the expected substrings appears verbatim in a retrieved chunk. Substring
    matching is a proxy — it under-counts paraphrase — but it needs no labelled
    chunk IDs, so the eval set stays cheap to write and extend.

    Returns ``None`` for refusal cases, where there is nothing to retrieve.
    """
    if case.expect_refusal:
        return None
    if case.expected_episode:
        return any(
            item.chunk.episode.lower() == case.expected_episode.lower() for item in retrieved
        )
    if not case.expected_answer_contains:
        return bool(retrieved)
    haystack = normalise("\n".join(item.chunk.text for item in retrieved))
    return any(normalise(needle) in haystack for needle in case.expected_answer_contains)


def grade_answer(case: EvalCase, answer: str) -> bool:
    """Did the generated answer contain what we expected?

    Refusal cases pass when the model declined; everything else requires every
    expected substring to be present (case-insensitive).
    """
    if case.expect_refusal:
        return is_refusal(answer)
    if is_refusal(answer):
        return False
    normalised = normalise(answer)
    return all(normalise(needle) in normalised for needle in case.expected_answer_contains)


def run_case(pipeline: RagPipeline, case: EvalCase, *, top_k: int | None = None) -> CaseResult:
    """Run one case end to end and grade it."""
    started = time.perf_counter()
    try:
        result = pipeline.answer(case.question, top_k=top_k)
    except Exception as exc:
        log.exception("eval.case_failed", extra={"event": "eval.case_failed", "question": case.question})
        return CaseResult(
            question=case.question,
            answer="",
            retrieval_hit=None,
            answer_hit=False,
            refused=False,
            top_score=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=str(exc),
        )

    retrieved = result.retrieved
    case_result = CaseResult(
        question=case.question,
        answer=result.answer,
        retrieval_hit=grade_retrieval(case, retrieved),
        answer_hit=grade_answer(case, result.answer),
        refused=result.refused,
        top_score=round(retrieved[0].score, 4) if retrieved else None,
        scores=[round(item.score, 4) for item in retrieved],
        episodes=sorted({item.chunk.episode for item in retrieved}),
        latency_ms=result.total_ms,
    )
    log.info(
        "eval.case_completed",
        extra={
            "event": "eval.case_completed",
            "question": case.question,
            "retrieval_hit": case_result.retrieval_hit,
            "answer_hit": case_result.answer_hit,
            "top_score": case_result.top_score,
            "latency_ms": round(case_result.latency_ms, 1),
        },
    )
    return case_result


def summarise(results: Sequence[CaseResult]) -> dict[str, Any]:
    """Aggregate case results into the headline numbers."""
    graded_retrieval = [r for r in results if r.retrieval_hit is not None]
    latencies = [r.latency_ms for r in results if r.error is None]
    return {
        "cases": len(results),
        "errors": sum(1 for r in results if r.error),
        "retrieval_recall": _ratio(sum(1 for r in graded_retrieval if r.retrieval_hit), len(graded_retrieval)),
        "retrieval_graded": len(graded_retrieval),
        "answer_accuracy": _ratio(sum(1 for r in results if r.passed), len(results)),
        "refusal_rate": _ratio(sum(1 for r in results if r.refused), len(results)),
        "mean_top_score": round(statistics.fmean([r.top_score for r in results if r.top_score is not None]), 4)
        if any(r.top_score is not None for r in results)
        else None,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "mean_latency_ms": round(statistics.fmean(latencies), 1) if latencies else None,
    }


def print_report(results: Sequence[CaseResult], summary: dict[str, Any]) -> None:
    """Print a plain-text table plus the aggregate scores."""
    print(f"\n{'':2} {'RETR':<5} {'ANS':<4} {'SCORE':<6} {'MS':>7}  QUESTION")
    for result in results:
        retrieval = {True: "hit", False: "MISS", None: "n/a"}[result.retrieval_hit]
        print(
            f"{'PASS' if result.passed else 'FAIL':<4} "
            f"{retrieval:<5} {'ok' if result.answer_hit else 'no':<4} "
            f"{result.top_score if result.top_score is not None else '-':<6} "
            f"{result.latency_ms:>7.0f}  {result.question[:70]}"
        )
        if result.error:
            print(f"      error: {result.error}")

    print("\n--- summary ---")
    print(f"cases              : {summary['cases']} ({summary['errors']} errored)")
    print(f"retrieval recall@k : {_pct(summary['retrieval_recall'])}  ({summary['retrieval_graded']} graded)")
    print(f"answer accuracy    : {_pct(summary['answer_accuracy'])}")
    print(f"refusal rate       : {_pct(summary['refusal_rate'])}")
    print(f"mean top score     : {summary['mean_top_score']}")
    print(f"latency p50 / mean : {summary['p50_latency_ms']} ms / {summary['mean_latency_ms']} ms\n")


def print_markdown(results: Sequence[CaseResult], summary: dict[str, Any]) -> None:
    """Emit a Markdown table, ready to paste into the README."""
    print("| Question | Retrieval | Answer | Top score | Latency |")
    print("| --- | --- | --- | --- | --- |")
    for result in results:
        retrieval = {True: "hit", False: "miss", None: "n/a"}[result.retrieval_hit]
        score = f"{result.top_score:.3f}" if result.top_score is not None else "-"
        answer = "pass" if result.answer_hit else "fail"
        print(f"| {result.question} | {retrieval} | {answer} | {score} | {result.latency_ms:.0f} ms |")
    print(
        f"\n**Retrieval recall@k** {_pct(summary['retrieval_recall'])} · "
        f"**Answer accuracy** {_pct(summary['answer_accuracy'])} · "
        f"**p50 latency** {summary['p50_latency_ms']} ms"
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    """Safe division that returns ``None`` rather than dividing by zero."""
    return round(numerator / denominator, 4) if denominator else None


def _pct(value: float | None) -> str:
    """Render a ratio as a percentage, or ``n/a``."""
    return "n/a" if value is None else f"{value * 100:.0f}%"


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline against a fixed question set.")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="Path to the eval set JSON.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval depth for the run.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    parser.add_argument("--json", type=Path, default=None, help="Also write full results to this JSON file.")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table instead of plain text.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    request_id_var.set(new_request_id())

    cases = load_eval_set(args.eval_set)[: args.limit]
    if not cases:
        print(f"No cases in {args.eval_set}.")
        return 1

    pipeline = build_pipeline(settings)
    results = [run_case(pipeline, case, top_k=args.top_k) for case in cases]
    summary = summarise(results)

    log.info("eval.completed", extra={"event": "eval.completed", **summary})

    if args.markdown:
        print_markdown(results, summary)
    else:
        print_report(results, summary)

    if args.json:
        args.json.write_text(
            json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")

    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
