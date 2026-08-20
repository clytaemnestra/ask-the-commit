"""The RAG core: embed a question, retrieve chunks, generate a grounded answer.

Run it directly to exercise the loop without the API::

    python rag.py "what did they say about burnout?"
    python rag.py --top-k 8 --json "who was the guest in episode 3?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Sequence

from app.config import Settings, get_settings
from app.factory import build_chat_model, build_embedder, build_vector_store
from app.interfaces import ChatModel, Embedder, GenerationError, VectorStore
from app.logging_config import configure_logging, get_logger, new_request_id, request_id_var
from app.models import RagAnswer, RetrievedChunk
from app.prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_user_prompt, is_refusal

log = get_logger(__name__)


class RagPipeline:
    """Retrieval-augmented question answering over indexed podcast transcripts.

    The pipeline owns no provider logic: it is handed an :class:`Embedder`, a
    :class:`VectorStore` and a :class:`ChatModel` and orchestrates them.

    Args:
        embedder: Encodes the question into the same space as the indexed chunks.
        store: Similarity search over those chunks.
        chat_model: Generation backend.
        top_k: Default number of chunks to retrieve.
        min_score: Cosine-similarity floor. When the best hit falls below it the
            pipeline refuses immediately instead of paying for a generation call
            that would (correctly) refuse anyway. Set below -1 to disable.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        chat_model: ChatModel,
        *,
        top_k: int = 5,
        min_score: float = 0.2,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chat_model = chat_model
        self._top_k = top_k
        self._min_score = min_score

    @property
    def model_name(self) -> str:
        """Identifier of the active generation backend."""
        return self._chat_model.name

    def index_stats(self) -> dict[str, object]:
        """Summary of what is indexed and which providers are wired up.

        Used by ``GET /health``; touches the store but performs no generation.
        """
        return {
            "indexed_chunks": self._store.count(),
            "indexed_episodes": len(self._store.episodes()),
            "embedding_model": self._embedder.name,
            "llm": self._chat_model.name,
        }

    def verify_index_compatibility(self) -> None:
        """Check the index was built by the embedder that will encode questions.

        A mismatch here does not raise at query time — it just returns nonsense
        neighbours — so it has to be caught at boot. Stores without provenance
        (Chroma) simply skip the check.

        Raises:
            IndexMismatchError: If the index names a different embedder.
        """
        verify = getattr(self._store, "verify_embedder", None)
        if verify is not None:
            verify(self._embedder.name)

    def warm_up(self) -> None:
        """Force lazy loads so the first request is not the slow one.

        Loading the ONNX session and running the first inference costs ~900ms;
        paying that at boot instead of on a visitor's first question is the whole
        point. For a hosted embedder this instead spends one tiny API call and
        warms the connection pool, which is a fair trade either way.
        """
        self._store.count()
        self._embedder.embed_query("warm up")

    def retrieve(self, question: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed the question and return the most similar chunks, best first."""
        vector = self._embedder.embed_query(question)
        return self._store.search(vector, top_k=top_k or self._top_k)

    def answer(self, question: str, *, top_k: int | None = None) -> RagAnswer:
        """Answer a question from the indexed transcripts.

        Args:
            question: The user's question.
            top_k: Override the configured retrieval depth.

        Returns:
            A :class:`~app.models.RagAnswer` with the answer text, the chunks it
            was grounded in, and per-stage latencies.

        Raises:
            GenerationError: If the generation backend cannot be reached.
        """
        question = question.strip()
        started = time.perf_counter()

        retrieval_started = time.perf_counter()
        chunks = self.retrieve(question, top_k=top_k)
        retrieval_ms = _elapsed_ms(retrieval_started)

        if self._below_threshold(chunks):
            result = self._refusal(question, chunks, retrieval_ms, started)
            self._log_query(result, chunks, short_circuited=True)
            return result

        generation_started = time.perf_counter()
        text = self._chat_model.complete(
            system=SYSTEM_PROMPT, user=build_user_prompt(question, chunks)
        )
        generation_ms = _elapsed_ms(generation_started)

        result = RagAnswer(
            question=question,
            answer=text,
            sources=[] if is_refusal(text) else chunks,
            retrieved=list(chunks),
            refused=is_refusal(text),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=_elapsed_ms(started),
            model=self._chat_model.name,
        )
        self._log_query(result, chunks, short_circuited=False)
        return result

    def _below_threshold(self, chunks: Sequence[RetrievedChunk]) -> bool:
        """Whether retrieval was too weak to be worth a generation call."""
        return not chunks or chunks[0].score < self._min_score

    def _refusal(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        retrieval_ms: float,
        started: float,
    ) -> RagAnswer:
        """Build the canned refusal without calling the generation backend."""
        return RagAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            sources=[],
            retrieved=list(chunks),
            refused=True,
            retrieval_ms=retrieval_ms,
            generation_ms=0.0,
            total_ms=_elapsed_ms(started),
            model=self._chat_model.name,
        )

    def _log_query(
        self, result: RagAnswer, retrieved: Sequence[RetrievedChunk], *, short_circuited: bool
    ) -> None:
        """Emit one structured record per query: scores, chunks, latencies."""
        log.info(
            "query.completed",
            extra={
                "event": "query.completed",
                "question": result.question,
                "model": result.model,
                "top_k": len(retrieved),
                "scores": [round(chunk.score, 4) for chunk in retrieved],
                "top_score": round(retrieved[0].score, 4) if retrieved else None,
                "chunks_used": [chunk.chunk.id for chunk in retrieved],
                "episodes": sorted({chunk.chunk.episode for chunk in retrieved}),
                "retrieval_ms": round(result.retrieval_ms, 1),
                "generation_ms": round(result.generation_ms, 1),
                "total_ms": round(result.total_ms, 1),
                "refused": result.refused,
                "short_circuited": short_circuited,
                "answer_chars": len(result.answer),
            },
        )


def build_pipeline(settings: Settings | None = None, *, read_only: bool = False) -> RagPipeline:
    """Wire a pipeline from settings. The one place providers are chosen.

    Args:
        settings: Application settings; defaults to the process singleton.
        read_only: Serving mode — the store rejects every write.
    """
    settings = settings or get_settings()
    return RagPipeline(
        embedder=build_embedder(settings),
        store=build_vector_store(settings, read_only=read_only),
        chat_model=build_chat_model(settings),
        top_k=settings.top_k,
        min_score=settings.min_score,
    )


def _elapsed_ms(since: float) -> float:
    """Milliseconds elapsed since a ``perf_counter`` reading."""
    return (time.perf_counter() - since) * 1000


def _render(result: RagAnswer) -> str:
    """Format an answer for the terminal."""
    lines = [result.answer, ""]
    if result.sources:
        lines.append("Sources:")
        lines += [
            f"  [{i}] {src.chunk.citation}  (score {src.score:.3f})"
            for i, src in enumerate(result.sources, start=1)
        ]
    lines.append(
        f"\n{result.model} | retrieval {result.retrieval_ms:.0f}ms | "
        f"generation {result.generation_ms:.0f}ms | total {result.total_ms:.0f}ms"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ad-hoc questions."""
    parser = argparse.ArgumentParser(description="Ask a question against the indexed podcast transcripts.")
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve.")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    request_id_var.set(new_request_id())

    try:
        result = build_pipeline(settings).answer(args.question, top_k=args.top_k)
    except GenerationError as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "refused": result.refused,
                    "model": result.model,
                    "sources": [
                        {
                            "episode": s.chunk.episode,
                            "timestamp": s.chunk.citation,
                            "score": round(s.score, 4),
                            "chunk_id": s.chunk.id,
                        }
                        for s in result.sources
                    ],
                    "latency_ms": round(result.total_ms, 1),
                },
                indent=2,
            )
        )
    else:
        print(_render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

