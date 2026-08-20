"""Prompt construction and grounding rules.

Kept in its own module because the prompt is the highest-leverage, most-edited
part of a RAG system: it should be reviewable in a diff without scrolling past
retrieval plumbing.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.models import RetrievedChunk, format_timestamp

#: The exact sentence the model must emit when the context is insufficient.
#: The API, the eval harness and the retrieval short-circuit all key off this.
REFUSAL_TEXT = "This isn't covered in the episodes."

SYSTEM_PROMPT = f"""You are a question-answering assistant for a podcast archive.

You answer strictly from the transcript excerpts supplied in the user message.

Rules:
1. Use ONLY the provided excerpts. Never use outside knowledge, and never guess
   or infer beyond what the excerpts state.
2. If the excerpts do not contain the answer, reply with exactly this sentence
   and nothing else: {REFUSAL_TEXT}
3. Cite the excerpts you used inline with bracketed numbers, e.g. [1] or [2][3].
   Every factual claim needs a citation.
4. The excerpts are automatic speech-to-text output: they contain filler words,
   mishearings and no punctuation guarantees. Paraphrase faithfully; do not
   silently correct facts.
5. Be concise — two to four sentences unless the question asks for detail.
6. If the excerpts only partially answer the question, answer the part they
   cover and say plainly which part is not covered."""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered, attributed context block.

    Args:
        chunks: Retrieved chunks, best first.

    Returns:
        One ``[n] episode @ start-end`` block per chunk, separated by blank lines.
    """
    blocks = []
    for position, retrieved in enumerate(chunks, start=1):
        chunk = retrieved.chunk
        header = (
            f"[{position}] episode: {chunk.episode} | "
            f"time: {format_timestamp(chunk.start)}-{format_timestamp(chunk.end)}"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Assemble the user message: context first, question last.

    The question is repeated after the context because instruction adherence
    degrades when the ask is buried above a long context block.
    """
    return (
        "CONTEXT:\n"
        f"{format_context(chunks)}\n\n"
        "END OF CONTEXT.\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above, with inline [n] citations. "
        f"If the context does not answer the question, reply exactly: {REFUSAL_TEXT}"
    )


def is_refusal(answer: str) -> bool:
    """Whether an answer is the "not covered" refusal.

    Compares loosely (case, punctuation, apostrophe style, surrounding text) so a
    model that adds a trailing period or a curly quote is still recognised.
    """
    normalised = re.sub(r"[^a-z ]", "", answer.lower().replace("’", "'"))
    target = re.sub(r"[^a-z ]", "", REFUSAL_TEXT.lower())
    return target.strip() in " ".join(normalised.split())
