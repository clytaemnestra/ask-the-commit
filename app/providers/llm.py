"""Generation adapters.

Groq, Ollama and OpenAI all speak the same OpenAI chat-completions wire format,
so one adapter covers all three — they differ only in base URL, model name and
credentials. Anything that does *not* speak that dialect (Anthropic, Bedrock,
llama.cpp's native server) can be added as a sibling class implementing
:class:`~app.interfaces.ChatModel` without touching the RAG pipeline.
"""

from __future__ import annotations

import re
import time
from typing import Any

from app.interfaces import GenerationError
from app.logging_config import get_logger

log = get_logger(__name__)


class OpenAICompatibleChatModel:
    """A :class:`~app.interfaces.ChatModel` for any OpenAI-compatible endpoint.

    Args:
        provider: Label used in logs and in the ``name`` property.
        model: Model identifier as the provider names it.
        base_url: Chat-completions base URL (``.../v1``).
        api_key: Credential; providers that need none (Ollama) accept a dummy.
        temperature: Sampling temperature. Low by default — this is extractive QA.
        max_tokens: Cap on generated tokens.
        timeout_s: Per-request timeout.
        max_retries: Retries for transient failures (429 / 5xx / connection).
    """

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float = 0.1,
        max_tokens: int = 700,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._provider = provider
        self._model = model
        self._base_url = base_url
        self._api_key = api_key or "not-needed"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: Any | None = None

    @property
    def name(self) -> str:
        """Provider-qualified model identifier, e.g. ``groq:llama-3.3-70b-versatile``."""
        return f"{self._provider}:{self._model}"

    @property
    def client(self) -> Any:
        """The OpenAI SDK client, constructed on first use."""
        if self._client is None:
            from openai import OpenAI

            # Retries are handled here so they can be logged; disable the SDK's.
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_s,
                max_retries=0,
            )
        return self._client

    def complete(self, *, system: str, user: str) -> str:
        """Generate a completion, retrying transient failures with backoff.

        Raises:
            GenerationError: If every attempt fails, or the response is empty.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return self._extract_text(response, started)
            except Exception as exc:  # narrowed by _is_retryable below
                last_error = exc
                retryable = _is_retryable(exc)
                log.warning(
                    "generation.attempt_failed",
                    extra={
                        "event": "generation.attempt_failed",
                        "model": self.name,
                        "attempt": attempt + 1,
                        "retryable": retryable,
                        "error": str(exc)[:300],
                    },
                )
                if not retryable or attempt == self._max_retries:
                    break
                time.sleep(_retry_delay(exc, attempt))

        raise GenerationError(f"{self.name} failed to generate a completion: {last_error}") from last_error

    def _extract_text(self, response: Any, started: float) -> str:
        """Pull the message text out of a chat-completions response."""
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        log.info(
            "generation.completed",
            extra={
                "event": "generation.completed",
                "model": self.name,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            },
        )
        if not text:
            raise GenerationError(f"{self.name} returned an empty completion")
        return text


class EchoChatModel:
    """Offline stand-in that returns the retrieved context verbatim.

    Useful for exercising ingestion, retrieval, the API and the eval harness with
    no network and no API key — set ``LLM_PROVIDER=echo``. It is *not* a real
    generator: it simply echoes the context it was given.
    """

    @property
    def name(self) -> str:
        """Identifier for logs."""
        return "echo:passthrough"

    def complete(self, *, system: str, user: str) -> str:
        """Return the context block from the prompt, unmodified."""
        marker = "CONTEXT:"
        body = user.split(marker, 1)[-1] if marker in user else user
        return f"[echo backend — no model called]\n{body.strip()[:2000]}"


#: Never sleep longer than this between attempts, whatever the provider asks for.
_MAX_BACKOFF_S = 60.0


def _retry_delay(exc: Exception, attempt: int, *, base: float = 2.0) -> float:
    """How long to wait before the next attempt.

    Providers that rate-limit by tokens-per-minute tell you exactly how long to
    wait; guessing with pure exponential backoff retries far too early and burns
    the attempt budget. Groq's free tier asks for ~9s while ``base ** attempt``
    would wait 1s, so the hint is worth honouring when present.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        seconds = _parse_duration(headers.get(header))
        if seconds is not None:
            return min(seconds + 0.5, _MAX_BACKOFF_S)  # small cushion for clock skew
    return min(base**attempt, _MAX_BACKOFF_S)


def _parse_duration(raw: str | None) -> float | None:
    """Parse a duration header into seconds.

    Handles the shapes these APIs actually emit: bare seconds (``"9"``), and
    suffixed values (``"8.865s"``, ``"500ms"``, ``"2m"``).
    """
    if not raw:
        return None
    text = str(raw).strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m|h)?", text)
    if not match:
        return None
    value = float(match.group(1))
    return value * {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]


def _is_retryable(exc: Exception) -> bool:
    """Decide whether a provider error is worth another attempt."""
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 408 or status == 429 or status >= 500
    # Connection/timeout errors carry no status code.
    return exc.__class__.__name__ in {"APIConnectionError", "APITimeoutError", "APIConnectionTimeoutError"}
