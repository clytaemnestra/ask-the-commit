"""Per-client rate limiting for the one endpoint that costs money.

``/ask`` embeds a question and calls a generation backend; ``/`` and ``/health``
are static reads. Only ``/ask`` is worth protecting, and what it is protected
against is narrow and specific: a public demo link that gets shared further than
intended, quietly draining a free-tier token budget.

The implementation is a sliding-window log — the timestamps of a client's recent
requests, with anything older than the window dropped on read. Compared with a
fixed window it costs a little more memory and refuses correctly at the boundary,
where a fixed window lets through two full bursts back to back.

Two properties worth stating plainly, because both are deliberate:

* **In-memory and per-process.** One free-tier instance is one process, so this
  is exact there. Behind more than one replica each would enforce its own quota;
  that needs shared state (Redis) and is not what this deployment is.
* **Keyed on an IP.** Which is a weak identity — a NAT shares one, a determined
  caller rotates through many. This raises the cost of casual abuse; it is not
  authentication. See ``trust_proxy_headers`` for the header-spoofing caveat.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass

from app.logging_config import get_logger

log = get_logger(__name__)

#: Header set by every reverse proxy worth the name, including Render's.
FORWARDED_FOR = "x-forwarded-for"


class RateLimitExceeded(Exception):
    """Raised when a client has spent its allowance for the current window.

    Args:
        retry_after: Seconds until the oldest request leaves the window.
        limit: Requests permitted per window, for the error message.
    """

    def __init__(self, *, retry_after: float, limit: int) -> None:
        super().__init__(f"rate limit exceeded: {limit} requests per window")
        self.retry_after = retry_after
        self.limit = limit


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one rate-limit check."""

    allowed: bool
    remaining: int
    retry_after: float


class RateLimiter:
    """A thread-safe sliding-window limiter, bounded in memory.

    Args:
        max_requests: Requests allowed per client per window. ``0`` disables the
            limiter entirely, which keeps the call site free of conditionals.
        window_seconds: Width of the sliding window.
        max_clients: How many distinct clients to track. Past this the least
            recently seen is forgotten, so a flood of unique IPs costs bounded
            memory rather than unbounded. Forgetting a client resets its
            allowance, which is the right way to fail: a limiter that could be
            pushed into swapping would be a denial of service in itself.
    """

    def __init__(
        self, *, max_requests: int, window_seconds: float, max_clients: int = 4096
    ) -> None:
        self._max_requests = max(0, max_requests)
        self._window = window_seconds
        self._max_clients = max(1, max_clients)
        self._seen: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether this limiter refuses anything."""
        return self._max_requests > 0

    @property
    def limit(self) -> int:
        """Requests permitted per window."""
        return self._max_requests

    @property
    def tracked_clients(self) -> int:
        """How many clients are currently held in memory."""
        with self._lock:
            return len(self._seen)

    def check(self, client: str) -> Decision:
        """Record a request from ``client`` and decide whether to allow it.

        A refused request is *not* recorded, so a client hammering a closed door
        does not push its own window further out.
        """
        if not self.enabled:
            return Decision(allowed=True, remaining=-1, retry_after=0.0)

        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            hits = self._seen.get(client)
            if hits is None:
                hits = deque()
                self._seen[client] = hits
            self._seen.move_to_end(client)

            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self._max_requests:
                retry_after = max(0.0, hits[0] + self._window - now)
                return Decision(allowed=False, remaining=0, retry_after=retry_after)

            hits.append(now)
            remaining = self._max_requests - len(hits)

            while len(self._seen) > self._max_clients:
                forgotten, _ = self._seen.popitem(last=False)
                log.debug(
                    "ratelimit.forgot_client",
                    extra={"event": "ratelimit.forgot_client", "client": forgotten},
                )

        return Decision(allowed=True, remaining=remaining, retry_after=0.0)

    def reset(self) -> None:
        """Forget every client. Used by tests."""
        with self._lock:
            self._seen.clear()


def client_key(
    *, forwarded_for: str | None, peer: str | None, trust_proxy_headers: bool
) -> str:
    """Identify the caller for rate-limiting purposes.

    Args:
        forwarded_for: Raw ``X-Forwarded-For`` header, if present.
        peer: The socket peer address.
        trust_proxy_headers: Whether the header may be believed.

    Returns:
        The client's IP, or ``"unknown"`` when there is nothing to key on.

    Note:
        With ``trust_proxy_headers`` on, the leftmost entry of
        ``X-Forwarded-For`` is used — that is the originating client when a
        proxy sets the header, and an arbitrary attacker-chosen string when
        nothing does. Behind Render, Fly or any load balancer the header is
        rewritten and this is correct. Directly exposed, it is a bypass, which
        is why it is a setting and not a default assumption.
    """
    if trust_proxy_headers and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"
