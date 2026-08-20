"""In-memory answer cache.

Demo traffic repeats heavily — visitors click the same example questions — and a
repeat costs ~1.5s of generation plus Groq quota for an answer we already have.
Caching by question removes both.

Deliberately in-memory only:

* the deployed host has no persistent disk, and the free tier idles the process
  out anyway, so anything on disk would be lost or stale;
* a cold start rebuilding an empty cache costs one generation call, which is the
  right trade against a persistence layer.

Thread-safe because FastAPI runs the pipeline in a threadpool, so several
requests can touch the cache at once.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

from app.logging_config import get_logger

log = get_logger(__name__)

T = TypeVar("T")


def normalise_question(question: str, top_k: int | None) -> str:
    """Build a cache key from a question.

    Case, surrounding whitespace and trailing punctuation are insignificant, so
    "What is chat control?" and "what is chat control" share an entry. ``top_k``
    is part of the key because it changes the retrieved context and therefore
    the answer.
    """
    text = re.sub(r"\s+", " ", question).strip().lower().rstrip("?!. ")
    return f"{top_k or 'default'}::{text}"


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Counters for observability."""

    hits: int
    misses: int
    size: int
    max_entries: int

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class AnswerCache(Generic[T]):
    """A bounded, optionally-expiring LRU cache.

    Args:
        max_entries: Capacity. ``0`` disables caching entirely, which keeps the
            call sites free of conditionals.
        ttl_seconds: Optional lifetime per entry. ``None`` means entries live
            until evicted. Answers only change when the index or model changes —
            both of which restart the process — so a TTL is optional here.
    """

    def __init__(self, max_entries: int = 256, ttl_seconds: float | None = None) -> None:
        self._max_entries = max(0, max_entries)
        self._ttl = ttl_seconds
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """Whether this cache stores anything."""
        return self._max_entries > 0

    def get(self, key: str) -> T | None:
        """Return a cached value, or ``None`` on miss or expiry."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            stored_at, value = entry
            if self._ttl is not None and time.monotonic() - stored_at > self._ttl:
                del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)  # most recently used
            self._hits += 1
            return value

    def put(self, key: str, value: T) -> None:
        """Store a value, evicting the least recently used entry if full."""
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                evicted, _ = self._entries.popitem(last=False)
                log.debug("cache.evicted", extra={"event": "cache.evicted", "key": evicted})

    def clear(self) -> None:
        """Drop every entry. Counters are kept."""
        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        """Snapshot of hit/miss counters and occupancy."""
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._entries),
                max_entries=self._max_entries,
            )
