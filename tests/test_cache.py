"""Tests for the answer cache."""

from __future__ import annotations

import time

import pytest

from app.cache import AnswerCache, normalise_question


def test_key_ignores_case_whitespace_and_trailing_punctuation() -> None:
    a = normalise_question("  What is CHAT   control? ", None)
    b = normalise_question("what is chat control", None)

    assert a == b


def test_key_separates_different_top_k() -> None:
    """top_k changes the retrieved context, so it must change the answer."""
    assert normalise_question("q", 3) != normalise_question("q", 5)


def test_key_distinguishes_different_questions() -> None:
    assert normalise_question("what is chat control", None) != normalise_question("who is kushal", None)


def test_hit_and_miss() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=4)

    assert cache.get("k") is None
    cache.put("k", "value")

    assert cache.get("k") == "value"
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.size) == (1, 1, 1)


def test_least_recently_used_is_evicted_first() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=2)
    cache.put("a", "1")
    cache.put("b", "2")
    cache.get("a")           # 'a' is now more recent than 'b'
    cache.put("c", "3")      # evicts 'b'

    assert cache.get("a") == "1"
    assert cache.get("b") is None
    assert cache.get("c") == "3"


def test_zero_size_disables_the_cache() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=0)
    cache.put("k", "value")

    assert cache.enabled is False
    assert cache.get("k") is None
    assert cache.stats().size == 0


def test_entries_expire_after_the_ttl() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=4, ttl_seconds=0.05)
    cache.put("k", "value")

    assert cache.get("k") == "value"
    time.sleep(0.06)
    assert cache.get("k") is None


def test_clear_empties_but_keeps_counters() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=4)
    cache.put("k", "v")
    cache.get("k")

    cache.clear()

    assert cache.get("k") is None
    assert cache.stats().hits == 1


def test_hit_rate() -> None:
    cache: AnswerCache[str] = AnswerCache(max_entries=4)
    cache.put("k", "v")
    cache.get("k")
    cache.get("k")
    cache.get("absent")

    assert cache.stats().hit_rate == pytest.approx(2 / 3)


def test_concurrent_access_is_safe() -> None:
    """FastAPI serves requests from a threadpool, so this is not hypothetical."""
    import threading

    cache: AnswerCache[int] = AnswerCache(max_entries=50)
    errors: list[Exception] = []

    def hammer(n: int) -> None:
        try:
            for i in range(200):
                cache.put(f"k{(n + i) % 80}", i)
                cache.get(f"k{i % 80}")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert cache.stats().size <= 50
