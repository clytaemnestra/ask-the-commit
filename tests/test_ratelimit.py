"""Tests for the sliding-window rate limiter and its wiring into /ask."""

from __future__ import annotations

import time

import pytest

from app.ratelimit import RateLimiter, client_key


def test_requests_under_the_limit_are_allowed() -> None:
    limiter = RateLimiter(max_requests=3, window_seconds=60)

    decisions = [limiter.check("1.2.3.4") for _ in range(3)]

    assert all(d.allowed for d in decisions)
    assert [d.remaining for d in decisions] == [2, 1, 0]


def test_the_next_request_over_the_limit_is_refused() -> None:
    limiter = RateLimiter(max_requests=2, window_seconds=60)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")

    decision = limiter.check("1.2.3.4")

    assert not decision.allowed
    assert 0 < decision.retry_after <= 60


def test_clients_are_limited_independently() -> None:
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.check("1.1.1.1").allowed
    assert not limiter.check("1.1.1.1").allowed
    assert limiter.check("2.2.2.2").allowed


def test_the_window_slides() -> None:
    """A window that has fully elapsed frees the whole allowance."""
    limiter = RateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.check("1.2.3.4").allowed
    assert not limiter.check("1.2.3.4").allowed

    time.sleep(0.06)

    assert limiter.check("1.2.3.4").allowed


def test_a_refused_request_does_not_extend_the_window() -> None:
    """Hammering a closed door must not push the client's reset further out."""
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check("1.2.3.4")

    first = limiter.check("1.2.3.4").retry_after
    for _ in range(5):
        limiter.check("1.2.3.4")
    last = limiter.check("1.2.3.4").retry_after

    assert last <= first


def test_zero_disables_the_limiter() -> None:
    limiter = RateLimiter(max_requests=0, window_seconds=60)

    assert not limiter.enabled
    assert all(limiter.check("1.2.3.4").allowed for _ in range(100))


def test_tracked_clients_are_bounded() -> None:
    """A flood of unique IPs costs bounded memory, not unbounded."""
    limiter = RateLimiter(max_requests=5, window_seconds=60, max_clients=10)

    for i in range(100):
        limiter.check(f"10.0.0.{i}")

    assert limiter.tracked_clients == 10


@pytest.mark.parametrize(
    ("forwarded", "peer", "trust", "expected"),
    [
        ("203.0.113.7", "10.0.0.1", True, "203.0.113.7"),
        ("203.0.113.7, 10.0.0.1", "10.0.0.1", True, "203.0.113.7"),
        ("  203.0.113.7  ", "10.0.0.1", True, "203.0.113.7"),
        # Not trusted: the header is ignored, so it cannot be used to bypass.
        ("203.0.113.7", "10.0.0.1", False, "10.0.0.1"),
        (None, "10.0.0.1", True, "10.0.0.1"),
        ("", "10.0.0.1", True, "10.0.0.1"),
        (None, None, True, "unknown"),
    ],
)
def test_client_key_resolution(
    forwarded: str | None, peer: str | None, trust: bool, expected: str
) -> None:
    assert client_key(forwarded_for=forwarded, peer=peer, trust_proxy_headers=trust) == expected
