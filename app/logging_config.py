"""Structured logging.

Every log line is a JSON object with a stable ``event`` key, so query logs can be
grepped, shipped to a log aggregator, or loaded into a dataframe without parsing
prose. Anything passed via ``extra=`` becomes a top-level field.

Example::

    log.info("query.completed", extra={"event": "query.completed", "latency_ms": 812})
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

#: Correlates every log line emitted while handling one request or CLI action.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}


def new_request_id() -> str:
    """Generate a short correlation ID."""
    return uuid4().hex[:12]


class _RequestIdFilter(logging.Filter):
    """Stamps the current correlation ID onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """Renders records as single-line JSON, preserving ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        payload.update(
            {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and k != "event"}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-friendly formatter for local development (``LOG_FORMAT=text``)."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS and k != "event"
        }
        suffix = " " + " ".join(f"{k}={v!r}" for k, v in extras.items()) if extras else ""
        return f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.getMessage()}{suffix}"


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install the root logging handler. Safe to call more than once.

    Args:
        level: Standard logging level name.
        fmt: ``"json"`` for machine-readable output, ``"text"`` for humans.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party libraries are chatty at INFO and drown out our own events.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper so modules do not import ``logging`` directly."""
    return logging.getLogger(name)
