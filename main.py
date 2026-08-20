"""FastAPI service exposing the RAG pipeline.

Run locally::

    uvicorn main:app --reload

Endpoints:
    GET  /        — minimal single-page query UI
    POST /ask     — ask a question, get a grounded answer plus sources
    GET  /health  — liveness plus index statistics
    GET  /docs    — generated OpenAPI docs
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app import __version__
from app.config import get_settings
from app.factory import ConfigurationError, validate_serving_config
from app.interfaces import GenerationError
from app.logging_config import configure_logging, get_logger, new_request_id, request_id_var
from app.schemas import AskRequest, AskResponse, ErrorResponse, HealthResponse
from rag import RagPipeline, build_pipeline

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration, load the index, and cache health facts.

    Anything expensive or fallible happens here rather than on the request path:

    * Missing credentials fail the boot with an actionable message instead of
      surfacing as a 500 on someone's first question.
    * The index is read from disk once. ``GET /health`` then answers from a
      cached dict, so an uptime pinger never touches the index or an API.
    """
    started = time.perf_counter()
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    validate_serving_config(settings)  # raises ConfigurationError -> boot fails loudly

    # read_only: the deployed host has no persistent disk, so writes must be
    # impossible rather than merely unused.
    pipeline = build_pipeline(settings, read_only=True)
    pipeline.verify_index_compatibility()  # wrong embedder -> silent garbage, so fail here
    pipeline.warm_up()  # pay the ~900ms model load here, not on the first question
    app.state.pipeline = pipeline
    app.state.stats = pipeline.index_stats()
    app.state.boot_seconds = round(time.perf_counter() - started, 2)

    log.info(
        "startup.completed",
        extra={
            "event": "startup.completed",
            "version": __version__,
            "boot_seconds": app.state.boot_seconds,
            **app.state.stats,
        },
    )
    yield
    log.info("shutdown.completed", extra={"event": "shutdown.completed"})


app = FastAPI(
    title="Ask the Commit",
    version=__version__,
    summary="Question answering grounded in local podcast transcripts.",
    description=(
        "Answers are generated only from transcript chunks retrieved from the local "
        "Chroma index. When the index has nothing relevant, the service says so "
        "rather than guessing."
    ),
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> RagPipeline:
    """Fetch the process-wide pipeline built during startup."""
    return request.app.state.pipeline


@app.middleware("http")
async def request_context(request: Request, call_next: Callable) -> Response:
    """Attach a correlation ID to every request and log one access record."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["x-request-id"] = request_id
    log.info(
        "http.request",
        extra={
            "event": "http.request",
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
        },
    )
    return response


@app.exception_handler(GenerationError)
async def _generation_error_handler(request: Request, exc: GenerationError) -> JSONResponse:
    """Upstream model failures are a bad gateway, not a server bug."""
    log.error("generation.failed", extra={"event": "generation.failed", "error": str(exc)})
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(detail=f"generation backend unavailable: {exc}", request_id=request_id_var.get()).model_dump(),
    )


@app.exception_handler(ConfigurationError)
async def _configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    """Missing credentials or an unknown provider: the service is misconfigured."""
    log.error("configuration.invalid", extra={"event": "configuration.invalid", "error": str(exc)})
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(detail=str(exc), request_id=request_id_var.get()).model_dump(),
    )


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={502: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Ask a question about the podcast archive",
)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question from the indexed transcripts.

    The response always includes the chunks the answer was grounded in. When the
    index contains nothing relevant, ``refused`` is true, ``sources`` is empty
    and the answer is the fixed "not covered" sentence.
    """
    pipeline = get_pipeline(request)
    # Embedding and the call to the generation backend both block; running them
    # inline would stall every other request in the process.
    result = await run_in_threadpool(pipeline.answer, payload.question, top_k=payload.top_k)
    return AskResponse.from_answer(result, request_id_var.get())


#: Single-page UI. Read once at import; it is a static asset, not a template.
_UI_PATH = Path(__file__).parent / "app" / "static" / "index.html"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the minimal query UI.

    Falls back to a pointer at ``/docs`` if the asset is missing, so a trimmed
    deployment image without the static folder still starts.
    """
    if not _UI_PATH.exists():
        return HTMLResponse('<h1>Ask the Commit</h1><p>API docs at <a href="/docs">/docs</a>.</p>')
    return HTMLResponse(_UI_PATH.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, summary="Liveness and index statistics")
async def health(request: Request) -> HealthResponse:
    """Liveness check. Answers from memory — no index access, no API calls.

    Free-tier hosts idle a service out after a few minutes without traffic, so
    this is designed to be hit by an external pinger every few minutes: it must
    stay cheap enough that keeping the service warm costs nothing.

    Returns ``status="degraded"`` when the index is empty — the service is up but
    every question would be refused until ``ingest.py`` has run.
    """
    stats = request.app.state.stats
    return HealthResponse(
        status="ok" if int(stats["indexed_chunks"]) > 0 else "degraded",
        version=__version__,
        indexed_chunks=int(stats["indexed_chunks"]),
        indexed_episodes=int(stats["indexed_episodes"]),
        embedding_model=str(stats["embedding_model"]),
        llm=str(stats["llm"]),
        boot_seconds=getattr(request.app.state, "boot_seconds", 0.0),
    )


if __name__ == "__main__":  # pragma: no cover - convenience for `python main.py`
    import uvicorn

    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=False)
