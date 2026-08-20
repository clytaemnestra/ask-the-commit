"""Ingestion pipeline: audio -> transcript -> chunks -> embeddings -> vector index.

Usage::

    python ingest.py                     # ingest anything in episodes/ that is not indexed yet
    python ingest.py --force             # re-chunk and re-embed everything (reuses cached transcripts)
    python ingest.py --retranscribe      # also throw away cached transcripts
    python ingest.py --dry-run           # show what would happen

Transcripts are cached under ``data/transcripts/`` as JSON. Transcription is by
far the most expensive step, so tuning chunk size and re-indexing is cheap: only
the chunk/embed/store stages re-run.

The output is the deployed artifact. With ``VECTOR_STORE=numpy`` (the default)
this writes ``data/index/`` — three small files that are committed to the repo
and memory-mapped read-only by the service. The index records which embedder
built it, and the service refuses to start if that does not match the embedder
configured to encode questions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.chunking import chunk_transcript
from app.config import Settings, get_settings
from app.factory import build_embedder, build_transcriber, build_vector_store
from app.interfaces import Embedder, Transcriber, VectorStore
from app.logging_config import configure_logging, get_logger, new_request_id, request_id_var
from app.models import Chunk, Transcript, slugify

log = get_logger(__name__)

#: Extensions faster-whisper (via ffmpeg) can decode.
AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".m4b", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm", ".mkv"}
)


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    """What happened to one episode during an ingest run."""

    episode: str
    status: str  # "indexed" | "skipped" | "empty" | "failed"
    chunks: int = 0
    tokens: int = 0
    duration_s: float = 0.0
    elapsed_s: float = 0.0
    transcribed: bool = False
    detail: str = ""


def discover_episodes(episodes_dir: Path) -> list[Path]:
    """Return every audio file in ``episodes_dir``, sorted by name.

    Hidden files and unsupported extensions are ignored, so a stray ``.DS_Store``
    or cover art does not blow up a run.
    """
    if not episodes_dir.exists():
        return []
    return sorted(
        path
        for path in episodes_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def episode_name(audio_path: Path) -> str:
    """Derive the citable episode name from a filename (the stem)."""
    return audio_path.stem


def transcript_cache_path(cache_dir: Path, episode: str) -> Path:
    """Where the cached transcript for ``episode`` lives."""
    return cache_dir / f"{slugify(episode)}.json"


def load_or_transcribe(
    audio_path: Path,
    *,
    transcriber: Transcriber,
    cache_dir: Path,
    retranscribe: bool = False,
) -> tuple[Transcript, bool]:
    """Return the episode transcript, using the on-disk cache when possible.

    Args:
        audio_path: The audio file.
        transcriber: Backend used when the cache misses.
        cache_dir: Directory holding cached transcript JSON.
        retranscribe: Ignore any cached transcript and transcribe again.

    Returns:
        ``(transcript, transcribed_now)`` — the flag says whether the expensive
        path was taken, which the run summary reports.
    """
    episode = episode_name(audio_path)
    cache_path = transcript_cache_path(cache_dir, episode)

    if cache_path.exists() and not retranscribe:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("transcriber") == transcriber.name:
            log.info(
                "transcript.cache_hit",
                extra={"event": "transcript.cache_hit", "episode": episode, "path": str(cache_path)},
            )
            return Transcript.from_dict(payload), False

    transcript = transcriber.transcribe(audio_path, episode=episode)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({**transcript.to_dict(), "transcriber": transcriber.name}, ensure_ascii=False),
        encoding="utf-8",
    )
    return transcript, True


def embed_and_store(
    chunks: Sequence[Chunk], *, embedder: Embedder, store: VectorStore, episode: str
) -> None:
    """Embed chunks and replace the episode's entries in the vector store."""
    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    log.info(
        "embed.completed",
        extra={
            "event": "embed.completed",
            "episode": episode,
            "chunks": len(chunks),
            "elapsed_s": round(time.perf_counter() - started, 2),
        },
    )
    store.delete_episode(episode)  # idempotent re-ingest: no duplicates, no stale chunks
    store.upsert(chunks, vectors)


def ingest_episode(
    audio_path: Path,
    *,
    settings: Settings,
    transcriber: Transcriber,
    embedder: Embedder,
    store: VectorStore,
    retranscribe: bool = False,
) -> EpisodeReport:
    """Run one episode through the full pipeline."""
    episode = episode_name(audio_path)
    started = time.perf_counter()
    log.info("ingest.started", extra={"event": "ingest.started", "episode": episode, "file": str(audio_path)})

    transcript, transcribed = load_or_transcribe(
        audio_path, transcriber=transcriber, cache_dir=settings.transcripts_dir, retranscribe=retranscribe
    )

    chunks = chunk_transcript(
        transcript,
        max_tokens=settings.chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        count_tokens=embedder.count_tokens,
    )
    if not chunks:
        log.warning("ingest.empty", extra={"event": "ingest.empty", "episode": episode})
        return EpisodeReport(episode, "empty", transcribed=transcribed, detail="no speech detected")

    embed_and_store(chunks, embedder=embedder, store=store, episode=episode)

    report = EpisodeReport(
        episode=episode,
        status="indexed",
        chunks=len(chunks),
        tokens=sum(chunk.n_tokens for chunk in chunks),
        duration_s=transcript.duration,
        elapsed_s=time.perf_counter() - started,
        transcribed=transcribed,
    )
    log.info(
        "ingest.completed",
        extra={
            "event": "ingest.completed",
            "episode": episode,
            "chunks": report.chunks,
            "mean_chunk_tokens": round(report.tokens / report.chunks, 1),
            "audio_duration_s": round(report.duration_s, 1),
            "elapsed_s": round(report.elapsed_s, 1),
            "transcribed": transcribed,
        },
    )
    return report


def ingest(
    *,
    settings: Settings,
    force: bool = False,
    retranscribe: bool = False,
    limit: int | None = None,
) -> list[EpisodeReport]:
    """Ingest every new episode in the configured folder.

    Args:
        settings: Application settings.
        force: Re-index episodes that are already in the store.
        retranscribe: Ignore cached transcripts.
        limit: Process at most this many episodes (handy for a first smoke test).

    Returns:
        One report per episode considered.
    """
    audio_files = discover_episodes(settings.episodes_dir)
    if not audio_files:
        log.warning(
            "ingest.no_audio",
            extra={
                "event": "ingest.no_audio",
                "episodes_dir": str(settings.episodes_dir),
                "supported": sorted(AUDIO_EXTENSIONS),
            },
        )
        return []

    transcriber = build_transcriber(settings)
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    _warn_if_chunks_exceed_context(settings, embedder)

    already_indexed = store.episodes()
    reports: list[EpisodeReport] = []

    for audio_path in audio_files[:limit]:
        episode = episode_name(audio_path)
        if episode in already_indexed and not force:
            log.info("ingest.skipped", extra={"event": "ingest.skipped", "episode": episode, "reason": "already indexed"})
            reports.append(EpisodeReport(episode, "skipped", detail="already indexed; use --force to redo"))
            continue
        try:
            reports.append(
                ingest_episode(
                    audio_path,
                    settings=settings,
                    transcriber=transcriber,
                    embedder=embedder,
                    store=store,
                    retranscribe=retranscribe,
                )
            )
        except Exception as exc:  # keep going: one bad file should not stop the run
            log.exception("ingest.failed", extra={"event": "ingest.failed", "episode": episode, "error": str(exc)})
            reports.append(EpisodeReport(episode, "failed", detail=str(exc)))

    _record_provenance(store, embedder=embedder, settings=settings)
    log.info(
        "ingest.run_completed",
        extra={
            "event": "ingest.run_completed",
            "episodes": len(reports),
            "indexed": sum(r.status == "indexed" for r in reports),
            "chunks": sum(r.chunks for r in reports),
            "total_chunks_in_store": store.count(),
        },
    )
    return reports


def _record_provenance(store: VectorStore, *, embedder: Embedder, settings: Settings) -> None:
    """Stamp the index with the embedder that built it.

    Query vectors must come from the same model as the indexed vectors, or
    retrieval quietly degrades instead of failing. The serving path compares its
    configured embedder against this record and refuses to start on a mismatch.

    Only the NumPy store carries a manifest; Chroma keeps its own metadata.
    """
    setter = getattr(store, "set_manifest", None)
    if setter is None:
        return
    setter(
        embedder=embedder.name,
        embedding_provider=settings.embedding_provider,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        transcriber=f"faster-whisper:{settings.whisper_model}",
    )
    log.info("index.provenance_recorded", extra={"event": "index.provenance_recorded", "embedder": embedder.name})


def _warn_if_chunks_exceed_context(settings: Settings, embedder: Embedder) -> None:
    """Warn when chunks are longer than the embedding model can actually read.

    ``all-MiniLM-L6-v2`` truncates at 256 word-piece tokens, so a 500-token chunk
    is stored with only its first half represented in the vector. Retrieval still
    works, but the tail of each chunk is invisible to search.
    """
    limit = embedder.max_input_tokens
    if settings.chunk_tokens > limit:
        log.warning(
            "chunking.exceeds_embedder_context",
            extra={
                "event": "chunking.exceeds_embedder_context",
                "chunk_tokens": settings.chunk_tokens,
                "embedder_max_tokens": limit,
                "model": embedder.name,
                "impact": "text beyond the model's limit is truncated before embedding",
                "suggestion": f"set CHUNK_TOKENS<={limit} (see README: chunking approach)",
            },
        )


def _print_summary(reports: Sequence[EpisodeReport], store: VectorStore) -> None:
    """Print a human-readable run summary."""
    if not reports:
        print("No episodes found. Drop audio files into the episodes/ folder and re-run.")
        return

    width = max(len(report.episode) for report in reports)
    print(f"\n{'EPISODE'.ljust(width)}  STATUS    CHUNKS  AUDIO     ELAPSED  DETAIL")
    for report in reports:
        print(
            f"{report.episode.ljust(width)}  {report.status:<8}  {report.chunks:>6}  "
            f"{report.duration_s / 60:>6.1f}m  {report.elapsed_s:>6.1f}s  {report.detail}"
        )
    indexed = sum(report.status == "indexed" for report in reports)
    print(f"\n{indexed} episode(s) indexed | {store.count()} chunks in the store\n")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Transcribe, chunk, embed and index podcast episodes.")
    parser.add_argument("--episodes-dir", type=Path, default=None, help="Override EPISODES_DIR.")
    parser.add_argument("--force", action="store_true", help="Re-index episodes that are already indexed.")
    parser.add_argument("--retranscribe", action="store_true", help="Ignore the transcript cache.")
    parser.add_argument("--limit", type=int, default=None, help="Ingest at most N episodes.")
    parser.add_argument("--chunk-tokens", type=int, default=None, help="Override CHUNK_TOKENS for this run.")
    parser.add_argument("--overlap-tokens", type=int, default=None, help="Override CHUNK_OVERLAP_TOKENS.")
    parser.add_argument("--dry-run", action="store_true", help="List the files that would be ingested and exit.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.episodes_dir:
        settings = settings.model_copy(update={"episodes_dir": args.episodes_dir})
    if args.chunk_tokens:
        settings = settings.model_copy(update={"chunk_tokens": args.chunk_tokens})
    if args.overlap_tokens is not None:
        settings = settings.model_copy(update={"chunk_overlap_tokens": args.overlap_tokens})

    configure_logging(settings.log_level, settings.log_format)
    request_id_var.set(new_request_id())

    if args.dry_run:
        files = discover_episodes(settings.episodes_dir)
        print(f"{len(files)} audio file(s) in {settings.episodes_dir}:")
        for path in files[: args.limit]:
            print(f"  {episode_name(path)}  ({path.suffix}, {path.stat().st_size / 1e6:.1f} MB)")
        return 0

    reports = ingest(
        settings=settings, force=args.force, retranscribe=args.retranscribe, limit=args.limit
    )
    _print_summary(reports, build_vector_store(settings))
    return 1 if any(report.status == "failed" for report in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
