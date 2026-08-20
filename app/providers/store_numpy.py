"""Vector store backed by a committed NumPy array — no database process.

At 96 chunks (and comfortably up to ~100k) an exhaustive dot product over a
normalised matrix is faster than an ANN index and has no server, no SQLite file
and no write path. That makes it the right shape for a free-tier deployment:
the index is three files, committed to the repo, memory-mapped read-only at
startup.

Layout under ``data/index/``:

===================  ==========================================================
``embeddings.npy``   ``float32`` array of shape ``(n_chunks, dim)``, L2-normalised
``chunks.json``      chunk metadata and text, in the same row order
``manifest.json``    which embedder produced it — checked before serving
===================  ==========================================================

The manifest matters: query vectors must come from the same model as the indexed
vectors, or retrieval degrades silently rather than failing. The server compares
its configured embedder against the manifest and refuses to start on a mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.logging_config import get_logger
from app.models import Chunk, RetrievedChunk

log = get_logger(__name__)

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class IndexMismatchError(RuntimeError):
    """Raised when the query embedder does not match the one that built the index."""


class ReadOnlyStoreError(RuntimeError):
    """Raised on any attempt to write while in read-only (serving) mode."""


class NumpyVectorStore:
    """A :class:`~app.interfaces.VectorStore` over committed NumPy files.

    Args:
        index_dir: Directory holding the three index files.
        read_only: When true (the serving path), every write raises. Render's
            free tier has no persistent disk, so a runtime write would either
            fail or silently vanish on the next deploy — better to make it
            impossible than to hope it never happens.
    """

    def __init__(self, index_dir: Path, *, read_only: bool = False) -> None:
        self._dir = index_dir
        self._read_only = read_only
        self._embeddings: np.ndarray | None = None
        self._chunks: list[Chunk] | None = None
        self._manifest: dict[str, Any] | None = None

    # --- loading -----------------------------------------------------------

    @property
    def exists(self) -> bool:
        """Whether a built index is present on disk."""
        return (self._dir / EMBEDDINGS_FILE).exists() and (self._dir / CHUNKS_FILE).exists()

    def load(self) -> None:
        """Read the index into memory. Idempotent.

        Uses ``mmap_mode="r"`` so the array is read-only by construction and
        large indexes do not have to be copied into the heap at boot.
        """
        if self._embeddings is not None:
            return
        if not self.exists:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            self._chunks = []
            self._manifest = {}
            log.warning(
                "store.index_missing",
                extra={"event": "store.index_missing", "path": str(self._dir)},
            )
            return

        self._embeddings = np.load(self._dir / EMBEDDINGS_FILE, mmap_mode="r" if self._read_only else None)
        payload = json.loads((self._dir / CHUNKS_FILE).read_text(encoding="utf-8"))
        self._chunks = [_chunk_from_dict(item) for item in payload]
        self._manifest = self._read_manifest()

        if len(self._chunks) != self._embeddings.shape[0]:
            raise IndexMismatchError(
                f"index is inconsistent: {self._embeddings.shape[0]} vectors but "
                f"{len(self._chunks)} chunks in {self._dir}"
            )
        log.info(
            "store.loaded",
            extra={
                "event": "store.loaded",
                "path": str(self._dir),
                "chunks": len(self._chunks),
                "dimension": int(self._embeddings.shape[1]) if self._embeddings.size else 0,
                "embedder": self._manifest.get("embedder"),
            },
        )

    def _read_manifest(self) -> dict[str, Any]:
        """Load the manifest, tolerating an index built before manifests existed."""
        path = self._dir / MANIFEST_FILE
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def manifest(self) -> dict[str, Any]:
        """Provenance of the loaded index."""
        self.load()
        return dict(self._manifest or {})

    def verify_embedder(self, embedder_name: str) -> None:
        """Fail loudly if this index was built by a different embedder.

        Args:
            embedder_name: Identity of the embedder that will encode questions.

        Raises:
            IndexMismatchError: If the manifest names a different embedder.
        """
        recorded = self.manifest.get("embedder")
        if recorded and recorded != embedder_name:
            raise IndexMismatchError(
                f"index at {self._dir} was built with embedder {recorded!r} but the service "
                f"is configured to embed questions with {embedder_name!r}. Query and document "
                "vectors must come from the same model. Re-run ingestion, or set the "
                "embedding settings back to match the index."
            )

    # --- reading -----------------------------------------------------------

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best first.

        Both sides are L2-normalised, so the dot product *is* cosine similarity.
        """
        self.load()
        assert self._embeddings is not None and self._chunks is not None
        if not self._chunks:
            return []

        query = np.asarray(embedding, dtype=np.float32)
        query = query / (np.linalg.norm(query) or 1.0)
        if query.shape[0] != self._embeddings.shape[1]:
            raise IndexMismatchError(
                f"question vector has {query.shape[0]} dimensions but the index has "
                f"{self._embeddings.shape[1]}; the query embedder does not match the index"
            )

        scores = np.asarray(self._embeddings) @ query
        k = min(top_k, scores.shape[0])
        # argpartition is O(n) vs O(n log n) for a full sort; only the top k is
        # then sorted properly.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def episodes(self) -> set[str]:
        """Names of every episode in the index."""
        self.load()
        return {chunk.episode for chunk in (self._chunks or [])}

    def count(self) -> int:
        """Number of stored chunks."""
        self.load()
        return len(self._chunks or [])

    # --- writing (ingest only) ---------------------------------------------

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks, then persist. Ingest-time only.

        Raises:
            ReadOnlyStoreError: If the store was opened read-only.
            ValueError: If the two sequences have different lengths.
        """
        self._require_writable()
        if len(chunks) != len(embeddings):
            raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        self.load()
        assert self._chunks is not None

        by_id = {chunk.id: (chunk, np.asarray(vec, dtype=np.float32)) for chunk, vec in zip(chunks, embeddings)}
        kept = [
            (chunk, np.asarray(self._embeddings[row], dtype=np.float32))
            for row, chunk in enumerate(self._chunks)
            if chunk.id not in by_id
        ]
        merged = kept + list(by_id.values())
        self._replace(merged)
        self._persist()

    def delete_episode(self, episode: str) -> None:
        """Remove every chunk belonging to ``episode``, then persist."""
        self._require_writable()
        self.load()
        assert self._chunks is not None
        kept = [
            (chunk, np.asarray(self._embeddings[row], dtype=np.float32))
            for row, chunk in enumerate(self._chunks)
            if chunk.episode != episode
        ]
        self._replace(kept)
        self._persist()

    def set_manifest(self, **fields: Any) -> None:
        """Record which embedder built this index."""
        self._require_writable()
        self.load()
        self._manifest = {**(self._manifest or {}), **fields}
        self._persist()

    def _replace(self, rows: list[tuple[Chunk, np.ndarray]]) -> None:
        """Swap in a new chunk/vector set, sorted for stable, diffable output."""
        rows.sort(key=lambda row: (row[0].episode, row[0].index))
        self._chunks = [chunk for chunk, _ in rows]
        if rows:
            matrix = np.vstack([vec for _, vec in rows]).astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._embeddings = matrix / np.where(norms == 0, 1.0, norms)
        else:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)

    def _persist(self) -> None:
        """Write all three files. Only ever called during ingestion."""
        assert self._chunks is not None and self._embeddings is not None
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._dir / EMBEDDINGS_FILE, self._embeddings)
        (self._dir / CHUNKS_FILE).write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        manifest = {
            **(self._manifest or {}),
            "chunks": len(self._chunks),
            "dimension": int(self._embeddings.shape[1]) if self._embeddings.size else 0,
        }
        self._manifest = manifest
        (self._dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _require_writable(self) -> None:
        """Guard every mutation against the read-only serving path."""
        if self._read_only:
            raise ReadOnlyStoreError(
                "this index is open read-only. The deployed service must never write to disk "
                "(Render's free tier has no persistent storage); rebuild the index locally "
                "with ingest.py and commit it."
            )


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Serialise a chunk for ``chunks.json``."""
    return {
        "id": chunk.id,
        "text": chunk.text,
        "episode": chunk.episode,
        "start": round(chunk.start, 3),
        "end": round(chunk.end, 3),
        "index": chunk.index,
        "n_tokens": chunk.n_tokens,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Rebuild a chunk from ``chunks.json``."""
    return Chunk(
        id=data["id"],
        text=data["text"],
        episode=data["episode"],
        start=float(data["start"]),
        end=float(data["end"]),
        index=int(data["index"]),
        n_tokens=int(data.get("n_tokens", 0)),
    )
