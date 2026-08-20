"""Vector store adapter backed by a local, persistent Chroma collection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence

from app.logging_config import get_logger
from app.models import Chunk, RetrievedChunk

log = get_logger(__name__)

#: SQLite-backed Chroma rejects very large writes; batch well under the limit.
_MAX_BATCH = 1000


class ChromaVectorStore:
    """Persistent :class:`~app.interfaces.VectorStore` over a Chroma collection.

    The collection is created with cosine space, and distances are converted to
    similarities (``1 - distance``) at this boundary so callers only ever deal
    with "higher is better".
    """

    def __init__(self, persist_dir: Path, collection_name: str = "podcast_chunks") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        """The Chroma collection, created on first access."""
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._get_or_create(client)
            log.info(
                "store.opened",
                extra={
                    "event": "store.opened",
                    "path": str(self._persist_dir),
                    "collection": self._collection_name,
                    "chunks": self._collection.count(),
                },
            )
        return self._collection

    def _get_or_create(self, client: Any) -> Any:
        """Fetch the collection, pinning cosine space and disabling Chroma's own embedder.

        We always supply vectors ourselves; passing ``embedding_function=None``
        stops Chroma from downloading its bundled ONNX model. Older/newer client
        versions disagree about that keyword, hence the fallback.
        """
        kwargs = {
            "name": self._collection_name,
            "metadata": {"hnsw:space": "cosine"},
        }
        try:
            return client.get_or_create_collection(**kwargs, embedding_function=None)
        except TypeError:  # pragma: no cover - depends on installed chromadb version
            return client.get_or_create_collection(**kwargs)

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Insert or replace chunks and their vectors.

        Args:
            chunks: Chunks to store; their deterministic IDs make this idempotent.
            embeddings: One vector per chunk, in the same order.

        Raises:
            ValueError: If the two sequences have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"got {len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return

        for batch_chunks, batch_vectors in _batched(chunks, embeddings, _MAX_BATCH):
            self.collection.upsert(
                ids=[chunk.id for chunk in batch_chunks],
                documents=[chunk.text for chunk in batch_chunks],
                metadatas=[chunk.to_metadata() for chunk in batch_chunks],
                embeddings=[list(vector) for vector in batch_vectors],
            )
        log.info(
            "store.upserted",
            extra={
                "event": "store.upserted",
                "chunks": len(chunks),
                "episode": chunks[0].episode,
                "total_chunks": self.collection.count(),
            },
        )

    def search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, best first."""
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return list(self._parse_results(result))

    @staticmethod
    def _parse_results(result: dict[str, Any]) -> Iterator[RetrievedChunk]:
        """Flatten Chroma's one-list-per-query response into domain objects."""
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            yield RetrievedChunk(
                chunk=Chunk.from_storage(chunk_id, document or "", metadata or {}),
                score=1.0 - float(distance),  # cosine distance -> cosine similarity
            )

    def delete_episode(self, episode: str) -> None:
        """Remove every chunk belonging to ``episode``."""
        self.collection.delete(where={"episode": episode})
        log.info("store.episode_deleted", extra={"event": "store.episode_deleted", "episode": episode})

    def episodes(self) -> set[str]:
        """Names of every episode currently indexed."""
        result = self.collection.get(include=["metadatas"])
        return {str(m.get("episode")) for m in (result.get("metadatas") or []) if m.get("episode")}

    def count(self) -> int:
        """Total number of stored chunks."""
        return int(self.collection.count())


def _batched(
    chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], size: int
) -> Iterator[tuple[Sequence[Chunk], Sequence[Sequence[float]]]]:
    """Yield aligned slices of chunks and embeddings."""
    for start in range(0, len(chunks), size):
        yield chunks[start : start + size], embeddings[start : start + size]
