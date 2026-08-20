"""Tests for the committed NumPy vector index."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.models import Chunk
from app.providers.store_numpy import (
    IndexMismatchError,
    NumpyVectorStore,
    ReadOnlyStoreError,
)


def chunk(idx: int, episode: str = "ep-01", text: str = "text") -> Chunk:
    """Build a chunk with a deterministic id."""
    return Chunk(
        id=f"{episode}:{idx:05d}", text=f"{text} {idx}", episode=episode,
        start=idx * 10.0, end=idx * 10.0 + 10.0, index=idx, n_tokens=5,
    )


@pytest.fixture
def store(tmp_path: Path) -> NumpyVectorStore:
    """A writable store seeded with three orthogonal unit vectors."""
    s = NumpyVectorStore(tmp_path / "index")
    s.upsert(
        [chunk(0), chunk(1), chunk(2)],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return s


def test_search_ranks_by_cosine_similarity(store: NumpyVectorStore) -> None:
    results = store.search([0.0, 1.0, 0.0], top_k=3)

    assert results[0].chunk.index == 1
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)


def test_search_normalises_the_query(store: NumpyVectorStore) -> None:
    """An unnormalised query must not inflate scores past 1.0."""
    results = store.search([0.0, 50.0, 0.0], top_k=1)

    assert results[0].score == pytest.approx(1.0)


def test_top_k_is_respected_and_capped(store: NumpyVectorStore) -> None:
    assert len(store.search([1.0, 0.0, 0.0], top_k=2)) == 2
    assert len(store.search([1.0, 0.0, 0.0], top_k=99)) == 3


def test_wrong_dimension_query_fails_loudly(store: NumpyVectorStore) -> None:
    with pytest.raises(IndexMismatchError, match="dimensions"):
        store.search([1.0, 0.0], top_k=1)


def test_upsert_replaces_by_id(store: NumpyVectorStore) -> None:
    store.upsert([chunk(1, text="replaced")], [[0.0, 1.0, 0.0]])

    assert store.count() == 3
    assert store.search([0.0, 1.0, 0.0], top_k=1)[0].chunk.text.startswith("replaced")


def test_delete_episode_removes_only_that_episode(store: NumpyVectorStore) -> None:
    store.upsert([chunk(0, episode="ep-02")], [[1.0, 1.0, 0.0]])
    assert store.episodes() == {"ep-01", "ep-02"}

    store.delete_episode("ep-01")

    assert store.episodes() == {"ep-02"}
    assert store.count() == 1


def test_index_survives_a_reload(tmp_path: Path, store: NumpyVectorStore) -> None:
    reopened = NumpyVectorStore(tmp_path / "index", read_only=True)

    assert reopened.count() == 3
    assert reopened.search([1.0, 0.0, 0.0], top_k=1)[0].chunk.index == 0


def test_read_only_store_refuses_every_write(tmp_path: Path, store: NumpyVectorStore) -> None:
    serving = NumpyVectorStore(tmp_path / "index", read_only=True)

    with pytest.raises(ReadOnlyStoreError):
        serving.upsert([chunk(9)], [[1.0, 0.0, 0.0]])
    with pytest.raises(ReadOnlyStoreError):
        serving.delete_episode("ep-01")
    with pytest.raises(ReadOnlyStoreError):
        serving.set_manifest(embedder="x")


def test_missing_index_reads_as_empty_rather_than_crashing(tmp_path: Path) -> None:
    empty = NumpyVectorStore(tmp_path / "absent", read_only=True)

    assert empty.exists is False
    assert empty.count() == 0
    assert empty.search([1.0, 0.0], top_k=3) == []


def test_manifest_records_provenance(tmp_path: Path, store: NumpyVectorStore) -> None:
    store.set_manifest(embedder="jina:jina-embeddings-v3")

    manifest = NumpyVectorStore(tmp_path / "index", read_only=True).manifest
    assert manifest["embedder"] == "jina:jina-embeddings-v3"
    assert manifest["chunks"] == 3
    assert manifest["dimension"] == 3


def test_verify_embedder_rejects_a_mismatch(tmp_path: Path, store: NumpyVectorStore) -> None:
    """The failure mode this guards against is silent, so it must be loud."""
    store.set_manifest(embedder="jina:jina-embeddings-v3")
    serving = NumpyVectorStore(tmp_path / "index", read_only=True)

    serving.verify_embedder("jina:jina-embeddings-v3")  # matching: no error

    with pytest.raises(IndexMismatchError, match="same model"):
        serving.verify_embedder("local:all-MiniLM-L6-v2")


def test_verify_embedder_tolerates_a_manifestless_index(tmp_path: Path, store: NumpyVectorStore) -> None:
    (tmp_path / "index" / "manifest.json").unlink()

    NumpyVectorStore(tmp_path / "index", read_only=True).verify_embedder("anything")


def test_row_count_mismatch_is_detected(tmp_path: Path, store: NumpyVectorStore) -> None:
    path = tmp_path / "index" / "chunks.json"
    payload = json.loads(path.read_text())
    path.write_text(json.dumps(payload[:2]))

    with pytest.raises(IndexMismatchError, match="inconsistent"):
        NumpyVectorStore(tmp_path / "index", read_only=True).load()


def test_stored_vectors_are_normalised(tmp_path: Path) -> None:
    s = NumpyVectorStore(tmp_path / "index")
    s.upsert([chunk(0)], [[3.0, 4.0, 0.0]])

    matrix = np.load(tmp_path / "index" / "embeddings.npy")
    assert np.linalg.norm(matrix[0]) == pytest.approx(1.0)
