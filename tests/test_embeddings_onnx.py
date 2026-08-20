"""Tests for the local ONNX embedder.

These run the committed model, so they are the slowest tests in the suite —
still under a second, and they cover the piece where a mistake is silent rather
than loud (pooling, masking, normalisation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import get_settings
from app.providers.embeddings_onnx import OnnxEmbedder, _mean_pool_and_normalise

MODEL_DIR = get_settings().onnx_model_dir
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(), reason="committed ONNX model not present"
)


@pytest.fixture(scope="module")
def embedder() -> OnnxEmbedder:
    """One session shared across the module; loading it is the expensive part."""
    return OnnxEmbedder(MODEL_DIR)


def test_query_vectors_are_unit_length(embedder: OnnxEmbedder) -> None:
    vector = np.array(embedder.embed_query("what did they say about privacy?"))

    assert vector.shape == (384,)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(embedder: OnnxEmbedder) -> None:
    a = embedder.embed_query("the same question")
    b = embedder.embed_query("the same question")

    assert a == pytest.approx(b)


def test_related_text_scores_higher_than_unrelated(embedder: OnnxEmbedder) -> None:
    query = np.array(embedder.embed_query("privacy and surveillance online"))
    related, unrelated = (
        np.array(v)
        for v in embedder.embed_documents(
            ["encryption protects people from surveillance", "a recipe for banana bread"]
        )
    )

    assert float(query @ related) > float(query @ unrelated)


def test_document_and_query_encoding_agree(embedder: OnnxEmbedder) -> None:
    """The property that makes retrieval meaningful at all.

    Chunks are embedded in bulk and questions one at a time. If those two paths
    disagreed, document and query vectors would sit in different spaces and every
    similarity score would be subtly wrong — with no error to notice.

    This regressed once: encoding in batches let dynamic int8 quantisation derive
    its scale from the batch, dropping self-similarity to 0.980.
    """
    texts = ["short one", "a considerably longer passage with many more tokens in it"]

    as_documents = np.array(embedder.embed_documents(texts))
    as_queries = np.array([embedder.embed_query(t) for t in texts])

    for i, text in enumerate(texts):
        assert float(as_documents[i] @ as_queries[i]) == pytest.approx(1.0, abs=1e-5), text


def test_neighbouring_texts_do_not_perturb_a_vector(embedder: OnnxEmbedder) -> None:
    """A text's vector must not depend on what it was embedded alongside."""
    target = "a passage about privacy and encryption"

    alone = np.array(embedder.embed_documents([target])[0])
    in_company = np.array(
        embedder.embed_documents(["something entirely unrelated about bread", target])[1]
    )

    assert float(alone @ in_company) == pytest.approx(1.0, abs=1e-5)


def test_empty_input_returns_nothing(embedder: OnnxEmbedder) -> None:
    assert embedder.embed_documents([]) == []


def test_token_counting_uses_the_real_tokenizer(embedder: OnnxEmbedder) -> None:
    assert embedder.count_tokens("hello world") < embedder.count_tokens("hello world " * 20)
    assert embedder.count_tokens("") == 0


def test_token_counting_is_not_truncated(embedder: OnnxEmbedder) -> None:
    """The chunker needs true length, or it cannot know to split a long segment."""
    long_text = "privacy surveillance encryption " * 200

    assert embedder.count_tokens(long_text) > embedder.max_input_tokens


def test_counting_leaves_the_tokenizer_usable(embedder: OnnxEmbedder) -> None:
    """count_tokens toggles truncation off; it must restore it."""
    embedder.count_tokens("x" * 5000)

    vector = np.array(embedder.embed_query("still works afterwards"))
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_missing_model_fails_with_a_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="SOURCE.md"):
        OnnxEmbedder(tmp_path / "absent").embed_query("x")


def test_identity_is_recorded_for_the_manifest(embedder: OnnxEmbedder) -> None:
    assert embedder.name == "onnx:all-MiniLM-L6-v2"


def test_mean_pooling_ignores_padding() -> None:
    """The unit that is easiest to get wrong, tested without the model."""
    hidden = np.array([[[1.0, 0.0], [3.0, 0.0], [99.0, 99.0]]])  # third token is padding
    mask = np.array([[1, 1, 0]])

    pooled = _mean_pool_and_normalise(hidden, mask)

    assert pooled.tolist()[0] == pytest.approx([1.0, 0.0])  # mean of 1 and 3 -> 2, normalised


def test_mean_pooling_survives_an_all_padding_row() -> None:
    pooled = _mean_pool_and_normalise(np.zeros((1, 3, 2)), np.zeros((1, 3), dtype=int))

    assert np.isfinite(pooled).all()
