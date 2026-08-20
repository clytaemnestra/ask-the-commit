"""Local embedding adapter using ONNX Runtime — the same model as torch, without torch.

This is what lets the deployed service embed questions locally on a 512MB free
tier. ``sentence-transformers`` measures at ~830MB resident because of torch;
ONNX Runtime plus an int8 MiniLM export does the same arithmetic in a fraction of
that, with a 23MB model file committed to the repo.

The maths reproduces sentence-transformers' ``all-MiniLM-L6-v2`` pipeline
exactly, and the order matters:

1. WordPiece tokenise, truncated to 256 tokens (the model's window)
2. run the encoder to get per-token hidden states
3. **mean-pool over tokens, weighted by the attention mask** — padding must not
   contribute, which is the step people usually get wrong
4. L2-normalise

Verified against the torch reference: the fp32 export scores cosine 1.000000, and
the int8 export used here gives identical retrieval recall on the eval set (see
``models/minilm-onnx/SOURCE.md``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.logging_config import get_logger

log = get_logger(__name__)

#: MiniLM's context window. Longer inputs are truncated before encoding.
MAX_SEQUENCE_LENGTH = 256


class OnnxEmbedder:
    """A local :class:`~app.interfaces.Embedder` backed by ONNX Runtime.

    Args:
        model_dir: Directory holding ``model.onnx`` and ``tokenizer.json``.
        model_name: Identity recorded in the index manifest and in logs.
        threads: Intra-op thread count. One is right for a small shared
            container: extra threads add contention and memory for no gain on
            single-question workloads.

    Note:
        Texts are encoded **one at a time, deliberately**. The int8 export uses
        dynamic quantisation, which derives its activation scale from the tensor
        it is given — so batching makes a text's vector depend on what else was
        in its batch. Measured, the same sentence embedded alone versus in a
        batch of two scored only 0.980 cosine against itself.

        That is fatal here: the index is built in bulk while questions arrive one
        at a time, so document and query vectors would sit in subtly different
        spaces. Encoding singly costs ~2s for a 96-chunk index (ingest-time only)
        and nothing at all at query time, where inputs are single anyway.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        model_name: str = "onnx:all-MiniLM-L6-v2",
        threads: int = 1,
    ) -> None:
        self._dir = model_dir
        self._model_name = model_name
        self._threads = threads
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._input_names: set[str] = set()

    @property
    def name(self) -> str:
        """Identity recorded in the index manifest."""
        return self._model_name

    @property
    def max_input_tokens(self) -> int:
        """Sequence length beyond which input is truncated."""
        return MAX_SEQUENCE_LENGTH

    def _load(self) -> None:
        """Create the inference session and tokenizer on first use."""
        if self._session is not None:
            return

        import onnxruntime as ort  # imported lazily to keep module import cheap
        from tokenizers import Tokenizer

        model_path = self._dir / "model.onnx"
        tokenizer_path = self._dir / "tokenizer.json"
        for path in (model_path, tokenizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}. The ONNX embedding model is committed to the repo; "
                    "see models/minilm-onnx/SOURCE.md for where it comes from."
                )

        started = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = self._threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self._tokenizer.enable_padding()

        log.info(
            "embedder.loaded",
            extra={
                "event": "embedder.loaded",
                "model": self._model_name,
                "path": str(self._dir),
                "load_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )

    def count_tokens(self, text: str) -> int:
        """Exact token count using the model's own WordPiece tokenizer.

        Truncation is disabled for counting, so the chunker sees the true length
        and can decide to split rather than silently losing the tail.
        """
        self._load()
        assert self._tokenizer is not None
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False).ids)
        finally:
            self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
            self._tokenizer.enable_padding()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, one per inference call.

        Not batched — see the class docstring. Batching would make each vector
        depend on its batch-mates under dynamic int8 quantisation.
        """
        return [self._encode_one(text).tolist() for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single question."""
        return self._encode_one(text).tolist()

    def _encode_one(self, text: str) -> np.ndarray:
        """Tokenise, run the encoder, mean-pool and normalise a single text."""
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Tokenise, run the encoder, mean-pool and normalise."""
        self._load()
        assert self._session is not None and self._tokenizer is not None

        encoded = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        hidden_states = self._session.run(None, feed)[0]
        return _mean_pool_and_normalise(hidden_states, attention_mask)


def _mean_pool_and_normalise(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mask-weighted mean pooling followed by L2 normalisation.

    Args:
        hidden_states: ``(batch, tokens, dim)`` encoder output.
        attention_mask: ``(batch, tokens)``; zeros mark padding.

    Returns:
        ``(batch, dim)`` unit vectors.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (hidden_states * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)  # never divide by zero
    pooled = summed / counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.where(norms == 0, 1.0, norms)
