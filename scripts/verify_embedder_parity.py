"""Check the committed ONNX embedder still reproduces the torch reference.

The failure mode this guards against is silent: a wrong pooling strategy, a
missing normalisation or a bad quantisation does not raise — it just returns
slightly wrong vectors, and retrieval quietly gets worse. So the check is
explicit and numeric.

Two things are measured:

1. **Vector parity** — cosine similarity between ONNX and torch embeddings of
   the same text. The fp32 export scores 1.000000; int8 drifts to ~0.98, which
   is expected and not disqualifying on its own.
2. **Retrieval agreement** — whether an index and queries embedded with the same
   ONNX model retrieve the same chunks as the torch reference does. This is what
   actually matters, because a shared rotation cancels in a dot product.

Needs the ingest dependencies (torch, for the reference):

    pip install -r requirements-ingest.txt
    python scripts/verify_embedder_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.providers.embeddings_onnx import OnnxEmbedder  # noqa: E402

#: Below this, the ONNX pipeline is doing something structurally different
#: (wrong pooling, missing normalisation) rather than merely quantised.
MIN_COSINE = 0.95

PROBES = [
    "Why does Kushal think privacy is a human right?",
    "journalists use a Qubes OS based system for sensitive documents",
    "PyPI was formerly known as the Cheese Shop, from the Monty Python sketch",
    "short",
    "a much longer passage that runs past the model's window " * 20,
]


def main() -> int:
    """Compare ONNX against torch and report. Non-zero exit on failure."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("needs the reference model: pip install -r requirements-ingest.txt")
        return 2

    settings = get_settings()
    onnx = OnnxEmbedder(settings.onnx_model_dir)
    torch_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    print("== vector parity ==")
    reference = torch_model.encode(PROBES, normalize_embeddings=True, show_progress_bar=False)
    candidate = np.array(onnx.embed_documents(PROBES))
    sims = [float(np.dot(candidate[i], reference[i])) for i in range(len(PROBES))]
    for probe, sim in zip(PROBES, sims):
        print(f"  {sim:.6f}  {probe[:60]}")
    print(f"  min={min(sims):.6f}  mean={float(np.mean(sims)):.6f}")

    print("\n== retrieval agreement on the eval set ==")
    agreement = _retrieval_agreement(onnx, torch_model)

    ok = min(sims) >= MIN_COSINE
    print(f"\n{'PASS' if ok else 'FAIL'}: min cosine {min(sims):.6f} (threshold {MIN_COSINE})")
    print(f"      retrieval top-5 set agreement {agreement:.0%}")
    if not ok:
        print("\nThe ONNX pipeline does not match the reference. Check mean pooling, the")
        print("attention mask, and L2 normalisation in app/providers/embeddings_onnx.py.")
    return 0 if ok else 1


def _retrieval_agreement(onnx: OnnxEmbedder, torch_model: object) -> float:
    """Fraction of eval questions where both embedders retrieve the same chunk set.

    Each embedder indexes the chunks *and* encodes the questions, so this
    compares two self-consistent systems rather than mixing vector spaces.
    """
    chunks = json.loads(Path("data/index/chunks.json").read_text())
    texts = [c["text"] for c in chunks]
    questions = [c["question"] for c in json.loads(Path("eval_set.json").read_text())]

    onnx_index = np.array(onnx.embed_documents(texts))
    torch_index = torch_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    onnx_q = np.array(onnx.embed_documents(questions))
    torch_q = torch_model.encode(questions, normalize_embeddings=True, show_progress_bar=False)

    same = 0
    for i, question in enumerate(questions):
        a = set(np.argsort(-(onnx_index @ onnx_q[i]))[:5])
        b = set(np.argsort(-(torch_index @ torch_q[i]))[:5])
        same += a == b
        if a != b:
            print(f"  differs: {question[:60]}  (overlap {len(a & b)}/5)")
    return same / len(questions)


if __name__ == "__main__":
    raise SystemExit(main())
