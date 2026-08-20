# all-MiniLM-L6-v2, ONNX int8

Committed so the deployed service can embed questions locally, with no ML
framework and no embedding API.

| File | Source | Size |
| --- | --- | --- |
| `model.onnx` | [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) → `onnx/model_quantized.onnx` | 23MB |
| `tokenizer.json` | same repo | 0.7MB |

Licence: Apache-2.0, same as the upstream `sentence-transformers/all-MiniLM-L6-v2`.

## Why int8 rather than fp32

The fp32 export (90MB) is bit-exact with the torch model — cosine similarity
1.000000 across test passages. The int8 export drifts to 0.977–0.993 per vector.

That drift sounds disqualifying, but it only matters if the *index* and the
*query* disagree. When both are embedded with int8, measured retrieval recall on
the eval set is identical (10/10, mean top score 0.5778 vs fp32's 0.5752). The
quantisation error is largely a shared rotation, so it cancels in the dot
product.

## The batching trap

Dynamic int8 quantisation derives its activation scale from the tensor it is
handed, so **batching makes a text's vector depend on its batch-mates**. Measured:
the same sentence embedded alone versus in a batch of two scored 0.980 cosine
against itself. Padding to a fixed length does not help — it is the batch
composition, not the sequence length.

That would be fatal here, because the index is built in bulk while questions
arrive one at a time. `OnnxEmbedder` therefore encodes **one text per inference
call**, which restores exact self-consistency (cosine 1.0000000). The cost is
~3.7s to build a 96-chunk index, and nothing at query time.

If you switch to the fp32 export, this constraint disappears and batching is safe.

Verified with `scripts/verify_embedder_parity.py`. If you swap this model, re-run
that script and re-run `ingest.py --force`.
