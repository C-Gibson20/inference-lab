# RUN-05 reranker candidate pilot

This directory records the local candidate comparison for RUN-05 PR1. Model
weights and tokenizer files are stored outside the repository under
`~/Models/inference-lab/run05-candidates/`; only their identities and SHA-256
digests are checked in. `artifact-manifest.json` lists every acquired file,
including the MiniCPM custom model code. The runner verifies the full file
inventory and hashes before loading any candidate.

## Pinned candidates

| Candidate | Immutable revision | Repository terms | Upstream terms |
| --- | --- | --- | --- |
| `BAAI/bge-reranker-v2-m3` | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` | Apache-2.0 | MIT (`BAAI/bge-m3`) |
| `BAAI/bge-reranker-v2-gemma` | `1787044f8b6fb740a9de4557c3a12377f84d9e17` | Apache-2.0 | Gemma Terms of Use (`google/gemma-2b`) |
| `BAAI/bge-reranker-v2-minicpm-layerwise` | `47b5332b296c4d8cb6ee2c60502cc62a0d708881` | Apache-2.0 | OpenBMB General Model License (`MiniCPM-2B-dpo-bf16`) |

These are the terms stated by the model repositories and their identified base
models. This inventory does not decide whether a candidate is cleared for a
particular deployment or use. Review the applicable upstream terms before
production use.

The official repositories are [M3](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[Gemma](https://huggingface.co/BAAI/bge-reranker-v2-gemma), and
[MiniCPM](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise). The
base model terms are published by [BAAI](https://huggingface.co/BAAI/bge-m3),
[Google](https://huggingface.co/google/gemma-2b), and
[OpenBMB](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16).

To acquire an artefact for this evaluation, use the official Hugging Face
repository and the revision above, with the local destination kept outside the
checkout. For example:

```python
from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    local_dir=str(
        Path.home() / "Models/inference-lab/run05-candidates/bge-reranker-v2-m3"
    ),
)
```

Substitute the repository and revision from the table for the other candidates.
The benchmark itself runs with Hugging Face offline flags and accepts explicit
local directories; it does not download or resolve model identifiers. The
MiniCPM model code runs only after its local files pass the checked-in digest
manifest, and is imported from that verified directory to avoid a remote-code
Hub lookup.

## Measurement

`candidate-fixture.jsonl` is a fixed synthetic development fixture: 8 queries,
5 candidate passages per query, 40 pairs per pass, and one labelled relevant
passage for each query. It contains no private corpus text. The runner performs
one warmup pass and three measured passes on CPU with an internal batch size of
4. Each model has its pinned revision, inference dtype, preprocessing limits,
environment, quality metrics, latency, throughput,
and RUN-03 memory observations in
[`results/candidate-comparison.json`](results/candidate-comparison.json).
The JSON records the digest of `artifact-manifest.json`; that manifest contains
the SHA-256 digest for each local file, avoiding a second copy of the inventory
in the results file.

| Candidate | Recall@1 | MRR | nDCG@5 | Median pass | Throughput | Measured model weights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3, float32 | 0.875 | 0.906 | 0.929 | 2.05 s | 19.45 pairs/s | 2.27 GB host |
| Gemma, bfloat16 | 0.750 | 0.854 | 0.891 | 84.06 s | 0.48 pairs/s | 5.01 GB host |
| MiniCPM layer 28, bfloat16 | 0.875 | 0.938 | 0.954 | 62.08 s | 0.64 pairs/s | 5.49 GB host |

The measurements used Python 3.13.12, PyTorch 2.8.0, Transformers 4.55.4, and
an arm64 Mac CPU. The candidate inference settings used no weight quantization;
bfloat16 is the load dtype for Gemma and MiniCPM. The values are specific to
this machine, fixture, software stack, and workload. RUN-03 reports inspectable
unique tensor storage as measured and accelerator KV cache as unavailable; it
does not attribute allocator reservation or activation peaks to each model.

The pilot does not establish a production winner. Eight synthetic queries are
not held-out retrieval evidence and do not represent corpus distributions,
languages, or long-passage truncation. The 40-pair pass is also smaller than the
planned approximately 100-candidate retrieval workload. Long-candidate
truncation is not exercised by this fixture. M3 is the clear CPU throughput
candidate in this run; MiniCPM and M3 tie on Recall@1, while the tiny fixture's
quality ordering is not a selection basis. Use a representative, labelled
development set and intended deployment hardware before fixing a production
revision.

## Reproduce

From the repository root, run the local benchmark with the three explicit model
directories and choose a report path under `benchmarks/run-05/results/`. The
checked-in runner is `scripts/inference_lab/backends/benchmark_local_reranker_candidates.py`.
It emits a compact JSON completion event, writes no query or passage text to
the report, and returns failures as bounded exception categories without
including dependency messages or local paths.
