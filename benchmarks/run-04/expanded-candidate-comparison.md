# RUN-04 expanded embedding candidate comparison

**Status:** Completed as development evidence on 2026-09-29. The results keep
Qwen3-Embedding-0.6B as the provisional RUN-04 adapter; they do not establish a
final model choice.

## Dataset and method

The run used the reviewed [`development-cases-expanded.jsonl`](development-cases-expanded.jsonl)
dataset (SHA-256
`c494342b0178b0b1aaaba72d94633d2da6fa173fcfb8d4b22b4801e650049676`). It
embedded every text-bearing physical page in the five audited PDFs: 521 pages
from 524 physical pages, plus all 28 labelled queries. The PDFs and extracted
text remained under the local corpus directory; neither was copied into the
repository or sent to a remote service. The runner checks each PDF digest and
the development dataset digest before inference.

Both candidates ran sequentially on CPU in float32, with batches of two and a
shared 4,096-token truncation ceiling. Qwen used the RUN-04 backend's pinned
query prefix and last-token pooling. BGE-M3 used unprefixed queries and
attention-mask-weighted mean pooling. Both vector sets were L2-normalised and
ranked by cosine similarity. Ties were resolved by document digest and
one-based physical page number. Each query has one directly supporting page
label; the second-pass review found additional related evidence for some cases,
so these scores do not credit every potentially relevant page.

The runner is [`compare_local_embedding_candidates.py`](../../scripts/inference_lab/evaluation/compare_local_embedding_candidates.py).
Example invocation, with model snapshots acquired separately:

```sh
uv run python scripts/inference_lab/evaluation/compare_local_embedding_candidates.py \
  --corpus-dir /path/to/corpus-sample \
  --qwen-model-dir /path/to/qwen3-embedding-0.6b \
  --bge-model-dir /path/to/bge-m3 \
  --output /tmp/run04-expanded-candidate-comparison.json
```

The runner is offline-only and writes aggregate metrics, model revisions and
weight digests. It does not write source paths, page text, or query text.

## Results

| Candidate | Recall@1 | Recall@5 | Recall@10 | MRR | Input tokens | Inference time | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B | 3/28 (0.107) | 7/28 (0.250) | 10/28 (0.357) | 0.193 | 556,105 | 870.91 s | 638.53 tokens/s |
| BGE-M3 | 4/28 (0.143) | 5/28 (0.179) | 7/28 (0.250) | 0.178 | 477,329 | 465.23 s | 1,026.01 tokens/s |

Qwen ranked more labelled pages in the top five and top ten and had the higher
MRR. BGE ranked one more label first and processed this workload faster. The
quality differences are small on only 28 manually authored cases. The token
counts differ because the candidates use different tokenizers and Qwen adds
its query instruction prefix.

## Candidate identity and environment

| Candidate | Pinned revision | Weight SHA-256 |
| --- | --- | --- |
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| BGE-M3 | `5617a9f61b028005a4858fdac845db406aefb181` | `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` |

The recorded environment was Apple Silicon (`arm64`, macOS 27.0), PyTorch
2.8.0, Transformers 4.55.4, and pypdf 6.19.0. The throughput values are from
one sequential pass over all pages and queries, without warmup or repeated
trials; they are observations for this machine, not performance guarantees.
The earlier pilot contains its separate cold-load measurements. Its
15-passage retrieval scores are not combined with this run.

## Interpretation and limits

The expanded results support keeping Qwen provisional because it retains the
stronger MRR and top-five/top-ten recall on the reviewed set. BGE remains a
meaningful alternative where CPU throughput or top-one ranking matters. The
scores are not broad retrieval evidence: all five PDFs concern SARM1, the
passage unit is a whole PDF page rather than production chunks, and the set
contains several related low-dose BEI questions. Single-page labels can also
under-count retrieval of other pages that support an answer. Held-out EVAL-04
questions were not used.
