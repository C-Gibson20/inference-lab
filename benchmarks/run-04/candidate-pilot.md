# RUN-04 embedding candidate pilot

**Status:** Exploratory development evidence. Qwen is provisionally selected
for the RUN-04 adapter, not as the final system-wide model.

## Purpose and data

This pilot checks whether either proposed embedder is promising enough to
continue evaluating against the supplied corpus sample. It is not a final
quality evaluation or a production benchmark.

The private local sample contained five English SARM1 documents: four research
articles and one patent. The run embedded the first three text-bearing pages
from each PDF as 15 page-level passages and scored the 16 development queries
in [`development-cases.jsonl`](development-cases.jsonl). Each query has one
manually assigned supporting page, identified by the source PDF's SHA-256
digest and one-based page number. The PDFs and their extracted text were not
copied into this repository or sent to a remote service.

The set was drafted from the opening pages and abstracts. Its questions and
page labels have not had independent domain review. The five-document sample
is narrow in topic and language, and opening pages are not representative of
the corpus's full passage distribution. Treat the figures below as a pilot
signal only.

## Candidate identities

| Candidate | Pinned repository revision | Declared licence | Weight file SHA-256 |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | Apache-2.0 | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| BGE-M3 | `5617a9f61b028005a4858fdac845db406aefb181` | MIT | `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` |

The licence and model capability descriptions are recorded in the [Qwen
model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3)
and [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181).
Both cards declare support for 100+ languages. This sample only exercises
English and cannot compare their multilingual quality. Qwen declares a 32K
context and up to 1024 dimensions; BGE-M3 declares an 8K context and 1024
dimensions. Model-card claims are not corpus evaluation results.

## Method

- Hardware: Apple Silicon host, 16 GB unified memory; PyTorch CUDA and MPS were
  unavailable, so inference used CPU.
- Software: PyTorch 2.8.0, Transformers 4.55.4; both checkpoints were loaded
  locally in float32 from the pinned revisions.
- Passage unit: each source page was embedded as one passage. Inputs were
  truncated at 4096 model tokens, padded in batches of two, and L2-normalised.
- Qwen query formatting: `Instruct: Given a scientific literature search
  query, retrieve passages that answer the query` followed by `Query: <text>`;
  document vectors used last-token pooling.
- BGE-M3 query formatting: unprefixed query text; passage and query vectors
  used attention-mask-weighted mean pooling.
- Ranking used cosine similarity. Metrics use one manually assigned relevant
  page per query. Timings are one cold local load and one CPU inference pass;
  there was no warmup or repeated throughput trial.

## Results

| Candidate | Recall@1 | Recall@5 | Recall@10 | MRR | Load time | Inference throughput | Peak process RSS | Parameter tensor bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B | 3/16 (0.188) | 14/16 (0.875) | 14/16 (0.875) | 0.464 | 1.6 s | 457 tokens/s | 5.11 GB | 2.38 GB |
| BGE-M3 | 0/16 (0.000) | 8/16 (0.500) | 13/16 (0.812) | 0.188 | 1.1 s | 776 tokens/s | 2.74 GB | 2.27 GB |

On this small, unreviewed query set, Qwen ranked more supporting pages in the
top five; BGE-M3 had higher single-pass CPU throughput and lower peak process
RSS. These measurements do not establish the candidates' expected GPU use or
full-corpus memory requirements. Page-level ranking also differs from the
production chunking and retrieval path.

## Decision and limitations

Qwen is provisionally selected for the adapter because it ranked more
supporting pages in the top five and had higher MRR on this small pilot. The
pilot is not enough to make a final system-wide selection: review or expand the
development queries and supporting-page labels, evaluate both candidates over
representative corpus passages, and complete the licence and language-profile
review. The held-out EVAL-04 data were not used.
