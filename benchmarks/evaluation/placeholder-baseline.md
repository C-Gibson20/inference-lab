# Provisional retrieval baseline

**Status:** Placeholder development baseline. This is an initial reference for
future phase work, not the final EVAL-04 held-out result or acceptance gate.

The current baseline uses the RUN-04 sample corpus and its 28 independently
reviewed development queries. It covers five English SARM1 documents and ranks
whole PDF pages, with one directly supporting page labelled per query. The
results below are copied from the [expanded candidate comparison](../run-04/expanded-candidate-comparison.md):

| Embedding candidate | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-0.6B | 3/28 (0.107) | 7/28 (0.250) | 10/28 (0.357) | 0.193 |
| BGE-M3 | 4/28 (0.143) | 5/28 (0.179) | 7/28 (0.250) | 0.178 |

These measurements provide a small, reproducible development reference while
the full corpus is being collected. They measure page-level embedding-candidate
ranking. They do not measure the Phase 6 lexical, dense, metadata, fused, or
RET-06 channel baselines, production chunk retrieval, all question categories,
or abstention behaviour. The sample is narrow in subject and its labels may
omit other relevant pages; see the [sample audit](../run-04/corpus-sample-audit.md)
for details.

Do not interpret these values as an EVAL-04 pass, reject, or threshold decision.
Once the full corpus and its frozen 30–50 case benchmark are available, run the
complete EVAL-04 channel comparison on that corpus and replace this placeholder
with the aggregate held-out report and its predeclared gate decision. Keep this
development baseline separate from the final report.
