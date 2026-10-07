# Local retrieval baseline

**Status:** Frozen held-out baseline; gate outcome is **inconclusive** because
three question categories do not meet the predeclared coverage minimum. This
is the pinned reference for later retrieval development. The gate policy was
set before the held-out run and was not adjusted from its results.

## Pinned inputs

- Dataset: `sarm1-retrieval-question-set@1.3.0`, schema `3.0.0`
- Dataset case-file checksum: `sha256:e94930c567a2544c85efb7282faed45bc97430f363a256ff6cebb6f49bac4eb6`
- Corpus: `53473b29-e7af-453a-8793-54e0d126613a`
- Corpus version: `4ac51a57-3f6d-4bdd-9a66-1425d0170b39` (18 documents)
- Held-out set: 30 cases, including 24 answerable cases and 6 expected
  abstentions
- Run fingerprint: `sha256:08a0164253c2244801b4ec2dd94f5b9182a523f4a97e8bc7f3a57fe8ba8cae1c`
- Aggregate report digest: `sha256:ff1bbc05ff7a15bbb48295954b82ee0298c034da592e8b9ab3391e9ac8172a01`

The private report bundles are retained in the ignored local evaluation
directory. An exact second run produced the same run fingerprint and report
digest and passed the byte-equivalence comparison.

## Channel results

Document recall counts relevant document targets across answerable cases;
expected-abstention cases remain in case coverage and are not relevant-target
denominator entries.

| Channel | Stage and cutoff | Document recall | Passage recall | Failures | Degraded | No result |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Lexical only | Candidate @10 | 0/24 | 0/24 | 0 | 0 | 30 |
| Dense only | Candidate @10 | 24/24 | 12/24 | 0 | 0 | 0 |
| Metadata only | Candidate @10 | Unavailable | Unavailable | 30 | 0 | 0 |
| Fused before reranking | Candidate @10 | 24/24 | 12/24 | 0 | 0 | 0 |
| Fused with reranking | Final evidence @8 | 23/24 | 17/24 | 0 | 0 | 0 |

The metadata channel is explicitly unavailable in this stack. Its 30 failed
outcomes remain visible in the comparison and are not presented as retrieval
successes. Fused candidate retrieval matched dense retrieval on document
recall, while reranking returned one fewer relevant document in the final
evidence set and more relevant passages.

## Gate and limitations

The frozen gate requires fused pre-rerank document Recall@10 of at least 9/10,
failure and degraded rates no greater than 1/20 each, and at least three cases
from three leakage groups for every category. Uncertainty uses a 95%
leakage-cluster bootstrap with 10,000 replicates. The fused channel retrieved
24/24 relevant document targets with no failed or degraded outcomes. The gate
is still **inconclusive**: comparison, follow-up, and difficult-terminology
each have only one held-out case and one leakage group. Their results do not
support a category-level conclusion.

The lexical AND-term baseline returned no results for any case. Metadata-only
retrieval is unavailable. The small held-out set limits conclusions for the
three sparse categories even though overall candidate recall is high. Do not
change retrieval settings based on these held-out results; use development
cases or a new versioned holdout for later selection.

See the [frozen gate policy](../../docs/operator/retrieval-baseline-gate.json)
and [retrieval evaluation runbook](../../docs/operator/retrieval-evaluation.md)
for the measurement contract and local rerun procedure. Questions, answers,
retrieved passages, and per-case diagnostics remain in the private local
bundles and are not part of this repository.
