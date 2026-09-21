# Retrieval-evaluation runbook

> **Status:** The reproducible retrieval-evaluation runner, publication bundle,
> comparison, and deterministic report are implemented. This checkout contains
> only synthetic contract fixtures; run a real evaluation only in the local
> deployment that owns the pinned dataset, corpus snapshot, and adapter.

This procedure measures one selected split of one immutable evaluation dataset
against exactly one pinned corpus version. It does not send queries to a
remote service, publish questions or candidate text, or select an active
corpus version implicitly.

## Before the run

1. Select a validated EVAL-01 dataset manifest and one explicit split. Confirm
   that its `corpus_id` and `corpus_version` identify the intended corpus
   generation.
2. Prepare a deployment-local zero-argument request factory returning a
   complete `RetrievalEvaluationRequest`. It owns dataset loading, the pinned
   relationship snapshot, the retrieval adapter, and any private query
   construction.
3. Ensure the adapter declares a complete path-free
   `RetrievalSystemIdentity`. Its algorithm, configuration, component
   generations, and model identities are part of reproducibility.
4. Choose a new output directory outside the repository. Do not commit the
   request factory, corpus paths, output directories, or private benchmark
   data.
5. Confirm the selected passage cut-offs and candidate depth in the request.
   Document Recall@10 is fixed; passage Recall@k is rendered for every
   configured cut-off.

The request factory is the deployment integration boundary. The command does
not discover files, open storage, choose a corpus version, or construct a
query from command-line text. It invokes the injected adapter with one shared
profiling context and keeps timing observations out of quality identity.

## Run and publish

From the repository root, run:

```sh
uv run python scripts/inference_lab/evaluation/run-retrieval-evaluation.py \
  --request-factory /deployment/private/retrieval_request.py:build_request \
  --output /deployment/private/retrieval-output/run-001
```

The command's stdout is one JSON result containing only run/result identities
and bounded comparison status:

```json
{"comparison":{"status":"not_requested"},"result_digest":"sha256:...","run_fingerprint":"sha256:..."}
```

The new directory contains exactly:

- `run.json`: semantic run identity, policies, selected cases, component
  identities, result digest, and limitations;
- `cases.jsonl`: ordered safe per-case results, sanitised references, metrics,
  failures, and diagnostics;
- `report.json`: exact aggregate metrics and reconciliation counts;
- `report.md`: deterministic human-readable metrics, identities, diagnostics,
  and limitations; and
- `manifest.json`: sizes and SHA-256 checksums for those four artefacts.

The destination must be absent. Publication stages and validates every file,
flushes file and directory metadata, and atomically promotes the complete
directory. It never overwrites an existing run.

## Review the report

Review `report.md` in this order:

1. Confirm the run fingerprint, result digest, dataset, split, corpus and
   corpus-version identities.
2. Check headline macro document Recall@10 and every configured macro passage
   Recall@k. Exact numerator/denominator values are authoritative; decimals
   use the pinned policy precision and rounding mode.
3. Review micro diagnostics separately. They are not substitutes for the
   named macro headlines.
4. Check outcome reconciliation for applicable/non-applicable, completed,
   partial, failed, and empty cases.
5. Review duplicate and family diagnostics. Families and duplicate copies are
   crowding diagnostics and never substitute for relevance.
6. Read the explicit limitations before interpreting a score. Failed, partial,
   empty, and non-applicable cases remain visible under the stated denominator
   rules.
7. Confirm that the report contains no questions, context, expected answers,
   candidate text, raw query values, filenames, paths, or raw exceptions.

## Repeatability check

Run the same request factory into a second absent directory and compare it with
the first run before publication:

```sh
uv run python scripts/inference_lab/evaluation/run-retrieval-evaluation.py \
  --request-factory /deployment/private/retrieval_request.py:build_request \
  --compare-to /deployment/private/retrieval-output/run-001 \
  --output /deployment/private/retrieval-output/run-002
```

An equivalent rerun publishes the second directory and reports
`comparison.equivalent: true`. Comparison covers the semantic run identity,
canonical case results, aggregate bytes, and result digest. Output location,
wall-clock timing, and profiling observations are excluded.

A mismatch exits non-zero, prints only bounded categories such as
`run_identity`, `case_results`, `aggregate`, and `result_digest`, and does not
create the requested output directory. Do not explain away a mismatch as
rounding or output ordering; inspect the pinned dataset, corpus version,
relationship snapshot, adapter identity, policy, and ranked results.

The comparison target must be a complete published bundle. Missing files,
unknown inventory entries, malformed JSON, checksum mismatches, mixed
identities, and inconsistent result digests fail closed with the fixed generic
runner error.

## Failure handling and privacy

- Treat a non-zero command status as an unsuccessful evaluation or comparison;
  do not use an unpublished or partially written directory.
- Keep deployment-local request factories and all output directories outside
  version control.
- Never place questions, conversation context, expected answers, document
  text, candidate text, query/filter values, paths, credentials, timing
  records, or raw exception messages in logs, CLI output, artefacts, or
  comparison errors.
- Use a new output directory for every immutable run. Preserve both local
  bundles when investigating a mismatch, subject to the deployment's data
  handling policy.

No real-corpus conclusion is recorded in this repository. The committed
acceptance tests use synthetic fixtures to establish reproducibility,
corruption rejection, atomic publication, and privacy boundaries.
