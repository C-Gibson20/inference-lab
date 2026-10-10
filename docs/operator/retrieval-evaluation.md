# Retrieval-evaluation runbook

> **Status:** The reproducible single-channel runner and frozen baseline runner,
> local channel composition, lexical, dense, fused, and reranked adapters,
> publication bundles, comparison, and deterministic reports are implemented.
> A 30-case held-out run has been repeated byte-equivalently against the pinned
> 18-document corpus. Its aggregate results and inconclusive gate decision are
> recorded in the [local retrieval baseline](../../benchmarks/evaluation/retrieval-baseline.md);
> detailed bundles remain in the ignored local evaluation directory.

This procedure measures one selected split of one immutable evaluation dataset
against exactly one pinned corpus version. It does not send queries to a
remote service, publish questions or candidate text, or select an active
corpus version implicitly.

## Before the run

1. Select a validated EVAL-01 dataset manifest and one explicit split. Confirm
   that its `corpus_id` and `corpus_version` identify the intended corpus
   generation.
2. Prepare a deployment-local zero-argument request factory returning a
   complete `RetrievalEvaluationRequest`. `LocalLexicalEvaluationAdapter`,
   `LocalDenseEvaluationAdapter`, `LocalFusedEvaluationAdapter`, and
   `LocalRerankedEvaluationAdapter` use the case question verbatim, pin the
   selected historical generation or generations, and declare their
   configuration and generation identities. Dense and fused adapters also
   require the local embedding backend used to build that generation. The
   reranked adapter composes the fused adapter with a locally loaded reranker
   and records its policy identity.
3. Ensure the adapter declares a complete path-free
   `RetrievalSystemIdentity`. Its algorithm, configuration, component
   generations, and model identities are part of reproducibility.
4. Choose a new output directory outside version control. Keep deployment
   configuration, corpus paths, output directories, and private benchmark
   data out of version control. A checked-in local request factory may load
   ignored private assets, but must not embed their contents.
5. Confirm the selected passage cut-offs and candidate depth in the request.
   Document Recall@10 is fixed; passage Recall@k is rendered for every
   configured cut-off.

The example below constructs the lexical-only baseline. A dense-only request
uses `LocalDenseEvaluationAdapter`; a fused pre-rerank request uses
`LocalFusedEvaluationAdapter` with a locally loaded embedding backend. A final
evidence request uses `LocalRerankedEvaluationAdapter` with the same fused
adapter and a locally loaded reranker. All paths use the same pinned corpus
version. The request factory is the deployment integration boundary. The
command does not discover files, open storage, choose a corpus version, or
construct a query from command-line text. It invokes the injected adapter with
one shared profiling context and keeps timing observations out of quality
identity.

For a real 30–50-question dataset, save the following file outside the
repository as `/deployment/private/retrieval_request.py`. Run it from the
directory containing that deployment's `config.toml`. Set `EVAL_DATASET_MANIFEST`
to the validated, immutable local dataset manifest. Keep the question file,
this factory, source storage, and reports private.

```python
import atexit
import os
from pathlib import Path

from inference_lab.evaluation import (
    LocalLexicalEvaluationAdapter,
    RetrievalEvaluationPolicy,
    RetrievalEvaluationRequest,
    lexical_relationship_snapshot,
    load_dataset,
    validate_final_benchmark,
)
from inference_lab.provenance import CorpusId, CorpusVersionId
from inference_lab.settings import get_settings
from inference_lab.storage import open_metadata_repository


def build_request():
    dataset = load_dataset(Path(os.environ["EVAL_DATASET_MANIFEST"]))
    validate_final_benchmark(dataset.cases)
    corpus_id = CorpusId.from_str(dataset.manifest.corpus_id)
    corpus_version = CorpusVersionId.from_str(dataset.manifest.corpus_version)
    settings = get_settings()
    repository = open_metadata_repository(
        settings.storage.metadata_db_path,
        source_dir=settings.storage.source_dir,
        apply_migrations=False,
    )
    atexit.register(repository.close)
    return RetrievalEvaluationRequest(
        dataset=dataset,
        selected_case_ids=tuple(dataset.manifest.splits["held_out"]),
        selected_split="held_out",
        corpus_id=corpus_id,
        corpus_version=corpus_version,
        policy=RetrievalEvaluationPolicy(
            passage_cutoffs=(1, 5, 10), candidate_depth=50
        ),
        relationship_snapshot=lexical_relationship_snapshot(
            repository, corpus_id, corpus_version
        ),
        adapter=LocalLexicalEvaluationAdapter(
            repository,
            settings.storage.index_dir,
            corpus_id,
            corpus_version,
        ),
    )
```

The snapshot helper reads the selected version's visible membership, duplicate,
and family records. Do not substitute the mutable corpus catalogue for this
version-scoped snapshot. For development iteration, change both
`selected_split` and `selected_case_ids` to `development`; never tune against
the held-out result. Verbatim terms mode applies the lexical engine's AND-term
policy to the whole question, including ordinary question words. This is an
intentionally strict, measurable baseline, not a query-rewriting or answer
quality claim. Record its no-result and failure denominators before comparing
future retrieval stages against it.

## Run and publish

From the deployment directory containing `config.toml`, set the absolute
checkout path and run:

```sh
export CHECKOUT=/absolute/path/to/inference-lab
export EVAL_DATASET_MANIFEST=/deployment/private/retrieval-dataset/manifest.json
uv run --project "$CHECKOUT" python "$CHECKOUT/scripts/inference_lab/evaluation/run-retrieval-evaluation.py" \
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
uv run --project "$CHECKOUT" python "$CHECKOUT/scripts/inference_lab/evaluation/run-retrieval-evaluation.py" \
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

### The approved gate policy artefact

`docs/operator/retrieval-baseline-gate.json` records the thresholds approved
before any held-out result was opened. It is decoded by
`deserialise_approved_held_out_gate`, which rejects unknown or missing fields,
an unsupported `schema_version`, any split other than `held_out`, any primary
metric semantics other than fused pre-rerank `document_recall@10` over all
relevant targets, an unapproved uncertainty method, and a malformed case-file
checksum. Relaxing a threshold therefore requires editing the artefact in a
reviewable change; it cannot be widened by adding a field the reader ignores.

The artefact also names the dataset and corpus it was approved against. A run
loads it only when that binding equals the local benchmark exactly, so an
approved policy can never be applied to a different dataset version, case file,
or corpus version.

For the pinned local corpus, the checked-in held-out request factory reads the
ignored bound benchmark and setup record. Run it from the local evaluation
directory so the application's `config.toml` is loaded:

```sh
cd .dev/local-evaluation
PYTHONPATH=../../src ../../.venv/bin/python ../../scripts/inference_lab/evaluation/run-held-out-baseline.py \
  --request-factory ../../scripts/inference_lab/evaluation/local_heldout_request.py:make_heldout_request \
  --output runs/held-out-baseline-next
```

To verify an exact rerun, use a fresh output name and compare it with the
previous bundle:

```sh
PYTHONPATH=../../src ../../.venv/bin/python ../../scripts/inference_lab/evaluation/run-held-out-baseline.py \
  --request-factory ../../scripts/inference_lab/evaluation/local_heldout_request.py:make_heldout_request \
  --compare-to runs/held-out-baseline-v1.3.0-20261007 \
  --output runs/held-out-baseline-repeat-next
```

## Frozen channel baseline

The frozen baseline command runs the predeclared channel identities and case
set through the aggregate report workflow. Its deployment-local factory returns
a `HeldOutBaselineRequest` containing the `HeldOutRunContract`, cases in the
contract's frozen order, and a `LocalHeldOutChannelRunner`. Construct the runner
with one held-out `RetrievalEvaluationRequest` and a mapping from each
`HeldOutChannel` to its adapter. It checks the dataset, case order, corpus
version, policy, relationship snapshot, and every adapter identity before
executing queries. Each path then runs through the same retrieval evaluation
contract. Use the local lexical, dense, fused, and reranked adapters for their
respective paths, sharing the pinned fused adapter with the reranker. Where no
metadata-only path exists, use `UnavailableMetadataEvaluationAdapter`; do not
invent a ranking method to make the comparison appear complete. The report
retains that outcome and the gate remains inconclusive when required coverage
is missing.

Build the runner after freezing the policy and channel identities. The identity
recorded for the metadata path must come from the unavailable adapter when that
path is not implemented:

```python
from inference_lab.evaluation import (
    HeldOutBaselineRequest,
    HeldOutChannel,
    LocalHeldOutChannelRunner,
    UnavailableMetadataEvaluationAdapter,
)

metadata_adapter = UnavailableMetadataEvaluationAdapter()
channel_runner = LocalHeldOutChannelRunner(
    contract,
    held_out_request,
    {
        HeldOutChannel.LEXICAL_ONLY: lexical_adapter,
        HeldOutChannel.DENSE_ONLY: dense_adapter,
        HeldOutChannel.METADATA: metadata_adapter,
        HeldOutChannel.FUSED_PRE_RERANK: fused_adapter,
        HeldOutChannel.FUSED_RERANKED: reranked_adapter,
    },
)
return HeldOutBaselineRequest(
    contract=contract,
    cases=held_out_cases,
    run_channel=channel_runner,
    limitations=("metadata-only retrieval is unavailable",),
)
```

Run from the deployment directory with a new output destination:

```sh
uv run --project "$CHECKOUT" python "$CHECKOUT/scripts/inference_lab/evaluation/run-held-out-baseline.py" \
  --request-factory /deployment/private/held_out_request.py:build_request \
  --output /deployment/private/retrieval-output/baseline-001
```

Repeat the exact request against the first aggregate report:

```sh
uv run --project "$CHECKOUT" python "$CHECKOUT/scripts/inference_lab/evaluation/run-held-out-baseline.py" \
  --request-factory /deployment/private/held_out_request.py:build_request \
  --compare-to /deployment/private/retrieval-output/baseline-001 \
  --output /deployment/private/retrieval-output/baseline-002
```

The command validates the comparison bundle and absent output destination
before executing any held-out query. It publishes only after an equivalent
repeat check, and emits bounded JSON containing run/report identities, gate
outcome, and comparison status. The aggregate bundle contains `report.json`,
`report.md`, and `manifest.json`; it excludes questions, answers, evidence, and
per-case diagnostics.

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

The committed baseline summary contains aggregate metrics, run identities,
repeatability evidence, and the gate limitation. The detailed report bundles
and per-case diagnostics remain local. Acceptance tests use synthetic fixtures
to establish reproducibility, corruption rejection, atomic publication, and
privacy boundaries.
