# Evaluation dataset operations

EVAL-01 exposes a Python library for creating and validating local evaluation
datasets. Evaluation execution, result records, and quality reports are added
by their later build-order issues.

## Validate a dataset

Always load a dataset through the public loader. It validates the manifest,
case-file checksum, case count, split membership, evidence corpus version, and
local-only path boundary.

```python
from pathlib import Path

from inference_lab.evaluation import load_dataset

dataset = load_dataset(Path("benchmarks/evaluation/manifest.json"))
print(dataset.manifest.dataset_id, len(dataset.cases))
```

Do not read case files around the loader or trust a caller-provided checksum.
Before using a dataset, confirm that the local corpus matches its `corpus_id`
and `corpus_version`.

Loading confirms that a dataset is structurally valid. Before designating one
as the final benchmark, also run the final acceptance gate from the repository
root:

```sh
uv run python \
  scripts/inference_lab/evaluation/validate-evaluation-benchmark.py \
  /local/path/to/evaluation/manifest.json
```

Exit status `0` confirms the 30–50 case limit, global question-type and
answerability coverage, representative held-out composition, and cross-split
leakage checks. A non-zero status rejects the dataset. The committed synthetic
fixture is expected to exit with status `1` because it is intentionally too
small for final acceptance.

## Write a dataset

Construct validated `EvaluationCase` and `DatasetManifest` values, then use
`write_dataset(...)`. The writer serializes cases deterministically and places
the calculated case-file checksum in the returned manifest.

The destination must be a new local directory that does not already exist. The
writer stages and validates the complete dataset before publishing that
directory atomically; it will not update or overwrite an existing dataset. Use
a new destination for every immutable dataset version.

The case file is relative to the dataset directory and cannot escape it.
Preserve the returned manifest as the authoritative version; do not edit split
membership or checksums manually. Interrupted or rejected writes leave no
published partial dataset, although an operating-system failure may require an
operator to inspect the destination parent for a writer-owned hidden `.staging`
directory.

The manifest exposes split membership, provenance metadata, and checksums as
immutable mappings. Create a new validated dataset version instead of mutating
an existing manifest.

## Publish the schema

Generate the Draft 2020-12 JSON Schema used by external dataset tooling with:

```sh
uv run python \
  scripts/inference_lab/evaluation/generate-evaluation-schema.py \
  /local/path/evaluation-schema.json
```

The schema accepts either a manifest object or one case object. Use the Python
loader as the final validation boundary because it additionally verifies
checksums, file containment, split consistency, and cross-record references.

## Held-out data

Development data may be used during implementation. Held-out membership is
immutable and must not be used for tuning or model selection. Do not expose
held-out questions or expected answers in logs, reports intended for routine
development, or debugging output.

The committed `benchmarks/evaluation` data is a synthetic contract fixture,
not the final corpus-backed question set.

## Failure handling and privacy

- Treat an invalid manifest, checksum, split, corpus reference, or local path
  as a rejected input.
- Do not continue with a partially validated dataset.
- Do not copy invalid payloads into exceptions intended for logs.
- Never log questions, expected answers, evidence, prompts, responses, model
  output, credentials, or sensitive paths.

Run identities, evaluation-unit records, metrics, and reporting workflows are
intentionally deferred until the corresponding evaluation issues define their
inputs and reproducibility requirements.
