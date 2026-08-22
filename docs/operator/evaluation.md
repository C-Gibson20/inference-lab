# Evaluation operations

Evaluation is currently exposed as a Python library workflow. A command-line
or service-level evaluation entry point will be documented when the serving
evaluation workflow is implemented.

## Before running evaluation

1. Confirm that the dataset manifest and case file are on the local filesystem.
2. Load the dataset through `load_dataset(...)`; do not read case files around
   the loader.
3. Confirm the intended split and purpose. Final evaluation uses `held_out`.
4. Confirm that the local corpus version in the manifest matches the evidence
   references used by the cases.
5. Prepare a finite JSON configuration and hash it with
   `configuration_digest(...)`.

The loader verifies manifest structure, case-file checksums, case count, split
membership, and corpus-version consistency. URI-shaped paths are rejected.

## Create a run

Use the run factory so policy checks and dataset digests are applied at one
boundary:

```python
from datetime import datetime, timezone
from pathlib import Path

from inference_lab.evaluation import (
    EvaluationRunRequest,
    configuration_digest,
    create_evaluation_run,
    load_dataset,
)

dataset = load_dataset(Path("benchmarks/evaluation/manifest.json"))
request = EvaluationRunRequest(split="held_out", purpose="final")
run = create_evaluation_run(
    dataset,
    request,
    system_version="system-1",
    configuration_digest=configuration_digest({"temperature": 0}),
    algorithm_versions={"matching": "1.0.0"},
    created_at=datetime.now(timezone.utc),
)
```

Store or pass the resulting `run_id` as the correlation identifier for the
evaluation. Do not construct a run ID from a timestamp or include case content
in the run metadata.

## Record units and failures

Use `EvaluationRecorder` for per-case results. Record machine-readable codes,
not exception messages or payloads:

```python
recorder.record(
    run_id=run.run_id,
    case_id="case-001",
    status="failed",
    error_code="backend_unavailable",
    timestamp=datetime.now(timezone.utc),
)
```

For an invalid input that prevents a unit or run from starting, emit a warning
with `invalid_input(...)` and an issue code. Continue only when the caller can
identify the affected scope and the result remains trustworthy. Otherwise fail
the run and retain the last valid run or report.

## Inspecting logs

Use JSON logging for scripts and tooling:

```toml
[logging]
format = "json"
level = "INFO"
```

Evaluation logs should expose event, level, timestamp, `run_id`, `case_id`,
status, duration, and issue codes as applicable. They must never expose the
question, expected answer, evidence passage, prompt, response, or model
output. The structured logging boundary rejects known sensitive field names;
fix the caller rather than bypassing that protection.

## Failure handling

- Treat an invalid manifest, checksum, split, corpus reference, or local path
  as an invalid input.
- Treat a backend exception or failed matching operation as a failed evaluation
  unit unless the workflow explicitly defines a safe skip.
- Do not convert failed units into passes or silently drop them.
- Preserve the run identity and machine-readable failure code for diagnosis.
- Do not retry a final run after changing its dataset, configuration, system,
  or algorithm inputs; create a new run instead.
