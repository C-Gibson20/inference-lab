# Evaluation conventions

Evaluation is a local, reproducible operation over an immutable dataset and
corpus version. Evaluation code must use validated dataset manifests, explicit
split membership, and the run-identity contract defined by
`inference_lab.evaluation.runs`.

## Local-only boundary

Evaluation inputs and outputs are filesystem paths on the local host. URI-shaped
locations, including HTTP, cloud-storage, and other remote schemes, are
rejected. Dataset loading and dataset serialization must validate this boundary
before reading or writing files.

Evaluation may read the selected local dataset and corpus references. It must
not send questions, evidence, prompts, answers, model traffic, or evaluation
artifacts to a remote service.

## Run identity

Every evaluation run has an immutable `EvaluationRun` record. The run ID is a
content-derived `run-<64 lowercase hex digits>` value over the semantic inputs:

- dataset, schema, and corpus identity and versions;
- selected split and evaluation purpose;
- system version and configuration digest;
- algorithm versions; and
- canonical dataset-manifest and case-file digests.

`created_at` is operational metadata and does not affect the run ID. Run
records contain identifiers and control-plane metadata only; they must not
contain questions, expected answers, evidence text, prompts, or model outputs.

Configuration digests use canonical, finite JSON and the format
`sha256:<64 lowercase hex digits>`. Raw configuration values must not be logged
or included in a run record.

## Split governance

- Development runs may be used for tuning and model selection.
- Final runs must use `held_out`.
- Held-out runs must not be used for tuning or model selection.
- Reporting runs must preserve the split and purpose recorded in the run.
- The dataset manifest is authoritative for split membership and held-out
  policy.

The run boundary validates the selected cases and rejects mismatches between
manifest membership and case data before a run is created.

## Structured evaluation records

Each evaluated case may emit one immutable `EvaluationRecord` with a stable
`eval-<64 lowercase hex digits>` ID derived from `run_id` and `case_id`.
Records contain only:

- `run_id` and `case_id`;
- `status`: `passed`, `failed`, or `skipped`;
- optional duration and machine-readable warning or error codes;
- UTC timestamp; and
- record schema version.

Failure records require an error code. A failed unit must not be represented as
a successful result, and its exception text must not be copied into structured
fields.

## Logging and privacy

Use structured logs with stable snake-case event names. Evaluation logging
follows the level policy below:

| Level | Evaluation use |
| --- | --- |
| `INFO` | Run lifecycle, aggregate summaries, and successful unit records |
| `WARNING` | Rejected or incomplete input that does not terminate the process |
| `ERROR` | A failed evaluation unit or run-level evaluation failure |
| `CRITICAL` | A process-wide condition that makes continued evaluation unsafe |

Logs must not contain questions, answers, passages, prompts, responses,
completions, model outputs, document text, credentials, or sensitive paths.
The logging processor rejects known content-bearing field names rather than
silently emitting them. Operational metrics such as token counts may be
recorded when they do not contain content.

Invalid inputs must produce a warning with a machine-readable issue code when
the operation can continue. Failed evaluation units must produce an error with
the stable run and case IDs plus a machine-readable error code. Neither event
may include the invalid payload or model output.
