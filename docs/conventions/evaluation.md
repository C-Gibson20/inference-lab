# Evaluation dataset conventions

EVAL-01 defines the immutable, versioned input contract used by later
evaluation work. It owns evaluation cases, expected-answer annotations,
evidence references, abstention expectations, dataset manifests, explicit
split membership, deterministic serialization, and validated local loading.
It does not execute an evaluation or decide whether a system answer is
correct.

## Local-only boundary

Evaluation inputs and outputs are filesystem paths on the local host. URI-shaped
locations, including HTTP, cloud-storage, and other remote schemes, are
rejected. Dataset loading and dataset serialization validate this boundary
before reading or writing files.

Evaluation may read the selected local dataset and corpus references. It must
not send questions, evidence, prompts, answers, model traffic, or evaluation
artifacts to a remote service.

## Versioned dataset contract

Every case and manifest declares a supported evaluation schema version. The
current format is `2.0.0`; the incompatible pre-release `1.0.0` format is not
accepted or migrated.
Unknown, missing, and unsupported schema versions are rejected before model
validation. A dataset manifest identifies one immutable dataset version and
one immutable corpus version.

The manifest records:

- dataset and corpus identity;
- the relative case-file location and its SHA-256 checksum;
- the expected case count;
- explicit, disjoint split membership;
- held-out governance policy; and
- creation, update, and provenance metadata.

Manifest split, provenance, and checksum mappings are defensively copied and
exposed as immutable mappings. Mutating the dictionaries used to construct a
manifest cannot alter the validated dataset identity or held-out membership.

The loader verifies the case-file checksum, unique case IDs, case count, split
membership, per-case split values, and evidence corpus versions. A case file
must remain inside its dataset directory.

## Evaluation cases

Each case records a stable case ID, question and question type, optional
conversation context, expected-answer characteristics, evidence references,
an explicit abstention expectation, and its persisted split.

Expected-answer data is an annotation format. It may describe accepted claims,
numeric values and tolerances, dates, or comparison ordering. EVAL-01 does not
define a generated-answer parser, semantic judge, matching algorithm, quality
metric, or pass/fail threshold.

Expectation keys are unique across claim, numeric, and date values. Numeric
values are finite and do not accept booleans or numeric strings. Dates use
`YYYY-MM-DD` without datetime coercion. Comparisons contain at least two unique
values, and every acceptable ordering contains exactly the same values.

Every evidence reference has a case-local `evidence_id`. Claims, numeric and
date expectations, and comparison expectations identify their supporting
evidence through these IDs. References must resolve within the same case and
must point to authoritative evidence. This makes claim-level citation and
completeness evaluation possible without making EVAL-01 responsible for the
later scoring algorithm.

Evidence references use PROV-01 identifiers and retain the corpus version,
document identity, and exactly the source-span or structured-record identity
required by their evidence kind. Document-level references are contextual and
cannot be authoritative. Duplicate case-local IDs and duplicate provenance
locations are rejected.

Answerable cases require at least one evidence-linked expected characteristic,
and every authoritative reference must support one of those characteristics.
An expected-abstention case uses reason-specific evidence rules:

- `evidence_absent` and `deliberately_unanswerable` have no authoritative
  evidence or expected answer;
- `evidence_insufficient` retains the available evidence and may annotate a
  supported partial answer; and
- `evidence_ambiguous` retains authoritative evidence that establishes the
  ambiguity.

`partial_answer_allowed` applies only to expected abstention and requires an
evidence-linked expected characteristic. Non-abstention cases use
`not_applicable`; expected-abstention cases must state a substantive reason.

Follow-up cases retain their preceding conversational context. Context is part
of the private dataset payload and must not be copied into operational logs.

## Held-out governance and leakage

Split membership is explicit in both the manifest and each case. The manifest
is authoritative, and every case belongs to exactly one split.

Deterministic split generation keeps exact questions, near-duplicate questions,
shared expected answers, and shared evidence in one leakage group. A persisted
dataset is rejected when a leakage group crosses splits.

Development cases may support iteration. Held-out membership is immutable and
must not be used for tuning or model selection. The final corpus-backed
question set contains 30–50 real questions spanning the question types in the
project specification; the committed synthetic dataset is only a contract
fixture.

## Serialization and privacy

Cases are serialized as deterministic UTF-8 JSONL. Manifests are serialized as
deterministic, human-reviewable JSON. Dataset writing calculates the case-file
checksum rather than trusting a caller-supplied digest.

`evaluation_json_schema()` publishes one standards-compliant JSON Schema using
Draft 2020-12. Its `oneOf` branches validate either a complete manifest or one
case record, all references resolve through the top-level `$defs`, and each
branch constrains `schema_version` to the current version. The schema artifact
is descriptive validation metadata; the Python loader remains authoritative
for cross-record and filesystem invariants that JSON Schema cannot express.

Questions, expected answers, evidence, prompts, responses, completions, model
outputs, document text, credentials, and sensitive paths must not appear in
operational logs. Dataset validation errors should identify the failed field or
record without copying its content into a log record.

## EVAL-01 scope boundary

EVAL-01 does not define retrieval metrics or reproducible run execution
(`EVAL-03`), a held-out retrieval report or Recall@10 gate (`EVAL-04`), answer
matching, citation and faithfulness metrics, or abstention accuracy (`EVAL-05`),
OCR evaluation (`EVAL-06`), extraction and aggregation metrics (`EVAL-07`), or
systems and session acceptance (`EVAL-09` and `EVAL-10`).

Those later issues consume the EVAL-01 dataset contract. They may introduce
versioned result records and algorithm identities without making scoring
policy part of the dataset format.
