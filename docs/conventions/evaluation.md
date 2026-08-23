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

Every case and manifest declares a supported evaluation schema version.
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

Evidence references use PROV-01 identifiers and retain the corpus version,
document identity, and the source-span or structured-record identity required
by their evidence kind. Answerable cases require authoritative evidence.
Abstention cases state why abstention is expected and whether a partial answer
may be acceptable.

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
