# Evaluation dataset conventions

The evaluation dataset layer defines the immutable, versioned input contract
used by evaluation workflows. It owns evaluation cases, expected-answer
annotations, evidence references, abstention expectations, dataset manifests,
explicit split membership, deterministic serialisation, and validated local
loading. It does not execute an evaluation or decide whether a system answer
is correct.

## Local-only boundary

Evaluation inputs and outputs are filesystem paths on the local host. URI-shaped
locations, including HTTP, cloud-storage, and other remote schemes, are
rejected. Dataset loading and dataset serialisation validate this boundary
before reading or writing files.

The boundary uses the shared local-path syntax policy described by the
configuration convention. It rejects URI schemes before and after `Path`
coercion, UNC-style network locations, empty paths, and NUL characters.
Rejected values are not included in errors intended for operational logs.

Evaluation may read the selected local dataset and corpus references. It must
not send questions, evidence, prompts, answers, model traffic, or evaluation
artefacts to a remote service.

## Versioned dataset contract

Every case and manifest declares a supported evaluation schema version. The
current format is `3.0.0`; the incompatible `1.0.0` and `2.0.0` formats are not
accepted or migrated.
Unknown, missing, and unsupported schema versions are rejected before model
validation. A dataset manifest identifies one immutable dataset version and
one `CorpusId` and one immutable `CorpusVersionId` owned by that corpus.

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
must remain inside its dataset directory. Checksum verification and JSONL
parsing use the same in-memory byte snapshot so a concurrent file change cannot
substitute unverified case content.

Every case and evidence reference belongs to the manifest's one selected
corpus version. Mixed-corpus evidence is invalid. Corpus comparisons use
separate datasets or separate runs whose outputs may be compared explicitly;
the first-build evaluator does not issue federated queries or score a
cross-corpus answer.

## Evaluation cases

Each case records a stable case ID, question and question type, optional
conversation context, expected-answer characteristics, evidence references,
an explicit abstention expectation, and its persisted split.

Expected-answer data is an annotation format. It may describe accepted claims,
numeric values and tolerances, dates, or comparison ordering. The dataset
contract does not define a generated-answer parser, semantic judge, matching
algorithm, quality metric, or pass/fail threshold.

Expectation keys are unique across claim, numeric, and date values. Numeric
values are finite and do not accept booleans or numeric strings. Dates use
`YYYY-MM-DD` without datetime coercion. Comparisons contain at least two unique
values, and every acceptable ordering contains exactly the same values.

Duplicate detection uses `normalise_for_duplicate_detection`, an intentionally
lossy comparison policy that applies NFKC, case folding, whitespace collapse,
and stripping. It is not a canonical document-text transformation and must not
be used for source text or provenance offsets.

Every evidence reference has a case-local `evidence_id`. Claims, numeric and
date expectations, and comparison expectations identify their supporting
evidence through these IDs. References must resolve within the same case and
must point to authoritative evidence. This makes claim-level citation and
completeness evaluation possible without making the dataset contract
responsible for scoring policy.

Evidence references use the canonical provenance identifier types and retain
the corpus version, document identity, and exactly the source-span or
structured-record identity required by their evidence kind. Document-level
references are contextual and cannot be authoritative. Duplicate case-local
IDs and duplicate provenance locations are rejected.

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
project specification. Its held-out subset contains every question type and
includes both answerable and expected-abstention cases, so final evaluation
cannot silently omit either a capability category or abstention behaviour. The
committed synthetic dataset is only a contract fixture.

## Serialisation and privacy

Cases are serialised as deterministic UTF-8 JSONL. Manifests are serialised as
deterministic, human-reviewable JSON. Dataset writing calculates the case-file
checksum rather than trusting a caller-supplied digest. A published manifest's
checksum inventory contains exactly its case file.

Dataset publication targets a new, absent directory. The writer creates a
sibling staging directory on the same filesystem, writes and flushes every
file, reloads the staged manifest through the normal validation boundary,
flushes directory metadata, and atomically renames the complete directory into
place. It never overwrites an existing dataset. A failure before publication
removes only the writer-owned staging directory and leaves any existing or
competing destination untouched.

`evaluation_json_schema()` publishes one standards-compliant JSON Schema using
Draft 2020-12. Its `oneOf` branches validate either a complete manifest or one
case record, all references resolve through the top-level `$defs`, and each
branch constrains `schema_version` to the current version. The schema artefact
is descriptive validation metadata; the Python loader remains authoritative
for cross-record and filesystem invariants that JSON Schema cannot express.

Questions, expected answers, evidence, prompts, responses, completions, model
outputs, document text, credentials, and sensitive paths must not appear in
operational logs. Dataset validation errors should identify the failed field or
record without copying its content into a log record.

## Scope boundary

The dataset contract does not define retrieval metrics, reproducible run
execution, held-out retrieval reports, answer matching, citation and
faithfulness metrics, abstention accuracy, OCR evaluation, extraction and
aggregation metrics, or system and session acceptance.

Those evaluation capabilities consume the dataset contract. They may introduce
versioned result records and algorithm identities without making scoring policy
part of the dataset format.
