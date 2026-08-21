# Provenance identifier conventions

Provenance identifiers provide the stable references that ingestion, storage,
retrieval, serving, evaluation, backup/restore, and logging use to refer to
corpus state. This document defines the identifier contract only; source
locations and record lineage are specified by their later build-order issues.

## Identity domains

The system uses a distinct opaque identifier type for each identity domain:

| Serialized field | Python type | Meaning | Stability boundary |
| --- | --- | --- | --- |
| `document_id` | `DocumentId` | One logical document across source versions | Stable across file moves, renames, and unchanged re-ingest |
| `document_version_id` | `DocumentVersionId` | One immutable source version | Changes when the source artifact changes |
| `record_id` | `RecordId` | One canonical or derived record | Stable for the lifetime of that record |
| `corpus_id` | `CorpusId` | One persistent corpus | Stable across corpus snapshots |
| `corpus_version` | `CorpusVersionId` | One immutable corpus snapshot | Changes when a new corpus state is committed |
| `source_span_id` | `SourceSpanId` | One source-location record | Stable only within its document-version representation |

Identifiers are canonical lowercase, hyphenated UUID strings. They do not
encode filenames, paths, dates, publication identifiers, titles, model names,
or other mutable metadata. Descriptive values belong in separate metadata
fields.

Identifier values are immutable, hashable, and domain-specific. Identifiers
from different domains are not equal or interchangeable, even when their UUID
strings match. Serialized field names use the full names in the table above;
use `document_id`, never `doc_id`, and `corpus_version`, never
`corpus_version_id`.

`from_str()` reconstructs an existing identifier and preserves its value.
`generate()` creates a new UUID4 identifier in the requested domain. Parsing
rejects non-string values and UUID spellings that are not canonical lowercase
hyphenated strings.

Generation does not decide whether an incoming document should reuse an
existing identity. Identity resolution belongs to ingest and storage. Content
hashes support unchanged-ingest detection and duplicate analysis but do not
replace primary identifiers.

## Stability rules

Document identity is distinct from document-version identity:

```text
document_id
  ├── document_version_id (original source)
  └── document_version_id (corrected or replacement source)
```

Re-ingesting unchanged source must resolve to the existing document and source
version. A changed source creates a new immutable version while retaining the
same logical document identity when the system determines that it is the same
document.

Corpus versions are immutable consistency boundaries. Metadata and persistent
indexes participating in a corpus snapshot must identify the same
`corpus_version`. Backup and restore operations must preserve this boundary.

`RecordId` and `SourceSpanId` identify records; they do not contain or replace
record lineage or resolvable source-location fields. Those records retain their
identifiers when moved between in-memory and persistent representations.

## Privacy and logging

Opaque provenance IDs may be used for structured operational correlation.
Logs must continue to follow the logging and backup/restore conventions:

- use `document_id` and `corpus_version` consistently;
- do not log document text, source passages, queries, prompts, model outputs,
  credentials, or sensitive paths;
- do not treat filenames or publication identifiers as substitutes for the
  stable ID fields.

## Scope boundaries

These conventions define identifier value objects only. They do not implement:

- document discovery or duplicate resolution;
- persistent metadata repositories;
- corpus-version transactions;
- source-location fields or citation resolution;
- record lineage, version metadata, invalidation, or graph traversal;
- parser, OCR, chunking, extraction, or summary generation.

Those behaviors are implemented by the dependency-ordered provenance, storage,
and ingest issues.
