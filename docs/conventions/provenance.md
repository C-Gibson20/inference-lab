# Provenance conventions

Provenance identifiers provide the stable references that ingestion, storage,
retrieval, serving, evaluation, backup/restore, and logging use to refer to
corpus state. Derived-record lineage records how immutable source evidence and
parent records were transformed into chunks, facts, summaries, OCR output, and
future derived-record kinds. These contracts are storage-neutral and remain
the same when values move between memory and persistent representations.

## Identity domains

The system uses a distinct opaque identifier type for each identity domain:

| Serialised field | Python type | Meaning | Stability boundary |
| --- | --- | --- | --- |
| `document_id` | `DocumentId` | One logical document across source versions | Stable across file moves, renames, and unchanged re-ingest |
| `document_version_id` | `DocumentVersionId` | One immutable source version | Changes when the source artefact changes |
| `source_observation_id` | `SourceObservationId` | One independently observed input source | Stable across moves and renames; distinct for independently citable duplicate sources |
| `document_family_id` | `DocumentFamilyId` | One publication or patent family independently of its member documents | Stable for the family membership represented by the record |
| `record_id` | `RecordId` | One canonical or derived record | Stable for the lifetime of that record |
| `corpus_id` | `CorpusId` | One named persistent corpus in a deployment | Stable across that corpus's snapshots |
| `corpus_version` | `CorpusVersionId` | One immutable snapshot of exactly one corpus | Changes when a new state of that corpus is committed |
| `source_span_id` | `SourceSpanId` | One source-location record | Stable only within its document-version representation |

Identifiers are canonical lowercase, hyphenated UUID strings. They do not
encode filenames, paths, dates, publication identifiers, titles, model names,
or other mutable metadata. Descriptive values belong in separate metadata
fields.

Identifier values are immutable, hashable, and domain-specific. Identifiers
from different domains are not equal or interchangeable, even when their UUID
strings match. Serialised field names use the full names in the table above;
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

## Corpus membership and isolation

One metadata deployment may contain multiple named corpora. `DocumentId`,
`DocumentVersionId`, `SourceObservationId`, `DocumentFamilyId`, and `RecordId`
values are deployment-global opaque identities; they do not encode a
`CorpusId`. Document and document-version visibility is instead represented by
explicit corpus membership. The same immutable document or version may be a
member of more than one corpus without being copied or mutated.

Each corpus has an operator-visible display name in addition to its stable
`CorpusId`. The name is mutable descriptive metadata, may be non-unique, and
must never replace the ID in persisted references or API contracts. Interfaces
must disambiguate equal names with the opaque ID. Corpus names may be sensitive
and are not part of the operational logging context.

Existence in the deployment-global catalogue never implies corpus membership.
Every ingest, retrieval, evaluation, session, and corpus-version operation
names exactly one `CorpusId`, either directly or through a validated
`CorpusVersionId` owned by that corpus. Reads and candidate discovery must be
scoped to that corpus, and derived records are visible only through the corpus
generation that includes them. Storage must reject a `CorpusVersionId` used
with a different `CorpusId`.

A source observation belongs to exactly one corpus because it records that
corpus's independently observed input and continuity. Source-continuity and
identity-candidate queries use only the target corpus. Reusing a known global
document or version in another corpus is an explicit membership operation, not
an implicit consequence of a hash or external-identifier match.

Duplicate and family relationships may refer to deployment-global immutable
records, but a corpus-scoped read exposes only relationships between records
visible in the selected corpus generation. These relationships never grant
membership. Federated or cross-corpus retrieval is outside the first-build
contract.

`SourceObservationId` supplies source-continuity evidence without making a
filesystem path an identity. Discovery allocates one for a newly observed
source and retains it when that source moves or is renamed. A separately
observed source receives a distinct identifier even when its bytes match an
existing source. The identity resolver therefore receives same-observation
continuity candidates separately from corpus-scoped content-hash and external-
identifier candidates.

Persistence records source continuity as append-only `SourceObservationState`
revisions plus one validated current-state pointer. Initial state is revision
one. Advancement names the expected current document version and fails on a
stale expectation or an attempt to switch logical documents; exact retry is a
no-op. Historical states remain available for audit, while identity resolution
uses only the current state as continuity evidence.

External identifiers are evidence, not primary identity. The default
`conservative` policy declares no namespace authoritative, so an external-
identifier-only match is ambiguous rather than silently creating a new source
version. A corpus may select authoritative namespaces only through an explicit
`ExternalIdentifierResolutionPolicy` with a canonical policy name and semantic
version. Corpus-specific authority must be supported by the corpus audit; the
provenance layer does not build in DOI, patent, publication, or other namespace
preferences.

For every namespace shared by an observation and a candidate, the complete set
of asserted values must agree. A match in one namespace does not override a
conflict in another. Conflicting evidence fails closed, while a consistent
match without an authoritative namespace remains ambiguous.

When unchanged bytes carry an identifier in a previously unasserted namespace,
the resolver returns an `ExternalIdentifierAssertion` in
`identifier_assertions_to_add` while retaining the immutable `DocumentVersion`.
Each assertion identifies the retained version, source observation, namespace,
and value. Persistence stores these values append-only, treating exact replay
as a no-op; it must not rewrite the original version record. Existing
assertions are supplied on later resolution requests and participate in match,
conflict, and enrichment decisions. A different value in an already asserted
namespace is a conflict, not an enrichment.

## Identity-record compatibility

PROV-02's document identity, source version, source observation, duplicate,
family, and identity-resolution values are versioned domain entities; they are
not independently schema-versioned wire records. They therefore do not carry a
record schema version or define a public canonical byte round trip. Their
in-memory constructors and invariants are the domain contract.

`IdentityResolutionRequest.as_dict()` and
`IdentityResolutionOutcome.as_dict()` provide deterministic JSON-compatible
projections for in-process inspection and tests. They are not a persistence,
interchange, or recovery format, have no compatible reader contract, and must
not be stored as durable identity records.

SQLite metadata begins at one current multi-corpus baseline revision.
Superseded pre-release schemas are not supported inputs and are rejected
without mutation; the repository does not infer corpus membership, source
ownership, or display names from those intermediate shapes. Future supported
schema evolution appends explicit forward migrations and still requires the
caller's schema-update permission.
STOR-08 owns any future portable backup/export record schema after corpus
generations and the complete persistent-state shape exist. The lineage
serialisation contract below remains independently versioned because PROV-03
explicitly exposes lineage records at a canonical byte boundary.

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

A `byte_identical` duplicate relationship is valid only when both immutable
source versions have the same content-hash algorithm and digest. Persistence
enforces this on relationship writes and rejects stored mismatches during
upgrade or reopen; downstream retrieval and evaluation may rely on the class.

Corpus versions are immutable, corpus-owned consistency boundaries. Metadata
and persistent indexes participating in a corpus snapshot must identify the
same `corpus_id` and `corpus_version`. Version advancement is independent for
each corpus. Backup and restore operations must preserve ownership and this
boundary. Publication, component participation, and reader lifecycle follow
the [storage conventions](storage.md).

`RecordId` and `SourceSpanId` identify records; they do not contain or replace
record lineage or resolvable source-location fields. Those records retain their
identifiers when moved between in-memory and persistent representations.

`DocumentFamilyId` identifies a family grouping, not a document. Family
membership does not merge member documents or their immutable source versions;
each member remains independently citable.

## Derived-record lineage

Every `DerivedRecordMetadata` value identifies one immutable record version and
contains its record kind, schema version, direct inputs, producing operation,
output digest, and derivation fingerprint. A record may also identify the
record version it supersedes. It does not contain document text, model output,
prompt content, raw configuration, paths, or other unvalidated payload fields.

Direct inputs are explicit typed references:

- `SourceAnchor` pairs a `source_span_id` with the `document_version_id` that
  owns that span.
- `ParentRecordReference` identifies another immutable derived record.
- `InputRole` and a non-negative input ordinal may describe the semantic role
  and order of either input without changing its identity.

Input collections are immutable, unique by referenced identity, and returned
in canonical order. An input role or ordinal cannot disguise a duplicate
source span or parent record. A derived record must have at least one direct
source anchor or parent record.

Edges point from evidence towards the record derived from it:

```text
document_version + source_span ──> chunk
document_version + source_span ──> fact
document-summary ──> topic-summary ──> corpus-summary
```

`LineageEdge.input_reference` is the source or parent and
`LineageEdge.child_record_id` is the derived child. A complete `LineageBundle`
contains the known source-span catalogue, records, and the exact edges implied
by every record's declared inputs. Supplied edges may not omit or add
relationships.

Complete lineage must be acyclic. Every parent branch must terminate at an
original `SourceAnchor`; a chain cannot terminate at a parentless derived
record. Validation also rejects missing source spans, source-span/document-
version mismatches, missing parent or child records, duplicate records or
edges, self-links, and direct or indirect cycles.

## Immutable versions and supersession

A `record_id` names one immutable derived-record version. Changing its domain
payload, inputs, producing operation, component metadata, or configuration
creates a new record rather than mutating the existing record. When the new
record replaces an earlier version, `supersedes_record_id` explicitly links the
new record to its predecessor.

Supersession is separate from derivation: it describes replacement history,
not an input-to-output relationship. A predecessor must exist in a complete
bundle, retain the same record kind, have at most one direct replacement, and
participate in an acyclic supersession chain.

`OutputDigest` identifies the immutable domain-owned payload without placing
that payload in provenance metadata. `DerivationFingerprint` identifies the
semantic derivation from the record kind, lineage schema version, direct
inputs, producing operation, components, configuration identity, and output
digest. It deliberately excludes `record_id` and supersession history, so two
record identities produced by the same semantic derivation have the same
fingerprint.

Any invalidation-relevant change to a source, parent, operation, model, parser,
prompt, normaliser, grouping component, contract version, artefact revision,
configuration identity, record kind, or output digest changes the derivation
fingerprint.

## Producing operations and components

`ProducingOperation` identifies the operation kind, project implementation,
semantic contract version, configuration identity, and ordered producing
components. Project-owned kinds, roles, and implementations use canonical
lowercase slugs. Contract versions use canonical semantic versions.

`ProducingComponent` records a component role, implementation, and contract
version. Optional artefact identity and revision fields identify external
models, parsers, prompts, normalisers, or grouping definitions without using
their contents. Artefact identity and revision are either both present or both
absent. Components are immutable, unique, and canonically ordered.

Configuration and output identities currently use lowercase hexadecimal
SHA-256 digests. Store only the digest of invalidation-relevant canonical
configuration; do not place raw settings, prompts, credentials, model paths, or
other private configuration in lineage metadata.

## Canonical serialisation

The current lineage schema version is published with the provenance API.
Readers reject missing, malformed, or unsupported versions before interpreting
deeper fields. Every record in a bundle must use the bundle's schema version.

Canonical serialisation is deterministic UTF-8 JSON with sorted object keys,
compact separators, full field names, and one trailing newline. Repeated
serialisation of the same semantic record or bundle is byte-identical.
Deserialisation is strict: it rejects invalid UTF-8, malformed JSON, duplicate
object keys, non-finite values, unknown or missing fields, implicit scalar
coercion, non-canonical identifiers, and invalid graph structure.

Serialisation and deserialisation operate on bytes at the public boundary.
JSON-compatible payload builders remain lower-level module helpers rather than
part of the top-level provenance API.

Wrong runtime types raise `TypeError`. Correctly typed values that violate
lineage invariants raise a named `InvalidLineageError` subclass, which also
derives from `ValueError`. Validation errors describe the failed invariant but
must not echo rejected identifiers, payload values, paths, prompts, or raw
configuration.

## Privacy and logging

Opaque provenance IDs may be used for structured operational correlation.
Logs must continue to follow the logging and backup/restore conventions:

- use `document_id`, `corpus_id`, and `corpus_version` consistently;
- do not log document text, source passages, queries, prompts, model outputs,
  credentials, or sensitive paths;
- do not treat filenames or publication identifiers as substitutes for the
  stable ID fields.

Pure provenance values, fingerprint construction, serialisation, and graph
validation do not emit operational logs. Persistence and transaction
boundaries may log lifecycle events using stable event names, opaque IDs,
bounded enums, canonical lowercase pipeline-stage slugs, and aggregate counts.
They must not log lineage payloads or rejected free-form values.

## Scope boundaries

These conventions define identifiers and storage-neutral derived-record
lineage. They do not implement:

- document discovery or duplicate resolution;
- persistent metadata repositories;
- corpus-version transactions;
- persisted source-location records or citation resolution;
- recomputation or invalidation scheduling;
- graph analytics or general-purpose traversal APIs;
- parser, OCR, chunking, extraction, or summary generation.

Provenance describes the output of those operations; it does not execute them.
Repository storage must preserve canonical values and validate complete bundles
at its transaction boundary. Recomputers may compare fingerprints and follow
lineage, but their scheduling and persistence policies belong to later storage
and ingest issues. PROV-03 adds no deployment setting or feature flag.
