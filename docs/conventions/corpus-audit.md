# Corpus-audit conventions

The corpus audit is a local, deterministic aggregation of exactly one
`CorpusId` and one immutable `CorpusVersionId`. Its input is a validated
`CorpusAuditRequest`; its output is an aggregate-only `CorpusAuditReport`.
The audit does not select a corpus, infer membership from a global catalogue,
parse unsupported formats, perform OCR, call a remote service, or publish
document-level diagnostics.

## Populations and denominators

The report keeps two populations separate:

- **Source intake** is every traversed or classified source, including
  unsupported, unknown, inaccessible, changed, ambiguous, conflicting, and
  failed sources. `files_seen` reconciles to classified files plus
  source-scoped traversal failures; format counts reconcile to classified
  files. Request-scoped failures remain explicit upstream limitations outside
  that source denominator.
- **Corpus membership** is every document version in the selected immutable
  corpus version. Member-level parse, acquisition, structure, span, and OCR
  coverage use `member_versions` as their population, including members with
  missing observations.

An intake source does not become a member merely because it was traversed, and
a member does not disappear because it was absent from the current intake.
`AuditReconciliation` records the source totals, member totals, parse slots,
acquisition slots, and section-population checks explicitly.

## Availability semantics

Every field preserves the reason its value is or is not available:

| State | Meaning |
| --- | --- |
| `observed` | A capable producer supplied a value. |
| `observed_absent` | A capable producer explicitly found no instance. |
| `unavailable` | The selected producer cannot measure the field. |
| `missing` | An expected observation was not supplied. |
| `failed` | The producing stage attempted and failed. |
| `invalid` | Retained evidence cannot be interpreted under the selected policy. |

Missing and failed observations remain in the denominator. A rate with no
denominator is rendered as `null`, never as zero. Known-only rates use only
observed and observed-absent values; population rates use the complete stated
population. Percentages use the fixed decimal precision and rounding policy in
`CorpusAuditPolicy`, not binary floating-point arithmetic.

For EVAL-02 acceptance, an explicit `unavailable` value is complete coverage
accounting when the selected versioned producer truthfully declares that it
cannot observe the field. It is not evidence that the feature is absent and
must remain visible in the report's limitations. EVAL-02 does not require
capabilities assigned to later parser, extraction, or OCR work. When such a
capability is added, its observations and producer identity must be versioned
and the corpus must be audited again before relying on the stronger evidence.

Source format support is classifier evidence, not file-signature proof. The
current classifier is the versioned suffix-based `text-native-baseline` policy:
`.txt` is supported text, `.pdf` is recognised but unsupported, and other or
missing suffixes are unknown.

Source byte summaries use safe classified byte counts and the versioned
nearest-rank quantile rule. Empty known populations have `null` extrema and
quantiles. Parse character totals are ING-01 canonical character counts, not
source bytes. Token totals are grouped by parser version and are never
recomputed with the RUN-02 tokeniser.

## Capability registry

Parser identity and parser capability identity are separate. Every parser
version used in a request must have a registered `ParserCapabilityProfile`; an
unknown profile rejects the request. A value or structure that a profile says
the parser cannot observe also rejects the request.

The current `text-native` profile (`capability-1.0.0`) can observe canonical
character counts, paragraph blocks, text-native acquisition for successful
`text/plain` input, and source spans. It cannot observe language, pages,
tokens, scan status, layout boxes, sections, tables, figures, claims,
examples, or references. Zero counts for those unsupported structures mean
“unavailable”, not “none present”.

## Report and identity

The canonical report contains:

- one corpus and corpus-version identity;
- audit, discovery, parser, capability, and policy identities;
- source-intake formats, sizes, failures, and traversal reconciliation;
- parse statuses, parser/media-type partitions, failures, field coverage, and
  integer distributions;
- acquisition statuses and known-only/population rates;
- capability-aware structure, source-span, and page-indicator summaries;
- in-scope duplicate and family aggregates, with advisory candidates separate;
- bounded limitations; and
- cross-section reconciliation checks.

`input_digest` identifies the canonical semantic input. `report_id` identifies
that input together with the report schema and implementation identity.
`report_digest` identifies the canonical report content. Repeating an audit
over the same immutable request and tool identities must produce byte-identical
`report.json` and `report.md`; changing the source inventory or corpus version
must change the input and report identity.

## Local publication and privacy

`run_and_publish_audit(...)` runs one validated request and publishes a new
directory containing exactly `report.json`, `report.md`, and `manifest.json`.
Publication stages, validates, flushes, and atomically renames the directory;
it never overwrites an existing destination. The report manifest contains
checksums for the published artefacts.

The runner does not discover files or open storage. A deployment-specific
adapter must pin one corpus snapshot and adapt its complete ING-02 inventory,
ING-01 observations, acquisition observations, and in-scope relationships into
the request. The adapter may inspect private source values in memory, but the
audit-owned request, report, logs, exceptions, and `repr` values must not
contain source text, paths, filenames, titles, external identifiers, content
hashes, raw exception messages, or corpus display names.

The command wrapper is:

```sh
uv run python scripts/inference_lab/evaluation/run-corpus-audit.py \
  --request-factory /deployment/private/audit_request.py:build_request \
  --output /deployment/private/audit-output/run-001
```

The request factory is zero-argument and must return a complete
`CorpusAuditRequest`. The command's final result line contains only
report/input identities; the normal privacy-safe lifecycle events may also be
emitted through the configured logging stream. Factory, aggregation, and
publication failures are reported with a fixed generic message; private
exception text is not copied to the process output.

## Focused validation and reruns

Run the focused audit tests and static checks from the repository root:

```sh
uv run pytest tests/inference_lab/evaluation -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

For a rerun, use a new output directory and compare the canonical report bytes
before treating the run as reproducible:

```sh
uv run python scripts/inference_lab/evaluation/run-corpus-audit.py \
  --request-factory /deployment/private/audit_request.py:build_request \
  --compare-to /deployment/private/audit-output/run-001 \
  --output /deployment/private/audit-output/run-002
```

The comparison target must contain the earlier `report.json`; a mismatch
fails before the new directory is published. Review `report.md` for explicit
limitations and inspect `manifest.json` before retaining the output.
