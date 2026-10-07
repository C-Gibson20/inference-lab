# Corpus-audit runbook

> **Status:** The audit engine and local publication runner are implemented.
> The repository does not contain a real corpus or a deployment metadata
> database. Execute the real-corpus procedure only in the local deployment that
> owns the pinned corpus snapshot and source roots.

This procedure produces one aggregate-only audit for one immutable corpus
version. It does not upload source material, invoke a remote model or judge,
or publish document-level diagnostics.

## Before the run

1. Select one already-published `CorpusId` and exact `CorpusVersionId`. Do not
   use an active-version lookup that can change during the run.
2. Pin the source-intake roots and the discovery/classifier policy used to
   produce the inventory. Exclude metadata databases, indexes, managed-source
   roots, backups, and staging directories.
3. Prepare a deployment-local zero-argument request factory that returns a
   complete `CorpusAuditRequest`. It must read one version-pinned snapshot,
   adapt ING-02 discovery/classification results, retain one parse slot per
   member, add explicit acquisition observations where a capable producer
   exists, and include only in-scope relationships.
4. Choose a new output directory outside the repository. Never commit the
   request factory, source paths, report output, or private corpus data.
5. Record the audit implementation, contract, discovery, parser capability,
   and policy identities from the resulting report. These identities are part
   of reproducibility.

The request factory is the storage integration point. The EVAL-02 runner does
not open SQLite, choose a corpus version, traverse source roots, or guess
membership. This prevents a deployment-global catalogue or a changing active
version from silently changing the audit population.

## Run and publish

Use the repository root and a deployment-local factory:

```sh
uv run python scripts/inference_lab/evaluation/run-corpus-audit.py \
  --request-factory /deployment/private/audit_request.py:build_request \
  --output /deployment/private/audit-output/run-001
```

The final result line from a successful command contains only `input_digest`,
`report_digest`, and `report_id`; configured structured lifecycle logs may also
be emitted. The output directory contains:

- `report.json`: canonical machine-readable aggregate report;
- `report.md`: deterministic human-readable rendering; and
- `manifest.json`: schema, report/input identities, and artefact checksums.

The destination must not already exist. A failed or interrupted publication
does not replace an existing report; inspect only runner-owned staging entries
under the output parent if cleanup is required.

## Review the result

Review `report.md` in this order:

1. Confirm the corpus and exact corpus version are the intended selection.
2. Confirm source and member populations are distinct and all reconciliation
   checks are true.
3. Check source format/support counts and traversal failures. Recognised PDF
   input is an unsupported-format finding, not a parse success or an audit
   failure.
4. Check parse status and field coverage. Missing, failed, and unavailable
   values must remain visible in their denominators.
5. Read acquisition, OCR/page-indicator, structure, and span limitations. Do
   not turn unsupported zero counts into claims about the corpus.
6. Check duplicate/family totals and confirm advisory candidates were not
   promoted to confirmed membership.
7. Verify the report is aggregate-only. Paths, filenames, text, titles,
   external identifiers, hashes, and raw exceptions must not appear.

The current text-native capability profile is intentionally limited. It can
measure canonical character counts, paragraph blocks, text-native acquisition
for successful plain-text parses, and source spans. Language, pages, tokens,
scan classification, layout, tables, figures, claims, examples, references,
and other unsupported structures must be reported as unavailable or missing,
not inferred from filenames, punctuation, headings, or heuristics.

Those explicit unavailable states are acceptable coverage for the EVAL-02
Phase 4 boundary; they do not assert that the corresponding corpus features are
absent. Do not add heuristic audit-only inference to turn an unavailable value
into an observation. Later versioned parser, extraction, and OCR producers may
strengthen the evidence, at which point this audit must be rerun before its
conclusions are reused.

## Repeatability check

Run the same request factory against a new destination and compare with the
first report:

```sh
uv run python scripts/inference_lab/evaluation/run-corpus-audit.py \
  --request-factory /deployment/private/audit_request.py:build_request \
  --compare-to /deployment/private/audit-output/run-001 \
  --output /deployment/private/audit-output/run-002
```

The command compares canonical `report.json` bytes before publishing the
second directory. A mismatch means the source inventory, pinned snapshot,
request construction, or tool identity changed; do not explain it away as
rounding or output ordering. Preserve both local outputs for investigation,
outside the repository and subject to the deployment's source-data policy.

## Reviewed conclusion for this checkout

No real-corpus audit conclusion is recorded in this repository. The pinned
retrieval corpus and deployment metadata database remain local and are not used
by the committed synthetic audit tests. Those tests establish deterministic
mixed-capability, reconciliation, privacy, and two-corpus isolation behaviour.
A deployment operator must execute the procedure above and review its
aggregate report before the corpus audit can be considered complete for a real
corpus. The retrieval baseline does not replace this separate audit.
