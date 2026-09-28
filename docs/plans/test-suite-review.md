# Test suite review and rework plan

**Status:** Consolidation complete for the implemented Phase 1–5 test tree.
Acceptance paths for Phases 6–13 remain reserved until those features land.

`build-details.md` is the acceptance source for this work. Each issue's
**Purpose**, **Scope**, **Boundary**, and **Done when** define the behavior to
protect. `build-order.md` defines which issues are implemented now and which
remain future work.

## Execution progress

- **Phase 1:** Consolidated settings, startup/logging, CI/hooks/commit
  validation, and graph refresh into seven phase-acceptance test functions.
  The former six modules had 73 test functions. Shared JSON and project-wide
  validation checks remain for the contract phase. The new acceptance file
  statically parses, and the tracked diff passes `git diff --check`. The full
  suite then had 1,956 test functions in 226 modules, down from 2,022 in 231
  modules before this work.
- **Phase 2:** Replaced separate profiling, identity, fake-runtime, dataset,
  and parser-contract matrices with four profiling journeys, three identity
  checks, six runtime acceptance paths, six dataset paths, and the existing
  text-native golden acceptance. Shared JSON/project validation and later
  phase parser persistence tests remain. At the Phase 2 checkpoint there were
  1,723 test functions in 207 modules. Changed Python files statically parse
  and the tracked diff passes `git diff --check`; tests have not been run.
- **Phase 3 (complete):** Replaced the SQLite per-method wrong-type read
  and write matrices with one read rollback path and one write rollback path.
  Lineage byte-stability and round-trip checks now share the source-to-summary
  lineage acceptance path, with a strict malformed-payload check retained.
  Private repository CRUD/lifecycle checks that duplicate the complete
  migration, transaction, and two-corpus journeys have been retired. The
  provenance tests now center on identity resolution, lineage acceptance,
  deterministic fingerprints, and graph validation instead of per-field and
  per-serializer modules. Phase 4's discovery result builders and count
  matrices have been retired where the discovery-to-identity path observes
  their output. At that checkpoint, static counting showed 1,528 test
  functions in 185 modules. Memory, retrieval, and the remaining discovery and
  persistence suites remain under review.
- **Phases 4–5 (complete):** Removed parser mapping and reconstruction unit
  matrices where the fresh-process persistence acceptance reconstructs
  successful, recoverable, and fatal outcomes byte-for-byte. The parse fault
  test now injects one late metadata failure, after prior rows are staged,
  instead of repeating the same rollback assertion for every table. The discovery privacy
  journey now traverses supported, recognised-unsupported, and unknown files
  through classification, hashing, identity persistence, and partial-result
  accounting; the separate result-lifecycle fixture tests were retired. The
  five baseline helper modules were retired against serving-ingest,
  publication, recovery, and chunk acceptance paths.
- **Phase 3 memory accounting:** Replaced five memory model, snapshot,
  serialisation, logging, and fake-adapter matrices (44 test functions) with
  four RUN-03 paths for attributed budget headroom, incomplete readings,
  canonical persistence, and safe runtime records. Kept the concrete PyTorch
  storage calibration tests.
- **Storage publication:** Retired thirteen lower-level corpus publication
  and update modules after mapping commit, conflict, rollback, race, snapshot,
  restart, and isolation guarantees to baseline publication acceptance.
  Retired parsed-record and parse-outcome write matrices after mapping
  aggregate publication, replay, conflict, supersession, and late-failure
  guarantees to parse persistence acceptance. Retired source-artifact and
  source-observation CRUD modules against fresh-process source persistence,
  identity continuity/isolation, publication, and failure acceptance. Retired
  SQLite chunk write/selection modules against chunk golden acceptance,
  restart-safe baseline ingest, and lexical retrieval journeys. Retired
  managed-source blob read/write helper matrices after mapping byte identity,
  corruption, interruption, cleanup, and privacy outcomes to complete source
  publication and cold-process recovery. Retired the
  corpus update outcome/value matrices and duplicate baseline checkpoint
  record/outcome modules; publication, recovery, checkpoint persistence, and
  one compact checkpoint value suite retain the state transitions and durable
  restart guarantees.
- **Phase 5 retrieval:** Replaced the fifteen-test metadata-filter contract
  matrix with four grouped RET-03 checks for every supported field and
  predicate, boolean and missing-value behavior, invalid combinations, and
  bounded/private validation. Actual query ordering, empty results, invalid
  inputs, and restart remain covered by filtered retrieval and serving search
  acceptance. Consolidated separate query, evidence, and result value matrices
  into two RET-01 acceptance checks for invalid boundaries and ordering,
  limits, and corpus-version mismatch; the shared multi-channel serialization
  acceptance remains.
- **Phase 4 parser outcomes:** Retired the direct outcome-construction matrix;
  success, recoverable failure, and fatal failure are observed through parser
  persistence, cold-process reconstruction, and restart acceptance.
- **Phase 4 parse publication:** Retired the mock-repository parse publication
  matrix where real SQLite source publication, parse restart, privacy, and
  fault-injection journeys exercise complete persisted outcomes and rollback.
- **Phase 4 chunking:** Retired separate chunk mapping, wire-format,
  instrumentation, performance-rescanning, and storage-record matrices after
  checking their required behaviors against golden chunk fixtures and the
  serving-ingest stage and restart journeys. Kept the token-window algorithm,
  tokeniser boundary, source-navigation, and deterministic chunk-ID checks.
- **Future-phase cleanup:** Removed the EVAL-10 final benchmark coverage and
  validator suites because build-order places EVAL-10 in Phase 13. The EVAL-01
  dataset contract and current Phase 4/5 evaluation paths remain.
- **Phase 4 source classification and parser records:** Retired separate
  classification-rule matrices after the filesystem discovery journey covered
  supported, recognised-unsupported, and unknown input classes. Retired the
  source-constructor matrix after the golden parser, persisted source-location,
  and chunk navigation journeys retained Unicode offsets and page coordinates.
- **Phase 3 identity resolution:** Retired the isolated resolver outcome
  matrix where new/unchanged/changed/moved/renamed/duplicate decisions are
  exercised through filesystem discovery and durable identity persistence.
  Dedicated ambiguity, conflict, retry, and two-corpus isolation paths remain.
- **Phase 3 repository lifecycle:** Retired separate SQLite initialisation,
  upgrade, and lifecycle wrappers after matching those cases to configured
  factory use, complete migration acceptance, and persisted source restart.
- **Phase 5 retrieval contracts:** Consolidated limit/query/evidence/result
  constructor matrices, metadata-filter algebra, query/filter serialization,
  and lexical query lifecycle checks into the channel-neutral RET-01 and RET-03
  acceptance paths plus persistent lexical-generation journeys.
- **Phase 5 baseline contracts:** Reduced the 24-case ingest DTO matrix to five
  grouped checks for request/stage identity, bounded failures, document
  outcomes, publication summaries, and the public API. End-to-end stage,
  checkpoint, restart, and publication scenarios remain in serving-ingest and
  baseline recovery/publication acceptance.
- **Phase 5 checkpoint persistence:** Retired the separate SQLite checkpoint
  CRUD/reopen matrix after confirming that serving ingest and baseline recovery
  exercise stage-by-stage resume, exact output reuse, incompatible proposals,
  restart, and corrupt durable work. Retained the compact checkpoint identity
  value suite.
- **Phase 4 source structures:** Retired the standalone parsed-representation
  and source-location constructor matrices after mapping canonical text, source
  offsets, coordinate systems, typed structures, and restart resolution to
  parser golden, cold-process persistence, chunk navigation, and SQLite read
  acceptance.
- **Phase 4 storage records:** Retired separate source, parse-outcome, and
  parsed-representation constructor/serialization matrices after checking
  canonical serialized output in cold-process persistence, source path and
  metadata integrity on repository reopen, parse failure privacy, and corrupt
  payload rejection in public storage acceptance.
- **Phase 3 corpus-version values:** Retired per-field component/manifest
  contract tests after matching semantic identity, compatibility, parent
  selection, replay, and cross-corpus publication rules to persistent lexical
  generation and corpus publication acceptance.
- **Phase 4 EVAL-02 audit:** Retired the 14-case aggregation matrix after
  mapping exact totals, unavailable-versus-absent coverage, numeric summaries,
  duplicate/family accounting, and report identity to the golden aggregate,
  two-corpus, and operator rerun acceptance paths. Shared report fixtures now
  live outside test modules; canonical serialization and publication checks
  remain. Also removed two internal parse-preflight checks for the absence of
  logging and result-constructor validation; source privacy and rejected-input
  behavior remain covered through persistence acceptance.
- **Phase 3 PROV-02 persistence:** Removed the two success-only helper tests
  for first identity creation and repeated no-op. The cold-process continuity
  acceptance already exercises new, unchanged, changed, moved, and duplicate
  identities through discovery and durable repository state.
- **Phase 4 ING-02 discovery contracts:** Removed three constructor-only
  matrices for strict input container/type shapes and filesystem identity-hint
  types. The discovery journeys still cover valid file/directory inputs, safe
  path handling, stable file evidence, parser support, and durable identity.
- **Phase 5 baseline summary:** Replaced four serializer unit checks with one
  summary-level contract that verifies deterministic output and bounded,
  payload-free failure fields.
- **Phase 4 audit publication:** Folded the standalone publication-error
  value check into the actual corrupt-write cleanup journey, where failure
  category, stage, and retryability are now asserted on the raised error.
- **Phase 3 PROV-02 resolution:** Removed four isolated new/unchanged/changed/
  duplicate outcome checks after confirming those identity transitions run
  through filesystem discovery and durable restart acceptance. Ambiguous and
  conflicting evidence plus policy-controlled matches remain explicit cases.
- **Phase 3 identity persistence:** Removed three direct success-only
  persistence checks for new identity, continuous version, and duplicate
  relationship writes; the filesystem-to-repository continuity/isolation
  journeys assert those durable outcomes. External-identifier persistence and
  transactional rollback cases remain.
- **Phase 5 checkpoint values:** Reduced twelve constructor-level checks to
  four grouped contracts for canonical work keys, closed lifecycle states,
  manifest identity/components, and checkpoint identity/state consistency.
  Restart/resume and corruption behavior remains in the recovery journey.
- **Phase 4 parse preflight:** Removed two checks for parser/OCR predecessor
  variants beyond ING-03's scope and one internal representation privacy check;
  public persistence failure and privacy paths remain.
- **Phase 4 parse requests:** Removed immutability/repr checks and a seven-field
  wrong-type matrix from the persistence request DTO suite. Failure attribution,
  parser/source agreement, and publication-state invariants remain.
- **Phase 5 filtered retrieval:** Removed five wrapper-level repeats for filter
  counters, empty-result counters, invalid-filter logging, fanout delivery, and
  corrupt-metadata rejection. The real snapshot query, public invalid-filter
  boundary, and profiled failure path retain these behaviors.
- **Phase 5 lexical indexing:** Removed seven repeated helper paths for ranked
  paging/page-size validation, filtered-query profiling, unavailable-generation
  profiling, exact-parent replay, the duplicate corpus-publication check, and
  no-parent selection. Query modes, filtered execution, update/removal,
  pinned readers, isolated corpora, publication rollback, generation integrity,
  and zero-chunk replay remain.
- **Phase 4 audit contracts:** Removed four local checks for distributions,
  payload representations, wrong-corpus inventory, and a closed enum after
  mapping those requirements to the golden aggregate, privacy, and report
  acceptance paths.
- **Phase 4 parse persistence:** Strengthened the real SQLite source-mismatch
  journey to assert that failed preflight publishes no source, representation,
  or outcome rows. Removed five preflight-only repeats for matching metadata,
  corpus-scope mismatch, source hashing, and malformed helper arguments; the
  remaining stale-source, size, predecessor, and failure-boundary cases stay.
  Also removed the duplicate membership-absence, compatible-predecessor, and
  invalid-size-configuration checks already represented by isolation,
  supersession, and settings boundaries.
- **Phase 4 traversal:** Removed five helper-level type, numeric-field, and
  repr checks and seven overlapping/out-of-scope cases for duplicate roots,
  hard links, special files, mount boundaries, conflicting duplicate-root
  metadata, kind-mismatched roots, and recursion-depth mechanics. Filesystem
  journeys retain deterministic ordering, symlink/junction boundaries, race
  handling, inaccessible entries, partial failures, and safe public results.
- **Phase 4 discovery failures:** Removed the immutability, full dataclass-field
  shape, and invalid-discriminant matrices. Category policy, one factory path,
  category/policy mismatch rejection, bounded media metadata, and private-path
  error behavior remain.
- **Phase 4 discovery contracts:** Reduced thirteen strict-constructor checks
  to six policy/path/family/classification/snapshot contracts. Explicit file
  metadata, directory scope, selected corpus, stable-read agreement, and
  identity handoff are now observed in the filesystem-to-repository journeys.
- **Phase 4 hashing:** Removed three helper-boundary checks for runtime type
  shapes, injected clock regression, and failure reprs. Streaming, size bounds,
  changing-file/symlink races, verified reacquisition, and safe discovery
  outcomes remain covered.
- **Phase 4 audit source inventory:** Removed one extra mixed-failure
  denominator case whose failure categories and exact totals appear in the
  golden audit fixture.
- **Phase 4 chunking:** Removed the eight direct window-helper tests after
  tracing each case to the parsed-document-to-chunk acceptance suite: golden
  size/overlap, tail handling, nested structures, tables, empty input, Unicode
  navigation, and fail-closed outcomes all remain there.
- **Phase 4 chunk records:** Removed four record-level success/failure repeats
  for source lineage, stable IDs across corpora, clipped references, and empty
  failed output, plus a zero-length source-reference constructor check; the
  chunk acceptance journey asserts the navigable records on its full output.
- **Phase 3 SQLite connection:** Reduced fourteen connection-helper cases to
  five checks for secure connection defaults, explicit WAL, local paths, error
  translation by SQLite result code, and retryable locks. Factory, inspection,
  and concurrency acceptance retain invalid database and reopen behavior.
- **Phase 5 checkpoint storage:** Removed ten direct SQL tests for checkpoint
  and manifest table constraints. The grouped checkpoint value contract and
  cold-restart resume/corruption acceptance retain lifecycle and durable-work
  behavior.
- **Phase 3 locator schema:** Retired five direct migration/constraint checks
  after mapping initialization, upgrade, revision history, scoped uniqueness,
  and hint persistence to repository migration and filesystem continuity/
  isolation acceptance.
- **Phase 3 locator behavior:** Reduced eleven direct storage checks to four
  focused boundaries for stale/conflicting writes, wrong-corpus writes, and
  corrupt current pointers. Valid moves, restart, ambiguity, privacy, and
  cross-corpus reads are observed through identity acceptance paths.
- **Phase 3/7 runtime and Phase 5 retrieval profiling:** Removed local runtime
  metadata/type checks already established at the shared fake-runtime
  boundary, and the local generator-only stream/cancellation cases owned by
  the later RUN-06 issue. Retained local token/context limits, model-artifact
  integrity, cache mechanics, and safe logging. Removed two lexical profiling
  checks duplicated by the shared build and query profiling journeys.
  Consolidated unloaded/closed/load lifecycle and prompt rendering/role-boundary
  checks into two local mechanics journeys.
- **Phase 3 provenance validation:** Removed six empty/value-shape or
  overlapping graph cases for empty bundles, DTO immutability, duplicate graph
  entries, self/direct cycles, and a redundant missing-child edge. Retained
  missing parents/spans, input-edge agreement, an indirect cycle, multiple
  parents, and version/supersession validation.
- **Phase 3 hash values:** Replaced eight content-hash DTO checks with one
  canonical round-trip and invalid-digest boundary. Real file hashing and
  repository restart paths remain the acceptance evidence for hash identity.
- **Phase 3 identity paths:** Removed eight resolver-context, allocator-collision,
  and operation-specific rollback helper tests. Conflict retry, bounded retry
  exhaustion, fatal failure, one injected late write rollback after earlier
  writes are staged, identity continuity/restart, and two-corpus isolation
  remain. Retired the now-empty resolver-boundary module and the per-write
  rollback parameter matrix.
- **Phase 3 storage boundaries:** Retired protocol-shape introspection and
  value-only chunk write tests after checking concrete repository, chunk
  publication, pinned-snapshot, and public-API journeys. Retired the separate
  transaction-contract matrix; repository migration and write-failure journeys
  retain complete-graph commit and rollback evidence. Removed the no-resource
  cleanup no-op check; the primary-exception cleanup path remains.
- **Phase 3 identity lookup:** Retired the direct candidate-query matrix and
  locator value-contract module after tracing hash/identifier selection,
  optional locator hints, rename continuity, and corpus isolation to real
  filesystem-to-repository restart journeys.
- **Phase 1 validation and Phase 3 migration inspection:** Reduced shared
  slug, media-type, and path-overlap predicates to representative accepted and
  rejected values; public settings and storage paths retain local-path checks.
  Migration inspection now keeps one checksum and one schema-drift rejection
  path instead of 15 nearly identical mutations.
- **Phase 5 RET-01 serialisation:** Removed query-serialisation cases already
  represented by the public retrieval acceptance path and trimmed malformed
  payload matrices to representative nested-value, identifier, query, and
  private-error cases. Full multi-channel result round-trip remains.
- **Phase 4 ING-01 versioning:** Consolidated twelve parser-version and span-ID
  helper checks into four behaviors: canonical/accepted versions, invalid
  version rejection, deterministic IDs sensitive to their source inputs, and
  representative invalid identity/location rejection.
- **Phase 4 EVAL-02 adapters:** Reduced seven adapter unit checks to one
  canonical-scope journey for preserving a member whose parse was not
  attempted. Exact totals, parser attribution, wrong-corpus rejection, and
  privacy remain in audit golden and report acceptance paths.
- **Phase 4 EVAL-02 report checks:** Replaced ten calls to private section
  renderers with one public report rendering assertion covering section order
  and required content. Consolidated five report-validation cases into one
  rejection path for schema, population, ordering, reconciliation, and digest
  drift.
- **Phase 4 ING-02 continuity:** Retired the twelve-case internal resolver
  suite after checking each branch against the filesystem-to-repository
  restart journeys for rename, changed bytes, move, duplicate observation,
  incomplete scans, and two-corpus isolation. Also removed the standalone
  empty-file digest check, empty-directory traversal check, duplicate
  root-order check, and junction-ancestor variant; nonempty file/directory,
  stable traversal, symlink policy, scan gaps, bounded hashing, and changing
  file behavior remain in the phase paths.
- **Phase 4 ING-03 contracts and Phase 5 STOR-03 snapshot:** Retired the
  request-only DTO matrix and two resolver helper tests because successful and
  failed parse publication, exact restart reconstruction, source mismatch, and
  incomplete-scan ambiguity already run through repository acceptance paths.
  Removed one duplicate lexical-store snapshot test; corpus publication
  acceptance retains pinned snapshot semantics, while lexical query acceptance
  verifies the old generation remains queryable during publication.
- **Phase 3 STOR-01 migration checks:** Removed exact internal migration
  checksum/schema-shape assertions already covered by the public factory and
  supported-upgrade journeys. Reduced malformed-identity and unsafe migration
  statement matrices to one representative rejection each. Retired the
  separate STOR-02 source-persistence schema matrix after mapping its restart,
  source reuse, span resolution, parse status, migration, and corruption
  behaviors to the public source persistence, parse restart, migration, and
  integrity acceptance paths.
- **Phase 3 STOR-01 schema matrix:** Retired three direct SQL checks for
  malformed UUID text, foreign-key version creation, and hash non-uniqueness;
  typed identity/value tests, repository referential-integrity recovery, and
  duplicate identity acceptance exercise those outcomes through their owning
  boundaries.
- **Phase 4 STOR-02 publication:** Removed two single-table rollback tests
  duplicated by the table-stage failure journey, which injects failures across
  the complete persisted source/parse bundle and verifies rollback, restart,
  corpus preservation, and privacy for every publication stage.
- **Phase 2 ING-01 failures and Phase 5 RET-02 indexing:** Kept one valid
  fatal/recoverable parser-failure contract path and one representative
  invalid-category/location contract; removed the remaining constructor-only
  shape variations. Removed the lexical generation helper's build-only
  assertion because the pinned-snapshot query journey and documented operator
  journey build and consume durable generations with source-linked results.
- **Phase 4 ING-04 tokeniser:** Removed four helper checks for boundary-map
  variants and invalid tokeniser offsets after mapping Unicode navigation,
  empty input, configuration identity, and bounded tokeniser failure to the
  golden chunk acceptance paths. Offline loading, symlink rejection, and the
  deterministic fake-tokeniser contract remain.
- **Phase 4 ING-03 preflight:** Retired the remaining mock-reader suite after
  mapping source mismatch/no-publication to the real SQLite privacy journey,
  predecessor success/conflict to parse publication, wrong-corpus membership
  to isolation, and interruptions to publication fault acceptance. This
  removes internal call-order assertions while keeping their public outcomes.
- **Phase 3/4 storage reads and writes:** Reduced parsed-representation read
  coverage to deterministic ordering, digest corruption, structural-kind
  round trips, repeated-text spans, and image-pixel layout; source restart,
  location, and corpus-isolation paths retain the complete workflow. Reduced
  source-history failure cases to creation/advance rollback, cross-corpus
  rollback, and commit recovery. Removed five direct corpus-version schema
  transitions already covered by publication, concurrency, and migration
  acceptance, retaining committed-version immutability.
- **Phase 4 STOR-02 publication faults:** Replaced the per-table database
  failure matrix with one late failure after source and parsed rows are staged,
  and reduced the four pre-promotion filesystem operation failures to one
  representative write interruption. Post-promotion orphan recovery,
  transaction rollback, and corrupt-source rejection remain separate paths.
- **Phase 3 STOR-01 factory and inspection:** Removed duplicate invalid-path
  shapes, invalid reconciliation-bound type matrices, and checks already
  covered by the public factory or migration journeys. Kept one path-overlap
  rejection, symlink-alias rejection, private path-error handling, explicit
  migration permission, and checksum/schema/history incompatibility paths.
- **Phase 3 RUN-02 local mechanics:** Reduced model-artifact tamper and cache
  helper variants to one representative artifact-integrity case and the
  rendered-prefix plus changed-prefix observations. Lifecycle, token/context
  limits, capability rejection, safe logs, and the fake backend remain.
- **Phase 4 ING-02 hashing:** Reduced the helper suite to six checks for
  streaming hash evidence, changed-source readback rejection, the size
  boundary, an open-time symlink race, post-read file mutation, and bounded
  I/O failure. Directory traversal still owns root/ancestor symlink policy.
- **Phase 4 ING-02/Phase 5 EVAL-01 contracts:** Removed the resolver's
  target-corpus constructor misuse check because corpus isolation is exercised
  through real two-corpus persistence. Reduced evaluation filesystem helper
  checks to the dataset loader's private-error boundaries; local-path and
  publication rejection remain in dataset/report acceptance.
- **Phase 3 STOR-01 schema:** Grouped eight raw-SQL schema checks into three
  storage-boundary scenarios: external-identifier ordering, duplicate/family
  graph integrity, and migration-registry safety. The grouped graph path still
  exercises mismatched hashes, invalid endpoints, cross-document membership,
  and duplicate family links.
- **Phase 5 RET-02/RET-03 retrieval filtering:** Removed five helper-level
  borrowed-snapshot and profiling wrapper cases already covered by pinned
  generation, corpus-isolation, query-failure, and shared profiling journeys.
  Filter ranking, empty results, identifier mode, provenance ranks, and invalid
  filter rejection remain in the retrieval paths.
- **Phase 4 ING-02 traversal:** Removed the local overlapping-root case because
  the restart/no-op identity journey already exercises overlapping roots,
  canonical order, and deduplication end to end. Combined missing and
  inaccessible root failures under one bounded-privacy path, and folded
  Unicode/reversed-filesystem ordering into the nested/hidden-directory path.
  Combined added/removed directory entries around one snapshot while retaining
  checks that deletion fails that scan and new input waits for the next run.
  Symlink policy, storage overlap, partial traversal, and unsupported inputs
  remain covered.
- **Phase 4 STOR-03 publication:** Removed the standalone commit/reopen smoke
  path because the cold-restart replay and documented operator workflow already
  assert a usable committed version after restart. Reduced the three SQL fault
  injection points to one late publication interruption with rollback and retry;
  removed the separate stale-parent fixture because the same-parent race already
  proves retryable lock contention followed by a non-retryable stale-parent
  conflict and one active winner. The phase criteria require interruption and
  conflict recovery, not a separate test per internal insert/update statement.
- **Phase 5 recovery:** Removed the parse/chunk “current migrated schema” matrix;
  repository migration acceptance proves the current schema, and recovery paths
  already exercise parse/chunk persistence and cold restart against it. Kept
  publication interruption at both ready and committed-but-unacknowledged
  boundaries, exact stage reuse, and fail-closed corruption recovery.
- **Phase 4 ING-02 failure contracts:** Folded the single factory-policy case
  into the complete failure-policy contract check; traversal and discovery
  journeys already observe category-created failures in realistic paths.
- **Phase 4 STOR-02 database failures:** Retained a late transactional failure
  that proves no source/parse records publish and a later retry works. Removed
  separate transaction-start locking, commit-cleanup failure, and rollback
  cleanup failure cases; the publication path and connection contract retain
  failure sanitization and repository usability without testing each SQLite
  cleanup branch independently.
- **Phase 5 RET-02 query:** Removed a second update/removal query path. The
  operator workflow exercises update/search, parent-diff acceptance asserts
  inserted/replaced/removed records, and the pinned-reader path proves the old
  selected generation remains queryable during publication.
- **Phase 5 ING-05 no-change publication:** Removed the storage-only no-change
  commit case because the serving restart path verifies no new sequence, no
  lexical rebuild, stable active version, and replay of the same no-change
  proposal through the public facade.
- **Phase 3 RUN-02 coverage audit:** The local-backend test module had become
  empty, leaving its required local lifecycle, capability, context-limit,
  repeated-prefix, cache-compatibility, and privacy behavior uncovered by the
  fake runtime acceptance. Restored six compact local-model acceptance paths;
  these exercise the behavior RUN-01's fake cannot prove.
- **Phase 3 STOR-01 connection setup:** Removed raw connection PRAGMA and WAL
  setup checks because repository factory/restart and migration journeys
  exercise connection setup at the public boundary. Kept local-path rejection,
  SQLite error translation, and retryable contention classification.
- **Phase 3 RET-01 serialization:** Removed the separate malformed-payload
  matrix after the combined retrieval contract acceptance proved result
  round-trip across channels, nested query/filter round-trip, strict value
  validation, version mismatch, and payload-private failure handling.
- **Phase 3 STOR-01 write atomicity:** Removed the extra first-source creation
  failure and generic commit-failure fixtures. The changed-source history path
  still proves partial writes roll back; source/parse publication retains a
  late database fault with reopen and retry, and cross-corpus rollback remains.
- **Final review and test run:** The complete suite contains 437 test function
  definitions across 98 modules and collects 577 pytest cases. The original
  tree had 2,022 functions in 231 modules. The last redundancy pass reviewed
  `retrieval/test_lexical_store.py`, `serving/test_ingest.py`,
  `evaluation/test_retrieval_contracts.py`, `provenance/test_validation.py`,
  `ingest/test_chunk_acceptance.py`, and
  `storage/test_sqlite_source_publication.py` against the 76-item
  `build-details.md` trace. It found no safe like-for-like removals: the
  remaining cases cover distinct restart, isolation, migration, privacy,
  scoring, and failure outcomes. The 150–250 planning estimate remains unmet
  and is directional, not a per-phase quota. `uv run pytest --cov` now passes:
  577 passed, 74.46% total coverage. Both the total and CI storage coverage
  floors are 70%; storage coverage reports 71%.

## Coverage trace by build item

These files are the current primary acceptance evidence after consolidation.
Smaller contract suites remain for boundaries that the public journeys cannot
observe cleanly. Criteria remain governed by each item in
`docs/plans/build-details.md`.

| Phase | Build item | Primary acceptance evidence |
| --- | --- | --- |
| 1 | TOOL-01 | `tests/tooling/test_phase1_acceptance.py` — clean-checkout hooks, CI configuration, and commit validation |
| 1 | TOOL-02 | `tests/tooling/test_phase1_acceptance.py` — precedence, nested overrides, validation, paths, and startup |
| 1 | TOOL-03 | `tests/tooling/test_phase1_acceptance.py` — renderers, context, privacy, level policy, and reconfiguration |
| 1 | TOOL-04 | `docs/conventions/backup-restore.md` and the operator runbook — strategy is a document gate; executable backup tests belong to STOR-08 |
| 1 | TOOL-05 | `tests/tooling/test_phase1_acceptance.py` — graph refresh output validation, safe replacement, and failure recovery |
| 1 | DOCS-01 | Operator config/query documentation plus `tests/inference_lab/serving/test_operator_workflow.py` — documented API steps are executed in fresh processes |
| 2 | PROV-01 | `tests/inference_lab/provenance/test_identifiers.py`, `test_values.py`, and the shared contract acceptance paths |
| 2 | PROF-01 | `tests/inference_lab/profiling/test_phase2_profile_contracts.py` |
| 2 | RUN-01 | `tests/inference_lab/backends/runtime/test_phase2_runtime_acceptance.py` |
| 2 | EVAL-01 | `tests/inference_lab/evaluation/test_phase2_dataset_acceptance.py` |
| 2 | ING-01 | `tests/inference_lab/ingest/test_text_native_acceptance.py`, `test_versioning.py`, and `test_errors.py` |
| 3 | PROV-02 | `tests/inference_lab/ingest/test_identity_continuity_acceptance.py`, `test_identity_isolation_acceptance.py`, and focused ambiguity/conflict outcomes |
| 3 | PROV-03 | `tests/inference_lab/provenance/test_lineage_acceptance.py`, `test_validation.py`, and `test_fingerprints.py` |
| 3 | STOR-01 | `tests/inference_lab/storage/test_source_persistence_acceptance.py`, repository migration, isolation, inspection, integrity, and concurrency acceptance |
| 3 | RUN-02 | `tests/inference_lab/backends/runtime/test_local.py` |
| 3 | RUN-03 | `tests/inference_lab/backends/runtime/test_phase3_memory_acceptance.py` and `test_memory_pytorch.py` |
| 3 | RET-01 | `tests/inference_lab/retrieval/test_acceptance.py` |
| 4 | STOR-02 | `tests/inference_lab/storage/test_source_persistence_acceptance.py`, `test_sqlite_parsed_representation_reads.py`, and source publication/reconciliation acceptance |
| 4 | STOR-03 | `tests/inference_lab/ingest/test_baseline_publication_acceptance.py`, repository concurrency, and the retained committed-snapshot immutability check |
| 4 | ING-02 | `tests/inference_lab/ingest/test_discovery_privacy_acceptance.py`, traversal, hashing, and identity restart/isolation acceptance |
| 4 | ING-03 | Parse publication, cold-process restart, privacy, corpus isolation, and fault-injection acceptance under `tests/inference_lab/ingest/` |
| 4 | ING-04 | `tests/inference_lab/ingest/test_chunk_acceptance.py` and `test_chunk_records.py`, with persisted chunk replay under baseline recovery/publication acceptance |
| 4 | EVAL-02 | Audit golden, runner, observability, and report acceptance under `tests/inference_lab/evaluation/`; the two identical runs on a real pinned corpus remain a deployment-owned gate |
| 5 | ING-05 | `tests/inference_lab/serving/test_ingest.py`, baseline recovery, and publication acceptance |
| 5 | PROF-02 | Shared profiling contracts plus serving ingest/search and SQLite observability journeys |
| 5 | RET-02 | Lexical build/query and generation acceptance under `tests/inference_lab/retrieval/`, plus the documented serving workflow |
| 5 | RET-03 | `tests/inference_lab/retrieval/test_phase5_metadata_filter_acceptance.py`, metadata adapter, filtered retrieval, and serving workflow |
| 5 | EVAL-03 | Retrieval evaluation contracts, execution, operator, and cold-publication acceptance under `tests/inference_lab/evaluation/` |
| 5 | SERV-01 | `tests/inference_lab/serving/test_operator_workflow.py`, `test_ingest.py`, and search/location acceptance |

The trace distinguishes fixture acceptance from deployment evidence. EVAL-02's
real-corpus repeatability requirement cannot be claimed from synthetic fixtures;
it remains an explicit deployment gate in the build specification.

## Final audit

- All 76 issue IDs in `build-details.md` have a corresponding row in the
  coverage trace. Phase 1–5 rows point to current acceptance or contract tests;
  Phases 6–13 describe the acceptance paths to add with those implementations.
- The only retired path still named as a file in the review is
  `tests/inference_lab/test_settings.py`; its behavior is covered by the Phase 1
  settings/startup acceptance path. `test_chunk_tokenisation.py` was removed
  after its token-boundary behavior moved into `test_chunk_acceptance.py`.
- The final test-tree inventory is 98 modules, 437 test function definitions,
  and 577 collected pytest cases. This is 1,585 fewer test functions and 133
  fewer modules than the original 2,022-function, 231-module inventory.
- The full suite passes with 577 cases and 74.46% total coverage. The targeted
  storage report is 71%, above its 70% CI floor. `git diff --check` passes.
- EVAL-02 still requires two byte-identical runs on a real pinned corpus. That
  gate is deployment-owned; synthetic fixture coverage does not satisfy it.

## Review conclusion

The current suite is still substantial: 98 test modules and 577 collected
cases, down from 231 modules and 2,022 test functions. A phase is the default
test boundary because it delivers a complete piece of work.
Phase acceptance tests should exercise the user or operator path through the
contracts and components delivered by that phase. Individual issues should
get separate tests only for a standalone contract, algorithm, failure
boundary, or risk that the phase workflow cannot observe meaningfully. Phase 2
does call for deterministic contract fixtures; later issues increasingly need
persistence, restart, isolation, recovery, operator workflows, and measured
quality.

Phase 1 through Phase 5 are represented in the current code and test tree.
Phases 6 through 13 are later build work; the tests for those issues belong
with those implementations. Do not add speculative test scaffolding for
unimplemented features.

Earlier review passes consolidated a small number of literal duplicate
checks, including the baseline API export assertion and source-blob key type
assertion. The final pass checked the remaining largest suites against the
issue trace and did not find a further safe removal. The test scale below is
directional; no fixed per-phase test budget applies.

## Test design rules

1. Tie each acceptance test to an issue ID and a concrete `Done when`
   requirement in its docstring or test name.
2. Prefer one test of a complete public workflow over multiple tests of
   private helper calls when both prove the same outcome.
3. Keep unit tests for pure algorithms and contracts whose behavior is itself
   an issue aim: identifiers, schemas, parser coordinates, deterministic
   chunking, scoring, resource records, and strict failure categories.
4. Use temporary local files and real SQLite repositories for storage,
   ingestion, restart, and isolation paths. Use deterministic fake model
   backends where model execution is not the behavior under test.
5. Inject a crash or fault at the owning boundary, then assert durable state,
   restart behavior, reuse, and the previous active version. Avoid testing each
   internal exception branch when one path proves the recovery guarantee.
6. Assert user-visible outcomes: returned records and locations, selected
   corpus/version, durable membership, logs' privacy properties, and whether
   work was reused. Do not retain tests solely to raise line coverage.
7. Keep phase gates explicit. A passing unit suite does not stand in for a
   restart, operator workflow, held-out evaluation, or hardware measurement
   required by the issue.

## Phase-level test shape and scale

Each phase gets a small set of complete acceptance journeys. A journey can
cover several related issue aims in one run. How many journeys a phase needs
depends on the amount and risk of its behavior; there is no fixed per-phase
cap. Keep issue-level tests when a standalone contract or failure boundary
cannot be observed through those journeys.

The desired scale is about **one order of magnitude fewer tests than the
original 2,022 test functions**. A rough planning range is 150–250 meaningful
acceptance scenarios and indispensable contract checks across all 13 phases.
The current 577 collected cases remain above that range after the final
redundancy pass. The range is directional, not a hard quota; each remaining
case was retained because it protects a distinct build criterion or failure
boundary. Revisit the count when implementation changes make a stronger phase
journey possible. Count independently exercised scenarios, including
parameterized variants; do not split every field or exception into a separate
test.

The phase-level journeys are:

| Phase | Complete phase acceptance journey |
| --- | --- |
| 1 — Cross-cutting tooling and infrastructure | From clean setup through validated settings and configured logging to local graph output and accurate operator guidance. |
| 2 — Shared contracts and test seams | Exercise public identity, profiling, fake runtime, evaluation dataset, and parser boundaries with deterministic fixtures. |
| 3 — Corpus identity and persistence | Create identities and lineage, persist a two-corpus repository, reopen it, and verify scoped reads plus local runtime/retrieval contracts. |
| 4 — Reproducible ingestion | Take filesystem inputs through discovery, identity, parsing, source persistence, chunking, corpus publication, and repeatable audit. |
| 5 — First runnable baseline | Ingest through the documented API, publish, restart, search/filter, resolve locations, resume interrupted work, and verify corpus isolation. |
| 6 — Hybrid retrieval baseline | Run local embedding/reranking through dense indexing, fusion, evidence selection, restart, and the held-out retrieval gate. |
| 7 — Evidence-grounded answers | Answer from bounded evidence, verify citations/abstention, resolve cited sources, and evaluate grounded-answer behavior. |
| 8 — Selective OCR | Process mixed native/scanned documents through selective OCR merge and restart, then measure downstream retrieval/citation change. |
| 9 — Structured knowledge | Persist/extract/normalize facts, query and aggregate them with provenance, and evaluate against verified records. |
| 10 — Summary knowledge | Generate and invalidate traceable summaries, retrieve them with source preference, and compare hybrid channels. |
| 11 — Sessions and context caching | Run a multi-turn session through restart, follow-up, caching, budget eviction, and corpus/version invalidation. |
| 12 — Hardware-aware runtime | Exercise scheduling, residency, and batching under simulated and representative hardware workloads; retain the full quality/systems result. |
| 13 — Interface and operational hardening | Restart, backup/restore, resume ingest, use chat/upload, enforce local-only operation, and run final held-out acceptance. |

## Issue acceptance inventory

Each row below is an acceptance obligation from `build-details.md`, not a
request for one separate test. Combine rows into the phase-level journeys
above. Keep a separate issue-level test only when it proves an independent
contract or failure boundary that the phase journey cannot reach.

| Build phase | Issue | Acceptance path to protect |
| --- | --- | --- |
| 1 | TOOL-01 | Clean-checkout quality gate, hook/commit rejection, CI event behavior, and cancellation of superseded runs. |
| 1 | TOOL-02 | Load example configuration through TOML, nested environment overrides, validation errors, path checks, and cache lifecycle. |
| 1 | TOOL-03 | Configure each renderer and level, start the app, emit a representative dependency record, and verify required fields and private-data rejection. |
| 1 | TOOL-04 | Keep as a reviewed recovery strategy and future acceptance contract; executable backup tests belong to STOR-08. |
| 1 | TOOL-05 | Run the graph wrapper against a small fixture tree; verify complete publication, old-output recovery on failure, and safe logs. |
| 1 | DOCS-01 | Validate documented clean-checkout setup and operator examples against the actual current command/API and settings. |
| 2 | PROV-01 | Keep compact value tests for generation, canonical parsing, domain separation, equality, hashing, and invalid spellings. |
| 2 | PROF-01 | Keep deterministic record/timer tests for UTC time, units, unavailable measurements, safe failures, and competing emission errors. |
| 2 | RUN-01 | Exercise fake backend lifecycle, capabilities, ordered streaming, limits, cancellation, and safe failure through the public protocol. |
| 2 | EVAL-01 | Keep dataset fixtures for schema, evidence, abstention, leakage, checksums, atomic publication, paths, and final composition. |
| 2 | ING-01 | Golden parser path from bytes to parsed structure; verify Unicode coordinates, empty/malformed/oversize input, repeatability, and payload-free logs. |
| 3 | PROV-02 | Resolve identities for new, unchanged, changed, moved, renamed, duplicate, and family-linked sources; verify stable identity and citable members. |
| 3 | PROV-03 | Trace a source through derived records and multiple parents; reject missing/cyclic lineage and verify versioned serialization. |
| 3 | STOR-01 | Create and reopen a repository, migrate it, roll back a transaction, and prove two-corpus membership and scoped reads. |
| 3 | RUN-02 | Run the small local mechanics fixture through lifecycle, token/context limits, cache reuse, and capability errors. |
| 3 | RUN-03 | Feed component observations into the memory budget and verify attribution, headroom, missing readings, and emitted records. |
| 3 | RET-01 | Keep focused contract fixtures for query/filter/result linkage, ordering, scores, empty results, and corpus-version mismatch. |
| 4 | STOR-02 | Persist source, parsed structure, and locations; close/reopen; resolve a span; detect corruption; prove interrupted writes publish nothing. |
| 4 | STOR-03 | Stage and commit a corpus version; race writers; read pinned snapshots; inject interruptions; retry/rollback; and prove independent corpora. |
| 4 | ING-02 | Discover real temporary files/directories, classify/hash/resolve them, repeat unchanged input, then verify changed/moved/duplicate and second-corpus outcomes. |
| 4 | ING-03 | Parse and persist a golden source, restart and replay it, then inject stale/mismatched/invalid-span/write-failure cases and inspect published state. |
| 4 | ING-04 | Chunk representative empty, short, nested, table, and long documents; verify token bounds, overlap, ordering, stable IDs, and exact source navigation. |
| 4 | EVAL-02 | Audit a pinned fixture corpus twice; compare exact totals, limitations, unavailable fields, and privacy-safe report output. |
| 5 | ING-05 | Use the serving ingest facade with multiple sources; cover new/unchanged/changed/duplicate and failed documents, crash/restart at each durable stage, resume, conflicts, and two-corpus isolation. |
| 5 | PROF-02 | Instrument one ingest and query run with an injected clock; assert stage counts, failure records, primary exception, and collected benchmark records. |
| 5 | RET-02 | Build and query a persistent lexical index across restart; exercise identifiers, Unicode, ties/top-k, updates/removals, corruption, and generation/corpus isolation. |
| 5 | RET-03 | Run field and combined-filter queries against a selected corpus version, including missing values, boundaries, ordering, invalid input, and restart. |
| 5 | EVAL-03 | Execute hand-calculated retrieval cases and verify ranks, denominators, duplicates, failures/no-results, stable run identity, and report configuration. |
| 5 | SERV-01 | Follow the documented Python API workflow in fresh processes: ingest, restart, query exact terms and filters, resolve locations, repeat unchanged input, and select two isolated corpora. |
| 6 | RUN-04 | Local embedding integration: fixed inputs, batch/order, empty/oversize behavior, lifecycle, local-only operation, dimensions, memory and throughput. |
| 6 | RUN-05 | Local reranking integration: candidate ranking, ties, truncation, batch/order, limits, locality, resources, and parity with the fake. |
| 6 | RET-04 | Persist dense vectors, query known neighbors, update/remove, restart, corrupt/rebuild, and verify generation and corpus isolation. |
| 6 | RET-05 | Run lexical, dense, and metadata candidates through fusion; hand-check ranks, useful channel evidence, family deduplication, and cross-corpus rejection. |
| 6 | RET-06 | Feed fused results to the reranker and verify selected evidence count, tie/duplicate behavior, degraded backend handling, and preserved provenance. |
| 6 | RET-07 | Fit selected passages to exact context budgets and verify diversity, deterministic ordering, Unicode/tokeniser-version handling, and reversible provenance. |
| 6 | STOR-04 | Cold-process lexical/dense queries during updates; prove readers see one generation, prior versions survive failures, and unchanged indexes are reused. |
| 6 | EVAL-04 | Repeat the frozen held-out retrieval run; retain failures and category coverage and record an evidence-based proceed/reject gate. |
| 7 | RUN-06 | Local generator lifecycle and streaming integration for ordering, cancellation, limits, usage, errors, repeatability, and feasibility. |
| 7 | SERV-02 | Generate direct and synthesized claims from bounded evidence; verify exact claim-to-span links, conflict/insufficient-evidence handling, and stable output. |
| 7 | SERV-03 | Exercise supported, unsupported, partial, contradictory, absent, ambiguous, and verifier-failure cases; ensure no unsupported factual claim is emitted. |
| 7 | SERV-04 | Open text and layout citations after restart; fail safely for stale/corrupt locations and keep absolute paths out of logs. |
| 7 | EVAL-05 | Hand-score grounded-answer cases for citation, faithfulness, utility, abstention, failures, denominators, and private-data handling. |
| 8 | ING-06 | Classify native, scanned, sparse, mixed, image-heavy, and failed pages; verify adapter contract, reasons, versions, locality, and safe logs. |
| 8 | ING-07 | Run mixed pages through selective OCR and parse merge; verify invocation scope, spans/boxes, cache reuse/invalidation, failure visibility, restart, and native-path stability. |
| 8 | EVAL-06 | Compare labelled OCR cases and a representative scanned subset on downstream retrieval/citation, quality, and cost. |
| 9 | STOR-05 | Persist fact types, mentions, ambiguity, confidence, source navigation, and replacement across restart and transaction boundaries. |
| 9 | ING-08 | Extract verified examples for every initial field; include absent/conflicting values, units, malformed output, retry/idempotency, exact spans, and locality. |
| 9 | ING-09 | Normalize aliases, Unicode, dates, identifiers, quantities/units, ambiguity, and no-match while preserving mentions and versioned source links. |
| 9 | RET-08 | Query and aggregate stored facts; hand-check counts/groups/units, duplicates, uncertainty, missing/conflicting values, ordering, restart, and provenance. |
| 9 | RET-11 | Filter verified date kinds and assignees through missing/ambiguous/conflicting observations, ranges, corpus isolation, updates, and restart. |
| 9 | EVAL-07 | Hand-score extraction and aggregation, including partial/duplicate records, spans, ambiguity, failures, and a representative held-out report. |
| 10 | STOR-06 | Persist document/topic/corpus summaries with claim support and lineage; restart, replace versions, invalidate, and reject unsupported claims. |
| 10 | ING-10 | Generate a document summary from source evidence; verify claim support, conflict/invalid output handling, repeatability, and unchanged-input reuse. |
| 10 | ING-11 | Summarise stable topic membership; verify cross-document support, conflicts, deterministic grouping, and membership/input invalidation. |
| 10 | ING-12 | Build corpus summaries from complete/partial topics; traverse support, handle conflicts/insufficient evidence, restart, and invalidate. |
| 10 | ING-13 | Change each source/model/prompt/normaliser/grouping input and verify only affected derived records recompute through a published corpus version. |
| 10 | RET-09 | Retrieve summaries at each level; verify original-source preference, summary-only discovery, duplicate support, restart, and citation targets. |
| 10 | RET-10 | Run lookup/comparison/thematic/aggregation queries through multi-channel planning; verify budgets, unavailable channels, empty evidence, and recorded execution. |
| 10 | EVAL-08 | Compare frozen hybrid results to each channel ablation; retain failures and explicitly accept or reject the quality/faithfulness hypothesis. |
| 11 | SERV-05 | Create and reopen sessions with ordered turns/evidence; test update conflicts, corpus-version changes, deletion/expiry, and corpus binding. |
| 11 | SERV-06 | Rewrite follow-ups for pronouns, comparisons, topic changes, corrections, ambiguity, stale/unsupported evidence, and explicit-intent preservation. |
| 11 | RUN-07 | Exercise cache lifecycle/capability through fake, small, and production integrations; verify reuse, incompatibility, cancellation, release, stale handles, fallback. |
| 11 | SERV-07 | Run ten turns under budget; verify intent/citations, deterministic eviction, model/version invalidation, corpus-switch reset, and uncached equivalence. |
| 12 | RUN-08 | Simulate interactive and batch loads; verify priority, fairness, overload, cancellation, failures, release, ordering, and no deadlock/starvation. |
| 12 | RUN-09 | Simulate and measure residency transitions under feasible/infeasible memory budgets; exercise interrupted loads and recovery with no partial use. |
| 12 | RUN-10 | Compare batched and unbatched outputs under timeout, mixed compatibility, cancellation, failures, overload, concurrency, and memory caps. |
| 12 | PROF-03 | Validate fake sampler timing/units/failures, calibrate supported hardware adapters, and verify degraded states, overhead, and shutdown. |
| 12 | PROF-04 | Repeat controlled experiment matrices with raw config-linked results, controlled cold/warm states, retained failures, and quality/systems comparisons. |
| 12 | EVAL-09 | Evaluate ten-turn quality and measured latency/throughput/cache/context on fixtures and representative hardware; reject performance gains that regress answers. |
| 13 | STOR-07 | Restart to serve without rebuild while readers query during publication; keep old generation on failure and close resources cleanly. |
| 13 | STOR-08 | Drill isolated per-corpus backup/restore, checksums, interruption, compatibility, activation rollback, retention, restart, and unaffected corpora. |
| 13 | ING-14 | Crash at each ingest stage, restart, resume duplicate commands/leases, and verify retries, partial batches, changed inputs, cleanup, and work reuse. |
| 13 | SERV-08 | Exercise local chat end to end: streaming, citations/jumps, abstention, follow-up, cancellation, backend/stale-source failure, restart, accessibility, privacy. |
| 13 | SERV-09 | Create/select named corpora and add/update sources through the interface; cover interruption/resume, partial failure, concurrent queries, permissions, and isolation. |
| 13 | SERV-10 | Run acceptance with network denied across ingest/query/chat/evaluation/recovery; inspect dependencies and canary privacy across logs/telemetry/errors. |
| 13 | EVAL-10 | Run the frozen full acceptance suite and preserve artefacts, limitations, thresholds, and an explicit accept/reject decision. |

## File-removal scope

Removal happens only after the replacement acceptance test exists and covers
the issue's required behavior. The 14 files below are the first concrete
retirement/consolidation batch. Their 103 test functions are not all slated for
one-for-one replacement; several acceptance paths already exist, and the new
discovery journey will replace repeated result-construction checks.

- Discovery result construction and observability:
  `test_discovery_result_builder.py`, `test_discovery_result_counts.py`,
  `test_discovery_result_invariants.py`, and
  `test_discovery_result_lifecycle.py`. Status/count/ordering/privacy outcomes
  now run through `test_discovery_privacy_acceptance.py`, from explicit
  filesystem inputs through the public discovery result.
- Baseline helper seams: `test_baseline_execution.py`,
  `test_baseline_generation.py`, `test_baseline_membership.py`,
  `test_baseline_parse.py`, and `test_baseline_chunk.py`. Keep the externally
  observable concurrency, stage failure, publication, and replay cases in the
  serving ingest and baseline recovery acceptance suites. Retain a compact
  contract suite for summary/checkpoint value invariants that have no workflow
  equivalent.
- Baseline persistence mapping: retire
  `test_baseline_checkpoint_mapping.py` after persisted checkpoint round-trip,
  restart, corruption, and incompatible-resume checks. Retain
  `test_baseline_checkpoints.py` as the compact value-invariant suite because
  its closed checkpoint-state and identity rules have no full workflow
  equivalent.
- Repeated SQLite wrong-argument matrices:
  `test_sqlite_assertion_input_failures.py`,
  `test_sqlite_read_input_failures.py`, and
  `test_sqlite_write_input_failures.py`. Replace per-method type matrices with
  a small number of representative public-boundary cases, including one read
  failure and one write failure that prove transaction rollback. Keep the
  domain-specific integrity, conflict, and atomicity tests.

The 14-file batch alone will not reach the order-of-magnitude reduction. Use
it as the first part of a phase-by-phase consolidation. The broader intended
scope is:

- **Phase 1:** Consolidate tooling, settings, logging, startup, and graph tests
  in `tests/tooling/`, `tests/inference_lab/test_settings.py`, and
  `tests/tooling/logging/` into a small set of phase journeys. Retire
  per-renderer, per-setting, and per-hook files after their behavior is
  represented in those journeys.
- **Phase 2:** Consolidate `tests/inference_lab/provenance/`,
  `tests/inference_lab/profiling/`, `tests/inference_lab/backends/runtime/`,
  evaluation dataset tests, and parser-contract tests into a compact set of
  fixture cases. Retire per-field/type/serialization modules when their required
  contract is asserted in the combined boundary scenarios. Preserve the
  issue-mandated strict identity, parser-coordinate, profile-record, dataset,
  and fake-stream behavior.
- **Phase 3:** Consolidate the remaining provenance, runtime, retrieval, and
  storage contract modules, including the granular
  `tests/inference_lab/storage/test_sqlite_*.py` suites, into a compact set of
  restart, migration, transaction, two-corpus, lineage, memory, and retrieval
  journeys. Retire one-test-per-table or one-test-per-method files after their
  distinct persistence guarantee is covered in a repository journey. Keep
  atomicity, migration, isolation, and integrity failure paths.
- **Phase 4:** Consolidate the remaining discovery, identity, parser
  persistence, chunking, and audit modules under
  `tests/inference_lab/ingest/` and `tests/inference_lab/evaluation/` into a
  compact set of end-to-end fixture journeys. Retire redundant helper/result/mapper
  tests once the filesystem-to-persisted-corpus journeys exercise those
  outcomes.
- **Phase 5:** Consolidate baseline, lexical retrieval, filters, evaluation,
  and serving suites into a small set of operator-path cases. Retire direct helper
  tests where `serving/test_ingest.py`, the baseline recovery/publication
  acceptance suites, and retrieval acceptance exercise the same behavior.
- **Phases 6–13:** Do not expand the current suite ahead of implementation.
  When each phase lands, add only the acceptance journeys its criteria require
  and retire any temporary focused fixtures that the complete journey
  supersedes.

File retirement should follow issue coverage, not directory ownership. A
module may remain when it owns a behavior that cannot be observed reliably
through an end-to-end path, such as deterministic token-window boundaries or
strict provenance identity separation. Keep those focused tests as deliberate
exceptions to phase-level consolidation.

Keep the following acceptance anchors and build-plan-critical contracts:

- `tests/inference_lab/serving/test_ingest.py`
- `tests/inference_lab/ingest/test_baseline_publication_acceptance.py`
- `tests/inference_lab/ingest/test_baseline_recovery_acceptance.py`
- `tests/inference_lab/ingest/test_parse_*_acceptance.py`
- `tests/inference_lab/ingest/test_chunk_acceptance.py`
- `tests/inference_lab/storage/test_source_persistence_acceptance.py`
- the corpus publication, migration, integrity, and concurrency tests needed
  for STOR-01 through STOR-04
- Phase 2 contract tests whose strictness, immutability, or deterministic fake
  behavior is explicitly named in `build-details.md`

Other contract tests remain until an issue-by-issue trace shows that their
behavior is both outside the build criteria and covered by a stronger public
path. No deletion is based on a `test_*.py` name or test count alone.

## End-to-end suite layout

Organize acceptance tests by user or operator journey, with issue IDs in
module docstrings:

1. **Repository/tooling journey:** clean setup, load settings, configure logs,
   run validation, and refresh the local graph.
2. **Corpus foundation journey:** create two corpora, ingest and persist
   sources/parsed spans, reopen the repository, and resolve each corpus's
   membership and source locations.
3. **Discovery journey:** traverse explicit file/directory inputs, classify,
   hash, resolve identity, repeat unchanged input, change/move a file, and
   inspect failures without leaking private data.
4. **Baseline ingest and search journey:** use the documented API in fresh
   processes to ingest several documents, publish a generation, restart,
   search/filter with source locations, and verify an unchanged rerun performs
   no reconstruction.
5. **Recovery journey:** inject process loss at each durable stage, resume,
   and assert work reuse, active-version stability, conflict behavior, and
   second-corpus isolation.
6. **Later product journeys:** add hybrid retrieval, cited answers, OCR,
   structured extraction, summaries, sessions, resource scheduling, backup,
   chat, and local-only controls when their respective build issues land.

Each journey should have a small number of named scenarios. Use parameterized
fixtures for genuine variations such as failure category or corpus identity.
Avoid one assertion per constructor field when the field behavior is already
observable in the workflow.

## Completion sequence

1. Label the existing acceptance tests with the issue IDs and `Done when`
   clauses they demonstrate.
2. Identify uncovered issue criteria before removing any existing test.
3. Extend or add the corresponding acceptance journey and retain deterministic
   fakes for external model/runtime boundaries.
4. Remove only the candidate low-level cases whose required behavior is now
   proven by those paths; preserve contract tests required by the issue spec.
5. Review the diff for lost acceptance criteria and stale fixtures. Run the
   repository quality gate when implementation of this plan is requested.
6. Report reduced modules/functions/lines alongside the issue criteria that
   remain covered. Coverage percentage is a diagnostic, not the goal.
