"""ING-02/PROV-02/STOR-01 cold-process source-continuity acceptance."""

from pathlib import Path

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    IdentityApplicationResult,
    IdentityPersistenceUnit,
    UuidDiscoveryIdentityIdAllocator,
    VerifiedSourceSnapshot,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
    persist_discovered_source_identities,
)
from inference_lab.provenance import (
    CorpusMetadata,
    ResolutionKind,
    ResolutionReason,
)
from tests.inference_lab.ingest.identity_acceptance_fixtures import (
    _CORPUS_ID,
    _IncompleteScanFileSystem,
    _open_test_repository,
    _persist_supported_sources,
)

from .fixtures import fixture_settings  # noqa: TID252


def test_rename_after_restart_preserves_identity_and_advances_locator(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    original_path = source_root / "original.txt"
    renamed_path = source_root / "renamed.txt"
    original_path.write_bytes(b"stable source bytes\n")
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    try:
        first_results = _persist_supported_sources(request, initial)
        assert len(first_results) == 1
        first = first_results[0]
        assert isinstance(first, IdentityApplicationResult)
        assert first.created_document_version
        assert first.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    finally:
        initial.close()

    original_path.rename(renamed_path)

    reopened = _open_test_repository(database_path)
    try:
        renamed_results = _persist_supported_sources(request, reopened)
        assert len(renamed_results) == 1
        renamed = renamed_results[0]
        assert isinstance(renamed, IdentityApplicationResult)
        assert not renamed.created_document_version
        assert renamed.source.identity_outcome.kind is ResolutionKind.UNCHANGED
        assert (
            renamed.source.source_observation_id == first.source.source_observation_id
        )
        assert (
            renamed.source.identity_outcome.document
            == first.source.identity_outcome.document
        )
        assert (
            renamed.source.identity_outcome.version
            == first.source.identity_outcome.version
        )

        outcome = first.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
        assert reopened.list_document_versions(outcome.document.document_id) == (
            outcome.version,
        )
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 1
        assert (
            len(
                reopened.list_source_observation_history(
                    _CORPUS_ID,
                    first.source.source_observation_id,
                )
            )
            == 1
        )

        locator_history = reopened.list_source_locator_history(
            _CORPUS_ID,
            first.source.source_observation_id,
        )
        assert tuple(locator.locator_revision for locator in locator_history) == (1, 2)
        assert tuple(locator.path for locator in locator_history) == (
            original_path.resolve(),
            renamed_path.resolve(),
        )
        assert locator_history[0].identity_hint == locator_history[1].identity_hint
        assert locator_history[0].fingerprint == locator_history[1].fingerprint
    finally:
        reopened.close()


def test_in_place_change_after_restart_creates_one_continuous_version(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / "changing.txt"
    source_path.write_bytes(b"first source bytes\n")
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    try:
        first_results = _persist_supported_sources(request, initial)
        assert len(first_results) == 1
        first = first_results[0]
        assert isinstance(first, IdentityApplicationResult)
        assert first.created_document_version
        assert first.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    finally:
        initial.close()

    source_path.write_bytes(b"second source byte\n")

    reopened = _open_test_repository(database_path)
    try:
        changed_results = _persist_supported_sources(request, reopened)
        assert len(changed_results) == 1
        changed = changed_results[0]
        assert isinstance(changed, IdentityApplicationResult)
        assert changed.created_document_version
        assert changed.source.identity_outcome.kind is ResolutionKind.NEW_VERSION
        assert (
            changed.source.identity_outcome.reason
            is ResolutionReason.SOURCE_CONTINUITY_MATCH
        )
        assert (
            changed.source.source_observation_id == first.source.source_observation_id
        )

        first_outcome = first.source.identity_outcome
        changed_outcome = changed.source.identity_outcome
        assert first_outcome.document is not None
        assert first_outcome.version is not None
        assert changed_outcome.document == first_outcome.document
        assert changed_outcome.version is not None
        assert changed_outcome.version != first_outcome.version
        assert (
            changed_outcome.version.content_hash != first_outcome.version.content_hash
        )

        versions = reopened.list_document_versions(first_outcome.document.document_id)
        assert frozenset(versions) == frozenset(
            (first_outcome.version, changed_outcome.version)
        )
        memberships = reopened.list_corpus_document_version_memberships(_CORPUS_ID)
        assert {item.document_version_id for item in memberships} == {
            first_outcome.version.document_version_id,
            changed_outcome.version.document_version_id,
        }

        observation_history = reopened.list_source_observation_history(
            _CORPUS_ID,
            first.source.source_observation_id,
        )
        assert tuple(
            observation.observation_revision for observation in observation_history
        ) == (1, 2)
        assert (
            tuple(observation.document_id for observation in observation_history)
            == (first_outcome.document.document_id,) * 2
        )
        assert tuple(
            observation.document_version_id for observation in observation_history
        ) == (
            first_outcome.version.document_version_id,
            changed_outcome.version.document_version_id,
        )

        locator_history = reopened.list_source_locator_history(
            _CORPUS_ID,
            first.source.source_observation_id,
        )
        assert tuple(locator.locator_revision for locator in locator_history) == (1, 2)
        assert (
            tuple(locator.path for locator in locator_history)
            == (source_path.resolve(),) * 2
        )
        assert locator_history[0].identity_hint == locator_history[1].identity_hint
        assert (
            locator_history[0].fingerprint.content_hash
            == first_outcome.version.content_hash
        )
        assert (
            locator_history[1].fingerprint.content_hash
            == changed_outcome.version.content_hash
        )
    finally:
        reopened.close()


def test_copy_delete_move_after_restart_uses_hash_continuity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    original_path = source_root / "original.txt"
    copied_path = source_root / "copied.txt"
    source_bytes = b"content-preserving copied source\n"
    original_path.write_bytes(source_bytes)
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    try:
        first_results = _persist_supported_sources(request, initial)
        assert len(first_results) == 1
        first = first_results[0]
        assert isinstance(first, IdentityApplicationResult)
        assert first.created_document_version
        assert first.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    finally:
        initial.close()

    copied_path.write_bytes(source_bytes)
    original_path.unlink()

    reopened = _open_test_repository(database_path)
    try:
        moved_results = _persist_supported_sources(request, reopened)
        assert len(moved_results) == 1
        moved = moved_results[0]
        assert isinstance(moved, IdentityApplicationResult)
        assert not moved.created_document_version
        assert moved.source.identity_outcome.kind is ResolutionKind.UNCHANGED
        assert moved.source.source_observation_id == first.source.source_observation_id
        assert (
            moved.source.identity_outcome.document
            == first.source.identity_outcome.document
        )
        assert (
            moved.source.identity_outcome.version
            == first.source.identity_outcome.version
        )

        outcome = first.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
        assert reopened.list_document_versions(outcome.document.document_id) == (
            outcome.version,
        )
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 1
        assert (
            len(
                reopened.list_source_observation_history(
                    _CORPUS_ID,
                    first.source.source_observation_id,
                )
            )
            == 1
        )
        assert (
            reopened.list_duplicate_relationships(
                _CORPUS_ID,
                outcome.version.document_version_id,
            )
            == ()
        )

        locator_history = reopened.list_source_locator_history(
            _CORPUS_ID,
            first.source.source_observation_id,
        )
        assert tuple(locator.locator_revision for locator in locator_history) == (1, 2)
        assert tuple(locator.path for locator in locator_history) == (
            original_path.resolve(),
            copied_path.resolve(),
        )
        assert locator_history[0].identity_hint is not None
        assert locator_history[1].identity_hint is not None
        assert locator_history[0].identity_hint != locator_history[1].identity_hint
        assert (
            locator_history[0].fingerprint.content_hash
            == locator_history[1].fingerprint.content_hash
            == outcome.version.content_hash
        )
    finally:
        reopened.close()


def test_removed_source_after_restart_leaves_durable_identity_unchanged(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    source_path = source_root / "removed.txt"
    source_path.write_bytes(b"source removed between runs\n")
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    try:
        first_results = _persist_supported_sources(request, initial)
        assert len(first_results) == 1
        first = first_results[0]
        assert isinstance(first, IdentityApplicationResult)
        outcome = first.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
        durable_before = (
            initial.list_document_versions(outcome.document.document_id),
            initial.list_corpus_document_version_memberships(_CORPUS_ID),
            initial.list_source_observation_history(
                _CORPUS_ID,
                first.source.source_observation_id,
            ),
            initial.list_source_locator_history(
                _CORPUS_ID,
                first.source.source_observation_id,
            ),
            initial.list_duplicate_relationships(
                _CORPUS_ID,
                outcome.version.document_version_id,
            ),
        )
    finally:
        initial.close()

    source_path.unlink()

    reopened = _open_test_repository(database_path)
    try:
        assert _persist_supported_sources(request, reopened) == ()
        assert (
            reopened.list_document_versions(outcome.document.document_id),
            reopened.list_corpus_document_version_memberships(_CORPUS_ID),
            reopened.list_source_observation_history(
                _CORPUS_ID,
                first.source.source_observation_id,
            ),
            reopened.list_source_locator_history(
                _CORPUS_ID,
                first.source.source_observation_id,
            ),
            reopened.list_duplicate_relationships(
                _CORPUS_ID,
                outcome.version.document_version_id,
            ),
        ) == durable_before
    finally:
        reopened.close()


def test_incomplete_scan_preserves_ambiguous_matching_source_after_restart(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    hidden_root = source_root / "a-inaccessible"
    hidden_root.mkdir(parents=True)
    original_path = hidden_root / "original.txt"
    visible_path = source_root / "z-visible-copy.txt"
    source_bytes = b"source hidden by an incomplete scan\n"
    original_path.write_bytes(source_bytes)
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    try:
        first_results = _persist_supported_sources(request, initial)
        assert len(first_results) == 1
        first = first_results[0]
        assert isinstance(first, IdentityApplicationResult)
        outcome = first.source.identity_outcome
        assert outcome.document is not None
        assert outcome.version is not None
    finally:
        initial.close()

    visible_path.write_bytes(source_bytes)
    filesystem = _IncompleteScanFileSystem(hidden_root)
    traversal = inventory_local_sources(
        request,
        excluded_paths=(),
        filesystem=filesystem,
    )
    assert traversal.complete_root_ordinals == ()
    assert tuple(failure.category for failure in traversal.failures) == (
        DiscoveryFailureCategory.TRAVERSAL_FAILED,
    )
    classification = classify_source_inventory(traversal)
    assert len(classification.hashable_candidates) == 1
    visible_source = classification.hashable_candidates[0]
    assert visible_source.candidate.locator.path == visible_path.resolve()
    snapshot = hash_source_candidate(
        visible_source,
        max_source_bytes=fixture_settings().ingest.max_source_bytes,
    )
    assert isinstance(snapshot, VerifiedSourceSnapshot)

    reopened = _open_test_repository(database_path)
    try:
        results = persist_discovered_source_identities(
            (IdentityPersistenceUnit(visible_source, snapshot),),
            request=request,
            inventory=traversal,
            repository=reopened,
            allocator=UuidDiscoveryIdentityIdAllocator(),
        )
        assert len(results) == 1
        failure = results[0]
        assert isinstance(failure, DiscoveryFailure)
        assert failure.category is DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS

        assert reopened.list_document_versions(outcome.document.document_id) == (
            outcome.version,
        )
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 1
        assert (
            len(
                reopened.list_source_observation_history(
                    _CORPUS_ID,
                    first.source.source_observation_id,
                )
            )
            == 1
        )
        assert (
            len(
                reopened.list_source_locator_history(
                    _CORPUS_ID,
                    first.source.source_observation_id,
                )
            )
            == 1
        )
        assert (
            reopened.list_duplicate_relationships(
                _CORPUS_ID,
                outcome.version.document_version_id,
            )
            == ()
        )
    finally:
        reopened.close()
