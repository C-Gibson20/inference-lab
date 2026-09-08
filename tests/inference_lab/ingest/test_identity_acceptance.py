"""Cold-process acceptance tests for persistent discovery identity."""

import os
from pathlib import Path

import pytest

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
    canonical_path_key,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
    persist_discovered_source_identities,
)
from inference_lab.provenance import (
    CorpusMetadata,
    DuplicateClass,
    ExternalIdentifier,
    ResolutionKind,
    ResolutionReason,
)
from tests.inference_lab.ingest.identity_acceptance_fixtures import (
    _CORPUS_ID,
    _OTHER_CORPUS_ID,
    _IncompleteScanFileSystem,
    _open_test_repository,
    _persist_supported_sources,
    _RecordingAllocator,
    _successful_results,
    _supported_persistence_units,
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


@pytest.mark.parametrize(
    ("use_hard_link", "shares_identity_hint"),
    ((False, False), (True, True)),
    ids=("byte-identical-copy", "hard-link"),
)
def test_live_identical_source_is_an_independent_duplicate_after_restart(
    tmp_path: Path,
    *,
    use_hard_link: bool,
    shares_identity_hint: bool,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    original_path = source_root / "a-original.txt"
    duplicate_path = source_root / "b-duplicate.txt"
    source_bytes = b"simultaneously live duplicate bytes\n"
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
        assert first.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    finally:
        initial.close()

    if use_hard_link:
        try:
            os.link(original_path, duplicate_path)
        except OSError as error:
            pytest.skip(f"hard links unavailable: {type(error).__name__}")
    else:
        duplicate_path.write_bytes(source_bytes)

    reopened = _open_test_repository(database_path)
    try:
        repeated_results = _persist_supported_sources(request, reopened)
        assert len(repeated_results) == 2
        original, duplicate = repeated_results
        assert isinstance(original, IdentityApplicationResult)
        assert isinstance(duplicate, IdentityApplicationResult)
        assert original.source.snapshot.locator.path == original_path.resolve()
        assert duplicate.source.snapshot.locator.path == duplicate_path.resolve()

        assert not original.created_document_version
        assert original.source.identity_outcome.kind is ResolutionKind.UNCHANGED
        assert (
            original.source.source_observation_id == first.source.source_observation_id
        )
        assert (
            original.source.identity_outcome.document
            == first.source.identity_outcome.document
        )
        assert (
            original.source.identity_outcome.version
            == first.source.identity_outcome.version
        )

        assert duplicate.created_document_version
        assert duplicate.source.identity_outcome.kind is ResolutionKind.DUPLICATE
        assert (
            duplicate.source.identity_outcome.reason
            is ResolutionReason.DUPLICATE_HASH_MATCH
        )
        assert (
            duplicate.source.source_observation_id != first.source.source_observation_id
        )

        first_outcome = first.source.identity_outcome
        duplicate_outcome = duplicate.source.identity_outcome
        assert first_outcome.document is not None
        assert first_outcome.version is not None
        assert duplicate_outcome.document is not None
        assert duplicate_outcome.version is not None
        assert duplicate_outcome.document != first_outcome.document
        assert duplicate_outcome.version != first_outcome.version
        assert (
            duplicate_outcome.version.content_hash == first_outcome.version.content_hash
        )
        assert duplicate_outcome.matched_version_ids == (
            first_outcome.version.document_version_id,
        )
        assert len(duplicate_outcome.duplicate_relationships) == 1
        relationship = duplicate_outcome.duplicate_relationships[0]
        assert relationship.duplicate_class is DuplicateClass.BYTE_IDENTICAL
        assert {
            relationship.first_version_id,
            relationship.second_version_id,
        } == {
            first_outcome.version.document_version_id,
            duplicate_outcome.version.document_version_id,
        }
        assert reopened.list_duplicate_relationships(
            _CORPUS_ID,
            first_outcome.version.document_version_id,
        ) == (relationship,)
        assert reopened.list_duplicate_relationships(
            _CORPUS_ID,
            duplicate_outcome.version.document_version_id,
        ) == (relationship,)
        assert len(reopened.list_corpus_document_version_memberships(_CORPUS_ID)) == 2

        original_locators = reopened.list_source_locator_history(
            _CORPUS_ID,
            first.source.source_observation_id,
        )
        duplicate_locators = reopened.list_source_locator_history(
            _CORPUS_ID,
            duplicate.source.source_observation_id,
        )
        assert len(original_locators) == len(duplicate_locators) == 1
        original_hint = original_locators[0].identity_hint
        duplicate_hint = duplicate_locators[0].identity_hint
        assert original_hint is not None
        assert duplicate_hint is not None
        assert (original_hint == duplicate_hint) is shares_identity_hint
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


def test_overlapping_roots_repeat_in_stable_order_as_an_exact_noop(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    nested_root = source_root / "nested"
    nested_root.mkdir(parents=True)
    paths = (
        source_root / "z-last.txt",
        nested_root / "a-first.txt",
        source_root / "m-middle.txt",
    )
    for ordinal, path in enumerate(paths):
        path.write_bytes(f"stable source {ordinal}\n".encode())
    expected_paths = tuple(
        sorted((path.resolve() for path in paths), key=canonical_path_key)
    )
    request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(
            DiscoveryInput(nested_root, DiscoveryInputKind.DIRECTORY),
            DiscoveryInput(source_root, DiscoveryInputKind.DIRECTORY),
        ),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "Acceptance corpus"))
    first_inventory, first_units = _supported_persistence_units(request)
    assert first_inventory.complete_root_ordinals == (0, 1)
    assert tuple(unit.snapshot.locator.path for unit in first_units) == expected_paths
    assert tuple(unit.ordinal for unit in first_units) == (0, 1, 2)
    first_allocator = _RecordingAllocator()
    try:
        first_results = persist_discovered_source_identities(
            first_units,
            request=request,
            inventory=first_inventory,
            repository=initial,
            allocator=first_allocator,
        )
        first_successes = tuple(
            result
            for result in first_results
            if isinstance(result, IdentityApplicationResult)
        )
        assert len(first_successes) == len(first_units)
        assert all(result.created_document_version for result in first_successes)
        assert all(
            result.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
            for result in first_successes
        )
        assert first_allocator.counts == (3, 3, 3)

        for unit, result in zip(first_units, first_successes, strict=True):
            assert result.source.snapshot == unit.snapshot
            locator = initial.get_source_locator(
                _CORPUS_ID,
                result.source.source_observation_id,
            )
            assert locator is not None
            opened = unit.snapshot.file_evidence.opened_after_read
            assert locator.fingerprint.content_hash == unit.snapshot.content_hash
            assert locator.fingerprint.byte_count == unit.snapshot.byte_count
            assert locator.fingerprint.modified_time_ns == opened.modified_time_ns
            assert opened.identity_hint is not None
            assert locator.identity_hint is not None
            assert locator.identity_hint.device_id == opened.identity_hint.device_id
            assert locator.identity_hint.file_id == opened.identity_hint.file_id

        durable_before = tuple(
            (
                initial.list_document_versions(
                    result.source.identity_outcome.document.document_id
                ),
                initial.list_source_observation_history(
                    _CORPUS_ID,
                    result.source.source_observation_id,
                ),
                initial.list_source_locator_history(
                    _CORPUS_ID,
                    result.source.source_observation_id,
                ),
                initial.list_duplicate_relationships(
                    _CORPUS_ID,
                    result.source.identity_outcome.version.document_version_id,
                ),
            )
            for result in first_successes
            if result.source.identity_outcome.document is not None
            and result.source.identity_outcome.version is not None
        )
        memberships_before = initial.list_corpus_document_version_memberships(
            _CORPUS_ID
        )
    finally:
        initial.close()

    reopened = _open_test_repository(database_path)
    second_inventory, second_units = _supported_persistence_units(request)
    assert second_inventory.candidates == first_inventory.candidates
    assert second_units == first_units
    second_allocator = _RecordingAllocator()
    try:
        repeated_results = persist_discovered_source_identities(
            second_units,
            request=request,
            inventory=second_inventory,
            repository=reopened,
            allocator=second_allocator,
        )
        repeated_successes = tuple(
            result
            for result in repeated_results
            if isinstance(result, IdentityApplicationResult)
        )
        assert len(repeated_successes) == len(first_successes)
        assert (
            tuple(result.source.snapshot.locator.path for result in repeated_successes)
            == expected_paths
        )
        assert all(not result.created_document_version for result in repeated_successes)
        assert all(
            result.source.identity_outcome.kind is ResolutionKind.UNCHANGED
            for result in repeated_successes
        )
        assert second_allocator.counts == (0, 0, 0)
        assert (
            tuple(
                (
                    reopened.list_document_versions(
                        result.source.identity_outcome.document.document_id
                    ),
                    reopened.list_source_observation_history(
                        _CORPUS_ID,
                        result.source.source_observation_id,
                    ),
                    reopened.list_source_locator_history(
                        _CORPUS_ID,
                        result.source.source_observation_id,
                    ),
                    reopened.list_duplicate_relationships(
                        _CORPUS_ID,
                        result.source.identity_outcome.version.document_version_id,
                    ),
                )
                for result in repeated_successes
                if result.source.identity_outcome.document is not None
                and result.source.identity_outcome.version is not None
            )
            == durable_before
        )
        assert (
            reopened.list_corpus_document_version_memberships(_CORPUS_ID)
            == memberships_before
        )
    finally:
        reopened.close()


def test_two_corpora_isolate_shared_candidates_observations_and_relationships(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    shared_path = source_root / "a-shared.txt"
    first_only_path = source_root / "b-first-only.txt"
    second_only_path = source_root / "c-second-only.txt"
    duplicate_path = source_root / "z-second-duplicate.txt"
    shared_bytes = b"bytes intentionally shared across corpora\n"
    shared_path.write_bytes(shared_bytes)
    first_only_path.write_bytes(b"first corpus only\n")
    second_only_path.write_bytes(b"second corpus only\n")
    shared_identifier = ExternalIdentifier("doi", "10.1000/shared")
    first_only_identifier = ExternalIdentifier("doi", "10.1000/first-only")
    second_only_identifier = ExternalIdentifier("doi", "10.1000/second-only")
    first_request = DiscoveryRequest(
        corpus_id=_CORPUS_ID,
        inputs=(
            DiscoveryInput(
                shared_path,
                DiscoveryInputKind.FILE,
                external_identifiers=(shared_identifier,),
            ),
            DiscoveryInput(
                first_only_path,
                DiscoveryInputKind.FILE,
                external_identifiers=(first_only_identifier,),
            ),
        ),
    )
    second_request = DiscoveryRequest(
        corpus_id=_OTHER_CORPUS_ID,
        inputs=(
            DiscoveryInput(
                shared_path,
                DiscoveryInputKind.FILE,
                external_identifiers=(shared_identifier,),
            ),
            DiscoveryInput(
                second_only_path,
                DiscoveryInputKind.FILE,
                external_identifiers=(second_only_identifier,),
            ),
        ),
    )
    database_path = tmp_path / "metadata.sqlite3"

    initial = _open_test_repository(database_path, apply_migrations=True)
    with initial.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "First corpus"))
        transaction.add_corpus(CorpusMetadata(_OTHER_CORPUS_ID, "Second corpus"))
    try:
        first_results = _successful_results(
            _persist_supported_sources(first_request, initial)
        )
        assert len(first_results) == 2
        assert all(
            result.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
            for result in first_results
        )
        first_by_path = {
            result.source.snapshot.locator.path: result for result in first_results
        }
        first_shared = first_by_path[shared_path.resolve()]
        first_only = first_by_path[first_only_path.resolve()]
    finally:
        initial.close()

    second = _open_test_repository(database_path)
    try:
        second_results = _successful_results(
            _persist_supported_sources(second_request, second)
        )
        assert len(second_results) == 2
        assert all(
            result.source.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
            for result in second_results
        )
        second_by_path = {
            result.source.snapshot.locator.path: result for result in second_results
        }
        second_shared = second_by_path[shared_path.resolve()]
        second_only = second_by_path[second_only_path.resolve()]

        first_shared_outcome = first_shared.source.identity_outcome
        first_only_outcome = first_only.source.identity_outcome
        second_shared_outcome = second_shared.source.identity_outcome
        second_only_outcome = second_only.source.identity_outcome
        for outcome in (
            first_shared_outcome,
            first_only_outcome,
            second_shared_outcome,
            second_only_outcome,
        ):
            assert outcome.document is not None
            assert outcome.version is not None
        assert first_shared_outcome.version is not None
        assert second_shared_outcome.version is not None

        assert (
            first_shared.source.snapshot.content_hash
            == second_shared.source.snapshot.content_hash
        )
        assert (
            first_shared.source.source_observation_id
            != second_shared.source.source_observation_id
        )
        assert first_shared_outcome.document != second_shared_outcome.document
        assert first_shared_outcome.version != second_shared_outcome.version

        assert second.find_document_versions_by_content_hash(
            _CORPUS_ID,
            first_shared.source.snapshot.content_hash,
        ) == (first_shared_outcome.version,)
        assert second.find_document_versions_by_content_hash(
            _OTHER_CORPUS_ID,
            second_shared.source.snapshot.content_hash,
        ) == (second_shared_outcome.version,)
        assert second.find_document_versions_by_external_identifiers(
            _CORPUS_ID,
            (shared_identifier,),
        ) == (first_shared_outcome.version,)
        assert second.find_document_versions_by_external_identifiers(
            _OTHER_CORPUS_ID,
            (shared_identifier,),
        ) == (second_shared_outcome.version,)
        assert (
            second.find_document_versions_by_content_hash(
                _CORPUS_ID,
                second_only.source.snapshot.content_hash,
            )
            == ()
        )
        assert (
            second.find_document_versions_by_external_identifiers(
                _CORPUS_ID,
                (second_only_identifier,),
            )
            == ()
        )
        assert (
            second.find_document_versions_by_content_hash(
                _OTHER_CORPUS_ID,
                first_only.source.snapshot.content_hash,
            )
            == ()
        )
        assert (
            second.find_document_versions_by_external_identifiers(
                _OTHER_CORPUS_ID,
                (first_only_identifier,),
            )
            == ()
        )

        first_locator = second.find_source_locator_by_path(
            _CORPUS_ID,
            shared_path.resolve(),
        )
        second_locator = second.find_source_locator_by_path(
            _OTHER_CORPUS_ID,
            shared_path.resolve(),
        )
        assert first_locator is not None
        assert second_locator is not None
        assert (
            first_locator.source_observation_id
            == first_shared.source.source_observation_id
        )
        assert (
            second_locator.source_observation_id
            == second_shared.source.source_observation_id
        )
        assert (
            second.get_source_observation(
                _CORPUS_ID,
                second_shared.source.source_observation_id,
            )
            is None
        )
        assert (
            second.get_source_observation(
                _OTHER_CORPUS_ID,
                first_shared.source.source_observation_id,
            )
            is None
        )
    finally:
        second.close()

    duplicate_path.write_bytes(shared_bytes)
    expanded_second_request = DiscoveryRequest(
        corpus_id=_OTHER_CORPUS_ID,
        inputs=second_request.inputs
        + (
            DiscoveryInput(
                duplicate_path,
                DiscoveryInputKind.FILE,
                external_identifiers=(shared_identifier,),
            ),
        ),
    )
    final = _open_test_repository(database_path)
    try:
        final_results = _successful_results(
            _persist_supported_sources(expanded_second_request, final)
        )
        assert len(final_results) == 3
        final_by_path = {
            result.source.snapshot.locator.path: result for result in final_results
        }
        assert (
            final_by_path[shared_path.resolve()].source.identity_outcome.kind
            is ResolutionKind.UNCHANGED
        )
        assert (
            final_by_path[second_only_path.resolve()].source.identity_outcome.kind
            is ResolutionKind.UNCHANGED
        )
        duplicate = final_by_path[duplicate_path.resolve()]
        duplicate_outcome = duplicate.source.identity_outcome
        assert duplicate.created_document_version
        assert duplicate_outcome.kind is ResolutionKind.DUPLICATE
        assert duplicate_outcome.reason is ResolutionReason.DUPLICATE_HASH_MATCH
        assert duplicate_outcome.document is not None
        assert duplicate_outcome.version is not None
        assert duplicate_outcome.matched_version_ids == (
            second_shared_outcome.version.document_version_id,
        )
        assert first_shared_outcome.version.document_version_id not in (
            duplicate_outcome.matched_version_ids
        )
        assert len(duplicate_outcome.duplicate_relationships) == 1
        relationship = duplicate_outcome.duplicate_relationships[0]
        assert final.list_duplicate_relationships(
            _OTHER_CORPUS_ID,
            duplicate_outcome.version.document_version_id,
        ) == (relationship,)
        assert final.list_duplicate_relationships(
            _OTHER_CORPUS_ID,
            second_shared_outcome.version.document_version_id,
        ) == (relationship,)
        assert (
            final.list_duplicate_relationships(
                _CORPUS_ID,
                duplicate_outcome.version.document_version_id,
            )
            == ()
        )
        assert (
            final.list_duplicate_relationships(
                _CORPUS_ID,
                first_shared_outcome.version.document_version_id,
            )
            == ()
        )

        first_memberships = final.list_corpus_document_version_memberships(_CORPUS_ID)
        second_memberships = final.list_corpus_document_version_memberships(
            _OTHER_CORPUS_ID
        )
        assert len(first_memberships) == 2
        assert len(second_memberships) == 3
        first_version_ids = {
            membership.document_version_id for membership in first_memberships
        }
        second_version_ids = {
            membership.document_version_id for membership in second_memberships
        }
        assert first_version_ids.isdisjoint(second_version_ids)
        assert duplicate_outcome.version.document_version_id in second_version_ids
        assert duplicate_outcome.version.document_version_id not in first_version_ids

        assert final.find_document_versions_by_external_identifiers(
            _CORPUS_ID,
            (shared_identifier,),
        ) == (first_shared_outcome.version,)
        assert frozenset(
            final.find_document_versions_by_external_identifiers(
                _OTHER_CORPUS_ID,
                (shared_identifier,),
            )
        ) == frozenset((second_shared_outcome.version, duplicate_outcome.version))
    finally:
        final.close()
