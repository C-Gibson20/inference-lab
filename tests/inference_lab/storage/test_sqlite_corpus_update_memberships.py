"""Tests for SQLite corpus-update document memberships."""

import sqlite3
from pathlib import Path

import pytest

from inference_lab.provenance import CorpusDocumentVersionMembership
from inference_lab.storage import StorageConflictError
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.storage.corpus_update_fixtures import (
    _CORPUS_A,
    _CORPUS_B,
    _FIRST_DOCUMENT_ID,
    _FIRST_DOCUMENT_VERSION_ID,
    _FIRST_MEMBERSHIP,
    _MANIFEST_DIGEST,
    _OTHER_VERSION,
    _PARENT_VERSION,
    _PROPOSED_VERSION,
    _REPLACEMENT_MEMBERSHIP,
    _SECOND_DOCUMENT_VERSION_ID,
    _SECOND_MEMBERSHIP,
    _add_catalogue,
    _create_committed_parent,
    _open,
)


def test_add_membership_is_ordered_idempotent_and_not_an_implicit_replace(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _REPLACEMENT_MEMBERSHIP,
                )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()


def test_add_requires_explicit_target_corpus_catalogue_membership(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_B.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        for membership in (
            _FIRST_MEMBERSHIP,
            CorpusDocumentVersionMembership(
                _CORPUS_B.corpus_id,
                _FIRST_DOCUMENT_ID,
                _FIRST_DOCUMENT_VERSION_ID,
            ),
        ):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.add_corpus_update_document_membership(
                        _CORPUS_B.corpus_id,
                        _PROPOSED_VERSION,
                        membership,
                    )

        assert (
            repository.list_corpus_update_document_memberships(
                _CORPUS_B.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


def test_replace_consumes_explicit_logical_document_agreement_and_replays(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
                _REPLACEMENT_MEMBERSHIP,
            )
            transaction.replace_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
                _REPLACEMENT_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_REPLACEMENT_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        assert reopened.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_REPLACEMENT_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("expected_membership", "replacement_membership"),
    [
        (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP),
        (_REPLACEMENT_MEMBERSHIP, _REPLACEMENT_MEMBERSHIP),
    ],
    ids=("different-logical-document", "stale-expected-membership"),
)
def test_replace_rejects_identity_disagreement_without_mutation(
    tmp_path: Path,
    expected_membership: CorpusDocumentVersionMembership,
    replacement_membership: CorpusDocumentVersionMembership,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.replace_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    expected_membership,
                    replacement_membership,
                )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP, _SECOND_MEMBERSHIP)
    finally:
        repository.close()


def test_remove_is_idempotent_and_preserves_global_catalogue_membership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _create_committed_parent(database_path)
    repository = _open(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=_PARENT_VERSION,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )

        assert repository.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP,)
        assert (
            repository.get_corpus_document_version_membership(
                _CORPUS_A.corpus_id,
                _SECOND_DOCUMENT_VERSION_ID,
            )
            == _SECOND_MEMBERSHIP
        )
    finally:
        repository.close()


def test_staging_mutation_rolls_back_without_losing_the_update(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        with pytest.raises(RuntimeError, match="private-primary"):
            with repository.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _FIRST_MEMBERSHIP,
                )
                raise RuntimeError("private-primary")

        assert (
            repository.get_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            is not None
        )
        assert (
            repository.list_corpus_update_document_memberships(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
            )
            == ()
        )
    finally:
        repository.close()


def test_sealed_membership_allows_exact_replay_but_rejects_changes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.begin_corpus_update(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            expected_corpus_version=None,
        )
        transaction.add_corpus_update_document_membership(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
            _FIRST_MEMBERSHIP,
        )
    repository.close()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """UPDATE corpus_updates
            SET state = 'ready', manifest_digest = ?
            WHERE corpus_id = ? AND corpus_version = ?""",
            (
                _MANIFEST_DIGEST,
                _CORPUS_A.corpus_id.as_str(),
                _PROPOSED_VERSION.as_str(),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    try:
        with reopened.transaction() as transaction:
            transaction.add_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _FIRST_MEMBERSHIP,
            )
            transaction.remove_corpus_update_document_membership(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                _SECOND_MEMBERSHIP,
            )

        with pytest.raises(StorageConflictError):
            with reopened.transaction() as transaction:
                transaction.add_corpus_update_document_membership(
                    _CORPUS_A.corpus_id,
                    _PROPOSED_VERSION,
                    _SECOND_MEMBERSHIP,
                )

        assert reopened.list_corpus_update_document_memberships(
            _CORPUS_A.corpus_id,
            _PROPOSED_VERSION,
        ) == (_FIRST_MEMBERSHIP,)
    finally:
        reopened.close()


def test_inspection_rejects_unknown_or_cross_corpus_update_identity(
    tmp_path: Path,
) -> None:
    repository = _open(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.begin_corpus_update(
                _CORPUS_A.corpus_id,
                _PROPOSED_VERSION,
                expected_corpus_version=None,
            )

        for corpus_id, corpus_version in (
            (_CORPUS_B.corpus_id, _PROPOSED_VERSION),
            (_CORPUS_A.corpus_id, _OTHER_VERSION),
        ):
            with pytest.raises(StorageConflictError):
                repository.list_corpus_update_document_memberships(
                    corpus_id,
                    corpus_version,
                )
    finally:
        repository.close()
