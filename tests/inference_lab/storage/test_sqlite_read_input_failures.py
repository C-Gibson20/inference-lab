"""Negative input contracts for SQLite repository and transaction reads."""

from pathlib import Path

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentId,
    DocumentVersionId,
    SourceObservationId,
)
from inference_lab.storage import (
    SourceFileIdentityHint,
)
from tests.inference_lab.storage.input_failure_fixtures import (
    _CORPUS,
    _CORPUS_ID,
    _SOURCE_ID,
    _UPDATE_VERSION_ID,
    _VERSION_ID,
    _invalid,
    _ReadOperation,
    _RepositoryReadOperation,
)

from .fixtures import open_repository as _open_repository  # noqa: TID252

_READ_FAILURE_CASES: tuple[tuple[str, _ReadOperation, str], ...] = (
    (
        "get-corpus",
        lambda owner: owner.get_corpus(_invalid(CorpusId)),
        "corpus_id",
    ),
    (
        "get-update-corpus",
        lambda owner: owner.get_corpus_update(_invalid(CorpusId), _UPDATE_VERSION_ID),
        "corpus_id",
    ),
    (
        "get-active-version",
        lambda owner: owner.get_active_corpus_version(_invalid(CorpusId)),
        "corpus_id",
    ),
    (
        "get-committed-version-corpus",
        lambda owner: owner.get_corpus_version(_invalid(CorpusId), _UPDATE_VERSION_ID),
        "corpus_id",
    ),
    (
        "get-committed-version-id",
        lambda owner: owner.get_corpus_version(_CORPUS_ID, _invalid(CorpusVersionId)),
        "corpus_version",
    ),
    (
        "list-committed-versions",
        lambda owner: owner.list_corpus_versions(_invalid(CorpusId)),
        "corpus_id",
    ),
    (
        "get-update-version",
        lambda owner: owner.get_corpus_update(_CORPUS_ID, _invalid(CorpusVersionId)),
        "corpus_version",
    ),
    (
        "list-update-memberships-corpus",
        lambda owner: owner.list_corpus_update_document_memberships(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "list-update-memberships-version",
        lambda owner: owner.list_corpus_update_document_memberships(
            _CORPUS_ID, _invalid(CorpusVersionId)
        ),
        "corpus_version",
    ),
    (
        "list-update-components-corpus",
        lambda owner: owner.list_corpus_update_component_generations(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "list-update-components-version",
        lambda owner: owner.list_corpus_update_component_generations(
            _CORPUS_ID, _invalid(CorpusVersionId)
        ),
        "corpus_version",
    ),
    (
        "get-membership-corpus",
        lambda owner: owner.get_corpus_document_version_membership(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "get-membership-version",
        lambda owner: owner.get_corpus_document_version_membership(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "list-memberships",
        lambda owner: owner.list_corpus_document_version_memberships(
            _invalid(CorpusId)
        ),
        "corpus_id",
    ),
    (
        "get-document",
        lambda owner: owner.get_document(_invalid(DocumentId)),
        "document_id",
    ),
    (
        "get-family",
        lambda owner: owner.get_document_family(_invalid(DocumentFamilyId)),
        "family_id",
    ),
    (
        "get-version",
        lambda owner: owner.get_document_version(_invalid(DocumentVersionId)),
        "document_version_id",
    ),
    (
        "list-versions",
        lambda owner: owner.list_document_versions(_invalid(DocumentId)),
        "document_id",
    ),
    (
        "source-artefact-corpus",
        lambda owner: owner.get_source_artefact(_invalid(CorpusId), _VERSION_ID),
        "corpus_id",
    ),
    (
        "source-artefact-version",
        lambda owner: owner.get_source_artefact(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "find-hash-corpus",
        lambda owner: owner.find_document_versions_by_content_hash(
            _invalid(CorpusId), _invalid(ContentHash)
        ),
        "corpus_id",
    ),
    (
        "find-hash-value",
        lambda owner: owner.find_document_versions_by_content_hash(
            _CORPUS_ID, _invalid(ContentHash)
        ),
        "content_hash",
    ),
    (
        "find-identifiers-corpus",
        lambda owner: owner.find_document_versions_by_external_identifiers(
            _invalid(CorpusId), ()
        ),
        "corpus_id",
    ),
    (
        "source-current-corpus",
        lambda owner: owner.get_source_observation(_invalid(CorpusId), _SOURCE_ID),
        "corpus_id",
    ),
    (
        "source-current-id",
        lambda owner: owner.get_source_observation(
            _CORPUS_ID, _invalid(SourceObservationId)
        ),
        "source_observation_id",
    ),
    (
        "source-history-corpus",
        lambda owner: owner.list_source_observation_history(
            _invalid(CorpusId), _SOURCE_ID
        ),
        "corpus_id",
    ),
    (
        "source-history-id",
        lambda owner: owner.list_source_observation_history(
            _CORPUS_ID, _invalid(SourceObservationId)
        ),
        "source_observation_id",
    ),
    (
        "source-by-version-corpus",
        lambda owner: owner.find_source_observations_by_document_version(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "source-by-version-version",
        lambda owner: owner.find_source_observations_by_document_version(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "locator-current-corpus",
        lambda owner: owner.get_source_locator(_invalid(CorpusId), _SOURCE_ID),
        "corpus_id",
    ),
    (
        "locator-current-source",
        lambda owner: owner.get_source_locator(
            _CORPUS_ID,
            _invalid(SourceObservationId),
        ),
        "source_observation_id",
    ),
    (
        "locator-history-corpus",
        lambda owner: owner.list_source_locator_history(
            _invalid(CorpusId),
            _SOURCE_ID,
        ),
        "corpus_id",
    ),
    (
        "locator-history-source",
        lambda owner: owner.list_source_locator_history(
            _CORPUS_ID,
            _invalid(SourceObservationId),
        ),
        "source_observation_id",
    ),
    (
        "locator-path-corpus",
        lambda owner: owner.find_source_locator_by_path(
            _invalid(CorpusId),
            Path("/private/source.txt"),
        ),
        "corpus_id",
    ),
    (
        "locator-path-value",
        lambda owner: owner.find_source_locator_by_path(
            _CORPUS_ID,
            _invalid(Path),
        ),
        "path",
    ),
    (
        "locator-hint-corpus",
        lambda owner: owner.find_source_locators_by_identity_hint(
            _invalid(CorpusId),
            SourceFileIdentityHint(1, 2),
        ),
        "corpus_id",
    ),
    (
        "locator-hint-value",
        lambda owner: owner.find_source_locators_by_identity_hint(
            _CORPUS_ID,
            _invalid(SourceFileIdentityHint),
        ),
        "identity_hint",
    ),
    (
        "assertions-corpus",
        lambda owner: owner.list_external_identifier_assertions(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "assertions-version",
        lambda owner: owner.list_external_identifier_assertions(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "duplicates-corpus",
        lambda owner: owner.list_duplicate_relationships(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "duplicates-version",
        lambda owner: owner.list_duplicate_relationships(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "families-by-family-corpus",
        lambda owner: owner.list_family_memberships_by_family(
            _invalid(CorpusId), _invalid(DocumentFamilyId)
        ),
        "corpus_id",
    ),
    (
        "families-by-family-id",
        lambda owner: owner.list_family_memberships_by_family(
            _CORPUS_ID, _invalid(DocumentFamilyId)
        ),
        "family_id",
    ),
    (
        "families-by-document-corpus",
        lambda owner: owner.list_family_memberships_by_document(
            _invalid(CorpusId), _invalid(DocumentId)
        ),
        "corpus_id",
    ),
    (
        "families-by-document-id",
        lambda owner: owner.list_family_memberships_by_document(
            _CORPUS_ID, _invalid(DocumentId)
        ),
        "document_id",
    ),
    (
        "families-by-version-corpus",
        lambda owner: owner.list_family_memberships_by_document_version(
            _invalid(CorpusId), _invalid(DocumentVersionId)
        ),
        "corpus_id",
    ),
    (
        "families-by-version-id",
        lambda owner: owner.list_family_memberships_by_document_version(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
)

_REPOSITORY_READ_FAILURE_CASES: tuple[
    tuple[str, _RepositoryReadOperation, str], ...
] = (
    (
        "source-bytes-corpus",
        lambda repository: repository.read_source_bytes(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "source-bytes-version",
        lambda repository: repository.read_source_bytes(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
    (
        "source-stream-corpus",
        lambda repository: repository.stream_source_bytes(
            _invalid(CorpusId), _VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "source-stream-version",
        lambda repository: repository.stream_source_bytes(
            _CORPUS_ID, _invalid(DocumentVersionId)
        ),
        "document_version_id",
    ),
)


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _READ_FAILURE_CASES],
    ids=[name for name, _, _ in _READ_FAILURE_CASES],
)
def test_repository_reads_reject_wrong_identity_types(
    tmp_path: Path,
    operation: _ReadOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            operation(repository)
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _REPOSITORY_READ_FAILURE_CASES],
    ids=[name for name, _, _ in _REPOSITORY_READ_FAILURE_CASES],
)
def test_repository_source_reads_reject_wrong_identity_types(
    tmp_path: Path,
    operation: _RepositoryReadOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            operation(repository)
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _READ_FAILURE_CASES],
    ids=[name for name, _, _ in _READ_FAILURE_CASES],
)
def test_transaction_read_failures_roll_back_prior_writes(
    tmp_path: Path,
    operation: _ReadOperation,
    expected_field: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(TypeError, match=expected_field):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                operation(transaction)

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()
