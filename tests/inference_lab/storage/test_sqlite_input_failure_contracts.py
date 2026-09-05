"""Negative input contracts for the SQLite repository and transactions."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentGenerationValidator,
    ComponentKind,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from inference_lab.storage._sqlite_transaction import SQLiteMetadataTransaction

from .fixtures import open_repository as _open_repository  # noqa: TID252

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS = CorpusMetadata(_CORPUS_ID, "Input failure corpus")
_VERSION_ID = DocumentVersionId("22222222-2222-4222-8222-222222222222")
_OTHER_VERSION_ID = DocumentVersionId("33333333-3333-4333-8333-333333333333")
_SOURCE_ID = SourceObservationId("44444444-4444-4444-8444-444444444444")
_OTHER_SOURCE_ID = SourceObservationId("55555555-5555-4555-8555-555555555555")
_IDENTIFIER = ExternalIdentifier("doi", "10.1000/input-failure")
_UPDATE_VERSION_ID = CorpusVersionId("66666666-6666-4666-8666-666666666666")
_OTHER_UPDATE_VERSION_ID = CorpusVersionId("77777777-7777-4777-8777-777777777777")
_DOCUMENT_ID = DocumentId("88888888-8888-4888-8888-888888888888")
_UPDATE_MEMBERSHIP = CorpusDocumentVersionMembership(
    _CORPUS_ID,
    _DOCUMENT_ID,
    _VERSION_ID,
)
_COMPONENT_KIND = ComponentKind("lexical")
_COMPONENT = ComponentGenerationReference(
    component_kind=_COMPONENT_KIND,
    generation_id="input-failure-generation",
    implementation="fake-index",
    format_version="1.0.0",
    contract_version="1.0.0",
    configuration_digest="a" * 64,
    source_manifest_digest="b" * 64,
    corpus_id=_CORPUS_ID,
    corpus_version=_UPDATE_VERSION_ID,
    state=ComponentGenerationState.PREPARING,
)


class _AcceptingComponentValidator:
    def validate_component_generation(
        self,
        generation: ComponentGenerationReference,
        document_memberships: tuple[CorpusDocumentVersionMembership, ...],
    ) -> bool:
        del generation, document_memberships
        return True


_COMPONENT_VALIDATOR = _AcceptingComponentValidator()

type _ReadOwner = SQLiteMetadataRepository | SQLiteMetadataTransaction
type _ReadOperation = Callable[[_ReadOwner], object]
type _WriteOperation = Callable[[SQLiteMetadataTransaction], object]


def _invalid[Value](value_type: type[Value]) -> Value:
    """Return a deliberately wrong runtime value with a requested static type."""
    del value_type
    return cast("Value", object())


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


_WRITE_FAILURE_CASES: tuple[tuple[str, _WriteOperation, str], ...] = (
    (
        "corpus",
        lambda transaction: transaction.add_corpus(_invalid(CorpusMetadata)),
        "corpus",
    ),
    (
        "corpus-name-id",
        lambda transaction: transaction.set_corpus_display_name(
            _invalid(CorpusId), "Renamed"
        ),
        "corpus_id",
    ),
    (
        "begin-update-corpus",
        lambda transaction: transaction.begin_corpus_update(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            expected_corpus_version=None,
        ),
        "corpus_id",
    ),
    (
        "begin-update-version",
        lambda transaction: transaction.begin_corpus_update(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            expected_corpus_version=None,
        ),
        "corpus_version",
    ),
    (
        "begin-update-expected",
        lambda transaction: transaction.begin_corpus_update(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            expected_corpus_version=_invalid(CorpusVersionId),
        ),
        "expected_corpus_version",
    ),
    (
        "begin-update-components",
        lambda transaction: transaction.begin_corpus_update(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            expected_corpus_version=None,
            required_component_kinds=_invalid(tuple),
        ),
        "required_component_kinds",
    ),
    (
        "begin-update-restore",
        lambda transaction: transaction.begin_corpus_update(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            expected_corpus_version=_OTHER_UPDATE_VERSION_ID,
            restores_corpus_version=_invalid(CorpusVersionId),
        ),
        "restores_corpus_version",
    ),
    (
        "abort-update-corpus",
        lambda transaction: transaction.abort_corpus_update(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "abort-update-version",
        lambda transaction: transaction.abort_corpus_update(
            _CORPUS_ID, _invalid(CorpusVersionId)
        ),
        "corpus_version",
    ),
    (
        "register-component-corpus",
        lambda transaction: transaction.register_corpus_update_component_generation(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            _COMPONENT,
            validator=_COMPONENT_VALIDATOR,
        ),
        "corpus_id",
    ),
    (
        "register-component-version",
        lambda transaction: transaction.register_corpus_update_component_generation(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            _COMPONENT,
            validator=_COMPONENT_VALIDATOR,
        ),
        "corpus_version",
    ),
    (
        "register-component-generation",
        lambda transaction: transaction.register_corpus_update_component_generation(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _invalid(ComponentGenerationReference),
            validator=_COMPONENT_VALIDATOR,
        ),
        "generation",
    ),
    (
        "register-component-validator",
        lambda transaction: transaction.register_corpus_update_component_generation(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _COMPONENT,
            validator=_invalid(ComponentGenerationValidator),
        ),
        "validator",
    ),
    (
        "replace-component-corpus",
        lambda transaction: transaction.replace_corpus_update_component_generation(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            _COMPONENT,
            _COMPONENT,
            validator=_COMPONENT_VALIDATOR,
        ),
        "corpus_id",
    ),
    (
        "replace-component-version",
        lambda transaction: transaction.replace_corpus_update_component_generation(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            _COMPONENT,
            _COMPONENT,
            validator=_COMPONENT_VALIDATOR,
        ),
        "corpus_version",
    ),
    (
        "replace-component-expected",
        lambda transaction: transaction.replace_corpus_update_component_generation(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _invalid(ComponentGenerationReference),
            _COMPONENT,
            validator=_COMPONENT_VALIDATOR,
        ),
        "expected_generation",
    ),
    (
        "replace-component-replacement",
        lambda transaction: transaction.replace_corpus_update_component_generation(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _COMPONENT,
            _invalid(ComponentGenerationReference),
            validator=_COMPONENT_VALIDATOR,
        ),
        "replacement_generation",
    ),
    (
        "replace-component-validator",
        lambda transaction: transaction.replace_corpus_update_component_generation(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _COMPONENT,
            _COMPONENT,
            validator=_invalid(ComponentGenerationValidator),
        ),
        "validator",
    ),
    (
        "seal-update-corpus",
        lambda transaction: transaction.seal_corpus_update(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "seal-update-version",
        lambda transaction: transaction.seal_corpus_update(
            _CORPUS_ID, _invalid(CorpusVersionId)
        ),
        "corpus_version",
    ),
    (
        "commit-update-corpus",
        lambda transaction: transaction.commit_corpus_update(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "commit-update-version",
        lambda transaction: transaction.commit_corpus_update(
            _CORPUS_ID, _invalid(CorpusVersionId)
        ),
        "corpus_version",
    ),
    (
        "add-update-membership-corpus",
        lambda transaction: transaction.add_corpus_update_document_membership(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_id",
    ),
    (
        "add-update-membership-version",
        lambda transaction: transaction.add_corpus_update_document_membership(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_version",
    ),
    (
        "add-update-membership-value",
        lambda transaction: transaction.add_corpus_update_document_membership(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _invalid(CorpusDocumentVersionMembership),
        ),
        "membership",
    ),
    (
        "replace-update-membership-corpus",
        lambda transaction: transaction.replace_corpus_update_document_membership(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            _UPDATE_MEMBERSHIP,
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_id",
    ),
    (
        "replace-update-membership-version",
        lambda transaction: transaction.replace_corpus_update_document_membership(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            _UPDATE_MEMBERSHIP,
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_version",
    ),
    (
        "replace-update-membership-expected",
        lambda transaction: transaction.replace_corpus_update_document_membership(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _invalid(CorpusDocumentVersionMembership),
            _UPDATE_MEMBERSHIP,
        ),
        "expected_membership",
    ),
    (
        "replace-update-membership-replacement",
        lambda transaction: transaction.replace_corpus_update_document_membership(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _UPDATE_MEMBERSHIP,
            _invalid(CorpusDocumentVersionMembership),
        ),
        "replacement_membership",
    ),
    (
        "remove-update-membership-corpus",
        lambda transaction: transaction.remove_corpus_update_document_membership(
            _invalid(CorpusId),
            _UPDATE_VERSION_ID,
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_id",
    ),
    (
        "remove-update-membership-version",
        lambda transaction: transaction.remove_corpus_update_document_membership(
            _CORPUS_ID,
            _invalid(CorpusVersionId),
            _UPDATE_MEMBERSHIP,
        ),
        "corpus_version",
    ),
    (
        "remove-update-membership-value",
        lambda transaction: transaction.remove_corpus_update_document_membership(
            _CORPUS_ID,
            _UPDATE_VERSION_ID,
            _invalid(CorpusDocumentVersionMembership),
        ),
        "membership",
    ),
    (
        "corpus-membership",
        lambda transaction: transaction.add_corpus_document_version_membership(
            _invalid(CorpusDocumentVersionMembership)
        ),
        "membership",
    ),
    (
        "document",
        lambda transaction: transaction.add_document(_invalid(DocumentIdentity)),
        "document",
    ),
    (
        "family",
        lambda transaction: transaction.add_document_family(_invalid(DocumentFamilyId)),
        "family_id",
    ),
    (
        "version",
        lambda transaction: transaction.add_document_version(_invalid(DocumentVersion)),
        "document_version",
    ),
    (
        "source",
        lambda transaction: transaction.add_source_observation(
            _invalid(SourceObservationState)
        ),
        "state",
    ),
    (
        "advance-corpus",
        lambda transaction: transaction.advance_source_observation(
            _invalid(CorpusId),
            _SOURCE_ID,
            expected_version_id=_VERSION_ID,
            new_version_id=_OTHER_VERSION_ID,
        ),
        "corpus_id",
    ),
    (
        "advance-source",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _invalid(SourceObservationId),
            expected_version_id=_VERSION_ID,
            new_version_id=_OTHER_VERSION_ID,
        ),
        "source_observation_id",
    ),
    (
        "advance-expected",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _SOURCE_ID,
            expected_version_id=_invalid(DocumentVersionId),
            new_version_id=_OTHER_VERSION_ID,
        ),
        "expected_version_id",
    ),
    (
        "advance-new",
        lambda transaction: transaction.advance_source_observation(
            _CORPUS_ID,
            _SOURCE_ID,
            expected_version_id=_VERSION_ID,
            new_version_id=_invalid(DocumentVersionId),
        ),
        "new_version_id",
    ),
    (
        "assertion-corpus",
        lambda transaction: transaction.add_external_identifier_assertions(
            _invalid(CorpusId), ()
        ),
        "corpus_id",
    ),
    (
        "duplicate",
        lambda transaction: transaction.add_duplicate_relationship(
            _invalid(DuplicateRelationship)
        ),
        "relationship",
    ),
    (
        "family-membership",
        lambda transaction: transaction.add_family_membership(
            _invalid(DocumentFamilyMembership)
        ),
        "membership",
    ),
)


@pytest.mark.parametrize(
    ("operation", "expected_field"),
    [(operation, field) for _, operation, field in _WRITE_FAILURE_CASES],
    ids=[name for name, _, _ in _WRITE_FAILURE_CASES],
)
def test_transaction_writes_reject_wrong_types_and_roll_back(
    tmp_path: Path,
    operation: _WriteOperation,
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


def _assertion(
    source_id: SourceObservationId,
    version_id: DocumentVersionId,
    namespace: str,
) -> ExternalIdentifierAssertion:
    """Build a deterministic assertion for malformed-batch tests."""
    return ExternalIdentifierAssertion(
        source_id,
        version_id,
        ExternalIdentifier(namespace, f"value-{namespace}"),
    )


@pytest.mark.parametrize(
    ("assertions", "expected_error", "expected_message"),
    [
        ([], TypeError, "tuple"),
        ((object(),), TypeError, "ExternalIdentifierAssertion"),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "duplicates",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "pmid"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "canonical order",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_OTHER_SOURCE_ID, _VERSION_ID, "pmid"),
            ),
            ValueError,
            "one source-version association",
        ),
    ],
    ids=["list", "wrong-member", "duplicate", "order", "mixed-association"],
)
def test_malformed_assertion_batches_fail_before_persistence(
    tmp_path: Path,
    assertions: object,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(expected_error, match=expected_message):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                transaction.add_external_identifier_assertions(
                    _CORPUS_ID,
                    cast("tuple[ExternalIdentifierAssertion, ...]", assertions),
                )

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()
