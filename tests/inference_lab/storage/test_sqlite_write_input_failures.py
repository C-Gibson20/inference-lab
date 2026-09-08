"""Negative input contracts for SQLite transaction writes."""

from pathlib import Path

import pytest

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentFamilyId,
    DocumentFamilyMembership,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateRelationship,
    SourceObservationId,
    SourceObservationState,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationValidator,
    ParsedRepresentation,
    SourceArtefact,
    SourceLocatorRevision,
)
from tests.inference_lab.storage.input_failure_fixtures import (
    _COMPONENT,
    _COMPONENT_VALIDATOR,
    _CORPUS,
    _CORPUS_ID,
    _OTHER_UPDATE_VERSION_ID,
    _OTHER_VERSION_ID,
    _SOURCE_ID,
    _UPDATE_MEMBERSHIP,
    _UPDATE_VERSION_ID,
    _VERSION_ID,
    _invalid,
    _WriteOperation,
)

from .fixtures import open_repository as _open_repository  # noqa: TID252

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
        "commit-rollback-corpus",
        lambda transaction: transaction.commit_corpus_rollback(
            _invalid(CorpusId), _UPDATE_VERSION_ID
        ),
        "corpus_id",
    ),
    (
        "commit-rollback-version",
        lambda transaction: transaction.commit_corpus_rollback(
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
        "source-artefact-corpus",
        lambda transaction: transaction.add_source_artefact(
            _invalid(CorpusId), _invalid(SourceArtefact)
        ),
        "corpus_id",
    ),
    (
        "source-artefact",
        lambda transaction: transaction.add_source_artefact(
            _CORPUS_ID, _invalid(SourceArtefact)
        ),
        "artefact",
    ),
    (
        "parsed-representation-corpus",
        lambda transaction: transaction.add_parsed_representation(
            _invalid(CorpusId), _invalid(ParsedRepresentation)
        ),
        "corpus_id",
    ),
    (
        "parsed-representation",
        lambda transaction: transaction.add_parsed_representation(
            _CORPUS_ID, _invalid(ParsedRepresentation)
        ),
        "representation",
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
        "locator-add",
        lambda transaction: transaction.add_source_locator(
            _invalid(SourceLocatorRevision)
        ),
        "revision",
    ),
    (
        "locator-advance",
        lambda transaction: transaction.advance_source_locator(
            _invalid(SourceLocatorRevision)
        ),
        "revision",
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
