"""Shared values and operation types for SQLite input-failure tests."""

from collections.abc import Callable
from typing import cast

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    CorpusVersionId,
    DocumentId,
    DocumentVersionId,
    ExternalIdentifier,
    SourceObservationId,
)
from inference_lab.storage import (
    ComponentGenerationReference,
    ComponentGenerationState,
    ComponentKind,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from inference_lab.storage._sqlite_transaction import SQLiteMetadataTransaction

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
type _RepositoryReadOperation = Callable[[SQLiteMetadataRepository], object]
type _WriteOperation = Callable[[SQLiteMetadataTransaction], object]


def _invalid[Value](value_type: type[Value]) -> Value:
    """Return a deliberately wrong runtime value with a requested static type."""
    del value_type
    return cast("Value", object())
