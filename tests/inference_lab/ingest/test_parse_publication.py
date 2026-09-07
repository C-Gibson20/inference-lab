"""Tests for bundled source parse publication coordination."""

import hashlib
from dataclasses import dataclass
from typing import Iterable

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParseOutcome,
    ParsePersistenceRequest,
    ParsePreflightError,
    ParserFailure,
    TextNativeParser,
    map_parse_persistence_request,
    persist_parse_request,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    DocumentId,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import (
    ParsedRepresentation,
    RepositoryUnavailableError,
    SourceArtefact,
    StorageConcurrencyError,
    StorageConflictError,
    build_source_blob_storage_key,
)
from inference_lab.storage import (
    ParseOutcome as StorageParseOutcome,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252


@dataclass(frozen=True, slots=True)
class _Publication:
    corpus_id: CorpusId
    artefact: SourceArtefact
    chunks: tuple[bytes, ...]
    max_source_bytes: int
    parse_outcome: StorageParseOutcome | None
    parsed_representation: ParsedRepresentation | None


class _Repository:
    def __init__(
        self,
        *,
        membership: CorpusDocumentVersionMembership,
        document_version: DocumentVersion,
        failure: Exception | None = None,
    ) -> None:
        self.membership = membership
        self.document_version = document_version
        self.failure = failure
        self.calls: list[str] = []
        self.publications: list[_Publication] = []

    def get_corpus_document_version_membership(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> CorpusDocumentVersionMembership | None:
        self.calls.append("membership")
        assert corpus_id == self.membership.corpus_id
        assert document_version_id == self.membership.document_version_id
        return self.membership

    def get_document_version(
        self,
        document_version_id: DocumentVersionId,
    ) -> DocumentVersion | None:
        self.calls.append("document_version")
        assert document_version_id == self.document_version.document_version_id
        return self.document_version

    def get_source_artefact(
        self,
        corpus_id: CorpusId,
        document_version_id: DocumentVersionId,
    ) -> SourceArtefact | None:
        self.calls.append("source_artefact")
        del corpus_id, document_version_id
        return None

    def get_parsed_representation(
        self,
        corpus_id: CorpusId,
        record_id: RecordId,
    ) -> ParsedRepresentation | None:
        self.calls.append("predecessor")
        del corpus_id, record_id
        return None

    def persist_source_artefact(
        self,
        corpus_id: CorpusId,
        artefact: SourceArtefact,
        chunks: Iterable[bytes],
        *,
        max_source_bytes: int,
        parse_outcome: StorageParseOutcome | None = None,
        parsed_representation: ParsedRepresentation | None = None,
    ) -> None:
        self.calls.append("publication")
        self.publications.append(
            _Publication(
                corpus_id=corpus_id,
                artefact=artefact,
                chunks=tuple(chunks),
                max_source_bytes=max_source_bytes,
                parse_outcome=parse_outcome,
                parsed_representation=parsed_representation,
            )
        )
        if self.failure is not None:
            raise self.failure


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    repository: _Repository
    document_version: DocumentVersion


def _case(*, failed: bool = False, failure: Exception | None = None) -> _Case:
    source = fixture_source("malformed_utf8.bin" if failed else "basic.txt")
    outcome = (
        ParseOutcome(
            status="failed",
            document=None,
            failures=(
                ParserFailure(
                    category="decode_failed",
                    severity="fatal",
                    stage="decode",
                ),
            ),
        )
        if failed
        else TextNativeParser().parse(source, settings=fixture_settings())
    )
    request = ParsePersistenceRequest(
        corpus_id=CorpusId.generate(),
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=None if failed else RecordId.generate(),
    )
    content_hash = ContentHash(
        HashAlgorithm.SHA256,
        hashlib.sha256(source.content).hexdigest(),
    )
    membership = CorpusDocumentVersionMembership(
        corpus_id=request.corpus_id,
        document_id=source.document_id,
        document_version_id=source.document_version_id,
    )
    document_version = DocumentVersion(
        document_version_id=source.document_version_id,
        document_id=source.document_id,
        content_hash=content_hash,
    )
    return _Case(
        request=request,
        repository=_Repository(
            membership=membership,
            document_version=document_version,
            failure=failure,
        ),
        document_version=document_version,
    )


def test_publishes_one_exact_mapped_bundle_after_all_preflight_reads() -> None:
    case = _case()
    settings = fixture_settings(max_source_bytes=1234)
    expected = map_parse_persistence_request(case.request)

    result = persist_parse_request(
        case.repository,
        case.request,
        settings=settings,
    )

    assert result.parse_id == case.request.parse_id
    assert result.parsed_record_id == case.request.parsed_record_id
    assert result.status == case.request.outcome.status
    assert case.repository.calls == [
        "membership",
        "document_version",
        "source_artefact",
        "publication",
    ]
    assert len(case.repository.publications) == 1
    publication = case.repository.publications[0]
    assert publication.corpus_id == case.request.corpus_id
    assert publication.artefact == SourceArtefact(
        document_id=case.document_version.document_id,
        document_version_id=case.document_version.document_version_id,
        content_hash=case.document_version.content_hash,
        byte_count=len(case.request.parser_input.content),
        media_type=case.request.parser_input.media_type,
        storage_key=build_source_blob_storage_key(case.document_version.content_hash),
    )
    assert publication.chunks == (case.request.parser_input.content,)
    assert publication.max_source_bytes == settings.ingest.max_source_bytes
    assert publication.parse_outcome == expected.outcome
    assert publication.parsed_representation == expected.representation


def test_fatal_parse_publishes_source_and_outcome_without_representation() -> None:
    case = _case(failed=True)

    result = persist_parse_request(
        case.repository,
        case.request,
        settings=fixture_settings(),
    )

    assert result.status == "failed"
    assert result.parsed_record_id is None
    publication = case.repository.publications[0]
    assert publication.chunks == (case.request.parser_input.content,)
    assert publication.parse_outcome is not None
    assert publication.parse_outcome.status.value == "failed"
    assert publication.parsed_representation is None


@pytest.mark.parametrize(
    "failure",
    [
        StorageConflictError(),
        StorageConcurrencyError(),
        RepositoryUnavailableError(),
    ],
)
def test_storage_errors_propagate_unchanged(failure: Exception) -> None:
    case = _case(failure=failure)

    with pytest.raises(type(failure)) as error:
        persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )

    assert error.value is failure
    assert len(case.repository.publications) == 1


def test_preflight_failure_prevents_publication() -> None:
    case = _case()
    case.repository.membership = CorpusDocumentVersionMembership(
        corpus_id=case.request.corpus_id,
        document_id=DocumentId.generate(),
        document_version_id=case.document_version.document_version_id,
    )

    with pytest.raises(ParsePreflightError):
        persist_parse_request(
            case.repository,
            case.request,
            settings=fixture_settings(),
        )

    assert case.repository.publications == []


def test_result_representation_contains_no_source_details() -> None:
    case = _case()

    result = persist_parse_request(
        case.repository,
        case.request,
        settings=fixture_settings(),
    )

    rendered = repr(result)
    assert case.request.parser_input.content.decode() not in rendered
    assert case.document_version.content_hash.digest not in rendered
    assert "storage_key" not in rendered
    assert "parser_input" not in rendered


def test_dependencies_are_validated_before_repository_reads() -> None:
    case = _case()

    with pytest.raises(TypeError, match="request must be"):
        persist_parse_request(
            case.repository,
            object(),  # type: ignore[arg-type]
            settings=fixture_settings(),
        )
    with pytest.raises(TypeError, match="settings must be"):
        persist_parse_request(
            case.repository,
            case.request,
            settings=object(),  # type: ignore[arg-type]
        )

    assert case.repository.calls == []
