"""SQLite acceptance for corpus-scoped ingest persistence."""

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParsePersistenceRequest,
    ParsePreflightError,
    ParsePreflightErrorCategory,
    TextNativeParser,
    persist_parse_request,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import MetadataRepository, open_metadata_repository

from .fixtures import fixture_settings, fixture_source  # noqa: TID252

_FIRST_CORPUS = CorpusId("c0000000-0000-4000-8000-000000000001")
_SECOND_CORPUS = CorpusId("c0000000-0000-4000-8000-000000000002")
_ABSENT_CORPUS = CorpusId("c0000000-0000-4000-8000-000000000003")

_SHARED_DOCUMENT_ID = DocumentId("d0000000-0000-4000-8000-000000000001")
_SHARED_VERSION_ID = DocumentVersionId("e0000000-0000-4000-8000-000000000001")
_FIRST_DOCUMENT_ID = DocumentId("d0000000-0000-4000-8000-000000000002")
_FIRST_VERSION_ID = DocumentVersionId("e0000000-0000-4000-8000-000000000002")
_SECOND_DOCUMENT_ID = DocumentId("d0000000-0000-4000-8000-000000000003")
_SECOND_VERSION_ID = DocumentVersionId("e0000000-0000-4000-8000-000000000003")

_SHARED_PARSE_ID = RecordId("f0000000-0000-4000-8000-000000000001")
_SHARED_PARSED_RECORD_ID = RecordId("f0000000-0000-4000-8000-000000000002")
_FIRST_PARSE_ID = RecordId("f0000000-0000-4000-8000-000000000003")
_FIRST_PARSED_RECORD_ID = RecordId("f0000000-0000-4000-8000-000000000004")
_SECOND_PARSE_ID = RecordId("f0000000-0000-4000-8000-000000000005")
_SECOND_PARSED_RECORD_ID = RecordId("f0000000-0000-4000-8000-000000000006")


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    version: DocumentVersion


def _case(
    source_name: str,
    *,
    document_id: DocumentId,
    document_version_id: DocumentVersionId,
    parse_id: RecordId,
    parsed_record_id: RecordId,
    corpus_id: CorpusId,
) -> _Case:
    source = replace(
        fixture_source(source_name),
        document_id=document_id,
        document_version_id=document_version_id,
    )
    outcome = TextNativeParser().parse(source, settings=fixture_settings())
    request = ParsePersistenceRequest(
        corpus_id=corpus_id,
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=parse_id,
        parsed_record_id=parsed_record_id,
    )
    version = DocumentVersion(
        document_version_id=document_version_id,
        document_id=document_id,
        content_hash=ContentHash(
            HashAlgorithm.SHA256,
            hashlib.sha256(source.content).hexdigest(),
        ),
    )
    return _Case(request=request, version=version)


@contextmanager
def _repository(
    tmp_path: Path,
    *,
    cases: tuple[_Case, ...],
    memberships: tuple[tuple[CorpusId, _Case], ...],
    corpora: tuple[CorpusId, ...],
) -> Iterator[MetadataRepository]:
    repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
        apply_migrations=True,
    )
    try:
        with repository.transaction() as transaction:
            for corpus_id in corpora:
                transaction.add_corpus(CorpusMetadata(corpus_id, "ING-03 isolation"))
            for case in cases:
                transaction.add_document(DocumentIdentity(case.version.document_id))
                transaction.add_document_version(case.version)
            for corpus_id, case in memberships:
                transaction.add_corpus_document_version_membership(
                    CorpusDocumentVersionMembership(
                        corpus_id,
                        case.version.document_id,
                        case.version.document_version_id,
                    )
                )
        yield repository
    finally:
        repository.close()


def test_non_member_corpus_is_indistinguishable_from_absence(
    tmp_path: Path,
) -> None:
    shared = _case(
        "basic.txt",
        document_id=_SHARED_DOCUMENT_ID,
        document_version_id=_SHARED_VERSION_ID,
        parse_id=_SHARED_PARSE_ID,
        parsed_record_id=_SHARED_PARSED_RECORD_ID,
        corpus_id=_FIRST_CORPUS,
    )
    with _repository(
        tmp_path,
        cases=(shared,),
        memberships=((_FIRST_CORPUS, shared),),
        corpora=(_FIRST_CORPUS, _SECOND_CORPUS),
    ) as repository:
        persist_parse_request(
            repository,
            shared.request,
            settings=fixture_settings(),
        )

        for corpus_id in (_SECOND_CORPUS, _ABSENT_CORPUS):
            request = replace(shared.request, corpus_id=corpus_id)
            with pytest.raises(ParsePreflightError) as error:
                persist_parse_request(
                    repository,
                    request,
                    settings=fixture_settings(),
                )

            assert error.value.category is (
                ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE
            )
            assert str(error.value) == "Parse persistence preflight failed."
            assert (
                repository.get_source_artefact(
                    corpus_id,
                    shared.version.document_version_id,
                )
                is None
            )
            assert (
                repository.list_parsed_representations(
                    corpus_id,
                    shared.version.document_version_id,
                )
                == ()
            )
            assert (
                repository.get_parse_outcome(
                    corpus_id,
                    shared.request.parse_id,
                )
                is None
            )


def test_explicit_shared_membership_reuses_one_immutable_bundle(
    tmp_path: Path,
) -> None:
    shared = _case(
        "basic.txt",
        document_id=_SHARED_DOCUMENT_ID,
        document_version_id=_SHARED_VERSION_ID,
        parse_id=_SHARED_PARSE_ID,
        parsed_record_id=_SHARED_PARSED_RECORD_ID,
        corpus_id=_FIRST_CORPUS,
    )
    with _repository(
        tmp_path,
        cases=(shared,),
        memberships=((_FIRST_CORPUS, shared), (_SECOND_CORPUS, shared)),
        corpora=(_FIRST_CORPUS, _SECOND_CORPUS),
    ) as repository:
        first = persist_parse_request(
            repository,
            shared.request,
            settings=fixture_settings(),
        )
        second = persist_parse_request(
            repository,
            replace(shared.request, corpus_id=_SECOND_CORPUS),
            settings=fixture_settings(),
        )

        assert second == first
        for corpus_id in (_FIRST_CORPUS, _SECOND_CORPUS):
            assert (
                repository.get_source_artefact(
                    corpus_id,
                    shared.version.document_version_id,
                )
                is not None
            )
            assert repository.list_parsed_representations(
                corpus_id,
                shared.version.document_version_id,
            ) == repository.list_parsed_representations(
                _FIRST_CORPUS,
                shared.version.document_version_id,
            )
            assert repository.get_parse_outcome(
                corpus_id,
                shared.request.parse_id,
            ) == repository.get_parse_outcome(
                _FIRST_CORPUS,
                shared.request.parse_id,
            )


def test_distinct_corpus_memberships_keep_parse_records_isolated(
    tmp_path: Path,
) -> None:
    first = _case(
        "basic.txt",
        document_id=_FIRST_DOCUMENT_ID,
        document_version_id=_FIRST_VERSION_ID,
        parse_id=_FIRST_PARSE_ID,
        parsed_record_id=_FIRST_PARSED_RECORD_ID,
        corpus_id=_FIRST_CORPUS,
    )
    second = _case(
        "unicode_newlines.bin",
        document_id=_SECOND_DOCUMENT_ID,
        document_version_id=_SECOND_VERSION_ID,
        parse_id=_SECOND_PARSE_ID,
        parsed_record_id=_SECOND_PARSED_RECORD_ID,
        corpus_id=_SECOND_CORPUS,
    )
    with _repository(
        tmp_path,
        cases=(first, second),
        memberships=((_FIRST_CORPUS, first), (_SECOND_CORPUS, second)),
        corpora=(_FIRST_CORPUS, _SECOND_CORPUS),
    ) as repository:
        first_result = persist_parse_request(
            repository,
            first.request,
            settings=fixture_settings(),
        )
        second_result = persist_parse_request(
            repository,
            second.request,
            settings=fixture_settings(),
        )

        assert first_result.parse_id != second_result.parse_id
        first_representations = repository.list_parsed_representations(
            _FIRST_CORPUS,
            first.version.document_version_id,
        )
        second_representations = repository.list_parsed_representations(
            _SECOND_CORPUS,
            second.version.document_version_id,
        )
        assert tuple(value.record_id for value in first_representations) == (
            first.request.parsed_record_id,
        )
        assert tuple(value.record_id for value in second_representations) == (
            second.request.parsed_record_id,
        )
        assert (
            repository.get_source_artefact(
                _FIRST_CORPUS,
                second.version.document_version_id,
            )
            is None
        )
        assert (
            repository.get_source_artefact(
                _SECOND_CORPUS,
                first.version.document_version_id,
            )
            is None
        )
        assert (
            repository.list_parsed_representations(
                _FIRST_CORPUS,
                second.version.document_version_id,
            )
            == ()
        )
        assert (
            repository.list_parsed_representations(
                _SECOND_CORPUS,
                first.version.document_version_id,
            )
            == ()
        )
        assert (
            repository.get_parse_outcome(
                _FIRST_CORPUS,
                second.request.parse_id,
            )
            is None
        )
        assert (
            repository.get_parse_outcome(
                _SECOND_CORPUS,
                first.request.parse_id,
            )
            is None
        )

        for request in (
            replace(first.request, corpus_id=_SECOND_CORPUS),
            replace(second.request, corpus_id=_FIRST_CORPUS),
        ):
            with pytest.raises(ParsePreflightError) as error:
                persist_parse_request(
                    repository,
                    request,
                    settings=fixture_settings(),
                )
            assert error.value.category is (
                ParsePreflightErrorCategory.SOURCE_NOT_AVAILABLE
            )
