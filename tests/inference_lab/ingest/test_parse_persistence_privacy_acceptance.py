"""Privacy acceptance for ingest persistence logs and public boundaries."""

import hashlib
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParsePersistenceRequest,
    ParsePreflightError,
    TextNativeParser,
    persist_parse_request,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    HashAlgorithm,
    RecordId,
)
from inference_lab.storage import (
    MetadataRepository,
    StorageConflictError,
    open_metadata_repository,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252

_CORPUS_ID = CorpusId("e0000000-0000-4000-8000-000000000001")
_PRIVATE_SOURCE_TEXT = "private-ingest-source-text-canary"
_PRIVATE_SOURCE_BYTES = (_PRIVATE_SOURCE_TEXT + "\n").encode("utf-8")
_PRIVATE_SOURCE_DIRECTORY = "private-ingest-source-directory"
_PRIVATE_DATABASE_DIRECTORY = "private-ingest-database-directory"
_PRIVATE_DATABASE_PAYLOAD = "private-ingest-database-payload"


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    catalogue_source: bytes
    database_path: Path
    source_dir: Path


def _case(
    tmp_path: Path,
    *,
    request_source: bytes = _PRIVATE_SOURCE_BYTES,
    catalogue_source: bytes = _PRIVATE_SOURCE_BYTES,
) -> _Case:
    parser_input = replace(
        fixture_source("basic.txt"),
        content=request_source,
    )
    outcome = TextNativeParser().parse(parser_input, settings=fixture_settings())
    request = ParsePersistenceRequest(
        corpus_id=_CORPUS_ID,
        parser_input=parser_input,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=RecordId.generate(),
        parsed_record_id=RecordId.generate(),
    )
    return _Case(
        request=request,
        catalogue_source=catalogue_source,
        database_path=tmp_path / _PRIVATE_DATABASE_DIRECTORY / "metadata.sqlite3",
        source_dir=tmp_path / _PRIVATE_SOURCE_DIRECTORY,
    )


def _open_repository(case: _Case) -> MetadataRepository:
    repository = open_metadata_repository(
        case.database_path,
        source_dir=case.source_dir,
        apply_migrations=True,
    )
    source = case.request.parser_input
    version = DocumentVersion(
        document_version_id=source.document_version_id,
        document_id=source.document_id,
        content_hash=ContentHash(
            HashAlgorithm.SHA256,
            hashlib.sha256(case.catalogue_source).hexdigest(),
        ),
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "ING-03 privacy"))
        transaction.add_document(DocumentIdentity(source.document_id))
        transaction.add_document_version(version)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                source.document_id,
                source.document_version_id,
            )
        )
    return repository


def _install_database_failure(case: _Case) -> None:
    connection = sqlite3.connect(case.database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_private_ingest_payload
            BEFORE INSERT ON source_artefacts
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""
        )
        connection.commit()
    finally:
        connection.close()


def _private_markers(case: _Case) -> tuple[str, ...]:
    return (
        _PRIVATE_SOURCE_TEXT,
        str(case.source_dir),
        case.source_dir.name,
        str(case.database_path),
        case.database_path.parent.name,
        _PRIVATE_DATABASE_PAYLOAD,
    )


def _assert_private_values_absent(rendered: str, case: _Case) -> None:
    assert all(marker not in rendered for marker in _private_markers(case))


def test_successful_persistence_hides_source_and_paths_from_public_boundaries(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    repository = _open_repository(case)
    try:
        with capture_logs() as logs:
            result = persist_parse_request(
                repository,
                case.request,
                settings=fixture_settings(),
            )

        _assert_private_values_absent(repr(logs), case)
        _assert_private_values_absent(repr(case.request), case)
        _assert_private_values_absent(repr(result), case)
    finally:
        repository.close()


def test_preflight_error_hides_changed_source_and_private_paths(
    tmp_path: Path,
) -> None:
    case = _case(
        tmp_path,
        request_source=_PRIVATE_SOURCE_BYTES,
        catalogue_source=b"catalogue-source-without-private-marker\n",
    )
    repository = _open_repository(case)
    try:
        with capture_logs() as logs:
            with pytest.raises(ParsePreflightError) as error:
                persist_parse_request(
                    repository,
                    case.request,
                    settings=fixture_settings(),
                )

        _assert_private_values_absent(str(error.value), case)
        _assert_private_values_absent(repr(error.value), case)
        _assert_private_values_absent(repr(logs), case)
        _assert_private_values_absent(repr(case.request), case)
        source = case.request.parser_input
        assert (
            repository.get_source_artefact(
                _CORPUS_ID,
                source.document_version_id,
            )
            is None
        )
        assert (
            repository.list_parsed_representations(
                _CORPUS_ID,
                source.document_version_id,
            )
            == ()
        )
        assert repository.get_parse_outcome(_CORPUS_ID, case.request.parse_id) is None
    finally:
        repository.close()


def test_storage_error_hides_database_payload_and_private_paths(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    repository = _open_repository(case)
    _install_database_failure(case)
    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as error:
                persist_parse_request(
                    repository,
                    case.request,
                    settings=fixture_settings(),
                )

        _assert_private_values_absent(str(error.value), case)
        _assert_private_values_absent(repr(error.value), case)
        _assert_private_values_absent(repr(logs), case)
        _assert_private_values_absent(repr(case.request), case)
    finally:
        repository.close()
