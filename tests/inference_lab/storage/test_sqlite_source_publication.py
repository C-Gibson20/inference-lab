"""Tests for fail-closed managed-source publication across both stores."""

import sqlite3
from pathlib import Path

import pytest
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
)
from inference_lab.storage import (
    ParseOutcome,
    ParseStatus,
    SourceArtefact,
    StorageConflictError,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SOURCE_BYTES,
    SOURCE_HASH,
    parsed_representation,
)

_CORPUS = CorpusMetadata(
    CorpusId("a0000000-0000-4000-8000-000000000001"),
    "Source publication corpus",
)
_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}",
)
_PARSE_OUTCOME = ParseOutcome(
    parse_id=PARSE_ID,
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    status=ParseStatus.SUCCEEDED,
    parser=PARSER,
    normalisation_version="1.0.0",
    parsed_record_id=PARSED_RECORD_ID,
    failures=(),
)
_PRIVATE_DATABASE_PAYLOAD = "private-source-publication-trigger"


def _open_repository(database_path: Path, source_dir: Path) -> SQLiteMetadataRepository:
    repository = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_CORPUS)
        transaction.add_document(DocumentIdentity(DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS.corpus_id,
                DOCUMENT_ID,
                DOCUMENT_VERSION_ID,
            )
        )
    return repository


def test_source_publication_becomes_visible_only_with_committed_metadata(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)

    with capture_logs(processors=(merge_contextvars,)) as logs:
        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES[:4], SOURCE_BYTES[4:]),
            max_source_bytes=len(SOURCE_BYTES),
        )
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        assert (
            reopened.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == _ARTEFACT
        )
        assert (
            reopened.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
    finally:
        reopened.close()

    assert [record["event"] for record in logs] == [
        "source_persistence_started",
        "transaction_completed",
        "source_persistence_completed",
    ]
    assert logs[-1]["byte_count"] == len(SOURCE_BYTES)
    assert all(record["corpus_id"] == _CORPUS.corpus_id.as_str() for record in logs)
    assert all(record["document_id"] == DOCUMENT_ID.as_str() for record in logs)
    assert all(str(source_dir) not in repr(record) for record in logs)
    assert all(SOURCE_HASH.digest not in repr(record) for record in logs)


def test_metadata_failure_leaves_promoted_blob_unreachable(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""CREATE TRIGGER reject_source_publication
            BEFORE INSERT ON source_artefacts
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""
        )

    try:
        with capture_logs(processors=(merge_contextvars,)) as logs:
            with pytest.raises(StorageConflictError) as captured:
                repository.persist_source_artefact(
                    _CORPUS.corpus_id,
                    _ARTEFACT,
                    (SOURCE_BYTES,),
                    max_source_bytes=len(SOURCE_BYTES),
                )

        assert (
            repository.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert (source_dir / _ARTEFACT.storage_key).is_file()
        assert _PRIVATE_DATABASE_PAYLOAD not in str(captured.value)
        assert _PRIVATE_DATABASE_PAYLOAD not in repr(logs)
        assert logs[-1]["event"] == "source_persistence_failed"
        assert logs[-1]["error_category"] == "conflict"
    finally:
        repository.close()


def test_invalid_membership_is_rejected_before_blob_publication(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = SQLiteMetadataRepository.open(
        database_path,
        source_dir=source_dir,
        apply_migrations=True,
    )
    try:
        with pytest.raises(StorageConflictError):
            repository.persist_source_artefact(
                _CORPUS.corpus_id,
                _ARTEFACT,
                (SOURCE_BYTES,),
                max_source_bytes=len(SOURCE_BYTES),
            )

        assert not (source_dir / _ARTEFACT.storage_key).exists()
    finally:
        repository.close()


def test_source_and_complete_parse_records_commit_together_across_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)
    representation = parsed_representation()
    repository.persist_source_artefact(
        _CORPUS.corpus_id,
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
        parse_outcome=_PARSE_OUTCOME,
        parsed_representation=representation,
    )
    repository.close()

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        assert (
            reopened.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == _ARTEFACT
        )
        assert (
            reopened.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            == representation
        )
        assert (
            reopened.get_parse_outcome(
                _CORPUS.corpus_id,
                PARSE_ID,
            )
            == _PARSE_OUTCOME
        )
    finally:
        reopened.close()


def test_parse_outcome_failure_rolls_back_all_metadata_after_blob_promotion(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"""CREATE TRIGGER reject_parse_publication
            BEFORE INSERT ON parse_outcomes
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""
        )

    try:
        with pytest.raises(StorageConflictError):
            repository.persist_source_artefact(
                _CORPUS.corpus_id,
                _ARTEFACT,
                (SOURCE_BYTES,),
                max_source_bytes=len(SOURCE_BYTES),
                parse_outcome=_PARSE_OUTCOME,
                parsed_representation=parsed_representation(),
            )

        assert (
            repository.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            is None
        )
        assert (
            repository.get_parse_outcome(
                _CORPUS.corpus_id,
                PARSE_ID,
            )
            is None
        )
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert (source_dir / _ARTEFACT.storage_key).is_file()
    finally:
        repository.close()


def test_mismatched_parse_bundle_is_rejected_before_blob_publication(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)
    mismatched = ParseOutcome(
        parse_id=PARSE_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        status=ParseStatus.SUCCEEDED,
        parser=PARSER,
        normalisation_version="2.0.0",
        parsed_record_id=PARSED_RECORD_ID,
        failures=(),
    )
    try:
        with pytest.raises(StorageConflictError):
            repository.persist_source_artefact(
                _CORPUS.corpus_id,
                _ARTEFACT,
                (SOURCE_BYTES,),
                max_source_bytes=len(SOURCE_BYTES),
                parse_outcome=mismatched,
                parsed_representation=parsed_representation(),
            )

        assert not (source_dir / _ARTEFACT.storage_key).exists()
    finally:
        repository.close()
