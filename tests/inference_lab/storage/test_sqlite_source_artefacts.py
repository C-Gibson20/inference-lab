"""Tests for corpus-authorised SQLite source-artefact persistence."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

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
)
from inference_lab.settings import StorageConfig
from inference_lab.storage import (
    CorruptStorageError,
    MetadataRepository,
    RepositoryClosedError,
    RepositoryUnavailableError,
    SourceArtefact,
    StorageConflictError,
    open_metadata_repository,
)
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .fixtures import open_repository as _open_repository  # noqa: TID252
from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    SOURCE_BYTES,
    SOURCE_HASH,
)

_FIRST_CORPUS = CorpusMetadata(
    CorpusId("10000000-0000-4000-8000-000000000010"),
    "First source corpus",
)
_SECOND_CORPUS = CorpusMetadata(
    CorpusId("10000000-0000-4000-8000-000000000020"),
    "Second source corpus",
)
_VERSION = DocumentVersion(
    DOCUMENT_VERSION_ID,
    DOCUMENT_ID,
    SOURCE_HASH,
)
_SHARED_DOCUMENT_ID = DocumentId("20000000-0000-4000-8000-000000000099")
_SHARED_VERSION_ID = DocumentVersionId("30000000-0000-4000-8000-000000000099")
_SHARED_VERSION = DocumentVersion(
    _SHARED_VERSION_ID,
    _SHARED_DOCUMENT_ID,
    SOURCE_HASH,
)
_ARTEFACT = SourceArtefact(
    document_id=DOCUMENT_ID,
    document_version_id=DOCUMENT_VERSION_ID,
    content_hash=SOURCE_HASH,
    byte_count=len(SOURCE_BYTES),
    media_type="application/pdf",
    storage_key=(f"sha256/{SOURCE_HASH.digest[:2]}/{SOURCE_HASH.digest}"),
)
_SHARED_ARTEFACT = replace(
    _ARTEFACT,
    document_id=_SHARED_DOCUMENT_ID,
    document_version_id=_SHARED_VERSION_ID,
)
_PRIVATE_DATABASE_PAYLOAD = "private-source-artefact-trigger-payload"


def _add_catalogue(
    repository: MetadataRepository,
    *,
    include_second_membership: bool = False,
) -> None:
    with repository.transaction() as transaction:
        transaction.add_corpus(_FIRST_CORPUS)
        transaction.add_corpus(_SECOND_CORPUS)
        transaction.add_document(DocumentIdentity(DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_ID,
                DOCUMENT_VERSION_ID,
            )
        )
        if include_second_membership:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _SECOND_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )


def _source_rows(path: Path) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            """SELECT document_version_id, document_id,
                record_schema_version, content_hash_algorithm,
                content_hash_digest, byte_count, media_type, storage_key
            FROM source_artefacts
            ORDER BY document_version_id"""
        ).fetchall()
    finally:
        connection.close()


def test_source_artefact_metadata_commits_and_survives_restart(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = _open_repository(database_path)
    _add_catalogue(first)
    with first.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    first.close()

    reopened = SQLiteMetadataRepository.open(database_path)
    with reopened.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    reopened.close()

    assert _source_rows(database_path) == [
        (
            DOCUMENT_VERSION_ID.as_str(),
            DOCUMENT_ID.as_str(),
            "1.0.0",
            "sha256",
            SOURCE_HASH.digest,
            len(SOURCE_BYTES),
            "application/pdf",
            _ARTEFACT.storage_key,
        )
    ]


def test_factory_returns_corpus_scoped_metadata_and_verified_bytes(
    tmp_path: Path,
) -> None:
    config = StorageConfig(
        index_dir=tmp_path / "indexes",
        metadata_db_path=tmp_path / "metadata.sqlite3",
        source_dir=tmp_path / "source-data",
    )
    ManagedSourceBlobStore(config.source_dir).write_chunks(
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    first = open_metadata_repository(
        config.metadata_db_path,
        source_dir=config.source_dir,
        apply_migrations=True,
    )
    _add_catalogue(first)
    with first.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    first.close()

    reopened = open_metadata_repository(
        config.metadata_db_path,
        source_dir=config.source_dir,
    )
    try:
        assert (
            reopened.get_source_artefact(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == _ARTEFACT
        )
        assert (
            reopened.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
        stream = reopened.stream_source_bytes(
            _FIRST_CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )
        assert stream is not None
        reopened.close()
        assert b"".join(stream) == SOURCE_BYTES
    finally:
        reopened.close()


def test_absent_and_non_member_reads_do_not_touch_managed_bytes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "missing-source-root"
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        source_dir=source_dir,
    )
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    try:
        assert (
            repository.get_source_artefact(
                _SECOND_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.read_source_bytes(
                _SECOND_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.stream_source_bytes(
                _SECOND_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert (
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DocumentVersionId("30000000-0000-4000-8000-000000000099"),
            )
            is None
        )
    finally:
        repository.close()


def test_reachable_missing_or_modified_blob_fails_closed(tmp_path: Path) -> None:
    source_dir = tmp_path / "source-data"
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        source_dir=source_dir,
    )
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    try:
        with pytest.raises(CorruptStorageError):
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
        with pytest.raises(CorruptStorageError):
            repository.stream_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )

        ManagedSourceBlobStore(source_dir).write_chunks(
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )
        blob_path = source_dir / _ARTEFACT.storage_key
        blob_path.chmod(0o600)
        blob_path.write_bytes(b"x" * len(SOURCE_BYTES))
        with pytest.raises(CorruptStorageError):
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
    finally:
        repository.close()


def test_metadata_read_does_not_require_a_configured_blob_store(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
    try:
        assert (
            repository.get_source_artefact(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == _ARTEFACT
        )
        with pytest.raises(RepositoryUnavailableError):
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
        with pytest.raises(RepositoryUnavailableError):
            repository.stream_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
    finally:
        repository.close()


def test_transaction_source_access_is_metadata_only(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source-data"
    ManagedSourceBlobStore(source_dir).write_chunks(
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    repository = SQLiteMetadataRepository.open(
        tmp_path / "metadata.sqlite3",
        apply_migrations=True,
        source_dir=source_dir,
    )
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
            assert (
                transaction.get_source_artefact(
                    _FIRST_CORPUS.corpus_id,
                    DOCUMENT_VERSION_ID,
                )
                == _ARTEFACT
            )
            assert not hasattr(transaction, "read_source_bytes")
            assert not hasattr(transaction, "stream_source_bytes")

        assert (
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
        stream = repository.stream_source_bytes(
            _FIRST_CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )
        assert stream is not None
        assert b"".join(stream) == SOURCE_BYTES
    finally:
        repository.close()


def test_source_reads_reject_repository_use_after_close(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    repository.close()

    with pytest.raises(RepositoryClosedError):
        repository.get_source_artefact(
            _FIRST_CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )
    with pytest.raises(RepositoryClosedError):
        repository.read_source_bytes(
            _FIRST_CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )
    with pytest.raises(RepositoryClosedError):
        repository.stream_source_bytes(
            _FIRST_CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        )


def test_distinct_versions_and_corpora_share_one_verified_blob(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    ManagedSourceBlobStore(source_dir).write_chunks(
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
    )
    repository = SQLiteMetadataRepository.open(
        database_path,
        apply_migrations=True,
        source_dir=source_dir,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(_FIRST_CORPUS)
        transaction.add_corpus(_SECOND_CORPUS)
        transaction.add_document(DocumentIdentity(DOCUMENT_ID))
        transaction.add_document(DocumentIdentity(_SHARED_DOCUMENT_ID))
        transaction.add_document_version(_VERSION)
        transaction.add_document_version(_SHARED_VERSION)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_ID,
                DOCUMENT_VERSION_ID,
            )
        )
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _SECOND_CORPUS.corpus_id,
                _SHARED_DOCUMENT_ID,
                _SHARED_VERSION_ID,
            )
        )
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)
        transaction.add_source_artefact(
            _SECOND_CORPUS.corpus_id,
            _SHARED_ARTEFACT,
        )

    try:
        with capture_logs() as logs:
            first = repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            first_repeat = repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            second = repository.read_source_bytes(
                _SECOND_CORPUS.corpus_id,
                _SHARED_VERSION_ID,
            )
            first_stream = repository.stream_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            second_stream = repository.stream_source_bytes(
                _SECOND_CORPUS.corpus_id,
                _SHARED_VERSION_ID,
            )
            first_stream_repeat = repository.stream_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )

        assert first == first_repeat == second == SOURCE_BYTES
        assert (
            repository.get_source_artefact(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == _ARTEFACT
        )
        assert (
            repository.get_source_artefact(
                _SECOND_CORPUS.corpus_id,
                _SHARED_VERSION_ID,
            )
            == _SHARED_ARTEFACT
        )
        assert first_stream is not None
        assert second_stream is not None
        assert first_stream_repeat is not None
        assert b"".join(first_stream) == b"".join(second_stream) == SOURCE_BYTES
        assert b"".join(first_stream_repeat) == SOURCE_BYTES
        assert _ARTEFACT.storage_key == _SHARED_ARTEFACT.storage_key
        assert len(_source_rows(database_path)) == 2
        assert (source_dir / _ARTEFACT.storage_key).is_file()
        assert logs == []
    finally:
        repository.close()


@pytest.mark.parametrize(
    "corruption_sql",
    [
        "UPDATE source_artefacts SET record_schema_version = '2.0.0'",
        """UPDATE source_artefacts
        SET document_id = '20000000-0000-4000-8000-000000000099'""",
    ],
    ids=("unsupported-record-version", "mismatched-owner"),
)
def test_corrupt_stored_metadata_fails_closed_on_every_public_read(
    tmp_path: Path,
    corruption_sql: str,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(corruption_sql)
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_source_artefact(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
        with pytest.raises(CorruptStorageError):
            repository.read_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
        with pytest.raises(CorruptStorageError):
            repository.stream_source_bytes(
                _FIRST_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
    finally:
        repository.close()


def test_source_artefact_reuse_requires_explicit_membership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_source_artefact(_SECOND_CORPUS.corpus_id, _ARTEFACT)

        with repository.transaction() as transaction:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _SECOND_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
            transaction.add_source_artefact(_SECOND_CORPUS.corpus_id, _ARTEFACT)

        assert len(_source_rows(database_path)) == 1
    finally:
        repository.close()


@pytest.mark.parametrize(
    ("corpus_id", "artefact"),
    [
        (
            CorpusId("10000000-0000-4000-8000-000000000099"),
            _ARTEFACT,
        ),
        (
            _FIRST_CORPUS.corpus_id,
            replace(
                _ARTEFACT,
                document_id=DocumentId("20000000-0000-4000-8000-000000000099"),
            ),
        ),
        (
            _FIRST_CORPUS.corpus_id,
            replace(
                _ARTEFACT,
                document_version_id=DocumentVersionId(
                    "30000000-0000-4000-8000-000000000099"
                ),
            ),
        ),
        (
            _FIRST_CORPUS.corpus_id,
            replace(
                _ARTEFACT,
                content_hash=ContentHash(HashAlgorithm.SHA256, "f" * 64),
                storage_key=f"sha256/ff/{'f' * 64}",
            ),
        ),
    ],
    ids=("unknown-corpus", "wrong-document", "unknown-version", "wrong-hash"),
)
def test_source_artefact_requires_an_agreeing_catalogue_membership(
    tmp_path: Path,
    corpus_id: CorpusId,
    artefact: SourceArtefact,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_source_artefact(corpus_id, artefact)

        assert _source_rows(database_path) == []
    finally:
        repository.close()


def test_reusing_version_identity_with_different_metadata_is_a_conflict(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    try:
        with repository.transaction() as transaction:
            transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_source_artefact(
                    _FIRST_CORPUS.corpus_id,
                    replace(_ARTEFACT, media_type="application/octet-stream"),
                )

        assert _source_rows(database_path)[0][6] == "application/pdf"
    finally:
        repository.close()


def test_source_artefact_rolls_back_with_its_metadata_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    try:
        with pytest.raises(RuntimeError, match="primary"):
            with repository.transaction() as transaction:
                transaction.set_corpus_display_name(
                    _FIRST_CORPUS.corpus_id,
                    "Uncommitted name",
                )
                transaction.add_source_artefact(
                    _FIRST_CORPUS.corpus_id,
                    _ARTEFACT,
                )
                raise RuntimeError("primary")

        assert repository.get_corpus(_FIRST_CORPUS.corpus_id) == _FIRST_CORPUS
        assert _source_rows(database_path) == []
    finally:
        repository.close()


def test_corrupt_stored_metadata_fails_closed_during_replay(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    with repository.transaction() as transaction:
        transaction.add_source_artefact(_FIRST_CORPUS.corpus_id, _ARTEFACT)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE source_artefacts SET media_type = 'INVALID'")
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            with repository.transaction() as transaction:
                transaction.add_source_artefact(
                    _FIRST_CORPUS.corpus_id,
                    _ARTEFACT,
                )
    finally:
        repository.close()


def test_database_failure_is_typed_and_payload_free(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    repository = _open_repository(database_path)
    _add_catalogue(repository)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_source_artefact
            BEFORE INSERT ON source_artefacts
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as raised:
                with repository.transaction() as transaction:
                    transaction.add_source_artefact(
                        _FIRST_CORPUS.corpus_id,
                        _ARTEFACT,
                    )

        assert _PRIVATE_DATABASE_PAYLOAD not in str(raised.value)
        assert _PRIVATE_DATABASE_PAYLOAD not in repr(logs)
        assert _ARTEFACT.storage_key not in repr(logs)
        assert _source_rows(database_path) == []
    finally:
        repository.close()
