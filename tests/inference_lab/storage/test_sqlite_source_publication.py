"""Tests for fail-closed managed-source publication across both stores."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, Self, cast

import pytest
from structlog.contextvars import merge_contextvars
from structlog.testing import capture_logs

import inference_lab.storage._source_blob_store as source_blob_store_module
from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
)
from inference_lab.storage import (
    CorruptStorageError,
    ParseOutcome,
    ParseStatus,
    SourceArtefact,
    StorageConcurrencyError,
    StorageConflictError,
    StorageTransactionError,
)
from inference_lab.storage._source_blob_store import ManagedSourceBlobStore
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository

from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SECOND_SPAN_ID,
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
_PRIVATE_FILESYSTEM_PAYLOAD = "private-source-filesystem-failure"

type _PrePromotionFailure = Literal["write", "flush", "hash", "rename"]
type _PublishedBlobCorruption = Literal[
    "missing", "truncated", "replaced", "mismatched"
]
type _ParsedOutputCorruption = Literal["digest", "payload"]


class _FailingBinaryWriter:
    """Delegate to a real staging stream while failing one requested operation."""

    def __init__(self, stream: BinaryIO, operation: Literal["write", "flush"]) -> None:
        self._stream = stream
        self._operation = operation

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stream.close()

    def write(self, content: bytes) -> int:
        if self._operation == "write":
            raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)
        return self._stream.write(content)

    def flush(self) -> None:
        if self._operation == "flush":
            raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


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


def _corrupt_published_blob(
    blob_path: Path,
    corruption: _PublishedBlobCorruption,
    *,
    replacement_path: Path,
) -> None:
    if corruption == "missing":
        blob_path.unlink()
    elif corruption == "truncated":
        blob_path.chmod(0o600)
        blob_path.write_bytes(SOURCE_BYTES[:-1])
    elif corruption == "replaced":
        replacement_path.write_bytes(SOURCE_BYTES)
        blob_path.unlink()
        blob_path.symlink_to(replacement_path)
    else:
        blob_path.chmod(0o600)
        blob_path.write_bytes(b"x" * len(SOURCE_BYTES))


def _corrupt_parsed_output(
    database_path: Path,
    corruption: _ParsedOutputCorruption,
) -> None:
    connection = sqlite3.connect(database_path)
    try:
        if corruption == "digest":
            connection.execute(
                """UPDATE parsed_representations
                SET output_digest = ?
                WHERE record_id = ?""",
                ("0" * 64, PARSED_RECORD_ID.as_str()),
            )
        else:
            connection.execute(
                """UPDATE parsed_representations
                SET token_count = token_count + 1
                WHERE record_id = ?""",
                (PARSED_RECORD_ID.as_str(),),
            )
        connection.commit()
    finally:
        connection.close()


def _inject_pre_promotion_failure(
    monkeypatch: pytest.MonkeyPatch,
    operation: _PrePromotionFailure,
) -> None:
    if operation in ("write", "flush"):
        real_fdopen = source_blob_store_module.os.fdopen

        def open_failing_stream(descriptor: int, mode: str) -> _FailingBinaryWriter:
            assert mode == "wb"
            stream = cast(BinaryIO, real_fdopen(descriptor, "wb"))
            return _FailingBinaryWriter(stream, operation)

        monkeypatch.setattr(source_blob_store_module.os, "fdopen", open_failing_stream)
        return
    if operation == "hash":

        class FailingDigest:
            def update(self, content: bytes) -> None:
                raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)

            def hexdigest(self) -> str:
                return ""

        def fail_hash_creation() -> object:
            return FailingDigest()

        monkeypatch.setattr(
            source_blob_store_module.hashlib,
            "sha256",
            fail_hash_creation,
        )
        return

    def fail_rename(*args: object, **kwargs: object) -> None:
        raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)

    monkeypatch.setattr(source_blob_store_module.os, "replace", fail_rename)


def _staging_files(source_dir: Path) -> tuple[Path, ...]:
    staging = source_dir / ".staging"
    return tuple(staging.iterdir()) if staging.exists() else ()


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


def test_pre_promotion_filesystem_failures_leave_no_visible_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    repository = _open_repository(database_path, source_dir)
    _inject_pre_promotion_failure(monkeypatch, "write")

    try:
        with capture_logs(processors=(merge_contextvars,)) as logs:
            with pytest.raises(StorageTransactionError) as raised:
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
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert not (source_dir / _ARTEFACT.storage_key).exists()
        assert _staging_files(source_dir) == ()
        assert logs[-1]["event"] == "source_persistence_failed"
        assert logs[-1]["error_category"] == "transaction"
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in str(raised.value)
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in repr(logs)
        assert str(source_dir) not in repr(logs)
    finally:
        repository.close()


def test_post_promotion_flush_failure_leaves_reusable_unreachable_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    repository = _open_repository(database_path, source_dir)
    real_replace = source_blob_store_module.os.replace
    real_fsync = source_blob_store_module.os.fsync
    promoted = False

    def record_replace(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal promoted
        real_replace(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )
        promoted = True

    def fail_post_promotion_flush(descriptor: int) -> None:
        if promoted:
            raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)
        real_fsync(descriptor)

    try:
        with monkeypatch.context() as failure:
            failure.setattr(source_blob_store_module.os, "replace", record_replace)
            failure.setattr(
                source_blob_store_module.os,
                "fsync",
                fail_post_promotion_flush,
            )
            with capture_logs(processors=(merge_contextvars,)) as logs:
                with pytest.raises(StorageTransactionError) as raised:
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
        assert (source_dir / _ARTEFACT.storage_key).read_bytes() == SOURCE_BYTES
        assert _staging_files(source_dir) == ()
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in str(raised.value)
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in repr(logs)
        assert str(source_dir) not in repr(logs)

        repository.persist_source_artefact(
            _CORPUS.corpus_id,
            _ARTEFACT,
            (SOURCE_BYTES,),
            max_source_bytes=len(SOURCE_BYTES),
        )
        assert (
            repository.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
    finally:
        repository.close()


def test_cleanup_failure_preserves_primary_and_startup_removes_abandoned_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    repository = _open_repository(database_path, source_dir)

    class SourceInterrupted(RuntimeError):
        pass

    primary = SourceInterrupted(_PRIVATE_FILESYSTEM_PAYLOAD)

    def interrupted_chunks() -> Iterator[bytes]:
        yield SOURCE_BYTES[:3]
        raise primary

    def fail_cleanup(path: str, *, dir_fd: int | None = None) -> None:
        raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)

    try:
        with monkeypatch.context() as failure:
            failure.setattr(source_blob_store_module.os, "unlink", fail_cleanup)
            with capture_logs(processors=(merge_contextvars,)) as failure_logs:
                with pytest.raises(SourceInterrupted) as raised:
                    repository.persist_source_artefact(
                        _CORPUS.corpus_id,
                        _ARTEFACT,
                        interrupted_chunks(),
                        max_source_bytes=len(SOURCE_BYTES),
                    )

        assert raised.value is primary
        assert len(_staging_files(source_dir)) == 1
        assert not (source_dir / _ARTEFACT.storage_key).exists()
        assert (
            repository.get_source_artefact(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            is None
        )
        assert repository.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in repr(failure_logs)
        assert str(source_dir) not in repr(failure_logs)
    finally:
        repository.close()

    with capture_logs() as recovery_logs:
        reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        assert _staging_files(source_dir) == ()
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
        completed = [
            record
            for record in recovery_logs
            if record["event"] == "source_reconciliation_completed"
        ]
        assert len(completed) == 1
        assert completed[0]["staging_files_removed"] == 1
        assert _PRIVATE_FILESYSTEM_PAYLOAD not in repr(recovery_logs)
        assert str(source_dir) not in repr(recovery_logs)
    finally:
        reopened.close()


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


def test_publication_contention_fails_before_blob_or_metadata_visibility(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "source-data"
    repository = _open_repository(database_path, source_dir)
    try:
        with ManagedSourceBlobStore(source_dir).publication_lock():
            with pytest.raises(StorageConcurrencyError):
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
        assert not (source_dir / _ARTEFACT.storage_key).exists()
    finally:
        repository.close()


@pytest.mark.parametrize(
    "corruption",
    ("missing", "truncated", "replaced", "mismatched"),
)
def test_published_blob_corruption_fails_closed_without_poisoning_metadata(
    tmp_path: Path,
    corruption: _PublishedBlobCorruption,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
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
    _corrupt_published_blob(
        source_dir / _ARTEFACT.storage_key,
        corruption,
        replacement_path=tmp_path / "private-replacement-source",
    )

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        with capture_logs() as logs:
            with pytest.raises(CorruptStorageError) as read_error:
                reopened.read_source_bytes(
                    _CORPUS.corpus_id,
                    DOCUMENT_VERSION_ID,
                )
            with pytest.raises(CorruptStorageError) as stream_error:
                reopened.stream_source_bytes(
                    _CORPUS.corpus_id,
                    DOCUMENT_VERSION_ID,
                )

        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
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
        assert reopened.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) == (
            _PARSE_OUTCOME
        )
        assert logs == []
        for private_value in (
            str(source_dir),
            _ARTEFACT.storage_key,
            SOURCE_HASH.digest,
            SOURCE_BYTES.decode(),
        ):
            assert private_value not in str(read_error.value)
            assert private_value not in str(stream_error.value)
    finally:
        reopened.close()


@pytest.mark.parametrize("corruption", ("digest", "payload"))
def test_parsed_output_corruption_fails_closed_without_poisoning_source_reads(
    tmp_path: Path,
    corruption: _ParsedOutputCorruption,
) -> None:
    database_path = tmp_path / "private-database" / "metadata.sqlite3"
    source_dir = tmp_path / "private-source-data"
    repository = _open_repository(database_path, source_dir)
    repository.persist_source_artefact(
        _CORPUS.corpus_id,
        _ARTEFACT,
        (SOURCE_BYTES,),
        max_source_bytes=len(SOURCE_BYTES),
        parse_outcome=_PARSE_OUTCOME,
        parsed_representation=parsed_representation(),
    )
    repository.close()
    _corrupt_parsed_output(database_path, corruption)

    reopened = SQLiteMetadataRepository.open(database_path, source_dir=source_dir)
    try:
        errors: list[CorruptStorageError] = []
        with capture_logs() as logs:
            with pytest.raises(CorruptStorageError) as get_error:
                reopened.get_parsed_representation(
                    _CORPUS.corpus_id,
                    PARSED_RECORD_ID,
                )
            errors.append(get_error.value)
            with pytest.raises(CorruptStorageError) as list_error:
                reopened.list_parsed_representations(
                    _CORPUS.corpus_id,
                    DOCUMENT_VERSION_ID,
                )
            errors.append(list_error.value)
            for source_span_id in (FIRST_SPAN_ID, SECOND_SPAN_ID):
                with pytest.raises(CorruptStorageError) as span_error:
                    reopened.resolve_source_span(
                        _CORPUS.corpus_id,
                        source_span_id,
                    )
                errors.append(span_error.value)

        assert (
            reopened.read_source_bytes(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == SOURCE_BYTES
        )
        assert reopened.get_corpus(_CORPUS.corpus_id) == _CORPUS
        assert reopened.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID) == (
            _PARSE_OUTCOME
        )
        assert logs == []
        for error in errors:
            assert str(database_path) not in str(error)
            assert str(source_dir) not in str(error)
            assert SOURCE_HASH.digest not in str(error)
    finally:
        reopened.close()
