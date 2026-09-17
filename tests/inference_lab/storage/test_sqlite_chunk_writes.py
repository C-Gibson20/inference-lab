"""Tests for atomic SQLite chunk-set writes."""

import hashlib
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    ChunkingRequest,
    FakeChunkTokeniser,
    build_chunk_set,
    map_chunk_set,
)
from inference_lab.provenance import (
    ConfigurationIdentity,
    ContentHash,
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DigestAlgorithm,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    HashAlgorithm,
    OutputDigest,
)
from inference_lab.storage import (
    ChunkSetSelectionRecord,
    ChunkSetWriteRequest,
    CorruptStorageError,
    StorageConflictError,
    WriteChunkSetOutcome,
)
from inference_lab.storage._sqlite_repository import SQLiteMetadataRepository
from tests.inference_lab.ingest.fixtures import (  # noqa: TID252
    fixture_parsed_document,
)
from tests.inference_lab.storage.fixtures import (  # noqa: TID252
    open_repository as _open_repository,
)

_CORPUS_ID = CorpusId("70000000-0000-4000-8000-000000000101")
_OTHER_CORPUS_ID = CorpusId("70000000-0000-4000-8000-000000000102")


@dataclass(frozen=True, slots=True)
class _WriteCase:
    """Complete graph values needed to exercise one chunk-set write."""

    corpus: CorpusMetadata
    other_corpus: CorpusMetadata
    document: DocumentIdentity
    version: DocumentVersion
    request: ChunkSetWriteRequest
    selection: ChunkSetSelectionRecord


def _case() -> _WriteCase:
    """Build one valid ING-04 output and its catalogue prerequisites."""
    document = fixture_parsed_document()
    chunk_set = build_chunk_set(
        ChunkingRequest(
            corpus_id=_CORPUS_ID,
            parsed_document=document,
            tokeniser=FakeChunkTokeniser(),
        )
    )
    selection = map_chunk_set(chunk_set)
    return _WriteCase(
        corpus=CorpusMetadata(_CORPUS_ID, "Chunk write corpus"),
        other_corpus=CorpusMetadata(_OTHER_CORPUS_ID, "Other chunk corpus"),
        document=DocumentIdentity(chunk_set.document_id),
        version=DocumentVersion(
            chunk_set.document_version_id,
            chunk_set.document_id,
            ContentHash(HashAlgorithm.SHA256, "a" * 64),
        ),
        request=ChunkSetWriteRequest(
            corpus_id=_CORPUS_ID,
            selection=selection,
        ),
        selection=selection,
    )


def _add_catalogue(
    repository: SQLiteMetadataRepository,
    case: _WriteCase,
    *,
    include_membership: bool = True,
) -> None:
    """Create the corpus/document/version graph required by the FK boundary."""
    with repository.transaction() as transaction:
        transaction.add_corpus(case.corpus)
        transaction.add_corpus(case.other_corpus)
        transaction.add_document(case.document)
        transaction.add_document_version(case.version)
        if include_membership:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    case.corpus.corpus_id,
                    case.version.document_id,
                    case.version.document_version_id,
                )
            )


def test_chunk_set_write_persists_complete_selection_atomically(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    try:
        outcome: WriteChunkSetOutcome | None = None
        with capture_logs() as logs:
            with repository.transaction() as transaction:
                outcome = transaction.write_chunk_set(case.request)

        assert outcome is not None
        assert outcome.selection == case.selection
        assert outcome.replayed is False
        assert (
            repository.get_chunk_set_selection(
                case.corpus.corpus_id,
                case.version.document_version_id,
                case.selection.configuration_identity,
            )
            == case.selection
        )
        rendered_logs = repr(logs)
        assert "chunk_set_written" in rendered_logs
        assert case.selection.payload_digest.digest not in rendered_logs
        assert case.selection.configuration_identity.digest not in rendered_logs
        assert case.selection.canonical_payload.decode("utf-8") not in rendered_logs
    finally:
        repository.close()


def test_chunk_set_write_requires_corpus_membership_and_leaves_no_row(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case, include_membership=False)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.write_chunk_set(case.request)

        assert (
            repository.get_chunk_set_selection(
                case.corpus.corpus_id,
                case.version.document_version_id,
                case.selection.configuration_identity,
            )
            is None
        )
    finally:
        repository.close()


def test_chunk_set_write_rolls_back_after_later_transaction_failure(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    try:
        with pytest.raises(RuntimeError, match="private write failure"):
            with repository.transaction() as transaction:
                transaction.write_chunk_set(case.request)
                raise RuntimeError("private write failure")

        assert (
            repository.get_chunk_set_selection(
                case.corpus.corpus_id,
                case.version.document_version_id,
                case.selection.configuration_identity,
            )
            is None
        )
    finally:
        repository.close()


def test_identical_chunk_set_write_is_an_exact_replay(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    try:
        first: WriteChunkSetOutcome | None = None
        replay: WriteChunkSetOutcome | None = None
        with repository.transaction() as transaction:
            first = transaction.write_chunk_set(case.request)
        with repository.transaction() as transaction:
            replay = transaction.write_chunk_set(case.request)

        assert first is not None
        assert replay is not None
        assert first.replayed is False
        assert replay == WriteChunkSetOutcome(
            selection=case.selection,
            replayed=True,
        )
    finally:
        repository.close()


def test_exact_replay_can_be_reused_by_another_corpus_membership(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    try:
        first: WriteChunkSetOutcome | None = None
        replay: WriteChunkSetOutcome | None = None
        with repository.transaction() as transaction:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    case.other_corpus.corpus_id,
                    case.version.document_id,
                    case.version.document_version_id,
                )
            )
            first = transaction.write_chunk_set(case.request)

        other_request = replace(
            case.request,
            corpus_id=case.other_corpus.corpus_id,
        )
        with repository.transaction() as transaction:
            replay = transaction.write_chunk_set(other_request)

        assert first is not None
        assert replay is not None
        assert first.replayed is False
        assert replay.replayed is True
        assert replay.selection == case.selection
    finally:
        repository.close()


def test_same_selection_key_with_changed_payload_is_a_conflict(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    changed_payload = b'{"changed":true}\n'
    changed_selection = replace(
        case.selection,
        canonical_payload=changed_payload,
        payload_digest=OutputDigest(
            algorithm=DigestAlgorithm.SHA256,
            digest=hashlib.sha256(changed_payload).hexdigest(),
        ),
    )
    changed_request = replace(case.request, selection=changed_selection)
    try:
        with repository.transaction() as transaction:
            transaction.write_chunk_set(case.request)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.write_chunk_set(changed_request)
    finally:
        repository.close()


def test_reused_payload_digest_with_changed_metadata_is_a_conflict(
    tmp_path: Path,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    changed_selection = replace(
        case.selection,
        configuration_identity=ConfigurationIdentity(
            algorithm=DigestAlgorithm.SHA256,
            digest="b" * 64,
        ),
    )
    changed_request = replace(case.request, selection=changed_selection)
    try:
        with repository.transaction() as transaction:
            transaction.write_chunk_set(case.request)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.write_chunk_set(changed_request)
    finally:
        repository.close()


def test_exact_replay_survives_repository_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    case = _case()
    repository = _open_repository(database_path)
    _add_catalogue(repository, case)
    first: WriteChunkSetOutcome | None = None
    with repository.transaction() as transaction:
        first = transaction.write_chunk_set(case.request)
    repository.close()

    reopened = _open_repository(database_path)
    try:
        replay: WriteChunkSetOutcome | None = None
        with reopened.transaction() as transaction:
            replay = transaction.write_chunk_set(case.request)
        assert first is not None
        assert replay is not None
        assert first.replayed is False
        assert replay.replayed is True
        assert replay.selection == case.selection
    finally:
        reopened.close()


def test_mismatched_document_ownership_is_a_conflict(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    case = _case()
    _add_catalogue(repository, case)
    wrong_selection = replace(
        case.selection,
        document_id=DocumentId("80000000-0000-4000-8000-000000000999"),
    )
    wrong_request = replace(case.request, selection=wrong_selection)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.write_chunk_set(wrong_request)
        assert (
            repository.get_chunk_set_selection(
                case.corpus.corpus_id,
                case.version.document_version_id,
                case.selection.configuration_identity,
            )
            is None
        )
    finally:
        repository.close()


def test_corrupt_existing_payload_fails_closed_for_read_and_replay(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    case = _case()
    repository = _open_repository(database_path)
    _add_catalogue(repository, case)
    with repository.transaction() as transaction:
        transaction.write_chunk_set(case.request)
    repository.close()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "UPDATE chunk_sets SET canonical_payload = ?",
            (b"tampered-payload",),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = _open_repository(database_path)
    try:
        with pytest.raises(CorruptStorageError):
            reopened.get_chunk_set_selection(
                case.corpus.corpus_id,
                case.version.document_version_id,
                case.selection.configuration_identity,
            )
        with pytest.raises(CorruptStorageError):
            with reopened.transaction() as transaction:
                transaction.write_chunk_set(case.request)
    finally:
        reopened.close()
