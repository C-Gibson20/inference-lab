"""Fault-injection acceptance for atomic ingest persistence."""

import hashlib
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, Self, cast

import pytest

import inference_lab.storage._source_blob_store as source_blob_store_module
from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    BoundingBox,
    PageLayoutDescriptor,
    ParseOutcome,
    ParsePersistenceRequest,
    ParserFailure,
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
from inference_lab.storage import (
    MetadataRepository,
    StorageConflictError,
    StorageTransactionError,
    open_metadata_repository,
)

from .fixtures import fixture_settings, fixture_source  # noqa: TID252

_CORPUS_ID = CorpusId("d0000000-0000-4000-8000-000000000001")
_DOCUMENT_ID = DocumentId("e0000000-0000-4000-8000-000000000001")
_DOCUMENT_VERSION_ID = DocumentVersionId("f0000000-0000-4000-8000-000000000001")
_PARSE_ID = RecordId("a0000000-0000-4000-8000-000000000001")
_PARSED_RECORD_ID = RecordId("a0000000-0000-4000-8000-000000000002")
_PRIVATE_DATABASE_PAYLOAD = "private-ingest-fault-trigger"
_PRIVATE_FILESYSTEM_PAYLOAD = "private-ingest-stream-failure"

type _MetadataTable = Literal[
    "source_artefacts",
    "parsed_representations",
    "page_layouts",
    "source_spans",
    "structural_blocks",
    "parse_outcomes",
    "parse_failures",
]


@dataclass(frozen=True, slots=True)
class _Case:
    request: ParsePersistenceRequest
    version: DocumentVersion
    source_dir: Path


class _FailingBinaryWriter:
    """Delegate to a staging stream while interrupting its first write."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream

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
        del content
        raise OSError(_PRIVATE_FILESYSTEM_PAYLOAD)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()


def _case(tmp_path: Path) -> _Case:
    source = replace(
        fixture_source("basic.txt"),
        document_id=_DOCUMENT_ID,
        document_version_id=_DOCUMENT_VERSION_ID,
    )
    parsed = TextNativeParser().parse(source, settings=fixture_settings())
    document = parsed.document
    assert document is not None
    first_block = document.blocks[0]
    first_span = replace(
        first_block.source_span,
        page_number=1,
        bounding_box=BoundingBox(
            left=10.5,
            top=20.25,
            right=80.75,
            bottom=90.5,
        ),
    )
    blocks = (replace(first_block, source_span=first_span), *document.blocks[1:])
    document = replace(
        document,
        blocks=blocks,
        paragraphs=blocks,
        page_count=1,
        page_layouts=(
            PageLayoutDescriptor(
                page_number=1,
                coordinate_unit="image_pixels",
                coordinate_origin="top_left",
                page_width=1200.0,
                page_height=1600.0,
                rotation=90,
            ),
        ),
    )
    outcome = ParseOutcome(
        status="completed_with_recoverable_failures",
        document=document,
        failures=(
            ParserFailure(
                category="unsupported_structure",
                severity="recoverable",
                stage="structure",
                unit_kind="table",
                unit_index=3,
            ),
        ),
    )
    request = ParsePersistenceRequest(
        corpus_id=_CORPUS_ID,
        parser_input=source,
        parser_version=CURRENT_PARSER_VERSION,
        outcome=outcome,
        parse_id=_PARSE_ID,
        parsed_record_id=_PARSED_RECORD_ID,
    )
    version = DocumentVersion(
        document_version_id=_DOCUMENT_VERSION_ID,
        document_id=_DOCUMENT_ID,
        content_hash=ContentHash(
            HashAlgorithm.SHA256,
            hashlib.sha256(source.content).hexdigest(),
        ),
    )
    return _Case(
        request=request,
        version=version,
        source_dir=tmp_path / "source-data",
    )


def _open_repository(tmp_path: Path, case: _Case) -> MetadataRepository:
    repository = open_metadata_repository(
        tmp_path / "metadata.sqlite3",
        source_dir=case.source_dir,
        apply_migrations=True,
    )
    with repository.transaction() as transaction:
        transaction.add_corpus(CorpusMetadata(_CORPUS_ID, "ING-03 faults"))
        transaction.add_document(DocumentIdentity(_DOCUMENT_ID))
        transaction.add_document_version(case.version)
        transaction.add_corpus_document_version_membership(
            CorpusDocumentVersionMembership(
                _CORPUS_ID,
                _DOCUMENT_ID,
                _DOCUMENT_VERSION_ID,
            )
        )
    return repository


def _install_metadata_trigger(database_path: Path, table: _MetadataTable) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_ingest_fault
            BEFORE INSERT ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_DATABASE_PAYLOAD}');
            END"""
        )
        connection.commit()
    finally:
        connection.close()


def _inject_stream_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fdopen = source_blob_store_module.os.fdopen

    def open_failing_stream(descriptor: int, mode: str) -> _FailingBinaryWriter:
        assert mode == "wb"
        stream = cast(BinaryIO, real_fdopen(descriptor, "wb"))
        return _FailingBinaryWriter(stream)

    monkeypatch.setattr(source_blob_store_module.os, "fdopen", open_failing_stream)


def _assert_no_visible_bundle(repository: MetadataRepository, case: _Case) -> None:
    assert (
        repository.get_source_artefact(
            _CORPUS_ID,
            _DOCUMENT_VERSION_ID,
        )
        is None
    )
    assert (
        repository.read_source_bytes(
            _CORPUS_ID,
            _DOCUMENT_VERSION_ID,
        )
        is None
    )
    assert (
        repository.list_parsed_representations(
            _CORPUS_ID,
            _DOCUMENT_VERSION_ID,
        )
        == ()
    )
    assert (
        repository.get_parsed_representation(
            _CORPUS_ID,
            _PARSED_RECORD_ID,
        )
        is None
    )
    assert (
        repository.list_known_source_spans(
            _CORPUS_ID,
            _DOCUMENT_VERSION_ID,
        )
        == ()
    )
    assert repository.get_parse_outcome(_CORPUS_ID, _PARSE_ID) is None


def test_metadata_subwrite_failure_leaves_no_visible_bundle(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    repository = _open_repository(tmp_path, case)
    # Fail late, after the source and parsed rows have been staged.
    _install_metadata_trigger(tmp_path / "metadata.sqlite3", "parse_outcomes")
    try:
        with pytest.raises(StorageConflictError) as error:
            persist_parse_request(
                repository,
                case.request,
                settings=fixture_settings(),
            )

        assert (
            str(error.value)
            == "The storage operation conflicts with existing metadata."
        )
        _assert_no_visible_bundle(repository, case)
    finally:
        repository.close()


def test_stream_interruption_leaves_no_visible_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    repository = _open_repository(tmp_path, case)
    _inject_stream_failure(monkeypatch)
    try:
        with pytest.raises(StorageTransactionError) as error:
            persist_parse_request(
                repository,
                case.request,
                settings=fixture_settings(),
            )

        assert str(error.value) == "The storage transaction failed."
        _assert_no_visible_bundle(repository, case)
        staging = case.source_dir / ".staging"
        assert not staging.exists() or tuple(staging.iterdir()) == ()
    finally:
        repository.close()
