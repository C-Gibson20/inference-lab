"""Tests for atomic SQLite parsed-representation aggregate writes."""

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
    DocumentIdentity,
    DocumentVersion,
    HashAlgorithm,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ParsedRepresentation,
    ProducerIdentity,
    StorageConflictError,
    StorageTransactionError,
    StructuralBlock,
    StructuralBlockKind,
)
from inference_lab.storage.source_locations import SourceLocation

from .fixtures import open_repository as _open_repository  # noqa: TID252
from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    PARSED_RECORD_ID,
    PARSER,
)

_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000001"),
    "Parsed representation corpus",
)
_OTHER_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000002"),
    "Other parsed representation corpus",
)
_VERSION = DocumentVersion(
    DOCUMENT_VERSION_ID,
    DOCUMENT_ID,
    ContentHash(HashAlgorithm.SHA256, "a" * 64),
)
_SECOND_RECORD_ID = RecordId("80000000-0000-4000-8000-000000000002")
_THIRD_RECORD_ID = RecordId("80000000-0000-4000-8000-000000000003")
_MISSING_RECORD_ID = RecordId("80000000-0000-4000-8000-000000000099")
_SECOND_SPAN_ID = SourceSpanId("90000000-0000-4000-8000-000000000002")
_THIRD_SPAN_ID = SourceSpanId("90000000-0000-4000-8000-000000000003")
_UPDATED_PARSER = ProducerIdentity("text-native", "3.0.0")
_OCR_PRODUCER = ProducerIdentity(
    "local-ocr",
    "1.0.0",
    "ocr-model",
    "revision-1",
)
_AGGREGATE_TABLES = (
    "parsed_representations",
    "page_layouts",
    "source_spans",
    "structural_blocks",
)
_PRIVATE_TRIGGER_PAYLOAD = "private aggregate insert payload"


def _representation() -> ParsedRepresentation:
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit=CoordinateUnit.PDF_POINTS,
        coordinate_origin=CoordinateOrigin.TOP_LEFT,
        page_width=612.0,
        page_height=792.0,
        rotation=PageRotation.DEGREES_0,
        content_origin=ContentOrigin.NATIVE,
        producer=PARSER,
    )
    location = SourceLocation(
        source_span_id=FIRST_SPAN_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        parsed_record_id=PARSED_RECORD_ID,
        start_offset=0,
        end_offset=8,
        content_origin=ContentOrigin.NATIVE,
        page_number=1,
        bounding_box=BoundingBox(left=10.0, top=20.0, right=110.0, bottom=40.0),
    )
    return ParsedRepresentation(
        record_id=PARSED_RECORD_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        source_text="Page one",
        text="Page one",
        parser=PARSER,
        normalisation_version="1.0.0",
        language="en",
        page_count=1,
        character_count=8,
        token_count=2,
        page_collection_availability=CollectionAvailability.AVAILABLE,
        section_collection_availability=CollectionAvailability.UNAVAILABLE,
        paragraph_collection_availability=CollectionAvailability.UNAVAILABLE,
        blocks=(
            StructuralBlock(
                ordinal=0,
                kind=StructuralBlockKind.PAGE,
                text="Page one",
                source_span_id=FIRST_SPAN_ID,
            ),
        ),
        source_locations=(location,),
        page_layouts=(layout,),
    )


def _reidentified_representation(
    representation: ParsedRepresentation,
    *,
    record_id: RecordId,
    source_span_id: SourceSpanId,
) -> ParsedRepresentation:
    location = replace(
        representation.source_locations[0],
        source_span_id=source_span_id,
        parsed_record_id=record_id,
    )
    block = replace(
        representation.blocks[0],
        source_span_id=source_span_id,
    )
    return replace(
        representation,
        record_id=record_id,
        blocks=(block,),
        source_locations=(location,),
    )


def _empty_representation() -> ParsedRepresentation:
    return ParsedRepresentation(
        record_id=_SECOND_RECORD_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        source_text="",
        text="",
        parser=_UPDATED_PARSER,
        normalisation_version="1.0.0",
        language=None,
        page_count=None,
        character_count=0,
        token_count=None,
        page_collection_availability=CollectionAvailability.AVAILABLE,
        section_collection_availability=CollectionAvailability.UNAVAILABLE,
        paragraph_collection_availability=CollectionAvailability.AVAILABLE,
        blocks=(),
        source_locations=(),
        page_layouts=(),
    )


def _add_catalogue(path: Path) -> None:
    repository = _open_repository(path)
    try:
        with repository.transaction() as transaction:
            transaction.add_corpus(_CORPUS)
            transaction.add_corpus(_OTHER_CORPUS)
            transaction.add_document(DocumentIdentity(DOCUMENT_ID))
            transaction.add_document_version(_VERSION)
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
    finally:
        repository.close()


def _table_count(path: Path, table: str) -> int:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            f"SELECT count(*) FROM {table}"  # noqa: S608
        ).fetchone()[0]
    finally:
        connection.close()


def _rows(path: Path, query: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def _reject_inserts(path: Path, table: str) -> None:
    if table not in _AGGREGATE_TABLES:
        raise ValueError
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_{table}_insert
            BEFORE INSERT ON {table}
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_TRIGGER_PAYLOAD}');
            END"""  # noqa: S608
        )
    finally:
        connection.close()


def test_complete_representation_commits_all_rows_and_accepts_exact_replay(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    representation = _representation()
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                representation,
            )
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                representation,
            )
    finally:
        repository.close()

    assert _rows(
        database_path,
        """SELECT record_id, document_id, document_version_id, source_text,
            canonical_text, parser_implementation, parser_contract_version,
            normalisation_version, language, page_count, character_count,
            token_count, page_collection_availability,
            section_collection_availability, paragraph_collection_availability,
            output_digest_algorithm, output_digest, supersedes_record_id
        FROM parsed_representations""",
    ) == [
        (
            PARSED_RECORD_ID.as_str(),
            DOCUMENT_ID.as_str(),
            DOCUMENT_VERSION_ID.as_str(),
            "Page one",
            "Page one",
            PARSER.implementation,
            PARSER.contract_version,
            "1.0.0",
            "en",
            1,
            8,
            2,
            "available",
            "unavailable",
            "unavailable",
            "sha256",
            representation.output_digest.digest,
            None,
        )
    ]
    assert _rows(
        database_path,
        """SELECT page_number, coordinate_unit, coordinate_origin,
            page_width, page_height, rotation, content_origin,
            producer_implementation, producer_contract_version
        FROM page_layouts""",
    ) == [
        (1, "pdf_points", "top_left", 612.0, 792.0, 0, "native", "text-native", "2.0.0")
    ]
    assert _rows(
        database_path,
        """SELECT source_span_id, start_offset, end_offset, content_origin,
            page_number, bounding_box_left, bounding_box_top,
            bounding_box_right, bounding_box_bottom
        FROM source_spans""",
    ) == [(FIRST_SPAN_ID.as_str(), 0, 8, "native", 1, 10.0, 20.0, 110.0, 40.0)]
    assert _rows(
        database_path,
        """SELECT ordinal, block_kind, block_text, source_span_id
        FROM structural_blocks""",
    ) == [(0, "page", "Page one", FIRST_SPAN_ID.as_str())]


def test_complete_representation_rolls_back_with_its_transaction(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    repository = _open_repository(database_path)
    try:
        with pytest.raises(RuntimeError, match="primary"):
            with repository.transaction() as transaction:
                transaction.add_parsed_representation(
                    _CORPUS.corpus_id,
                    _representation(),
                )
                raise RuntimeError("primary")
    finally:
        repository.close()

    assert _table_count(database_path, "parsed_representations") == 0
    assert _table_count(database_path, "page_layouts") == 0
    assert _table_count(database_path, "source_spans") == 0
    assert _table_count(database_path, "structural_blocks") == 0


@pytest.mark.parametrize("rejected_table", _AGGREGATE_TABLES)
def test_each_insert_failure_rolls_back_and_poisons_the_transaction(
    tmp_path: Path,
    rejected_table: str,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    _add_catalogue(database_path)
    representation = _representation()
    repository = _open_repository(database_path)
    try:
        _reject_inserts(database_path, rejected_table)
        with capture_logs() as logs:
            with pytest.raises(StorageTransactionError):
                with repository.transaction() as transaction:
                    with pytest.raises(StorageConflictError) as insert_error:
                        transaction.add_parsed_representation(
                            _CORPUS.corpus_id,
                            representation,
                        )
                    assert (
                        str(insert_error.value) == StorageConflictError.default_message
                    )
                    with pytest.raises(StorageTransactionError):
                        transaction.add_parsed_representation(
                            _CORPUS.corpus_id,
                            representation,
                        )
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "transaction",
                "retryable": False,
                "log_level": "error",
            }
        ]
        assert _PRIVATE_TRIGGER_PAYLOAD not in repr(logs)
        assert "private-path" not in repr(logs)
        assert representation.source_text not in repr(logs)
    finally:
        repository.close()

    for table in _AGGREGATE_TABLES:
        assert _table_count(database_path, table) == 0


@pytest.mark.parametrize(
    "corpus_id",
    [
        _OTHER_CORPUS.corpus_id,
        CorpusId("70000000-0000-4000-8000-000000000099"),
    ],
    ids=("non-member", "absent"),
)
def test_complete_representation_requires_explicit_corpus_membership(
    tmp_path: Path,
    corpus_id: CorpusId,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    repository = _open_repository(database_path)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_parsed_representation(corpus_id, _representation())
    finally:
        repository.close()

    assert _table_count(database_path, "parsed_representations") == 0
    assert _table_count(database_path, "page_layouts") == 0
    assert _table_count(database_path, "source_spans") == 0
    assert _table_count(database_path, "structural_blocks") == 0


def test_record_and_natural_identity_reuse_with_different_payload_conflicts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    representation = _representation()
    changed_record = replace(representation, language="fr")
    changed_natural_identity = _reidentified_representation(
        representation,
        record_id=_SECOND_RECORD_ID,
        source_span_id=_SECOND_SPAN_ID,
    )
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                representation,
            )

        for conflicting in (changed_record, changed_natural_identity):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.add_parsed_representation(
                        _CORPUS.corpus_id,
                        conflicting,
                    )

        with repository.transaction() as transaction:
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                representation,
            )
    finally:
        repository.close()

    assert _table_count(database_path, "parsed_representations") == 1
    assert _table_count(database_path, "page_layouts") == 1
    assert _table_count(database_path, "source_spans") == 1
    assert _table_count(database_path, "structural_blocks") == 1


def test_exact_representation_reuse_requires_explicit_second_membership(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                _representation(),
            )

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_parsed_representation(
                    _OTHER_CORPUS.corpus_id,
                    _representation(),
                )

        with repository.transaction() as transaction:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _OTHER_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
            transaction.add_parsed_representation(
                _OTHER_CORPUS.corpus_id,
                _representation(),
            )
    finally:
        repository.close()

    assert _table_count(database_path, "parsed_representations") == 1
    assert _table_count(database_path, "page_layouts") == 1
    assert _table_count(database_path, "source_spans") == 1
    assert _table_count(database_path, "structural_blocks") == 1


def test_parser_ocr_variants_and_supersession_are_persisted(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    original = _representation()
    successor = replace(
        _reidentified_representation(
            original,
            record_id=_SECOND_RECORD_ID,
            source_span_id=_SECOND_SPAN_ID,
        ),
        parser=_UPDATED_PARSER,
        supersedes_record_id=PARSED_RECORD_ID,
    )
    ocr_identity = _reidentified_representation(
        original,
        record_id=_THIRD_RECORD_ID,
        source_span_id=_THIRD_SPAN_ID,
    )
    ocr_location = replace(
        ocr_identity.source_locations[0],
        content_origin=ContentOrigin.OCR,
    )
    ocr_layout = replace(
        ocr_identity.page_layouts[0],
        content_origin=ContentOrigin.OCR,
        producer=_OCR_PRODUCER,
    )
    ocr = replace(
        ocr_identity,
        ocr_producer=_OCR_PRODUCER,
        source_locations=(ocr_location,),
        page_layouts=(ocr_layout,),
        supersedes_record_id=_SECOND_RECORD_ID,
    )

    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parsed_representation(_CORPUS.corpus_id, original)
            transaction.add_parsed_representation(_CORPUS.corpus_id, successor)
            transaction.add_parsed_representation(_CORPUS.corpus_id, ocr)
    finally:
        repository.close()

    assert _rows(
        database_path,
        """SELECT record_id, parser_contract_version, ocr_implementation,
            ocr_contract_version, ocr_artefact_identity,
            ocr_artefact_revision, supersedes_record_id
        FROM parsed_representations ORDER BY record_id""",
    ) == [
        (PARSED_RECORD_ID.as_str(), "2.0.0", None, None, None, None, None),
        (
            _SECOND_RECORD_ID.as_str(),
            "3.0.0",
            None,
            None,
            None,
            None,
            PARSED_RECORD_ID.as_str(),
        ),
        (
            _THIRD_RECORD_ID.as_str(),
            "2.0.0",
            "local-ocr",
            "1.0.0",
            "ocr-model",
            "revision-1",
            _SECOND_RECORD_ID.as_str(),
        ),
    ]
    assert _rows(
        database_path,
        """SELECT parsed_record_id, content_origin, producer_implementation
        FROM page_layouts ORDER BY parsed_record_id""",
    )[-1] == (_THIRD_RECORD_ID.as_str(), "ocr", "local-ocr")
    assert _rows(
        database_path,
        """SELECT parsed_record_id, content_origin
        FROM source_spans ORDER BY parsed_record_id""",
    )[-1] == (_THIRD_RECORD_ID.as_str(), "ocr")


def test_missing_superseded_representation_rolls_back_every_row(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    successor = replace(
        _reidentified_representation(
            _representation(),
            record_id=_SECOND_RECORD_ID,
            source_span_id=_SECOND_SPAN_ID,
        ),
        parser=_UPDATED_PARSER,
        supersedes_record_id=_MISSING_RECORD_ID,
    )
    repository = _open_repository(database_path)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_parsed_representation(
                    _CORPUS.corpus_id,
                    successor,
                )
    finally:
        repository.close()

    assert _table_count(database_path, "parsed_representations") == 0
    assert _table_count(database_path, "page_layouts") == 0
    assert _table_count(database_path, "source_spans") == 0
    assert _table_count(database_path, "structural_blocks") == 0


def test_optional_values_and_available_empty_collections_remain_explicit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue(database_path)
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                _empty_representation(),
            )
    finally:
        repository.close()

    assert _rows(
        database_path,
        """SELECT language, page_count, token_count, ocr_implementation,
            page_collection_availability, section_collection_availability,
            paragraph_collection_availability
        FROM parsed_representations""",
    ) == [(None, None, None, None, "available", "unavailable", "available")]
    assert _table_count(database_path, "page_layouts") == 0
    assert _table_count(database_path, "source_spans") == 0
    assert _table_count(database_path, "structural_blocks") == 0


def test_repository_exposes_no_partial_aggregate_writer(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert not hasattr(repository, "add_parsed_representation")
        assert not hasattr(repository, "add_page_layout")
        assert not hasattr(repository, "add_source_span")
        assert not hasattr(repository, "add_structural_block")
    finally:
        repository.close()
