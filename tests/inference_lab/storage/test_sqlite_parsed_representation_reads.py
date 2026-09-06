"""Tests for corpus-scoped SQLite parsed-representation reads."""

from dataclasses import replace
from pathlib import Path

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage import (
    BoundingBox,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    ParsedRepresentation,
    ProducerIdentity,
    ResolvedSourceSpan,
)

from .fixtures import open_repository as _open_repository  # noqa: TID252
from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    FIRST_SPAN_ID,
    PARSED_RECORD_ID,
    PARSER,
    SECOND_SPAN_ID,
    SOURCE_HASH,
    SOURCE_TEXT,
    parsed_representation,
)

_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000001"),
    "Parsed representation read corpus",
)
_OTHER_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000002"),
    "Other parsed representation read corpus",
)
_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_SECOND_RECORD_ID = RecordId("40000000-0000-4000-8000-000000000002")
_THIRD_RECORD_ID = RecordId("40000000-0000-4000-8000-000000000003")
_THIRD_SPAN_ID = SourceSpanId("50000000-0000-4000-8000-000000000003")
_FOURTH_SPAN_ID = SourceSpanId("50000000-0000-4000-8000-000000000004")
_MISSING_SPAN_ID = SourceSpanId("50000000-0000-4000-8000-000000000099")
_MISSING_VERSION_ID = DocumentVersionId("20000000-0000-4000-8000-000000000099")


def _variant() -> ParsedRepresentation:
    original = parsed_representation()
    span_ids = (_THIRD_SPAN_ID, _FOURTH_SPAN_ID)
    locations = tuple(
        replace(
            location,
            source_span_id=source_span_id,
            parsed_record_id=_SECOND_RECORD_ID,
        )
        for location, source_span_id in zip(
            original.source_locations,
            span_ids,
            strict=True,
        )
    )
    blocks = tuple(
        replace(block, source_span_id=source_span_id)
        for block, source_span_id in zip(original.blocks, span_ids, strict=True)
    )
    return replace(
        original,
        record_id=_SECOND_RECORD_ID,
        parser=ProducerIdentity("text-native", "3.0.0"),
        source_locations=locations,
        blocks=blocks,
    )


def _representation_with_layout() -> ParsedRepresentation:
    original = parsed_representation()
    layout = PageLayoutDescriptor(
        page_number=1,
        coordinate_unit=CoordinateUnit.PDF_POINTS,
        coordinate_origin=CoordinateOrigin.BOTTOM_LEFT,
        page_width=612.0,
        page_height=792.0,
        rotation=PageRotation.DEGREES_90,
        content_origin=ContentOrigin.NATIVE,
        producer=PARSER,
    )
    first_location = replace(
        original.source_locations[0],
        page_number=1,
        bounding_box=BoundingBox(left=10.0, top=20.0, right=110.0, bottom=40.0),
    )
    return replace(
        original,
        page_count=1,
        source_locations=(first_location, original.source_locations[1]),
        page_layouts=(layout,),
    )


def _persist(path: Path, representations: tuple[ParsedRepresentation, ...]) -> None:
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
            for representation in representations:
                transaction.add_parsed_representation(
                    _CORPUS.corpus_id,
                    representation,
                )
    finally:
        repository.close()


def test_complete_representation_round_trips_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    expected = parsed_representation()
    _persist(database_path, (expected,))

    repository = _open_repository(database_path)
    try:
        assert (
            repository.get_parsed_representation(
                _CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            == expected
        )
        assert repository.list_parsed_representations(
            _CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        ) == (expected,)
        with repository.transaction() as transaction:
            assert (
                transaction.get_parsed_representation(
                    _CORPUS.corpus_id,
                    PARSED_RECORD_ID,
                )
                == expected
            )
            assert transaction.list_parsed_representations(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            ) == (expected,)

        reconstructed = repository.get_parsed_representation(
            _CORPUS.corpus_id,
            PARSED_RECORD_ID,
        )
        assert reconstructed is not None
        assert reconstructed.source_text == SOURCE_TEXT
        assert tuple(
            location.source_span_id for location in reconstructed.source_locations
        ) == (FIRST_SPAN_ID, SECOND_SPAN_ID)
    finally:
        repository.close()


def test_representation_list_uses_canonical_record_identity_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    first = parsed_representation()
    second = _variant()
    _persist(database_path, (second, first))

    repository = _open_repository(database_path)
    try:
        assert repository.list_parsed_representations(
            _CORPUS.corpus_id,
            DOCUMENT_VERSION_ID,
        ) == (first, second)
        expected_spans = tuple(
            sorted(
                first.known_source_spans() + second.known_source_spans(),
                key=lambda span: span.key(),
            )
        )
        assert (
            repository.list_known_source_spans(
                _CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == expected_spans
        )
    finally:
        repository.close()


def test_source_span_resolution_returns_decoded_offsets_and_coordinates(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    representation = _representation_with_layout()
    _persist(database_path, (representation,))
    expected = ResolvedSourceSpan(
        document_id=DOCUMENT_ID,
        location=representation.source_locations[0],
        page_layout=representation.page_layouts[0],
    )

    repository = _open_repository(database_path)
    try:
        assert (
            repository.resolve_source_span(
                _CORPUS.corpus_id,
                FIRST_SPAN_ID,
            )
            == expected
        )
        assert repository.resolve_source_span(
            _CORPUS.corpus_id,
            SECOND_SPAN_ID,
        ) == ResolvedSourceSpan(
            document_id=DOCUMENT_ID,
            location=representation.source_locations[1],
            page_layout=None,
        )
        with repository.transaction() as transaction:
            assert (
                transaction.resolve_source_span(
                    _CORPUS.corpus_id,
                    FIRST_SPAN_ID,
                )
                == expected
            )
        assert (
            SOURCE_TEXT[expected.location.start_offset : expected.location.end_offset]
            == "Cafe\u0301"
        )
        assert expected.as_known_source_span() == representation.known_source_spans()[0]
    finally:
        repository.close()


def test_representation_reads_do_not_disclose_absence_or_other_corpus_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _persist(database_path, (parsed_representation(),))
    repository = _open_repository(database_path)
    try:
        assert (
            repository.get_parsed_representation(
                _OTHER_CORPUS.corpus_id,
                PARSED_RECORD_ID,
            )
            is None
        )
        assert (
            repository.get_parsed_representation(
                _CORPUS.corpus_id,
                _THIRD_RECORD_ID,
            )
            is None
        )
        assert (
            repository.list_parsed_representations(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == ()
        )
        assert (
            repository.list_parsed_representations(
                _CORPUS.corpus_id,
                _MISSING_VERSION_ID,
            )
            == ()
        )
        assert (
            repository.resolve_source_span(
                _OTHER_CORPUS.corpus_id,
                FIRST_SPAN_ID,
            )
            is None
        )
        assert (
            repository.resolve_source_span(
                _CORPUS.corpus_id,
                _MISSING_SPAN_ID,
            )
            is None
        )
        assert (
            repository.list_known_source_spans(
                _OTHER_CORPUS.corpus_id,
                DOCUMENT_VERSION_ID,
            )
            == ()
        )
    finally:
        repository.close()
