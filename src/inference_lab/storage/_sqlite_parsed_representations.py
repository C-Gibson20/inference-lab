"""Internal SQLite persistence for complete parsed representations."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import (
    CorpusId,
    DocumentId,
    DocumentVersionId,
    RecordId,
    SourceSpanId,
)
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
)
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.producers import ProducerIdentity
from inference_lab.storage.source_locations import (
    BoundingBox,
    CollectionAvailability,
    ContentOrigin,
    CoordinateOrigin,
    CoordinateUnit,
    PageLayoutDescriptor,
    PageRotation,
    SourceLocation,
    StructuralBlock,
    StructuralBlockKind,
)


def _producer_values(
    producer: ProducerIdentity | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if producer is None:
        return (None, None, None, None)
    return (
        producer.implementation,
        producer.contract_version,
        producer.artefact_identity,
        producer.artefact_revision,
    )


def _bounding_box_values(
    bounding_box: BoundingBox | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    if bounding_box is None:
        return (None, None, None, None)
    return (
        float(bounding_box.left),
        float(bounding_box.top),
        float(bounding_box.right),
        float(bounding_box.bottom),
    )


def _reconstruct_producer(
    implementation: str | None,
    contract_version: str | None,
    artefact_identity: str | None,
    artefact_revision: str | None,
) -> ProducerIdentity | None:
    if implementation is None:
        if any(
            value is not None
            for value in (contract_version, artefact_identity, artefact_revision)
        ):
            raise ValueError
        return None
    if contract_version is None:
        raise ValueError
    return ProducerIdentity(
        implementation,
        contract_version,
        artefact_identity,
        artefact_revision,
    )


def _reconstruct_layout(row: sqlite3.Row) -> PageLayoutDescriptor:
    producer = _reconstruct_producer(
        row["producer_implementation"],
        row["producer_contract_version"],
        row["producer_artefact_identity"],
        row["producer_artefact_revision"],
    )
    if producer is None:
        raise ValueError
    return PageLayoutDescriptor(
        page_number=row["page_number"],
        coordinate_unit=CoordinateUnit(row["coordinate_unit"]),
        coordinate_origin=CoordinateOrigin(row["coordinate_origin"]),
        page_width=row["page_width"],
        page_height=row["page_height"],
        rotation=PageRotation(row["rotation"]),
        content_origin=ContentOrigin(row["content_origin"]),
        producer=producer,
    )


def _reconstruct_bounding_box(row: sqlite3.Row) -> BoundingBox | None:
    values = (
        row["bounding_box_left"],
        row["bounding_box_top"],
        row["bounding_box_right"],
        row["bounding_box_bottom"],
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError
    return BoundingBox(
        left=values[0],
        top=values[1],
        right=values[2],
        bottom=values[3],
    )


def _reconstruct_location(row: sqlite3.Row) -> SourceLocation:
    return SourceLocation(
        source_span_id=SourceSpanId.from_str(row["source_span_id"]),
        document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
        parsed_record_id=RecordId.from_str(row["parsed_record_id"]),
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        content_origin=ContentOrigin(row["content_origin"]),
        page_number=row["page_number"],
        bounding_box=_reconstruct_bounding_box(row),
    )


def _reconstruct_block(row: sqlite3.Row) -> StructuralBlock:
    return StructuralBlock(
        ordinal=row["ordinal"],
        kind=StructuralBlockKind(row["block_kind"]),
        text=row["block_text"],
        source_span_id=SourceSpanId.from_str(row["source_span_id"]),
    )


def _header_values(representation: ParsedRepresentation) -> tuple[object, ...]:
    return (
        representation.record_id.as_str(),
        representation.document_id.as_str(),
        representation.document_version_id.as_str(),
        representation.record_schema_version,
        representation.source_text,
        representation.text,
        *_producer_values(representation.parser),
        representation.normalisation_version,
        *_producer_values(representation.ocr_producer),
        representation.language,
        representation.page_count,
        representation.character_count,
        representation.token_count,
        representation.page_collection_availability.value,
        representation.section_collection_availability.value,
        representation.paragraph_collection_availability.value,
        representation.output_digest.algorithm.value,
        representation.output_digest.digest,
        (
            representation.supersedes_record_id.as_str()
            if representation.supersedes_record_id is not None
            else None
        ),
    )


def _layout_values(
    representation: ParsedRepresentation,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            representation.record_id.as_str(),
            representation.document_version_id.as_str(),
            layout.page_number,
            layout.coordinate_unit.value,
            layout.coordinate_origin.value,
            float(layout.page_width),
            float(layout.page_height),
            layout.rotation.value,
            layout.content_origin.value,
            *_producer_values(layout.producer),
        )
        for layout in representation.page_layouts
    )


def _span_values(
    representation: ParsedRepresentation,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            location.source_span_id.as_str(),
            representation.record_id.as_str(),
            representation.document_version_id.as_str(),
            location.start_offset,
            location.end_offset,
            location.content_origin.value,
            location.page_number,
            *_bounding_box_values(location.bounding_box),
        )
        for location in representation.source_locations
    )


def _block_values(
    representation: ParsedRepresentation,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            representation.record_id.as_str(),
            representation.document_version_id.as_str(),
            block.ordinal,
            block.kind.value,
            block.text,
            block.source_span_id.as_str(),
        )
        for block in representation.blocks
    )


def _validated_representation(
    representation: ParsedRepresentation,
) -> ParsedRepresentation:
    if not isinstance(representation, ParsedRepresentation):
        raise TypeError("representation must be a ParsedRepresentation.")
    validated = replace(representation)
    if validated != representation:
        raise ValueError("representation must contain its canonical output digest.")
    return validated


def _is_authorised(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    representation: ParsedRepresentation,
) -> bool:
    row = connection.execute(
        """SELECT 1
        FROM corpus_document_version_memberships
        WHERE corpus_id = ? AND document_id = ? AND document_version_id = ?""",
        (
            corpus_id.as_str(),
            representation.document_id.as_str(),
            representation.document_version_id.as_str(),
        ),
    ).fetchone()
    return row is not None


def _reconstruct_representation(
    connection: sqlite3.Connection,
    header: sqlite3.Row,
) -> ParsedRepresentation:
    record_id = header["record_id"]
    layouts = connection.execute(
        """SELECT page_number, coordinate_unit, coordinate_origin,
            page_width, page_height, rotation, content_origin,
            producer_implementation, producer_contract_version,
            producer_artefact_identity, producer_artefact_revision
        FROM page_layouts WHERE parsed_record_id = ? ORDER BY page_number""",
        (record_id,),
    ).fetchall()
    locations = connection.execute(
        """SELECT source_span_id, parsed_record_id, document_version_id,
            start_offset, end_offset, content_origin, page_number,
            bounding_box_left, bounding_box_top, bounding_box_right,
            bounding_box_bottom
        FROM source_spans WHERE parsed_record_id = ? ORDER BY source_span_id""",
        (record_id,),
    ).fetchall()
    blocks = connection.execute(
        """SELECT ordinal, block_kind, block_text, source_span_id
        FROM structural_blocks WHERE parsed_record_id = ? ORDER BY ordinal""",
        (record_id,),
    ).fetchall()
    parser = _reconstruct_producer(
        header["parser_implementation"],
        header["parser_contract_version"],
        header["parser_artefact_identity"],
        header["parser_artefact_revision"],
    )
    if parser is None:
        raise ValueError
    ocr_producer = _reconstruct_producer(
        header["ocr_implementation"],
        header["ocr_contract_version"],
        header["ocr_artefact_identity"],
        header["ocr_artefact_revision"],
    )
    supersedes_record_id = header["supersedes_record_id"]
    representation = ParsedRepresentation(
        record_id=RecordId.from_str(record_id),
        document_id=DocumentId.from_str(header["document_id"]),
        document_version_id=DocumentVersionId.from_str(header["document_version_id"]),
        record_schema_version=header["record_schema_version"],
        source_text=header["source_text"],
        text=header["canonical_text"],
        parser=parser,
        normalisation_version=header["normalisation_version"],
        ocr_producer=ocr_producer,
        language=header["language"],
        page_count=header["page_count"],
        character_count=header["character_count"],
        token_count=header["token_count"],
        page_collection_availability=CollectionAvailability(
            header["page_collection_availability"]
        ),
        section_collection_availability=CollectionAvailability(
            header["section_collection_availability"]
        ),
        paragraph_collection_availability=CollectionAvailability(
            header["paragraph_collection_availability"]
        ),
        page_layouts=tuple(_reconstruct_layout(row) for row in layouts),
        source_locations=tuple(_reconstruct_location(row) for row in locations),
        blocks=tuple(_reconstruct_block(row) for row in blocks),
        supersedes_record_id=(
            RecordId.from_str(supersedes_record_id)
            if supersedes_record_id is not None
            else None
        ),
    )
    if (
        header["output_digest_algorithm"]
        != representation.output_digest.algorithm.value
        or header["output_digest"] != representation.output_digest.digest
    ):
        raise CorruptStorageError()
    return representation


def _reconstruct_rows(
    connection: sqlite3.Connection,
    rows: list[sqlite3.Row],
) -> tuple[ParsedRepresentation, ...]:
    try:
        return tuple(_reconstruct_representation(connection, row) for row in rows)
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None


def get_parsed_representation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    record_id: RecordId,
) -> ParsedRepresentation | None:
    """Return a complete representation through explicit corpus membership."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(record_id, RecordId):
        raise TypeError("record_id must be a RecordId.")
    try:
        row = connection.execute(
            """SELECT representation.record_id, representation.document_id,
                representation.document_version_id,
                representation.record_schema_version,
                representation.source_text, representation.canonical_text,
                representation.parser_implementation,
                representation.parser_contract_version,
                representation.parser_artefact_identity,
                representation.parser_artefact_revision,
                representation.normalisation_version,
                representation.ocr_implementation,
                representation.ocr_contract_version,
                representation.ocr_artefact_identity,
                representation.ocr_artefact_revision,
                representation.language, representation.page_count,
                representation.character_count, representation.token_count,
                representation.page_collection_availability,
                representation.section_collection_availability,
                representation.paragraph_collection_availability,
                representation.output_digest_algorithm,
                representation.output_digest,
                representation.supersedes_record_id
            FROM parsed_representations AS representation
            JOIN corpus_document_version_memberships AS membership
                ON membership.document_version_id = representation.document_version_id
                AND membership.document_id = representation.document_id
            WHERE membership.corpus_id = ? AND representation.record_id = ?""",
            (corpus_id.as_str(), record_id.as_str()),
        ).fetchone()
        if row is None:
            return None
        return _reconstruct_rows(connection, [row])[0]
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def list_parsed_representations(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    document_version_id: DocumentVersionId,
) -> tuple[ParsedRepresentation, ...]:
    """Return corpus-visible representations in canonical identity order."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(document_version_id, DocumentVersionId):
        raise TypeError("document_version_id must be a DocumentVersionId.")
    try:
        rows = connection.execute(
            """SELECT representation.record_id, representation.document_id,
                representation.document_version_id,
                representation.record_schema_version,
                representation.source_text, representation.canonical_text,
                representation.parser_implementation,
                representation.parser_contract_version,
                representation.parser_artefact_identity,
                representation.parser_artefact_revision,
                representation.normalisation_version,
                representation.ocr_implementation,
                representation.ocr_contract_version,
                representation.ocr_artefact_identity,
                representation.ocr_artefact_revision,
                representation.language, representation.page_count,
                representation.character_count, representation.token_count,
                representation.page_collection_availability,
                representation.section_collection_availability,
                representation.paragraph_collection_availability,
                representation.output_digest_algorithm,
                representation.output_digest,
                representation.supersedes_record_id
            FROM parsed_representations AS representation
            JOIN corpus_document_version_memberships AS membership
                ON membership.document_version_id = representation.document_version_id
                AND membership.document_id = representation.document_id
            WHERE membership.corpus_id = ?
                AND representation.document_version_id = ?
            ORDER BY representation.record_id""",
            (corpus_id.as_str(), document_version_id.as_str()),
        ).fetchall()
        return _reconstruct_rows(connection, rows)
    except CorruptStorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


def _stored_aggregate_matches(
    connection: sqlite3.Connection,
    representation: ParsedRepresentation,
) -> bool:
    header = connection.execute(
        """SELECT record_id, document_id, document_version_id,
            record_schema_version, source_text, canonical_text,
            parser_implementation, parser_contract_version,
            parser_artefact_identity, parser_artefact_revision,
            normalisation_version, ocr_implementation, ocr_contract_version,
            ocr_artefact_identity, ocr_artefact_revision, language, page_count,
            character_count, token_count, page_collection_availability,
            section_collection_availability, paragraph_collection_availability,
            output_digest_algorithm, output_digest, supersedes_record_id
        FROM parsed_representations WHERE record_id = ?""",
        (representation.record_id.as_str(),),
    ).fetchone()
    if header is None:
        return False
    layouts = connection.execute(
        """SELECT parsed_record_id, document_version_id, page_number,
            coordinate_unit, coordinate_origin, page_width, page_height,
            rotation, content_origin, producer_implementation,
            producer_contract_version, producer_artefact_identity,
            producer_artefact_revision
        FROM page_layouts WHERE parsed_record_id = ? ORDER BY page_number""",
        (representation.record_id.as_str(),),
    ).fetchall()
    spans = connection.execute(
        """SELECT source_span_id, parsed_record_id, document_version_id,
            start_offset, end_offset, content_origin, page_number,
            bounding_box_left, bounding_box_top, bounding_box_right,
            bounding_box_bottom
        FROM source_spans WHERE parsed_record_id = ? ORDER BY source_span_id""",
        (representation.record_id.as_str(),),
    ).fetchall()
    blocks = connection.execute(
        """SELECT parsed_record_id, document_version_id, ordinal, block_kind,
            block_text, source_span_id
        FROM structural_blocks WHERE parsed_record_id = ? ORDER BY ordinal""",
        (representation.record_id.as_str(),),
    ).fetchall()
    return (
        tuple(header) == _header_values(representation)
        and tuple(map(tuple, layouts)) == _layout_values(representation)
        and tuple(map(tuple, spans)) == _span_values(representation)
        and tuple(map(tuple, blocks)) == _block_values(representation)
    )


def _insert_complete_aggregate(
    connection: sqlite3.Connection,
    representation: ParsedRepresentation,
) -> None:
    connection.execute(
        """INSERT INTO parsed_representations (
            record_id, document_id, document_version_id,
            record_schema_version, source_text, canonical_text,
            parser_implementation, parser_contract_version,
            parser_artefact_identity, parser_artefact_revision,
            normalisation_version, ocr_implementation, ocr_contract_version,
            ocr_artefact_identity, ocr_artefact_revision, language, page_count,
            character_count, token_count, page_collection_availability,
            section_collection_availability, paragraph_collection_availability,
            output_digest_algorithm, output_digest, supersedes_record_id
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?
        )""",
        _header_values(representation),
    )
    connection.executemany(
        """INSERT INTO page_layouts (
            parsed_record_id, document_version_id, page_number,
            coordinate_unit, coordinate_origin, page_width, page_height,
            rotation, content_origin, producer_implementation,
            producer_contract_version, producer_artefact_identity,
            producer_artefact_revision
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        _layout_values(representation),
    )
    connection.executemany(
        """INSERT INTO source_spans (
            source_span_id, parsed_record_id, document_version_id,
            start_offset, end_offset, content_origin, page_number,
            bounding_box_left, bounding_box_top, bounding_box_right,
            bounding_box_bottom
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        _span_values(representation),
    )
    connection.executemany(
        """INSERT INTO structural_blocks (
            parsed_record_id, document_version_id, ordinal, block_kind,
            block_text, source_span_id
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        _block_values(representation),
    )


def add_parsed_representation(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    representation: ParsedRepresentation,
) -> None:
    """Add one complete corpus-authorised aggregate or accept exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    representation = _validated_representation(representation)

    try:
        if not _is_authorised(connection, corpus_id, representation):
            raise StorageConflictError()
        if _stored_aggregate_matches(connection, representation):
            return
        _insert_complete_aggregate(connection, representation)
    except StorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
