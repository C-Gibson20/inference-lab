"""Internal SQLite writes for complete immutable parsed representations."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import CorpusId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
)
from inference_lab.storage.parsed_records import ParsedRepresentation
from inference_lab.storage.producers import ProducerIdentity
from inference_lab.storage.source_locations import BoundingBox


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
