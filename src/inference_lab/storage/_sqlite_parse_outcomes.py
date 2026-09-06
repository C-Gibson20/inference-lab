"""Internal SQLite writes for immutable parse outcomes and bounded failures."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import CorpusId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
)
from inference_lab.storage.parse_records import (
    ParseFailureRecord,
    ParseOutcome,
)
from inference_lab.storage.producers import ProducerIdentity


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


def _outcome_values(outcome: ParseOutcome) -> tuple[object, ...]:
    return (
        outcome.parse_id.as_str(),
        outcome.document_id.as_str(),
        outcome.document_version_id.as_str(),
        outcome.record_schema_version,
        outcome.status.value,
        *_producer_values(outcome.parser),
        outcome.normalisation_version,
        *_producer_values(outcome.ocr_producer),
        (
            outcome.parsed_record_id.as_str()
            if outcome.parsed_record_id is not None
            else None
        ),
    )


def _failure_values(failure: ParseFailureRecord) -> tuple[object, ...]:
    return (
        failure.category.value,
        failure.severity.value,
        failure.stage.value,
        failure.unit_kind.value if failure.unit_kind is not None else None,
        failure.unit_index,
    )


def _validated_outcome(outcome: ParseOutcome) -> ParseOutcome:
    if not isinstance(outcome, ParseOutcome):
        raise TypeError("outcome must be a ParseOutcome.")
    return replace(outcome)


def _is_authorised(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    outcome: ParseOutcome,
) -> bool:
    return (
        connection.execute(
            """SELECT 1
            FROM corpus_document_version_memberships
            WHERE corpus_id = ? AND document_id = ? AND document_version_id = ?""",
            (
                corpus_id.as_str(),
                outcome.document_id.as_str(),
                outcome.document_version_id.as_str(),
            ),
        ).fetchone()
        is not None
    )


def _linked_representation_matches(
    connection: sqlite3.Connection,
    outcome: ParseOutcome,
) -> bool:
    if outcome.parsed_record_id is None:
        return True
    row = connection.execute(
        """SELECT document_id, document_version_id, record_schema_version,
            parser_implementation, parser_contract_version,
            parser_artefact_identity, parser_artefact_revision,
            normalisation_version, ocr_implementation, ocr_contract_version,
            ocr_artefact_identity, ocr_artefact_revision
        FROM parsed_representations WHERE record_id = ?""",
        (outcome.parsed_record_id.as_str(),),
    ).fetchone()
    expected = (
        outcome.document_id.as_str(),
        outcome.document_version_id.as_str(),
        outcome.record_schema_version,
        *_producer_values(outcome.parser),
        outcome.normalisation_version,
        *_producer_values(outcome.ocr_producer),
    )
    return row is not None and tuple(row) == expected


def _stored_outcome_matches(
    connection: sqlite3.Connection,
    outcome: ParseOutcome,
) -> bool:
    header = connection.execute(
        """SELECT parse_id, document_id, document_version_id,
            record_schema_version, parse_status, parser_implementation,
            parser_contract_version, parser_artefact_identity,
            parser_artefact_revision, normalisation_version,
            ocr_implementation, ocr_contract_version, ocr_artefact_identity,
            ocr_artefact_revision, parsed_record_id
        FROM parse_outcomes WHERE parse_id = ?""",
        (outcome.parse_id.as_str(),),
    ).fetchone()
    if header is None:
        return False
    failures = connection.execute(
        """SELECT failure_category, failure_severity, failure_stage,
            unit_kind, unit_index
        FROM parse_failures WHERE parse_id = ? ORDER BY ordinal""",
        (outcome.parse_id.as_str(),),
    ).fetchall()
    return tuple(header) == _outcome_values(outcome) and tuple(
        map(tuple, failures)
    ) == tuple(_failure_values(failure) for failure in outcome.failures)


def _insert_outcome(connection: sqlite3.Connection, outcome: ParseOutcome) -> None:
    connection.execute(
        """INSERT INTO parse_outcomes (
            parse_id, document_id, document_version_id,
            record_schema_version, parse_status, parser_implementation,
            parser_contract_version, parser_artefact_identity,
            parser_artefact_revision, normalisation_version,
            ocr_implementation, ocr_contract_version, ocr_artefact_identity,
            ocr_artefact_revision, parsed_record_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        _outcome_values(outcome),
    )
    connection.executemany(
        """INSERT INTO parse_failures (
            parse_id, ordinal, failure_category, failure_severity,
            failure_stage, unit_kind, unit_index
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            (outcome.parse_id.as_str(), ordinal, *_failure_values(failure))
            for ordinal, failure in enumerate(outcome.failures)
        ),
    )


def add_parse_outcome(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    outcome: ParseOutcome,
) -> None:
    """Add one corpus-authorised parse outcome or accept exact replay."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    outcome = _validated_outcome(outcome)

    try:
        if not _is_authorised(connection, corpus_id, outcome):
            raise StorageConflictError()
        if not _linked_representation_matches(connection, outcome):
            raise StorageConflictError()
        if _stored_outcome_matches(connection, outcome):
            return
        _insert_outcome(connection, outcome)
    except StorageError:
        raise
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None
