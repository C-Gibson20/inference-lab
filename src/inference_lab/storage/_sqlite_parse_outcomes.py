"""Internal SQLite writes for immutable parse outcomes and bounded failures."""

import sqlite3
from dataclasses import replace

from inference_lab.provenance import CorpusId, DocumentId, DocumentVersionId, RecordId
from inference_lab.storage._sqlite_connection import translate_sqlite_error
from inference_lab.storage.errors import (
    CorruptStorageError,
    RepositoryUnavailableError,
    StorageConflictError,
    StorageError,
)
from inference_lab.storage.parse_records import (
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
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


def _reconstruct_failure(row: sqlite3.Row) -> ParseFailureRecord:
    unit_kind = row["unit_kind"]
    return ParseFailureRecord(
        category=ParseFailureCategory(row["failure_category"]),
        severity=ParseFailureSeverity(row["failure_severity"]),
        stage=ParseFailureStage(row["failure_stage"]),
        unit_kind=(ParseFailureUnitKind(unit_kind) if unit_kind is not None else None),
        unit_index=row["unit_index"],
    )


def _reconstruct_outcome(
    row: sqlite3.Row,
    failure_rows: list[sqlite3.Row],
) -> ParseOutcome:
    if any(
        failure_row["ordinal"] != ordinal
        for ordinal, failure_row in enumerate(failure_rows)
    ):
        raise ValueError
    parser = _reconstruct_producer(
        row["parser_implementation"],
        row["parser_contract_version"],
        row["parser_artefact_identity"],
        row["parser_artefact_revision"],
    )
    if parser is None:
        raise ValueError
    ocr_producer = _reconstruct_producer(
        row["ocr_implementation"],
        row["ocr_contract_version"],
        row["ocr_artefact_identity"],
        row["ocr_artefact_revision"],
    )
    parsed_record_id = row["parsed_record_id"]
    return ParseOutcome(
        parse_id=RecordId.from_str(row["parse_id"]),
        document_id=DocumentId.from_str(row["document_id"]),
        document_version_id=DocumentVersionId.from_str(row["document_version_id"]),
        record_schema_version=row["record_schema_version"],
        status=ParseStatus(row["parse_status"]),
        parser=parser,
        normalisation_version=row["normalisation_version"],
        ocr_producer=ocr_producer,
        parsed_record_id=(
            RecordId.from_str(parsed_record_id)
            if parsed_record_id is not None
            else None
        ),
        failures=tuple(_reconstruct_failure(value) for value in failure_rows),
    )


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


def get_parse_outcome(
    connection: sqlite3.Connection,
    corpus_id: CorpusId,
    parse_id: RecordId,
) -> ParseOutcome | None:
    """Return a complete outcome only through explicit corpus membership."""
    if not isinstance(corpus_id, CorpusId):
        raise TypeError("corpus_id must be a CorpusId.")
    if not isinstance(parse_id, RecordId):
        raise TypeError("parse_id must be a RecordId.")

    try:
        row = connection.execute(
            """SELECT outcome.parse_id, outcome.document_id,
                outcome.document_version_id, outcome.record_schema_version,
                outcome.parse_status, outcome.parser_implementation,
                outcome.parser_contract_version,
                outcome.parser_artefact_identity,
                outcome.parser_artefact_revision,
                outcome.normalisation_version, outcome.ocr_implementation,
                outcome.ocr_contract_version, outcome.ocr_artefact_identity,
                outcome.ocr_artefact_revision, outcome.parsed_record_id
            FROM parse_outcomes AS outcome
            JOIN corpus_document_version_memberships AS membership
                ON membership.document_version_id = outcome.document_version_id
                AND membership.document_id = outcome.document_id
            WHERE membership.corpus_id = ? AND outcome.parse_id = ?""",
            (corpus_id.as_str(), parse_id.as_str()),
        ).fetchone()
        if row is None:
            return None
        failure_rows = connection.execute(
            """SELECT ordinal, failure_category, failure_severity,
                failure_stage, unit_kind, unit_index
            FROM parse_failures WHERE parse_id = ? ORDER BY ordinal""",
            (parse_id.as_str(),),
        ).fetchall()
        outcome = _reconstruct_outcome(row, failure_rows)
        if not _linked_representation_matches(connection, outcome):
            raise CorruptStorageError()
        return outcome
    except CorruptStorageError:
        raise
    except (TypeError, ValueError, KeyError, IndexError):
        raise CorruptStorageError() from None
    except sqlite3.Error as error:
        raise translate_sqlite_error(
            error,
            default=RepositoryUnavailableError,
        ) from None


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
