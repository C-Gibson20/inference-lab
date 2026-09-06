"""Tests for atomic SQLite parse-outcome writes."""

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.provenance import (
    CorpusDocumentVersionMembership,
    CorpusId,
    CorpusMetadata,
    DocumentIdentity,
    DocumentVersion,
    RecordId,
)
from inference_lab.storage import (
    CorruptStorageError,
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
    ProducerIdentity,
    StorageConflictError,
)

from .fixtures import open_repository as _open_repository  # noqa: TID252
from .source_fixtures import (  # noqa: TID252
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
    SOURCE_HASH,
    SOURCE_TEXT,
    parsed_representation,
)

_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000001"),
    "Parse outcome corpus",
)
_OTHER_CORPUS = CorpusMetadata(
    CorpusId("70000000-0000-4000-8000-000000000002"),
    "Other parse outcome corpus",
)
_VERSION = DocumentVersion(DOCUMENT_VERSION_ID, DOCUMENT_ID, SOURCE_HASH)
_RECOVERABLE_PARSE_ID = RecordId("30000000-0000-4000-8000-000000000002")
_FAILED_PARSE_ID = RecordId("30000000-0000-4000-8000-000000000003")
_MISSING_PARSED_RECORD_ID = RecordId("40000000-0000-4000-8000-000000000099")
_ABSENT_CORPUS_ID = CorpusId("70000000-0000-4000-8000-000000000099")
_PRIVATE_TRIGGER_PAYLOAD = "private rejected parse payload"

_RECOVERABLE_FAILURES = (
    ParseFailureRecord(
        category=ParseFailureCategory.MALFORMED_UNIT,
        severity=ParseFailureSeverity.RECOVERABLE,
        stage=ParseFailureStage.STRUCTURE,
        unit_kind=ParseFailureUnitKind.PARAGRAPH,
        unit_index=4,
    ),
    ParseFailureRecord(
        category=ParseFailureCategory.UNSUPPORTED_STRUCTURE,
        severity=ParseFailureSeverity.RECOVERABLE,
        stage=ParseFailureStage.STRUCTURE,
        unit_kind=ParseFailureUnitKind.TABLE,
        unit_index=1,
    ),
)
_FATAL_FAILURES = (
    ParseFailureRecord(
        category=ParseFailureCategory.INVALID_ENCODING,
        severity=ParseFailureSeverity.FATAL,
        stage=ParseFailureStage.INPUT,
    ),
    ParseFailureRecord(
        category=ParseFailureCategory.INVALID_OUTPUT,
        severity=ParseFailureSeverity.FATAL,
        stage=ParseFailureStage.OUTPUT,
    ),
)


def _outcome(
    parse_id: RecordId,
    status: ParseStatus,
    parsed_record_id: RecordId | None,
    failures: tuple[ParseFailureRecord, ...],
) -> ParseOutcome:
    return ParseOutcome(
        parse_id=parse_id,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        status=status,
        parser=PARSER,
        normalisation_version="1.0.0",
        parsed_record_id=parsed_record_id,
        failures=failures,
    )


def _add_catalogue_and_representation(path: Path) -> None:
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
            transaction.add_parsed_representation(
                _CORPUS.corpus_id,
                parsed_representation(),
            )
    finally:
        repository.close()


def _rows(path: Path, query: str) -> list[tuple[object, ...]]:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def test_all_outcome_states_and_ordered_failures_persist_with_exact_replay(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    outcomes = (
        _outcome(PARSE_ID, ParseStatus.SUCCEEDED, PARSED_RECORD_ID, ()),
        _outcome(
            _RECOVERABLE_PARSE_ID,
            ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
            PARSED_RECORD_ID,
            _RECOVERABLE_FAILURES,
        ),
        _outcome(_FAILED_PARSE_ID, ParseStatus.FAILED, None, _FATAL_FAILURES),
    )
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            for outcome in outcomes:
                transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)
                transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)
    finally:
        repository.close()

    assert _rows(
        database_path,
        """SELECT parse_id, parse_status, parser_implementation,
            parser_contract_version, normalisation_version, parsed_record_id
        FROM parse_outcomes ORDER BY parse_id""",
    ) == [
        (
            PARSE_ID.as_str(),
            "succeeded",
            "text-native",
            "2.0.0",
            "1.0.0",
            PARSED_RECORD_ID.as_str(),
        ),
        (
            _RECOVERABLE_PARSE_ID.as_str(),
            "completed_with_recoverable_failures",
            "text-native",
            "2.0.0",
            "1.0.0",
            PARSED_RECORD_ID.as_str(),
        ),
        (_FAILED_PARSE_ID.as_str(), "failed", "text-native", "2.0.0", "1.0.0", None),
    ]
    assert _rows(
        database_path,
        """SELECT parse_id, ordinal, failure_category, failure_severity,
            failure_stage, unit_kind, unit_index
        FROM parse_failures ORDER BY parse_id, ordinal""",
    ) == [
        (
            _RECOVERABLE_PARSE_ID.as_str(),
            0,
            "malformed_unit",
            "recoverable",
            "structure",
            "paragraph",
            4,
        ),
        (
            _RECOVERABLE_PARSE_ID.as_str(),
            1,
            "unsupported_structure",
            "recoverable",
            "structure",
            "table",
            1,
        ),
        (
            _FAILED_PARSE_ID.as_str(),
            0,
            "invalid_encoding",
            "fatal",
            "input",
            None,
            None,
        ),
        (_FAILED_PARSE_ID.as_str(), 1, "invalid_output", "fatal", "output", None, None),
    ]


def test_outcome_and_failures_roll_back_with_their_transaction(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    repository = _open_repository(database_path)
    try:
        with pytest.raises(RuntimeError, match="primary"):
            with repository.transaction() as transaction:
                transaction.add_parse_outcome(
                    _CORPUS.corpus_id,
                    _outcome(
                        _RECOVERABLE_PARSE_ID,
                        ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
                        PARSED_RECORD_ID,
                        _RECOVERABLE_FAILURES,
                    ),
                )
                raise RuntimeError("primary")
    finally:
        repository.close()

    assert _rows(database_path, "SELECT parse_id FROM parse_outcomes") == []
    assert _rows(database_path, "SELECT parse_id FROM parse_failures") == []


def test_all_outcome_states_round_trip_after_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    outcomes = (
        _outcome(PARSE_ID, ParseStatus.SUCCEEDED, PARSED_RECORD_ID, ()),
        _outcome(
            _RECOVERABLE_PARSE_ID,
            ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
            PARSED_RECORD_ID,
            _RECOVERABLE_FAILURES,
        ),
        _outcome(_FAILED_PARSE_ID, ParseStatus.FAILED, None, _FATAL_FAILURES),
    )
    repository = _open_repository(database_path)
    with repository.transaction() as transaction:
        for outcome in outcomes:
            transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)
    repository.close()

    reopened = _open_repository(database_path)
    try:
        for outcome in outcomes:
            assert (
                reopened.get_parse_outcome(
                    _CORPUS.corpus_id,
                    outcome.parse_id,
                )
                == outcome
            )
        with reopened.transaction() as transaction:
            assert (
                transaction.get_parse_outcome(_CORPUS.corpus_id, PARSE_ID)
                == outcomes[0]
            )
    finally:
        reopened.close()


def test_outcome_reads_do_not_disclose_absence_or_other_corpus_data(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    outcome = _outcome(PARSE_ID, ParseStatus.SUCCEEDED, PARSED_RECORD_ID, ())
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)

        assert repository.get_parse_outcome(_OTHER_CORPUS.corpus_id, PARSE_ID) is None
        assert repository.get_parse_outcome(_CORPUS.corpus_id, _FAILED_PARSE_ID) is None
    finally:
        repository.close()


@pytest.mark.parametrize(
    "corruption_sql",
    [
        "UPDATE parse_failures SET ordinal = ordinal + 4",
        "UPDATE parse_outcomes SET parser_contract_version = '3.0.0'",
    ],
    ids=("non-contiguous-failures", "mismatched-representation"),
)
def test_corrupt_outcomes_fail_closed_during_reconstruction(
    tmp_path: Path,
    corruption_sql: str,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    outcome = _outcome(
        _RECOVERABLE_PARSE_ID,
        ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
        PARSED_RECORD_ID,
        _RECOVERABLE_FAILURES,
    )
    repository = _open_repository(database_path)
    with repository.transaction() as transaction:
        transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(corruption_sql)
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(CorruptStorageError):
            repository.get_parse_outcome(
                _CORPUS.corpus_id,
                _RECOVERABLE_PARSE_ID,
            )
    finally:
        repository.close()


def test_parse_identity_reuse_with_different_failures_conflicts(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    original = _outcome(_FAILED_PARSE_ID, ParseStatus.FAILED, None, _FATAL_FAILURES)
    conflicting = replace(original, failures=(_FATAL_FAILURES[1],))
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parse_outcome(_CORPUS.corpus_id, original)

        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_parse_outcome(_CORPUS.corpus_id, conflicting)

        assert (
            repository.get_parse_outcome(
                _CORPUS.corpus_id,
                _FAILED_PARSE_ID,
            )
            == original
        )
    finally:
        repository.close()


@pytest.mark.parametrize(
    "outcome",
    [
        _outcome(PARSE_ID, ParseStatus.SUCCEEDED, _MISSING_PARSED_RECORD_ID, ()),
        replace(
            _outcome(PARSE_ID, ParseStatus.SUCCEEDED, PARSED_RECORD_ID, ()),
            parser=ProducerIdentity("text-native", "3.0.0"),
        ),
    ],
    ids=("missing-representation", "mismatched-representation"),
)
def test_outcome_writes_require_an_agreeing_representation(
    tmp_path: Path,
    outcome: ParseOutcome,
) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    repository = _open_repository(database_path)
    try:
        with pytest.raises(StorageConflictError):
            with repository.transaction() as transaction:
                transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)
    finally:
        repository.close()

    assert _rows(database_path, "SELECT parse_id FROM parse_outcomes") == []
    assert _rows(database_path, "SELECT parse_id FROM parse_failures") == []


def test_outcome_reuse_requires_explicit_membership(tmp_path: Path) -> None:
    database_path = tmp_path / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    outcome = _outcome(_FAILED_PARSE_ID, ParseStatus.FAILED, None, _FATAL_FAILURES)
    repository = _open_repository(database_path)
    try:
        with repository.transaction() as transaction:
            transaction.add_parse_outcome(_CORPUS.corpus_id, outcome)

        for corpus_id in (_OTHER_CORPUS.corpus_id, _ABSENT_CORPUS_ID):
            with pytest.raises(StorageConflictError):
                with repository.transaction() as transaction:
                    transaction.add_parse_outcome(corpus_id, outcome)

        with repository.transaction() as transaction:
            transaction.add_corpus_document_version_membership(
                CorpusDocumentVersionMembership(
                    _OTHER_CORPUS.corpus_id,
                    DOCUMENT_ID,
                    DOCUMENT_VERSION_ID,
                )
            )
            transaction.add_parse_outcome(_OTHER_CORPUS.corpus_id, outcome)

        assert (
            repository.get_parse_outcome(
                _OTHER_CORPUS.corpus_id,
                _FAILED_PARSE_ID,
            )
            == outcome
        )
    finally:
        repository.close()


def test_failure_payload_and_source_data_are_absent_from_errors_and_logs(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "private-path" / "metadata.sqlite3"
    _add_catalogue_and_representation(database_path)
    repository = _open_repository(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            f"""CREATE TRIGGER reject_parse_failure
            BEFORE INSERT ON parse_failures
            BEGIN
                SELECT RAISE(ABORT, '{_PRIVATE_TRIGGER_PAYLOAD}');
            END"""  # noqa: S608
        )
        connection.commit()
    finally:
        connection.close()

    try:
        with capture_logs() as logs:
            with pytest.raises(StorageConflictError) as raised:
                with repository.transaction() as transaction:
                    transaction.add_parse_outcome(
                        _CORPUS.corpus_id,
                        _outcome(
                            _RECOVERABLE_PARSE_ID,
                            ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
                            PARSED_RECORD_ID,
                            _RECOVERABLE_FAILURES,
                        ),
                    )

        assert str(raised.value) == StorageConflictError.default_message
        assert logs == [
            {
                "event": "transaction_failed",
                "pipeline_stage": "storage",
                "error_category": "conflict",
                "retryable": False,
                "log_level": "error",
            }
        ]
        for private_value in (_PRIVATE_TRIGGER_PAYLOAD, "private-path", SOURCE_TEXT):
            assert private_value not in str(raised.value)
            assert private_value not in repr(logs)
    finally:
        repository.close()

    assert _rows(database_path, "SELECT parse_id FROM parse_outcomes") == []
    assert _rows(database_path, "SELECT parse_id FROM parse_failures") == []


def test_repository_exposes_no_parse_outcome_writer(tmp_path: Path) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        assert not hasattr(repository, "add_parse_outcome")
    finally:
        repository.close()
