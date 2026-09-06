"""Tests for producer identity, parse outcome, and bounded failure contracts."""

from dataclasses import FrozenInstanceError

import pytest

from inference_lab.provenance import RecordId
from inference_lab.storage import (
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
    ProducerIdentity,
)
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    OCR_PRODUCER,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
)


def _fatal_failure() -> ParseFailureRecord:
    return ParseFailureRecord(
        category=ParseFailureCategory.DECODE_FAILED,
        severity=ParseFailureSeverity.FATAL,
        stage=ParseFailureStage.DECODE,
    )


def _recoverable_failure() -> ParseFailureRecord:
    return ParseFailureRecord(
        category=ParseFailureCategory.MALFORMED_UNIT,
        severity=ParseFailureSeverity.RECOVERABLE,
        stage=ParseFailureStage.STRUCTURE,
        unit_kind=ParseFailureUnitKind.PARAGRAPH,
        unit_index=2,
    )


def _outcome(
    *,
    status: ParseStatus = ParseStatus.SUCCEEDED,
    parsed_record_id: RecordId | None = PARSED_RECORD_ID,
    failures: tuple[ParseFailureRecord, ...] = (),
) -> ParseOutcome:
    return ParseOutcome(
        parse_id=PARSE_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        status=status,
        parser=PARSER,
        normalisation_version="1.0.0",
        parsed_record_id=parsed_record_id,
        failures=failures,
    )


def test_producer_identity_is_versioned_and_immutable() -> None:
    producer = ProducerIdentity(
        implementation="layout-parser",
        contract_version="1.2.0",
        artefact_identity="layout-model",
        artefact_revision="sha-abc",
    )

    assert producer.artefact_revision == "sha-abc"
    with pytest.raises(FrozenInstanceError):
        producer.contract_version = "2.0.0"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"implementation": "Layout Parser"}, ValueError),
        ({"contract_version": "1"}, ValueError),
        ({"artefact_identity": "model", "artefact_revision": None}, ValueError),
        ({"artefact_identity": None, "artefact_revision": "revision"}, ValueError),
        ({"artefact_identity": "\x00", "artefact_revision": "revision"}, ValueError),
        ({"implementation": 1}, TypeError),
    ],
)
def test_producer_identity_rejects_invalid_metadata(
    changes: dict[str, object], error_type: type[Exception]
) -> None:
    values: dict[str, object] = {
        "implementation": "text-native",
        "contract_version": "1.0.0",
        "artefact_identity": None,
        "artefact_revision": None,
    }
    values.update(changes)

    with pytest.raises(error_type):
        ProducerIdentity(**values)  # type: ignore[arg-type]


def test_bounded_failures_preserve_only_contract_fields() -> None:
    failure = _recoverable_failure()

    assert failure.unit_kind is ParseFailureUnitKind.PARAGRAPH
    assert failure.unit_index == 2
    assert not hasattr(failure, "message")
    assert not hasattr(failure, "exception")


@pytest.mark.parametrize(
    "failure",
    [
        {
            "category": ParseFailureCategory.DECODE_FAILED,
            "severity": ParseFailureSeverity.RECOVERABLE,
            "stage": ParseFailureStage.DECODE,
        },
        {
            "category": ParseFailureCategory.MALFORMED_UNIT,
            "severity": ParseFailureSeverity.RECOVERABLE,
            "stage": ParseFailureStage.STRUCTURE,
        },
        {
            "category": ParseFailureCategory.INVALID_INPUT,
            "severity": ParseFailureSeverity.FATAL,
            "stage": ParseFailureStage.INPUT,
            "unit_kind": ParseFailureUnitKind.PAGE,
            "unit_index": 0,
        },
        {
            "category": ParseFailureCategory.MALFORMED_UNIT,
            "severity": ParseFailureSeverity.RECOVERABLE,
            "stage": ParseFailureStage.STRUCTURE,
            "unit_kind": ParseFailureUnitKind.PAGE,
            "unit_index": -1,
        },
    ],
)
def test_bounded_failure_rejects_invalid_policy_combinations(
    failure: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ParseFailureRecord(**failure)  # type: ignore[arg-type]


def test_parse_outcomes_retain_success_recoverable_and_failure_distinctions() -> None:
    succeeded = _outcome()
    recoverable = _outcome(
        status=ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
        failures=(_recoverable_failure(),),
    )
    failed = _outcome(
        status=ParseStatus.FAILED,
        parsed_record_id=None,
        failures=(_fatal_failure(),),
    )

    assert succeeded.parsed_record_id == PARSED_RECORD_ID
    assert recoverable.failures[0].severity is ParseFailureSeverity.RECOVERABLE
    assert failed.parsed_record_id is None


def test_parse_outcome_retains_optional_ocr_producer_identity() -> None:
    outcome = ParseOutcome(
        parse_id=PARSE_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        status=ParseStatus.SUCCEEDED,
        parser=PARSER,
        normalisation_version="1.0.0",
        parsed_record_id=PARSED_RECORD_ID,
        failures=(),
        ocr_producer=OCR_PRODUCER,
    )

    assert outcome.ocr_producer == OCR_PRODUCER


@pytest.mark.parametrize(
    ("status", "parsed_record_id", "failures"),
    [
        (ParseStatus.SUCCEEDED, None, ()),
        (ParseStatus.SUCCEEDED, PARSED_RECORD_ID, (_recoverable_failure(),)),
        (ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES, PARSED_RECORD_ID, ()),
        (
            ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
            PARSED_RECORD_ID,
            (_fatal_failure(),),
        ),
        (ParseStatus.FAILED, PARSED_RECORD_ID, (_fatal_failure(),)),
        (ParseStatus.FAILED, None, ()),
        (ParseStatus.FAILED, None, (_recoverable_failure(),)),
    ],
)
def test_parse_outcome_rejects_inconsistent_status(
    status: ParseStatus,
    parsed_record_id: RecordId | None,
    failures: tuple[ParseFailureRecord, ...],
) -> None:
    with pytest.raises(ValueError):
        _outcome(
            status=status,
            parsed_record_id=parsed_record_id,
            failures=failures,
        )


@pytest.mark.parametrize(
    ("changes", "error_type", "match"),
    [
        ({"parse_id": "parse"}, TypeError, "parse_id"),
        ({"normalisation_version": "1"}, ValueError, "normalisation_version"),
        ({"failures": []}, TypeError, "failures"),
        ({"ocr_producer": "ocr"}, TypeError, "ocr_producer"),
        ({"record_schema_version": "2.0.0"}, ValueError, "record_schema_version"),
    ],
)
def test_parse_outcome_rejects_invalid_fields(
    changes: dict[str, object], error_type: type[Exception], match: str
) -> None:
    values: dict[str, object] = {
        "parse_id": PARSE_ID,
        "document_id": DOCUMENT_ID,
        "document_version_id": DOCUMENT_VERSION_ID,
        "status": ParseStatus.SUCCEEDED,
        "parser": PARSER,
        "normalisation_version": "1.0.0",
        "parsed_record_id": PARSED_RECORD_ID,
        "failures": (),
    }
    values.update(changes)

    with pytest.raises(error_type, match=match):
        ParseOutcome(**values)  # type: ignore[arg-type]
