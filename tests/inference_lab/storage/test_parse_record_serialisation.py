"""Tests for deterministic and private parse-outcome serialisation."""

import json

import pytest
from structlog.testing import capture_logs

from inference_lab.storage import (
    ParseFailureCategory,
    ParseFailureRecord,
    ParseFailureSeverity,
    ParseFailureStage,
    ParseFailureUnitKind,
    ParseOutcome,
    ParseStatus,
    serialise_parse_outcome,
)
from inference_lab.storage.parse_serialisation import parse_outcome_payload
from tests.inference_lab.storage.source_fixtures import (
    DOCUMENT_ID,
    DOCUMENT_VERSION_ID,
    OCR_PRODUCER,
    PARSE_ID,
    PARSED_RECORD_ID,
    PARSER,
)


def _parse_outcome() -> ParseOutcome:
    return ParseOutcome(
        parse_id=PARSE_ID,
        document_id=DOCUMENT_ID,
        document_version_id=DOCUMENT_VERSION_ID,
        status=ParseStatus.COMPLETED_WITH_RECOVERABLE_FAILURES,
        parser=PARSER,
        normalisation_version="1.0.0",
        ocr_producer=OCR_PRODUCER,
        parsed_record_id=PARSED_RECORD_ID,
        failures=(
            ParseFailureRecord(
                category=ParseFailureCategory.UNSUPPORTED_STRUCTURE,
                severity=ParseFailureSeverity.RECOVERABLE,
                stage=ParseFailureStage.STRUCTURE,
                unit_kind=ParseFailureUnitKind.TABLE,
                unit_index=3,
            ),
            ParseFailureRecord(
                category=ParseFailureCategory.MALFORMED_UNIT,
                severity=ParseFailureSeverity.RECOVERABLE,
                stage=ParseFailureStage.STRUCTURE,
                unit_kind=ParseFailureUnitKind.FIGURE_CAPTION,
                unit_index=7,
            ),
        ),
    )


def test_parse_outcome_serialisation_is_deterministic_and_ordered() -> None:
    outcome = _parse_outcome()

    first = serialise_parse_outcome(outcome)
    second = serialise_parse_outcome(outcome)
    payload = json.loads(first)

    assert first == second
    assert first.endswith(b"\n")
    assert payload["record_schema_version"] == "1.0.0"
    assert payload["parse_id"] == PARSE_ID.as_str()
    assert payload["parser"]["implementation"] == "text-native"
    assert payload["ocr_producer"]["artefact_revision"] == "revision-1"
    assert [failure["unit_index"] for failure in payload["failures"]] == [3, 7]
    assert set(payload["failures"][0]) == {
        "category",
        "severity",
        "stage",
        "unit_kind",
        "unit_index",
    }


def test_parse_outcome_payload_rejects_wrong_runtime_type() -> None:
    with pytest.raises(TypeError, match="ParseOutcome"):
        parse_outcome_payload(object())  # type: ignore[arg-type]


def test_parse_outcome_contract_and_serialisation_emit_no_logs() -> None:
    with capture_logs() as logs:
        serialise_parse_outcome(_parse_outcome())

    assert logs == []


def test_serialised_failure_cannot_contain_free_form_payloads() -> None:
    private_marker = "private-source-text-marker"
    encoded = serialise_parse_outcome(_parse_outcome())

    assert private_marker.encode() not in encoded
    assert b"message" not in encoded
    assert b"exception" not in encoded
