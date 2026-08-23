"""Tests parser error handling."""

import pytest

from inference_lab.ingest import ParserFailure


def test_fatal_failure_is_document_level() -> None:
    failure = ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )

    assert failure.unit_index is None


def test_recoverable_failure_identifies_omitted_unit() -> None:
    failure = ParserFailure(
        category="malformed_unit",
        severity="recoverable",
        stage="structure",
        unit_kind="paragraph",
        unit_index=2,
    )

    assert failure.unit_kind == "paragraph"
    assert failure.unit_index == 2


def test_failure_rejects_invalid_unit_kind() -> None:
    with pytest.raises(ValueError, match="invalid_failure_unit_kind"):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
            unit_kind="source text",  # type: ignore[arg-type]
            unit_index=0,
        )


def test_failure_rejects_negative_unit_index() -> None:
    with pytest.raises(ValueError, match="failure_unit_index_must_be_non_negative"):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
            unit_kind="paragraph",
            unit_index=-1,
        )


def test_fatal_failure_cannot_identify_partial_unit() -> None:
    with pytest.raises(
        ValueError,
        match="fatal_failure_must_not_identify_partial_unit",
    ):
        ParserFailure(
            category="decode_failed",
            severity="fatal",
            stage="decode",
            unit_kind="paragraph",
            unit_index=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("category", "unknown", "invalid_parser_failure_category"),
        ("severity", "warning", "invalid_parser_failure_severity"),
        ("stage", "parse", "invalid_parser_failure_stage"),
    ],
)
def test_failure_rejects_unknown_discriminants(
    field: str,
    value: str,
    category: str,
) -> None:
    values = {
        "category": "decode_failed",
        "severity": "fatal",
        "stage": "decode",
    }
    values[field] = value

    with pytest.raises(ValueError, match=category):
        ParserFailure(**values)  # type: ignore[arg-type]


def test_failure_rejects_category_policy_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="failure_stage_does_not_match_category",
    ):
        ParserFailure(
            category="decode_failed",
            severity="fatal",
            stage="input",
        )

    with pytest.raises(
        ValueError,
        match="failure_severity_does_not_match_category",
    ):
        ParserFailure(
            category="decode_failed",
            severity="recoverable",
            stage="decode",
        )


def test_recoverable_failure_requires_complete_unit_location() -> None:
    with pytest.raises(
        ValueError,
        match="failure_unit_location_must_be_complete",
    ):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
            unit_kind="paragraph",
        )

    with pytest.raises(
        ValueError,
        match="recoverable_failure_requires_unit_location",
    ):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
        )
