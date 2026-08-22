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


def test_failure_rejects_empty_unit_kind() -> None:
    with pytest.raises(ValueError, match="failure_unit_kind_must_not_be_empty"):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
            unit_kind=" ",
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
            unit_kind="document",
            unit_index=0,
        )
