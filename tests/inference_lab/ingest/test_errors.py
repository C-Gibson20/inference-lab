"""Phase 2 ING-01 failure records: document failures and omitted units."""

import pytest

from inference_lab.ingest import ParserFailure


def test_parser_failures_distinguish_fatal_and_recoverable_units() -> None:
    fatal = ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )
    recoverable = ParserFailure(
        category="malformed_unit",
        severity="recoverable",
        stage="structure",
        unit_kind="paragraph",
        unit_index=2,
    )

    assert fatal.unit_kind is None
    assert fatal.unit_index is None
    assert recoverable.unit_kind == "paragraph"
    assert recoverable.unit_index == 2


def test_parser_failure_contract_rejects_invalid_category_and_unit_location() -> None:
    with pytest.raises(ValueError, match="invalid_parser_failure_category"):
        ParserFailure(
            category="unknown",  # type: ignore[arg-type]
            severity="fatal",
            stage="decode",
        )

    with pytest.raises(ValueError, match="failure_unit_location_must_be_complete"):
        ParserFailure(
            category="malformed_unit",
            severity="recoverable",
            stage="structure",
            unit_kind="paragraph",
        )
