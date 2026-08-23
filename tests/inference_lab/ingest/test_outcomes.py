"""Test parser outcomes."""

import pytest

from inference_lab.ingest.errors import ParserFailure
from inference_lab.ingest.outcomes import ParseOutcome

from .fixtures import fixture_parsed_document  # noqa: TID252


def _recoverable_failure() -> ParserFailure:
    return ParserFailure(
        category="malformed_unit",
        severity="recoverable",
        stage="structure",
        unit_kind="paragraph",
        unit_index=1,
    )


def _fatal_failure() -> ParserFailure:
    return ParserFailure(
        category="decode_failed",
        severity="fatal",
        stage="decode",
    )


def test_success_requires_document_and_no_failures() -> None:
    outcome = ParseOutcome(
        status="succeeded",
        document=fixture_parsed_document(),
        failures=(),
    )

    assert outcome.document is not None


def test_success_without_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="successful_parse_requires_document"):
        ParseOutcome(
            status="succeeded",
            document=None,
            failures=(),
        )


def test_success_rejects_failures() -> None:
    with pytest.raises(ValueError, match="successful_parse_must_not_have_failures"):
        ParseOutcome(
            status="succeeded",
            document=fixture_parsed_document(),
            failures=(_recoverable_failure(),),
        )


def test_partial_success_requires_recoverable_failure() -> None:
    outcome = ParseOutcome(
        status="completed_with_recoverable_failures",
        document=fixture_parsed_document(),
        failures=(_recoverable_failure(),),
    )

    assert outcome.document is not None


def test_partial_success_rejects_fatal_failure() -> None:
    with pytest.raises(
        ValueError,
        match="partial_parse_must_not_have_fatal_failures",
    ):
        ParseOutcome(
            status="completed_with_recoverable_failures",
            document=fixture_parsed_document(),
            failures=(_fatal_failure(),),
        )


def test_partial_success_requires_document() -> None:
    with pytest.raises(ValueError, match="partial_parse_requires_document"):
        ParseOutcome(
            status="completed_with_recoverable_failures",
            document=None,
            failures=(_recoverable_failure(),),
        )


def test_partial_success_requires_failures() -> None:
    with pytest.raises(ValueError, match="partial_parse_requires_failures"):
        ParseOutcome(
            status="completed_with_recoverable_failures",
            document=fixture_parsed_document(),
            failures=(),
        )


def test_failed_outcome_requires_fatal_failure_and_no_document() -> None:
    outcome = ParseOutcome(
        status="failed",
        document=None,
        failures=(_fatal_failure(),),
    )

    assert outcome.status == "failed"


def test_failed_outcome_cannot_contain_document() -> None:
    with pytest.raises(ValueError, match="failed_parse_must_not_have_document"):
        ParseOutcome(
            status="failed",
            document=fixture_parsed_document(),
            failures=(_fatal_failure(),),
        )


def test_failed_outcome_requires_failures() -> None:
    with pytest.raises(ValueError, match="failed_parse_requires_failures"):
        ParseOutcome(
            status="failed",
            document=None,
            failures=(),
        )


def test_failed_outcome_requires_fatal_failure() -> None:
    with pytest.raises(ValueError, match="failed_parse_requires_fatal_failure"):
        ParseOutcome(
            status="failed",
            document=None,
            failures=(_recoverable_failure(),),
        )


def test_parse_outcome_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="invalid_parse_status"):
        ParseOutcome(
            status="unknown",  # type: ignore[arg-type]
            document=None,
            failures=(),
        )


def test_parse_outcome_requires_immutable_failures() -> None:
    with pytest.raises(ValueError, match="parse_failures_must_be_tuple"):
        ParseOutcome(
            status="failed",
            document=None,
            failures=[_fatal_failure()],  # type: ignore[arg-type]
        )


def test_parse_outcome_rejects_invalid_document() -> None:
    with pytest.raises(ValueError, match="invalid_parsed_document"):
        ParseOutcome(
            status="succeeded",
            document="not-a-document",  # type: ignore[arg-type]
            failures=(),
        )


def test_parse_outcome_rejects_invalid_failure_member() -> None:
    with pytest.raises(ValueError, match="invalid_parser_failure"):
        ParseOutcome(
            status="failed",
            document=None,
            failures=("not-a-failure",),  # type: ignore[arg-type]
        )
