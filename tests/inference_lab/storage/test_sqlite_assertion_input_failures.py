"""Negative input contracts for external-identifier assertion batches."""

from pathlib import Path
from typing import cast

import pytest

from inference_lab.provenance import (
    DocumentVersionId,
    ExternalIdentifier,
    ExternalIdentifierAssertion,
    SourceObservationId,
)
from tests.inference_lab.storage.input_failure_fixtures import (
    _CORPUS,
    _CORPUS_ID,
    _OTHER_SOURCE_ID,
    _SOURCE_ID,
    _VERSION_ID,
)

from .fixtures import open_repository as _open_repository  # noqa: TID252


def _assertion(
    source_id: SourceObservationId,
    version_id: DocumentVersionId,
    namespace: str,
) -> ExternalIdentifierAssertion:
    """Build a deterministic assertion for malformed-batch tests."""
    return ExternalIdentifierAssertion(
        source_id,
        version_id,
        ExternalIdentifier(namespace, f"value-{namespace}"),
    )


@pytest.mark.parametrize(
    ("assertions", "expected_error", "expected_message"),
    [
        ([], TypeError, "tuple"),
        ((object(),), TypeError, "ExternalIdentifierAssertion"),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "duplicates",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "pmid"),
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
            ),
            ValueError,
            "canonical order",
        ),
        (
            (
                _assertion(_SOURCE_ID, _VERSION_ID, "doi"),
                _assertion(_OTHER_SOURCE_ID, _VERSION_ID, "pmid"),
            ),
            ValueError,
            "one source-version association",
        ),
    ],
    ids=["list", "wrong-member", "duplicate", "order", "mixed-association"],
)
def test_malformed_assertion_batches_fail_before_persistence(
    tmp_path: Path,
    assertions: object,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    repository = _open_repository(tmp_path / "metadata.sqlite3")
    try:
        with pytest.raises(expected_error, match=expected_message):
            with repository.transaction() as transaction:
                transaction.add_corpus(_CORPUS)
                transaction.add_external_identifier_assertions(
                    _CORPUS_ID,
                    cast("tuple[ExternalIdentifierAssertion, ...]", assertions),
                )

        assert repository.get_corpus(_CORPUS_ID) is None
    finally:
        repository.close()
