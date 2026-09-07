"""Tests for parser versioning."""

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
    generate_source_span_id,
)
from inference_lab.provenance import DocumentVersionId


def test_parser_version_has_canonical_value() -> None:
    assert CURRENT_PARSER_VERSION.value == (
        "text-native/contract-3.0.0/normalisation-1.0.0"
    )


def test_source_span_id_generation_is_deterministic() -> None:
    document_version_id = DocumentVersionId.generate()

    first = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )
    second = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )

    assert first == second


def test_source_span_id_changes_for_different_representation() -> None:
    document_version_id = DocumentVersionId.generate()

    first = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )
    second = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=1,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )

    assert first != second


def test_source_span_id_is_namespaced_by_document_version() -> None:
    document_version_id = DocumentVersionId.generate()
    other_document_version_id = DocumentVersionId.generate()

    first = generate_source_span_id(
        other_document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )
    second = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=5,
    )

    assert first != second


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("implementation", "", "parser_implementation_must_be_slug"),
        ("implementation", "Text Native", "parser_implementation_must_be_slug"),
        (
            "contract_version",
            "1.0",
            "parser_contract_version_must_be_semantic_version",
        ),
        (
            "contract_version",
            "01.0.0",
            "parser_contract_version_must_be_semantic_version",
        ),
        (
            "normalisation_version",
            "1.0.0-01",
            "normalisation_version_must_be_semantic_version",
        ),
    ],
)
def test_parser_version_rejects_noncanonical_components(
    field: str,
    value: str,
    category: str,
) -> None:
    values = {
        "implementation": "text-native",
        "contract_version": "1.0.0",
        "normalisation_version": "1.0.0",
    }
    values[field] = value

    with pytest.raises(ValueError, match=category):
        ParserVersion(**values)


def test_parser_version_accepts_semantic_prerelease_and_build_metadata() -> None:
    version = ParserVersion(
        implementation="text-native",
        contract_version="1.0.0-rc.1",
        normalisation_version="2.1.0+unicode-15.1",
    )

    assert version.value == (
        "text-native/contract-1.0.0-rc.1/normalisation-2.1.0+unicode-15.1"
    )


def test_source_span_id_changes_with_parser_version() -> None:
    document_version_id = DocumentVersionId.generate()
    other_parser_version = ParserVersion(
        implementation="text-native",
        contract_version="1.0.1",
        normalisation_version="1.0.0",
    )

    first = generate_source_span_id(
        document_version_id,
        parser_version=CURRENT_PARSER_VERSION,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=1,
    )
    second = generate_source_span_id(
        document_version_id,
        parser_version=other_parser_version,
        ordinal=0,
        kind="paragraph",
        start_offset=0,
        end_offset=1,
    )

    assert first != second


def test_source_span_id_rejects_invalid_parser_version() -> None:
    with pytest.raises(ValueError, match="invalid_parser_version"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version="1.0.0",  # type: ignore[arg-type]
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        )


def test_source_span_id_rejects_invalid_document_version_id() -> None:
    with pytest.raises(ValueError, match="invalid_document_version_id"):
        generate_source_span_id(
            "not-a-version-id",  # type: ignore[arg-type]
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        )


def test_source_span_id_rejects_invalid_ordinal() -> None:
    with pytest.raises(ValueError, match="source_span_ordinal_must_be_integer"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0.5,  # type: ignore[arg-type]
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        )


def test_source_span_id_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError, match="source_span_kind_must_be_string"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind=1,  # type: ignore[arg-type]
            start_offset=0,
            end_offset=1,
        )

    with pytest.raises(ValueError, match="source_span_kind_must_not_be_empty"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="   ",
            start_offset=0,
            end_offset=1,
        )


def test_source_span_id_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError, match="source_span_offsets_must_be_integers"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=0.5,  # type: ignore[arg-type]
            end_offset=1,
        )

    with pytest.raises(ValueError, match="source_span_start_must_be_non_negative"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=-1,
            end_offset=1,
        )

    with pytest.raises(ValueError, match="source_span_end_must_not_precede_start"):
        generate_source_span_id(
            DocumentVersionId.generate(),
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=2,
            end_offset=1,
        )
