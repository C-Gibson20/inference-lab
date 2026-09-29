"""Focused ING-01 parser-version and source-span identity contracts."""

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_VERSION,
    ParserVersion,
    generate_source_span_id,
)
from inference_lab.provenance import DocumentVersionId, SourceSpanId


def test_parser_version_has_canonical_value_and_accepts_semver_metadata() -> None:
    assert CURRENT_PARSER_VERSION.value == (
        "text-native/contract-4.0.0/normalisation-1.0.0"
    )
    version = ParserVersion(
        implementation="text-native",
        contract_version="1.0.0-rc.1",
        normalisation_version="2.1.0+unicode-15.1",
    )
    assert version.value == (
        "text-native/contract-1.0.0-rc.1/normalisation-2.1.0+unicode-15.1"
    )


def test_parser_version_rejects_noncanonical_semver() -> None:
    with pytest.raises(
        ValueError,
        match="parser_contract_version_must_be_semantic_version",
    ):
        ParserVersion(
            implementation="text-native",
            contract_version="01.0.0",
            normalisation_version="1.0.0",
        )


def test_source_span_ids_are_deterministic_and_scoped_to_source_inputs() -> None:
    version = DocumentVersionId.generate()
    other_version = DocumentVersionId.generate()
    other_parser = ParserVersion(
        implementation="text-native",
        contract_version="1.0.1",
        normalisation_version="1.0.0",
    )

    def span_id(
        document_version: DocumentVersionId = version,
        parser: ParserVersion = CURRENT_PARSER_VERSION,
        *,
        ordinal: int = 0,
        kind: str = "paragraph",
        start: int = 0,
        end: int = 5,
    ) -> SourceSpanId:
        return generate_source_span_id(
            document_version,
            parser_version=parser,
            ordinal=ordinal,
            kind=kind,
            start_offset=start,
            end_offset=end,
        )

    original = span_id()
    assert span_id() == original
    assert span_id(other_version) != original
    assert span_id(parser=other_parser) != original
    assert span_id(ordinal=1) != original
    assert span_id(kind="heading") != original
    assert span_id(end=6) != original


def test_source_span_id_rejects_invalid_identity_and_location_inputs() -> None:
    version = DocumentVersionId.generate()
    with pytest.raises(ValueError, match="invalid_document_version_id"):
        generate_source_span_id(
            "not-a-version-id",  # type: ignore[arg-type]
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        )
    with pytest.raises(ValueError, match="source_span_ordinal_must_be_integer"):
        generate_source_span_id(
            version,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0.5,  # type: ignore[arg-type]
            kind="paragraph",
            start_offset=0,
            end_offset=1,
        )
    with pytest.raises(ValueError, match="source_span_kind_must_not_be_empty"):
        generate_source_span_id(
            version,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="   ",
            start_offset=0,
            end_offset=1,
        )
    with pytest.raises(ValueError, match="source_span_end_must_not_precede_start"):
        generate_source_span_id(
            version,
            parser_version=CURRENT_PARSER_VERSION,
            ordinal=0,
            kind="paragraph",
            start_offset=2,
            end_offset=1,
        )
