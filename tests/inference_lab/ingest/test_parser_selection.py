"""Tests for versioned parser-capability selection."""

import pytest

from inference_lab.ingest import (
    CURRENT_PARSER_REGISTRY,
    CURRENT_PARSER_VERSION,
    ParserRegistration,
    ParserRegistry,
    ParserSelectionError,
    ParserSelectionErrorCategory,
    SourceClassification,
    SourceSupport,
    TextNativeParser,
    select_parser,
)
from inference_lab.ingest.versioning import ParserVersion


def _classification(
    *,
    support: SourceSupport = SourceSupport.SUPPORTED,
    capability: str | None = "text-native",
) -> SourceClassification:
    return SourceClassification(
        media_type="text/plain"
        if support is SourceSupport.SUPPORTED
        else "application/pdf",
        support=support,
        classifier_name="test-classifier",
        classifier_version="1.0.0",
        parser_capability=capability,
        encoding="utf-8" if support is SourceSupport.SUPPORTED else None,
    )


def test_current_registry_selects_text_native_parser() -> None:
    parser = select_parser(
        _classification(),
        parser_version=CURRENT_PARSER_VERSION,
    )

    assert isinstance(parser, TextNativeParser)
    assert CURRENT_PARSER_REGISTRY.registrations[0].capability == "text-native"


def test_selection_rejects_unsupported_source() -> None:
    with pytest.raises(ParserSelectionError) as raised:
        select_parser(
            _classification(
                support=SourceSupport.RECOGNISED_UNSUPPORTED,
                capability=None,
            ),
            parser_version=CURRENT_PARSER_VERSION,
        )

    assert raised.value.category is ParserSelectionErrorCategory.UNSUPPORTED_SOURCE
    assert "application/pdf" not in str(raised.value)


def test_selection_rejects_missing_capability() -> None:
    classification = _classification()
    object.__setattr__(classification, "parser_capability", None)

    with pytest.raises(ParserSelectionError) as raised:
        ParserRegistry(registrations=()).select(
            classification,
            parser_version=CURRENT_PARSER_VERSION,
        )

    assert raised.value.category is ParserSelectionErrorCategory.MISSING_CAPABILITY


def test_selection_rejects_unavailable_capability() -> None:
    with pytest.raises(ParserSelectionError) as raised:
        ParserRegistry(registrations=()).select(
            _classification(capability="other-parser"),
            parser_version=CURRENT_PARSER_VERSION,
        )

    assert raised.value.category is ParserSelectionErrorCategory.PARSER_UNAVAILABLE


def test_selection_rejects_parser_version_mismatch() -> None:
    registry = ParserRegistry(
        registrations=(
            ParserRegistration(
                capability="text-native",
                parser_version=CURRENT_PARSER_VERSION,
                parser=TextNativeParser(),
            ),
        )
    )
    requested = ParserVersion(
        implementation="text-native",
        contract_version="999.0.0",
        normalisation_version="1.0.0",
    )

    with pytest.raises(ParserSelectionError) as raised:
        registry.select(_classification(), parser_version=requested)

    assert raised.value.category is ParserSelectionErrorCategory.VERSION_MISMATCH


def test_registry_rejects_duplicate_or_noncanonical_registrations() -> None:
    registration = ParserRegistration(
        capability="text-native",
        parser_version=CURRENT_PARSER_VERSION,
        parser=TextNativeParser(),
    )
    with pytest.raises(ValueError, match="unique"):
        ParserRegistry(registrations=(registration, registration))

    with pytest.raises(TypeError, match="implement parse"):
        ParserRegistration(
            capability="text-native",
            parser_version=CURRENT_PARSER_VERSION,
            parser=object(),  # type: ignore[arg-type]
        )
