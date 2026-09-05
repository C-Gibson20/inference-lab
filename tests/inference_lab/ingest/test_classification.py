"""Tests for versioned, content-free source classification."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    CURRENT_SOURCE_CLASSIFIER,
    FileIdentityHint,
    FileStatEvidence,
    SourceClassification,
    SourceClassifierRegistry,
    SourceClassifierRule,
    SourceLocator,
    SourceSupport,
    TraversalCandidate,
    classify_source_candidate,
)
from inference_lab.provenance import ExternalIdentifier


def _candidate(
    name: str,
    *,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> TraversalCandidate:
    return TraversalCandidate(
        ordinal=0,
        locator=SourceLocator(
            path=Path("private") / name,
            request_root_ordinal=0,
            relative_path=Path(name),
        ),
        stat_evidence=FileStatEvidence(
            byte_count=10,
            modified_time_ns=20,
            identity_hint=FileIdentityHint(device_id=30, file_id=40),
        ),
        external_identifiers=external_identifiers,
    )


@pytest.mark.parametrize("name", ["source.txt", "source.TXT", "source.TxT"])
def test_text_suffix_selects_utf8_text_native_capability(name: str) -> None:
    result = classify_source_candidate(_candidate(name))

    assert result.media_type == "text/plain"
    assert result.support is SourceSupport.SUPPORTED
    assert result.parser_capability == "text-native"
    assert result.encoding == "utf-8"


@pytest.mark.parametrize("name", ["paper.pdf", "paper.PDF", "paper.PdF"])
def test_pdf_is_recognised_without_claiming_parser_support(name: str) -> None:
    result = classify_source_candidate(_candidate(name))

    assert result.media_type == "application/pdf"
    assert result.support is SourceSupport.RECOGNISED_UNSUPPORTED
    assert result.parser_capability is None
    assert result.encoding is None


@pytest.mark.parametrize(
    "name",
    ["source", "source.bin", "source.ＴＸＴ", ".txt", "source.txt."],
)
def test_unknown_or_unicode_lookalike_suffix_uses_binary_fallback(name: str) -> None:
    result = classify_source_candidate(_candidate(name))

    assert result.media_type == "application/octet-stream"
    assert result.support is SourceSupport.UNKNOWN
    assert result.parser_capability is None
    assert result.encoding is None


def test_suffix_classification_is_advisory_and_content_free() -> None:
    candidate = _candidate("misleading.txt")

    result = classify_source_candidate(candidate)

    assert result.support is SourceSupport.SUPPORTED
    assert result.encoding == "utf-8"
    assert candidate.stat_evidence.byte_count == 10


def test_filename_does_not_create_or_replace_source_metadata() -> None:
    explicit = ExternalIdentifier("doi", "10.1000/explicit")
    candidate = _candidate(
        "doi-10.9999-inferred.txt",
        external_identifiers=(explicit,),
    )

    classify_source_candidate(candidate)

    assert candidate.external_identifiers == (explicit,)
    assert candidate.family_candidates == ()


def test_current_registry_matches_discovery_policy_identity() -> None:
    result = classify_source_candidate(_candidate("source.txt"))

    assert CURRENT_SOURCE_CLASSIFIER.classifier_name == (
        CURRENT_DISCOVERY_POLICY.classifier_name
    )
    assert CURRENT_SOURCE_CLASSIFIER.classifier_version == (
        CURRENT_DISCOVERY_POLICY.classifier_version
    )
    assert result.classifier_name == CURRENT_SOURCE_CLASSIFIER.classifier_name
    assert result.classifier_version == CURRENT_SOURCE_CLASSIFIER.classifier_version


def test_custom_registry_version_is_retained_in_result() -> None:
    classification = SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name="custom-classifier",
        classifier_version="2.0.0",
        parser_capability="text-native",
        encoding="utf-8",
    )
    registry = SourceClassifierRegistry(
        classifier_name="custom-classifier",
        classifier_version="2.0.0",
        rules=(SourceClassifierRule(".txt", classification),),
    )

    result = classify_source_candidate(_candidate("source.TXT"), registry=registry)

    assert result is classification
    assert result.classifier_version == "2.0.0"


def test_registry_and_rules_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        CURRENT_SOURCE_CLASSIFIER.classifier_version = "2.0.0"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(FrozenInstanceError):
        CURRENT_SOURCE_CLASSIFIER.rules[0].suffix = ".txt"  # pyright: ignore[reportAttributeAccessIssue]


def test_registry_rejects_invalid_rules_and_identity_mismatch() -> None:
    text_rule = CURRENT_SOURCE_CLASSIFIER.rules[1]
    with pytest.raises(ValueError, match="unique and ordered"):
        SourceClassifierRegistry(
            classifier_name=CURRENT_SOURCE_CLASSIFIER.classifier_name,
            classifier_version=CURRENT_SOURCE_CLASSIFIER.classifier_version,
            rules=(text_rule, text_rule),
        )
    with pytest.raises(ValueError, match="match the registry identity"):
        SourceClassifierRegistry(
            classifier_name="different-classifier",
            classifier_version="1.0.0",
            rules=(text_rule,),
        )
    with pytest.raises(ValueError, match="lowercase ASCII suffix"):
        SourceClassifierRule(".ＴＸＴ", text_rule.classification)
    unknown = SourceClassification(
        media_type="application/octet-stream",
        support=SourceSupport.UNKNOWN,
        classifier_name=CURRENT_SOURCE_CLASSIFIER.classifier_name,
        classifier_version=CURRENT_SOURCE_CLASSIFIER.classifier_version,
    )
    with pytest.raises(ValueError, match="recognised formats"):
        SourceClassifierRule(".bin", unknown)


def test_registry_rejects_mutable_empty_or_untyped_rule_collections() -> None:
    identity = {
        "classifier_name": CURRENT_SOURCE_CLASSIFIER.classifier_name,
        "classifier_version": CURRENT_SOURCE_CLASSIFIER.classifier_version,
    }
    with pytest.raises(TypeError, match="rules must be a tuple"):
        SourceClassifierRegistry(
            **identity,
            rules=[],  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="at least one"):
        SourceClassifierRegistry(**identity, rules=())
    with pytest.raises(TypeError, match="SourceClassifierRule values"):
        SourceClassifierRegistry(
            **identity,
            rules=(object(),),  # pyright: ignore[reportArgumentType]
        )


def test_rule_rejects_untyped_suffix_and_classification() -> None:
    text_rule = CURRENT_SOURCE_CLASSIFIER.rules[1]
    with pytest.raises(TypeError, match="suffix must be a string"):
        SourceClassifierRule(
            1,  # pyright: ignore[reportArgumentType]
            text_rule.classification,
        )
    with pytest.raises(TypeError, match="classification must be"):
        SourceClassifierRule(
            ".txt",
            object(),  # pyright: ignore[reportArgumentType]
        )


def test_classification_boundary_requires_typed_candidate_and_registry() -> None:
    with pytest.raises(TypeError, match="candidate must be a TraversalCandidate"):
        classify_source_candidate(
            object(),  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(TypeError, match="registry must be"):
        classify_source_candidate(
            _candidate("source.txt"),
            registry=object(),  # pyright: ignore[reportArgumentType]
        )
