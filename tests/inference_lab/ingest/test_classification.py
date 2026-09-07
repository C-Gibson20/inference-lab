"""Tests for versioned, content-free source classification."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import BinaryIO

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    CURRENT_SOURCE_CLASSIFIER,
    ClassificationInventory,
    ClassifiedSourceCandidate,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FileIdentityHint,
    FileStatEvidence,
    LocalStableReadFileSystem,
    SourceClassification,
    SourceClassifierRegistry,
    SourceClassifierRule,
    SourceFormatCount,
    SourceLocator,
    SourceSupport,
    TraversalCandidate,
    TraversalInventory,
    VerifiedSourceSnapshot,
    classify_source_candidate,
    classify_source_inventory,
    hash_source_candidate,
    inventory_local_sources,
)
from inference_lab.provenance import CorpusId, ExternalIdentifier


class _OpenRecordingFileSystem(LocalStableReadFileSystem):
    """Record opened paths without retaining any source bytes."""

    def __init__(self) -> None:
        self.opened_paths: list[Path] = []

    def open_binary(self, path: Path) -> BinaryIO:
        self.opened_paths.append(path)
        return super().open_binary(path)


def _candidate(
    name: str,
    *,
    ordinal: int = 0,
    byte_count: int = 10,
    external_identifiers: tuple[ExternalIdentifier, ...] = (),
) -> TraversalCandidate:
    return TraversalCandidate(
        ordinal=ordinal,
        locator=SourceLocator(
            path=Path("private") / name,
            request_root_ordinal=0,
            relative_path=Path(name),
        ),
        stat_evidence=FileStatEvidence(
            byte_count=byte_count,
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


def test_inventory_classifies_every_file_and_exposes_only_supported_for_hashing() -> (
    None
):
    traversal = TraversalInventory(
        root_count=1,
        candidates=(
            _candidate("paper.pdf", ordinal=0, byte_count=20),
            _candidate("notes.txt", ordinal=1, byte_count=10),
            _candidate("unknown.bin", ordinal=2, byte_count=7),
            _candidate("appendix.TXT", ordinal=3, byte_count=5),
        ),
        failures=(),
        complete_root_ordinals=(0,),
    )

    with capture_logs() as logs:
        result = classify_source_inventory(traversal)

    assert tuple(
        item.classification.support for item in result.classified_candidates
    ) == (
        SourceSupport.RECOGNISED_UNSUPPORTED,
        SourceSupport.SUPPORTED,
        SourceSupport.UNKNOWN,
        SourceSupport.SUPPORTED,
    )
    assert tuple(item.candidate.ordinal for item in result.hashable_candidates) == (
        1,
        3,
    )
    assert tuple(failure.ordinal for failure in result.failures) == (0, 2)
    assert tuple(failure.category for failure in result.failures) == (
        DiscoveryFailureCategory.RECOGNISED_UNSUPPORTED_FORMAT,
        DiscoveryFailureCategory.UNKNOWN_FORMAT,
    )
    assert tuple(failure.media_type for failure in result.failures) == (
        "application/pdf",
        "application/octet-stream",
    )
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("source_skipped", "info"),
        ("source_skipped", "info"),
    ]
    assert [record["source_ordinal"] for record in logs] == [0, 2]
    assert [record["support"] for record in logs] == [
        "recognised_unsupported",
        "unknown",
    ]
    assert [record["media_type"] for record in logs] == [
        "application/pdf",
        "application/octet-stream",
    ]
    log_payload = repr(logs)
    assert "paper.pdf" not in log_payload
    assert "unknown.bin" not in log_payload


def test_format_counts_aggregate_only_safe_stat_metadata() -> None:
    traversal = TraversalInventory(
        root_count=1,
        candidates=(
            _candidate("paper.pdf", ordinal=0, byte_count=20),
            _candidate("notes.txt", ordinal=1, byte_count=10),
            _candidate("unknown.bin", ordinal=2, byte_count=7),
            _candidate("appendix.TXT", ordinal=3, byte_count=5),
        ),
        failures=(),
        complete_root_ordinals=(0,),
    )

    result = classify_source_inventory(traversal)

    assert result.format_counts == (
        SourceFormatCount(
            media_type="application/octet-stream",
            support=SourceSupport.UNKNOWN,
            file_count=1,
            total_bytes=7,
        ),
        SourceFormatCount(
            media_type="application/pdf",
            support=SourceSupport.RECOGNISED_UNSUPPORTED,
            file_count=1,
            total_bytes=20,
        ),
        SourceFormatCount(
            media_type="text/plain",
            support=SourceSupport.SUPPORTED,
            file_count=2,
            total_bytes=15,
        ),
    )


def test_existing_traversal_failure_is_retained_in_ordinal_order() -> None:
    locator = SourceLocator(
        path=Path("private/linked"),
        request_root_ordinal=0,
        relative_path=Path("linked"),
    )
    traversal_failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.SOURCE_SYMLINK,
        locator=locator,
        file_kind=DiscoveryFileKind.SYMLINK,
    )
    traversal = TraversalInventory(
        root_count=1,
        candidates=(_candidate("unknown.bin", ordinal=1),),
        failures=(traversal_failure,),
        complete_root_ordinals=(0,),
    )

    result = classify_source_inventory(traversal)

    assert result.failures[0] is traversal_failure
    assert result.failures[1].category is DiscoveryFailureCategory.UNKNOWN_FORMAT
    assert tuple(failure.ordinal for failure in result.failures) == (0, 1)
    assert result.hashable_candidates == ()


def test_empty_traversal_retains_policy_and_completion_evidence() -> None:
    traversal = TraversalInventory(
        root_count=1,
        candidates=(),
        failures=(),
        complete_root_ordinals=(0,),
    )

    result = classify_source_inventory(traversal)

    assert result.classifier_name == CURRENT_SOURCE_CLASSIFIER.classifier_name
    assert result.classifier_version == CURRENT_SOURCE_CLASSIFIER.classifier_version
    assert result.classified_candidates == ()
    assert result.hashable_candidates == ()
    assert result.failures == ()
    assert result.format_counts == ()
    assert result.traversal is traversal


def test_batch_result_rejects_incomplete_failure_or_count_accounting() -> None:
    traversal = TraversalInventory(
        root_count=1,
        candidates=(_candidate("unknown.bin"),),
        failures=(),
        complete_root_ordinals=(0,),
    )
    result = classify_source_inventory(traversal)

    with pytest.raises(ValueError, match="failures must exactly account"):
        replace(result, failures=())
    with pytest.raises(ValueError, match="format_counts must match"):
        replace(result, format_counts=())


def test_batch_value_contracts_reject_invalid_types_and_counts() -> None:
    candidate = _candidate("source.txt")
    classification = classify_source_candidate(candidate)
    with pytest.raises(TypeError, match="candidate must be a TraversalCandidate"):
        ClassifiedSourceCandidate(
            candidate=object(),  # pyright: ignore[reportArgumentType]
            classification=classification,
        )
    with pytest.raises(TypeError, match="classification must be"):
        ClassifiedSourceCandidate(
            candidate=candidate,
            classification=object(),  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(TypeError, match="file_count must be an integer"):
        SourceFormatCount("text/plain", SourceSupport.SUPPORTED, True, 1)
    with pytest.raises(ValueError, match="file_count must be positive"):
        SourceFormatCount("text/plain", SourceSupport.SUPPORTED, 0, 0)
    with pytest.raises(ValueError, match="total_bytes must be non-negative"):
        SourceFormatCount("text/plain", SourceSupport.SUPPORTED, 1, -1)


def test_classification_inventory_requires_typed_traversal_and_registry() -> None:
    with pytest.raises(TypeError, match="traversal must be a TraversalInventory"):
        classify_source_inventory(
            object(),  # pyright: ignore[reportArgumentType]
        )
    traversal = TraversalInventory(
        root_count=1,
        candidates=(),
        failures=(),
        complete_root_ordinals=(0,),
    )
    with pytest.raises(TypeError, match="registry must be"):
        classify_source_inventory(
            traversal,
            registry=object(),  # pyright: ignore[reportArgumentType]
        )


def test_batch_representations_hide_private_source_names() -> None:
    marker = "confidential-unsupported-source-7741.pdf"
    traversal = TraversalInventory(
        root_count=1,
        candidates=(_candidate(marker),),
        failures=(),
        complete_root_ordinals=(0,),
    )

    result = classify_source_inventory(traversal)

    assert marker not in repr(result)
    assert set(ClassificationInventory.__dataclass_fields__) == {
        "classifier_name",
        "classifier_version",
        "traversal",
        "classified_candidates",
        "failures",
        "format_counts",
    }


def test_only_supported_classifications_reach_source_hashing(tmp_path: Path) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    (root / "a-paper.pdf").write_bytes(b"private pdf payload")
    (root / "b-unknown.bin").write_bytes(b"private binary payload")
    supported_path = root / "c-notes.txt"
    supported_path.write_bytes(b"supported text")
    traversal = inventory_local_sources(
        DiscoveryRequest(
            corpus_id=CorpusId.generate(),
            inputs=(DiscoveryInput(root, DiscoveryInputKind.DIRECTORY),),
        ),
        excluded_paths=(),
    )
    classified = classify_source_inventory(traversal)
    filesystem = _OpenRecordingFileSystem()

    hashed = tuple(
        hash_source_candidate(
            item,
            max_source_bytes=50 * 1024 * 1024,
            filesystem=filesystem,
        )
        for item in classified.hashable_candidates
    )

    assert len(hashed) == 1
    assert isinstance(hashed[0], VerifiedSourceSnapshot)
    assert filesystem.opened_paths == [supported_path.resolve()]
