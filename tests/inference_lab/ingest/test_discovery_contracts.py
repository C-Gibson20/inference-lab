"""Tests for immutable source-discovery contracts and privacy boundaries."""

from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    CURRENT_SOURCE_CLASSIFIER,
    DirectorySnapshotPolicy,
    DiscoveredSource,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    FamilyCandidate,
    FamilyEvidenceKind,
    FileIdentityHint,
    FileStatEvidence,
    HiddenEntryPolicy,
    NestedMountPolicy,
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    StableReadPolicy,
    SymlinkPolicy,
    VerifiedSourceSnapshot,
    classify_source_inventory,
    inventory_local_sources,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentFamilyId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    HashAlgorithm,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
)

_DIGEST = "a" * 64


def _locator(marker: str = "confidential-source-name.txt") -> SourceLocator:
    return SourceLocator(
        path=Path("private") / marker,
        request_root_ordinal=0,
        relative_path=Path(marker),
    )


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _file_evidence(*, byte_count: int = 12) -> StableFileEvidence:
    stat = FileStatEvidence(
        byte_count=byte_count,
        modified_time_ns=123_456,
        identity_hint=FileIdentityHint(device_id=4, file_id=9),
    )
    return StableFileEvidence(
        pre_open=stat,
        opened=stat,
        opened_after_read=stat,
        path_after_read=stat,
    )


def _snapshot() -> VerifiedSourceSnapshot:
    return VerifiedSourceSnapshot(
        locator=_locator(),
        media_type="text/plain",
        encoding="utf-8",
        byte_count=12,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
        file_evidence=_file_evidence(),
    )


def _new_document_outcome() -> IdentityResolutionOutcome:
    document_id = DocumentId.generate()
    version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=document_id,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
    )
    return IdentityResolutionOutcome(
        kind=ResolutionKind.NEW_DOCUMENT,
        reason=ResolutionReason.NO_MATCH,
        document=DocumentIdentity(document_id),
        version=version,
    )


def _discovered_source() -> DiscoveredSource:
    return DiscoveredSource(
        ordinal=0,
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        source_observation_id=SourceObservationId.generate(),
        snapshot=_snapshot(),
        external_identifiers=(),
        family_candidates=(),
        classification=_classification(),
        identity_outcome=_new_document_outcome(),
    )


def test_current_policy_pins_every_approved_baseline_behaviour() -> None:
    assert CURRENT_DISCOVERY_POLICY.policy_name == "local-source-discovery"
    assert CURRENT_DISCOVERY_POLICY.policy_version == "1.0.0"
    assert CURRENT_DISCOVERY_POLICY.classifier_version == "2.0.0"
    assert CURRENT_DISCOVERY_POLICY.symlink_policy is SymlinkPolicy.EXCLUDE
    assert CURRENT_DISCOVERY_POLICY.nested_mount_policy is NestedMountPolicy.EXCLUDE
    assert CURRENT_DISCOVERY_POLICY.hidden_entry_policy is HiddenEntryPolicy.INCLUDE
    assert (
        CURRENT_DISCOVERY_POLICY.directory_snapshot_policy
        is DirectorySnapshotPolicy.SORTED
    )
    assert (
        CURRENT_DISCOVERY_POLICY.stable_read_policy
        is StableReadPolicy.PRE_OPEN_POST_STAT
    )


def test_current_classifier_admits_pdf_to_the_text_native_parser(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"%PDF-1.4")
    request = DiscoveryRequest(
        corpus_id=CorpusId.generate(),
        inputs=(DiscoveryInput(source_path, DiscoveryInputKind.FILE),),
    )

    traversal = inventory_local_sources(request, excluded_paths=())
    classification = classify_source_inventory(
        traversal,
        registry=CURRENT_SOURCE_CLASSIFIER,
    )

    assert classification.failures == ()
    assert len(classification.hashable_candidates) == 1
    pdf_classification = classification.hashable_candidates[0].classification
    assert pdf_classification.media_type == "application/pdf"
    assert pdf_classification.support is SourceSupport.SUPPORTED
    assert pdf_classification.parser_capability == "text-native"
    assert pdf_classification.encoding is None


@pytest.mark.parametrize("root", [Path("https:/example.test/source"), Path(" ")])
def test_discovery_input_rejects_non_local_path_shapes(root: Path) -> None:
    with pytest.raises(ValueError, match="root must name a local filesystem path"):
        DiscoveryInput(root=root, kind=DiscoveryInputKind.FILE)


def test_family_candidates_are_canonical_and_versions_require_documents() -> None:
    family_id = DocumentFamilyId.generate()
    with pytest.raises(ValueError, match="requires a document_id"):
        FamilyCandidate(
            document_family_id=family_id,
            evidence_kind=FamilyEvidenceKind.EXPLICIT_ASSOCIATION,
            document_version_id=DocumentVersionId.generate(),
        )

    first, second = sorted(
        (
            FamilyCandidate(
                document_family_id=DocumentFamilyId.generate(),
                evidence_kind=FamilyEvidenceKind.EXPLICIT_ASSOCIATION,
            ),
            FamilyCandidate(
                document_family_id=DocumentFamilyId.generate(),
                evidence_kind=FamilyEvidenceKind.EXPLICIT_ASSOCIATION,
            ),
        ),
        key=lambda candidate: candidate.key(),
    )
    with pytest.raises(ValueError, match="canonical order"):
        DiscoveryInput(
            root=Path("source.txt"),
            kind=DiscoveryInputKind.FILE,
            family_candidates=(second, first),
        )


def test_source_locator_rejects_escape_and_root_relative_sentinels() -> None:
    with pytest.raises(ValueError, match="remain inside"):
        SourceLocator(
            path=Path("private/source.txt"),
            request_root_ordinal=0,
            relative_path=Path("../source.txt"),
        )
    with pytest.raises(ValueError, match="root-relative entry"):
        SourceLocator(
            path=Path("private/source.txt"),
            request_root_ordinal=0,
            relative_path=Path("."),
        )


def test_unsupported_classification_cannot_claim_parser_details() -> None:
    with pytest.raises(ValueError, match="cannot declare parser capability"):
        SourceClassification(
            media_type="application/pdf",
            support=SourceSupport.RECOGNISED_UNSUPPORTED,
            classifier_name="baseline",
            classifier_version="1.0.0",
            parser_capability="pdf",
        )


def test_verified_snapshot_requires_supported_matching_stable_evidence() -> None:
    with pytest.raises(ValueError, match="snapshot byte count must match"):
        VerifiedSourceSnapshot(
            locator=_locator(),
            media_type="text/plain",
            encoding="utf-8",
            byte_count=11,
            content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
            file_evidence=_file_evidence(byte_count=12),
        )
