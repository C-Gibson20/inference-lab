"""Tests for discovery result status, ordering, and aggregate invariants."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveredSource,
    DiscoveryCounts,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryResult,
    DiscoveryStatus,
    FileIdentityHint,
    FileStatEvidence,
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    VerifiedSourceSnapshot,
)
from inference_lab.provenance import (
    ContentHash,
    CorpusId,
    DocumentId,
    DocumentIdentity,
    DocumentVersion,
    DocumentVersionId,
    DuplicateClass,
    DuplicateRelationship,
    HashAlgorithm,
    IdentityResolutionOutcome,
    ResolutionKind,
    ResolutionReason,
    SourceObservationId,
)

_DIGEST = "b" * 64


def _classification() -> SourceClassification:
    return SourceClassification(
        media_type="text/plain",
        support=SourceSupport.SUPPORTED,
        classifier_name=CURRENT_DISCOVERY_POLICY.classifier_name,
        classifier_version=CURRENT_DISCOVERY_POLICY.classifier_version,
        parser_capability="text-native",
        encoding="utf-8",
    )


def _snapshot(ordinal: int) -> VerifiedSourceSnapshot:
    classification = _classification()
    locator = SourceLocator(
        path=Path("private") / f"source-{ordinal}.txt",
        request_root_ordinal=0,
        relative_path=Path(f"source-{ordinal}.txt"),
    )
    stat = FileStatEvidence(
        byte_count=1,
        modified_time_ns=ordinal + 1,
        identity_hint=FileIdentityHint(device_id=1, file_id=ordinal + 1),
    )
    return VerifiedSourceSnapshot(
        locator=locator,
        media_type=classification.media_type,
        encoding=classification.encoding,
        byte_count=1,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
        file_evidence=StableFileEvidence(stat, stat, stat, stat),
    )


def _assigned_values() -> tuple[DocumentIdentity, DocumentVersion]:
    document_id = DocumentId.generate()
    version = DocumentVersion(
        document_version_id=DocumentVersionId.generate(),
        document_id=document_id,
        content_hash=ContentHash(HashAlgorithm.SHA256, _DIGEST),
    )
    return DocumentIdentity(document_id), version


def _outcome(kind: ResolutionKind) -> IdentityResolutionOutcome:
    document, version = _assigned_values()
    if kind is ResolutionKind.UNCHANGED:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.EXACT_HASH_MATCH,
            document=document,
            version=version,
            matched_version_ids=(version.document_version_id,),
        )
    if kind is ResolutionKind.NEW_VERSION:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.SOURCE_CONTINUITY_MATCH,
            document=document,
            version=version,
            matched_version_ids=(DocumentVersionId.generate(),),
        )
    if kind is ResolutionKind.NEW_DOCUMENT:
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.NO_MATCH,
            document=document,
            version=version,
        )
    if kind is ResolutionKind.DUPLICATE:
        matched_id = DocumentVersionId.generate()
        first_id, second_id = sorted(
            (version.document_version_id, matched_id),
            key=DocumentVersionId.as_str,
        )
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.DUPLICATE_HASH_MATCH,
            document=document,
            version=version,
            matched_version_ids=(matched_id,),
            duplicate_relationships=(
                DuplicateRelationship(
                    first_version_id=first_id,
                    second_version_id=second_id,
                    duplicate_class=DuplicateClass.BYTE_IDENTICAL,
                ),
            ),
        )
    if kind is ResolutionKind.AMBIGUOUS:
        matched_ids = tuple(
            sorted(
                (DocumentVersionId.generate(), DocumentVersionId.generate()),
                key=DocumentVersionId.as_str,
            )
        )
        return IdentityResolutionOutcome(
            kind=kind,
            reason=ResolutionReason.MULTIPLE_CANDIDATES,
            matched_version_ids=matched_ids,
        )
    return IdentityResolutionOutcome(
        kind=ResolutionKind.CONFLICT,
        reason=ResolutionReason.CONFLICTING_EVIDENCE,
    )


def _source(ordinal: int, kind: ResolutionKind) -> DiscoveredSource:
    return DiscoveredSource(
        ordinal=ordinal,
        source_observation_id=SourceObservationId.generate(),
        snapshot=_snapshot(ordinal),
        external_identifiers=(),
        family_candidates=(),
        classification=_classification(),
        identity_outcome=_outcome(kind),
    )


def test_empty_directory_is_a_successful_zero_source_result() -> None:
    result = DiscoveryResult(
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        status=DiscoveryStatus.COMPLETED,
        sources=(),
        failures=(),
        counts=DiscoveryCounts.empty(),
    )

    assert result.status is DiscoveryStatus.COMPLETED
    assert result.counts.files_seen == 0


def test_counts_factory_maps_every_identity_outcome_exactly() -> None:
    kinds = tuple(ResolutionKind)
    sources = tuple(_source(ordinal, kind) for ordinal, kind in enumerate(kinds))

    counts = DiscoveryCounts.from_outcomes(
        files_seen=len(sources),
        sources=sources,
        failures=(),
    )

    assert counts.hashed == 6
    assert counts.unchanged == 1
    assert counts.changed == 1
    assert counts.new == 1
    assert counts.duplicate == 1
    assert counts.ambiguous == 1
    assert counts.conflicting == 1


def test_partial_result_combines_usable_source_and_isolated_failure() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_TOO_LARGE,
    )
    result = DiscoveryResult(
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        status=DiscoveryStatus.PARTIAL,
        sources=(source,),
        failures=(failure,),
        counts=DiscoveryCounts.from_outcomes(
            files_seen=2,
            sources=(source,),
            failures=(failure,),
        ),
    )

    assert result.counts.supported == 2
    assert result.counts.failed == 1


def test_unsupported_only_scope_is_a_failed_result_without_operational_failure() -> (
    None
):
    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.UNKNOWN_FORMAT,
        media_type="application/octet-stream",
    )
    counts = DiscoveryCounts.from_outcomes(
        files_seen=1,
        sources=(),
        failures=(failure,),
    )
    result = DiscoveryResult(
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        status=DiscoveryStatus.FAILED,
        sources=(),
        failures=(failure,),
        counts=counts,
    )

    assert result.counts.unsupported == 1
    assert result.counts.failed == 0


def test_failed_result_can_retain_an_unusable_conflict_outcome() -> None:
    source = _source(0, ResolutionKind.CONFLICT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
    )
    result = DiscoveryResult(
        corpus_id=CorpusId.generate(),
        policy=CURRENT_DISCOVERY_POLICY,
        status=DiscoveryStatus.FAILED,
        sources=(source,),
        failures=(failure,),
        counts=DiscoveryCounts.from_outcomes(
            files_seen=2,
            sources=(source,),
            failures=(failure,),
        ),
    )

    assert result.counts.conflicting == 1


def test_request_failure_requires_failed_status_and_no_usable_source() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.REPOSITORY_UNAVAILABLE,
    )
    counts = DiscoveryCounts.from_outcomes(
        files_seen=1,
        sources=(source,),
        failures=(failure,),
    )

    with pytest.raises(ValueError, match="request failure requires a failed"):
        DiscoveryResult(
            CorpusId.generate(),
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.PARTIAL,
            (source,),
            (failure,),
            counts,
        )


def test_completed_partial_and_failed_status_invariants_are_enforced() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
    )
    counts = DiscoveryCounts.from_outcomes(
        files_seen=2,
        sources=(source,),
        failures=(failure,),
    )
    with pytest.raises(ValueError, match="completed discovery must not"):
        DiscoveryResult(
            CorpusId.generate(),
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.COMPLETED,
            (source,),
            (failure,),
            counts,
        )
    with pytest.raises(ValueError, match="failed discovery must not contain usable"):
        DiscoveryResult(
            CorpusId.generate(),
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.FAILED,
            (source,),
            (failure,),
            counts,
        )
    with pytest.raises(ValueError, match="partial discovery requires"):
        DiscoveryResult(
            CorpusId.generate(),
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.PARTIAL,
            (),
            (failure,),
            DiscoveryCounts.from_outcomes(
                files_seen=1,
                sources=(),
                failures=(failure,),
            ),
        )


def test_result_rejects_unordered_or_reused_ordinals() -> None:
    first = _source(1, ResolutionKind.NEW_DOCUMENT)
    second = _source(0, ResolutionKind.NEW_DOCUMENT)
    with pytest.raises(ValueError, match="sources must be in discovery ordinal order"):
        DiscoveryResult(
            corpus_id=CorpusId.generate(),
            policy=CURRENT_DISCOVERY_POLICY,
            status=DiscoveryStatus.COMPLETED,
            sources=(first, second),
            failures=(),
            counts=DiscoveryCounts.from_outcomes(
                files_seen=2,
                sources=(first, second),
                failures=(),
            ),
        )

    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.UNKNOWN_FORMAT,
    )
    with pytest.raises(ValueError, match="ordinals must be unique"):
        DiscoveryResult(
            corpus_id=CorpusId.generate(),
            policy=CURRENT_DISCOVERY_POLICY,
            status=DiscoveryStatus.PARTIAL,
            sources=(second,),
            failures=(failure,),
            counts=DiscoveryCounts.from_outcomes(
                files_seen=2,
                sources=(second,),
                failures=(failure,),
            ),
        )


def test_counts_reject_invalid_totals_and_implicit_numbers() -> None:
    with pytest.raises(ValueError, match="outcome counts must equal hashed"):
        DiscoveryCounts(
            files_seen=1,
            supported=1,
            unsupported=0,
            hashed=1,
            unchanged=0,
            changed=0,
            new=0,
            duplicate=0,
            ambiguous=0,
            conflicting=0,
            failed=0,
        )
    with pytest.raises(TypeError, match="counts must be integers"):
        DiscoveryCounts(
            files_seen=True,  # pyright: ignore[reportArgumentType]
            supported=0,
            unsupported=0,
            hashed=0,
            unchanged=0,
            changed=0,
            new=0,
            duplicate=0,
            ambiguous=0,
            conflicting=0,
            failed=0,
        )


def test_result_rejects_aggregate_mismatch() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    incorrect = DiscoveryCounts(
        files_seen=2,
        supported=2,
        unsupported=0,
        hashed=1,
        unchanged=0,
        changed=0,
        new=1,
        duplicate=0,
        ambiguous=0,
        conflicting=0,
        failed=0,
    )

    with pytest.raises(ValueError, match="counts must match result outcomes"):
        DiscoveryResult(
            corpus_id=CorpusId.generate(),
            policy=CURRENT_DISCOVERY_POLICY,
            status=DiscoveryStatus.COMPLETED,
            sources=(source,),
            failures=(),
            counts=incorrect,
        )


def test_result_requires_immutable_collections_and_is_itself_immutable() -> None:
    with pytest.raises(TypeError, match="sources must be a tuple"):
        DiscoveryResult(
            CorpusId.generate(),
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.COMPLETED,
            [],  # pyright: ignore[reportArgumentType]
            (),
            DiscoveryCounts.empty(),
        )

    result = DiscoveryResult(
        CorpusId.generate(),
        CURRENT_DISCOVERY_POLICY,
        DiscoveryStatus.COMPLETED,
        (),
        (),
        DiscoveryCounts.empty(),
    )
    with pytest.raises(FrozenInstanceError):
        result.status = DiscoveryStatus.FAILED  # type: ignore[misc]
