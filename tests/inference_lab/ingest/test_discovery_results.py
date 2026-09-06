"""Tests for discovery result status, ordering, and aggregate invariants."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveredSource,
    DiscoveryCounts,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryInput,
    DiscoveryInputKind,
    DiscoveryRequest,
    DiscoveryResult,
    DiscoveryStatus,
    FileIdentityHint,
    FileStatEvidence,
    IdentityApplicationResult,
    SourceClassification,
    SourceLocator,
    SourceSupport,
    StableFileEvidence,
    VerifiedSourceSnapshot,
    build_discovery_result,
    observe_discovery,
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


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        corpus_id=CorpusId.generate(),
        inputs=(DiscoveryInput(Path("private"), DiscoveryInputKind.DIRECTORY),),
    )


def _applied_source(
    ordinal: int,
    kind: ResolutionKind,
) -> IdentityApplicationResult:
    return IdentityApplicationResult(
        source=_source(ordinal, kind),
        parse_eligible=kind
        in {
            ResolutionKind.NEW_VERSION,
            ResolutionKind.NEW_DOCUMENT,
            ResolutionKind.DUPLICATE,
        },
    )


def test_builder_orders_outcomes_and_derives_partial_summary() -> None:
    request = _request()
    unsupported = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.UNKNOWN_FORMAT,
        media_type="application/octet-stream",
    )
    persistence_failure = DiscoveryFailure.for_category(
        ordinal=3,
        category=DiscoveryFailureCategory.IDENTITY_WRITE_FAILED,
    )

    result = build_discovery_result(
        request,
        files_seen=4,
        stage_failures=(unsupported,),
        identity_results=(
            persistence_failure,
            _applied_source(2, ResolutionKind.NEW_DOCUMENT),
            _applied_source(0, ResolutionKind.UNCHANGED),
        ),
    )

    assert result.corpus_id == request.corpus_id
    assert result.policy == CURRENT_DISCOVERY_POLICY
    assert result.status is DiscoveryStatus.PARTIAL
    assert tuple(source.ordinal for source in result.sources) == (0, 2)
    assert tuple(failure.ordinal for failure in result.failures) == (1, 3)
    assert result.counts == DiscoveryCounts(
        files_seen=4,
        supported=3,
        unsupported=1,
        hashed=2,
        unchanged=1,
        changed=0,
        new=1,
        duplicate=0,
        ambiguous=0,
        conflicting=0,
        failed=1,
    )


def test_builder_treats_empty_directory_as_completed() -> None:
    request = _request()

    result = build_discovery_result(
        request,
        files_seen=0,
        stage_failures=(),
        identity_results=(),
    )

    assert result.status is DiscoveryStatus.COMPLETED
    assert result.sources == ()
    assert result.failures == ()
    assert result.counts == DiscoveryCounts.empty()


@pytest.mark.parametrize(
    "failure",
    (
        DiscoveryFailure.for_category(
            ordinal=0,
            category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
        ),
        DiscoveryFailure.for_category(
            ordinal=0,
            category=DiscoveryFailureCategory.ROOT_NOT_FOUND,
        ),
    ),
)
def test_builder_derives_failed_status_without_usable_sources(
    failure: DiscoveryFailure,
) -> None:
    files_seen = 1 if failure.scope is DiscoveryFailureScope.SOURCE else 0

    result = build_discovery_result(
        _request(),
        files_seen=files_seen,
        stage_failures=(failure,),
        identity_results=(),
    )

    assert result.status is DiscoveryStatus.FAILED


def test_builder_rejects_incomplete_file_accounting() -> None:
    with pytest.raises(ValueError, match="files_seen must exactly match"):
        build_discovery_result(
            _request(),
            files_seen=2,
            stage_failures=(),
            identity_results=(_applied_source(0, ResolutionKind.NEW_DOCUMENT),),
        )


def test_lifecycle_binds_context_and_logs_privacy_safe_partial_summary() -> None:
    request = _request()
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
    )
    clock_values = iter((1_000_000_000, 1_025_000_000))

    with capture_logs() as logs:
        with observe_discovery(request, clock=clock_values.__next__) as lifecycle:
            assert get_contextvars() == {
                "corpus_id": request.corpus_id.as_str(),
                "corpus_version": None,
                "document_id": None,
            }
            result = lifecycle.complete(
                files_seen=2,
                stage_failures=(failure,),
                identity_results=(_applied_source(0, ResolutionKind.NEW_DOCUMENT),),
            )

    assert result.status is DiscoveryStatus.PARTIAL
    assert get_contextvars() == {}
    assert [(record["event"], record["log_level"]) for record in logs] == [
        ("discovery_started", "info"),
        ("discovery_completed", "warning"),
    ]
    assert logs[0] == {
        "pipeline_stage": "ingest",
        "policy_name": CURRENT_DISCOVERY_POLICY.policy_name,
        "policy_version": CURRENT_DISCOVERY_POLICY.policy_version,
        "input_count": 1,
        "event": "discovery_started",
        "log_level": "info",
    }
    assert logs[1]["status"] == "partial"
    assert logs[1]["duration_ms"] == 25.0
    assert logs[1]["files_seen"] == 2
    assert logs[1]["supported"] == 1
    assert logs[1]["failed"] == 1
    log_payload = repr(logs)
    assert "source-0.txt" not in log_payload
    assert _DIGEST not in log_payload


@pytest.mark.parametrize(
    ("stage_failures", "files_seen", "expected_status", "expected_level"),
    (
        ((), 0, DiscoveryStatus.COMPLETED, "info"),
        (
            (
                DiscoveryFailure.for_category(
                    ordinal=0,
                    category=DiscoveryFailureCategory.ROOT_NOT_FOUND,
                ),
            ),
            0,
            DiscoveryStatus.FAILED,
            "error",
        ),
    ),
)
def test_lifecycle_completion_level_matches_result_status(
    stage_failures: tuple[DiscoveryFailure, ...],
    files_seen: int,
    expected_status: DiscoveryStatus,
    expected_level: str,
) -> None:
    clock_values = iter((10, 20))

    with capture_logs() as logs:
        with observe_discovery(
            _request(),
            clock=clock_values.__next__,
        ) as lifecycle:
            result = lifecycle.complete(
                files_seen=files_seen,
                stage_failures=stage_failures,
                identity_results=(),
            )

    assert result.status is expected_status
    assert logs[-1]["log_level"] == expected_level


def test_lifecycle_requires_a_terminal_result_without_masking_primary_errors() -> None:
    with pytest.raises(RuntimeError, match="without a terminal result"):
        with observe_discovery(_request(), clock=lambda: 0):
            pass

    primary = RuntimeError("private-primary-error")
    with pytest.raises(RuntimeError, match="private-primary-error") as caught:
        with observe_discovery(_request(), clock=lambda: 0):
            raise primary

    assert caught.value is primary
    assert get_contextvars() == {}


def test_lifecycle_rejects_a_clock_that_moves_backwards() -> None:
    clock_values = iter((20, 10))

    with capture_logs() as logs:
        with pytest.raises(ValueError, match="must not move backwards"):
            with observe_discovery(
                _request(),
                clock=clock_values.__next__,
            ) as lifecycle:
                lifecycle.complete(
                    files_seen=0,
                    stage_failures=(),
                    identity_results=(),
                )

    assert [record["event"] for record in logs] == ["discovery_started"]


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
