"""Tests for discovery result and count invariants."""

from dataclasses import FrozenInstanceError, replace

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryCounts,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryResult,
    DiscoveryStatus,
)
from inference_lab.provenance import (
    CorpusId,
    ResolutionKind,
)
from tests.inference_lab.ingest.discovery_result_fixtures import (
    _source,
)


def test_request_failure_requires_failed_status_and_no_usable_source() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.REPOSITORY_UNAVAILABLE,
    )
    counts = DiscoveryCounts.from_outcomes(
        files_seen=2,
        sources=(source,),
        failures=(failure,),
    )

    with pytest.raises(ValueError, match="request failure requires a failed"):
        DiscoveryResult(
            source.corpus_id,
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
            source.corpus_id,
            CURRENT_DISCOVERY_POLICY,
            DiscoveryStatus.COMPLETED,
            (source,),
            (failure,),
            counts,
        )
    with pytest.raises(ValueError, match="failed discovery must not contain usable"):
        DiscoveryResult(
            source.corpus_id,
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
            corpus_id=first.corpus_id,
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
            corpus_id=second.corpus_id,
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
    with pytest.raises(ValueError, match="outcome counts cannot exceed hashed"):
        DiscoveryCounts(
            files_seen=1,
            supported=1,
            unsupported=0,
            hashed=0,
            unchanged=0,
            changed=0,
            new=1,
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
            corpus_id=source.corpus_id,
            policy=CURRENT_DISCOVERY_POLICY,
            status=DiscoveryStatus.COMPLETED,
            sources=(source,),
            failures=(),
            counts=incorrect,
        )


def test_result_rejects_cross_corpus_or_cross_policy_sources() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    counts = DiscoveryCounts.from_outcomes(
        files_seen=1,
        sources=(source,),
        failures=(),
    )

    with pytest.raises(ValueError, match="source corpus must match"):
        DiscoveryResult(
            corpus_id=CorpusId.generate(),
            policy=source.policy,
            status=DiscoveryStatus.COMPLETED,
            sources=(source,),
            failures=(),
            counts=counts,
        )

    with pytest.raises(ValueError, match="source policy must match"):
        DiscoveryResult(
            corpus_id=source.corpus_id,
            policy=replace(source.policy, policy_version="1.0.1"),
            status=DiscoveryStatus.COMPLETED,
            sources=(source,),
            failures=(),
            counts=counts,
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
