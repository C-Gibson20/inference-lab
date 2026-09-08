"""Tests for discovery counts and outcome-derived status."""

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


@pytest.mark.parametrize(
    "category",
    (
        DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS,
        DiscoveryFailureCategory.CANDIDATE_METADATA_INVALID,
        DiscoveryFailureCategory.REPOSITORY_UNAVAILABLE,
        DiscoveryFailureCategory.CONCURRENT_WRITE_CONFLICT,
        DiscoveryFailureCategory.IDENTITY_WRITE_FAILED,
    ),
)
def test_counts_include_post_hash_failures_as_hashed(
    category: DiscoveryFailureCategory,
) -> None:
    failure = DiscoveryFailure.for_category(ordinal=0, category=category)

    counts = DiscoveryCounts.from_outcomes(
        files_seen=1,
        sources=(),
        failures=(failure,),
    )

    assert counts.supported == 1
    assert counts.hashed == 1
    assert counts.failed == 1


@pytest.mark.parametrize(
    "category",
    (
        DiscoveryFailureCategory.SOURCE_TOO_LARGE,
        DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ,
        DiscoveryFailureCategory.SOURCE_READ_FAILED,
    ),
)
def test_counts_exclude_hashing_failures_from_hashed(
    category: DiscoveryFailureCategory,
) -> None:
    failure = DiscoveryFailure.for_category(ordinal=0, category=category)

    counts = DiscoveryCounts.from_outcomes(
        files_seen=1,
        sources=(),
        failures=(failure,),
    )

    assert counts.supported == 1
    assert counts.hashed == 0
    assert counts.failed == 1


def test_partial_result_combines_usable_source_and_isolated_failure() -> None:
    source = _source(0, ResolutionKind.NEW_DOCUMENT)
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_TOO_LARGE,
    )
    result = DiscoveryResult(
        corpus_id=source.corpus_id,
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
        corpus_id=source.corpus_id,
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
