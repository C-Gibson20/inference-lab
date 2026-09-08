"""Tests for building aggregate discovery results."""

from dataclasses import replace

import pytest

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryCounts,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryStatus,
    build_discovery_result,
)
from inference_lab.provenance import (
    CorpusId,
    ResolutionKind,
)
from tests.inference_lab.ingest.discovery_result_fixtures import (
    _applied_source,
    _classified_inventory,
    _custom_registry,
    _request,
)


def test_builder_orders_outcomes_and_derives_partial_summary() -> None:
    request = _request()
    classification = _classified_inventory(
        request.corpus_id,
        supported_ordinals=(0, 2, 3),
        unknown_ordinals=(1,),
    )
    persistence_failure = DiscoveryFailure.for_category(
        ordinal=3,
        category=DiscoveryFailureCategory.IDENTITY_WRITE_FAILED,
    )

    result = build_discovery_result(
        classification,
        identity_results=(
            persistence_failure,
            _applied_source(
                2,
                ResolutionKind.NEW_DOCUMENT,
                corpus_id=request.corpus_id,
            ),
            _applied_source(
                0,
                ResolutionKind.UNCHANGED,
                corpus_id=request.corpus_id,
            ),
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
        hashed=3,
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
    classification = _classified_inventory(request.corpus_id)

    result = build_discovery_result(
        classification,
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
    request = _request()
    classification = _classified_inventory(
        request.corpus_id,
        traversal_failures=(failure,),
    )

    result = build_discovery_result(
        classification,
        identity_results=(),
    )

    assert result.status is DiscoveryStatus.FAILED
    assert result.counts.files_seen == files_seen


def test_builder_rejects_incomplete_hashable_candidate_accounting() -> None:
    request = _request()
    classification = _classified_inventory(
        request.corpus_id,
        supported_ordinals=(0, 1),
    )
    with pytest.raises(ValueError, match="must exactly match hashable"):
        build_discovery_result(
            classification,
            identity_results=(
                _applied_source(
                    0,
                    ResolutionKind.NEW_DOCUMENT,
                    corpus_id=request.corpus_id,
                ),
            ),
        )


def test_builder_derives_effective_policy_from_the_classification_inventory() -> None:
    registry = _custom_registry()
    corpus_id = CorpusId.generate()
    classification = _classified_inventory(
        corpus_id,
        unknown_ordinals=(0,),
        registry=registry,
    )

    result = build_discovery_result(classification, identity_results=())

    assert result.corpus_id == corpus_id
    assert result.policy.classifier_name == "custom-classifier"
    assert result.policy.classifier_version == "2.0.0"


def test_builder_rejects_cross_corpus_or_cross_policy_source_outcomes() -> None:
    corpus_id = CorpusId.generate()
    classification = _classified_inventory(
        corpus_id,
        supported_ordinals=(0,),
    )
    applied = _applied_source(
        0,
        ResolutionKind.NEW_DOCUMENT,
        corpus_id=corpus_id,
    )

    with pytest.raises(ValueError, match="source corpus must match"):
        build_discovery_result(
            classification,
            identity_results=(
                replace(
                    applied,
                    source=replace(applied.source, corpus_id=CorpusId.generate()),
                ),
            ),
        )

    mismatched_policy = replace(CURRENT_DISCOVERY_POLICY, policy_version="1.0.1")
    with pytest.raises(ValueError, match="source policy must match"):
        build_discovery_result(
            classification,
            identity_results=(
                replace(
                    applied,
                    source=replace(applied.source, policy=mismatched_policy),
                ),
            ),
        )
