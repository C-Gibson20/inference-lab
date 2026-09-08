"""Tests for discovery result status, ordering, and aggregate invariants."""

from dataclasses import FrozenInstanceError, replace

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
    DiscoveryCounts,
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryResult,
    DiscoveryStatus,
    build_discovery_result,
    observe_discovery,
)
from inference_lab.provenance import (
    CorpusId,
    ResolutionKind,
)
from tests.inference_lab.ingest.discovery_result_fixtures import (
    _DIGEST,
    _applied_source,
    _classified_inventory,
    _custom_registry,
    _request,
    _source,
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


def test_lifecycle_uses_and_validates_the_selected_classifier_policy() -> None:
    request = _request()
    registry = _custom_registry()
    classification = _classified_inventory(
        request.corpus_id,
        unknown_ordinals=(0,),
        registry=registry,
    )
    clock_values = iter((10, 20))

    with capture_logs() as logs:
        with observe_discovery(
            request,
            classifier=registry,
            clock=clock_values.__next__,
        ) as lifecycle:
            result = lifecycle.complete(
                classification=classification,
                identity_results=(),
            )

    assert result.policy == classification.policy
    assert logs[0]["classifier_name"] == "custom-classifier"
    assert logs[0]["classifier_version"] == "2.0.0"
    assert logs[-1]["classifier_name"] == "custom-classifier"
    assert logs[-1]["classifier_version"] == "2.0.0"

    with pytest.raises(ValueError, match="classification policy must match"):
        with observe_discovery(request, clock=lambda: 0) as lifecycle:
            lifecycle.complete(
                classification=classification,
                identity_results=(),
            )


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


def test_lifecycle_binds_context_and_logs_privacy_safe_partial_summary() -> None:
    request = _request()
    failure = DiscoveryFailure.for_category(
        ordinal=1,
        category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
    )
    clock_values = iter((1_000_000_000, 1_025_000_000))
    classification = _classified_inventory(
        request.corpus_id,
        supported_ordinals=(0,),
        traversal_failures=(failure,),
    )

    with capture_logs() as logs:
        with observe_discovery(request, clock=clock_values.__next__) as lifecycle:
            assert get_contextvars() == {
                "corpus_id": request.corpus_id.as_str(),
                "corpus_version": None,
                "document_id": None,
            }
            result = lifecycle.complete(
                classification=classification,
                identity_results=(
                    _applied_source(
                        0,
                        ResolutionKind.NEW_DOCUMENT,
                        corpus_id=request.corpus_id,
                    ),
                ),
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
        "classifier_name": CURRENT_DISCOVERY_POLICY.classifier_name,
        "classifier_version": CURRENT_DISCOVERY_POLICY.classifier_version,
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
    request = _request()
    classification = _classified_inventory(
        request.corpus_id,
        traversal_failures=stage_failures,
    )

    with capture_logs() as logs:
        with observe_discovery(
            request,
            clock=clock_values.__next__,
        ) as lifecycle:
            result = lifecycle.complete(
                classification=classification,
                identity_results=(),
            )

    assert result.status is expected_status
    assert result.counts.files_seen == files_seen
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
    request = _request()
    classification = _classified_inventory(request.corpus_id)

    with capture_logs() as logs:
        with pytest.raises(ValueError, match="must not move backwards"):
            with observe_discovery(
                request,
                clock=clock_values.__next__,
            ) as lifecycle:
                lifecycle.complete(
                    classification=classification,
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
