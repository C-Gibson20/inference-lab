"""Tests for discovery lifecycle coordination and observability."""

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryStatus,
    observe_discovery,
)
from inference_lab.provenance import (
    ResolutionKind,
)
from tests.inference_lab.ingest.discovery_result_fixtures import (
    _DIGEST,
    _applied_source,
    _classified_inventory,
    _custom_registry,
    _request,
)


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
    assert result.policy.classifier_name == "custom-classifier"
    assert result.policy.classifier_version == "2.0.0"
    assert [
        (record["event"], record["status"])
        for record in logs
        if record["event"] == "stage_completed"
    ] == [("stage_completed", "failed")]
    assert [record["event"] for record in logs if record["event"] == "unit_failed"] == [
        "unit_failed"
    ]

    with pytest.raises(ValueError, match="classification policy must match"):
        with observe_discovery(request, clock=lambda: 0) as lifecycle:
            lifecycle.complete(
                classification=classification,
                identity_results=(),
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
            contextvars = get_contextvars()
            assert contextvars["corpus_id"] == request.corpus_id.as_str()
            assert contextvars["corpus_version"] is None
            assert contextvars["document_id"] is None
            assert isinstance(contextvars["run_id"], str)
            assert contextvars["request_id"] is None
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
    completions = [record for record in logs if record["event"] == "stage_completed"]
    failures = [record for record in logs if record["event"] == "unit_failed"]
    assert len(completions) == 1
    assert completions[0]["status"] == "failed"
    assert completions[0]["duration_ms"] == 25.0
    assert completions[0]["items_processed"] == 2
    assert completions[0]["records_emitted"] == 2
    assert len(failures) == 1
    assert failures[0]["failure_category"] == "dependency"
    assert failures[0]["failure_code"] == "source_inaccessible"
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

    assert [record["event"] for record in logs] == ["stage_started"]
