"""Tests for discovery lifecycle coordination and observability."""

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

from inference_lab.ingest import (
    CURRENT_DISCOVERY_POLICY,
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
