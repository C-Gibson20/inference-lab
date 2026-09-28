"""Acceptance tests for adapting complete ING-02 source inventories."""

from dataclasses import replace
from pathlib import Path

import pytest

from inference_lab.evaluation import (
    AuditContractError,
    AuditErrorCategory,
    AuditOverallStatus,
    AuditSourceInventory,
    CorpusAuditReport,
    CorpusAuditRequest,
    CorpusAuditScope,
    adapt_source_inventory,
)
from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    FileIdentityHint,
    FileStatEvidence,
    SourceLocator,
    TraversalCandidate,
    TraversalInventory,
    build_discovery_result,
    classify_source_inventory,
)
from inference_lab.provenance import CorpusId, CorpusVersionId

_CORPUS_ID = CorpusId("11111111-1111-4111-8111-111111111111")
_CORPUS_VERSION_ID = CorpusVersionId("22222222-2222-4222-8222-222222222222")


def _adapt_failures(
    *categories: DiscoveryFailureCategory,
) -> AuditSourceInventory:
    failures = tuple(
        DiscoveryFailure.for_category(ordinal=ordinal, category=category)
        for ordinal, category in enumerate(categories)
    )
    traversal = TraversalInventory(
        corpus_id=_CORPUS_ID,
        root_count=1,
        candidates=(),
        failures=failures,
        complete_root_ordinals=(),
    )
    classification = classify_source_inventory(traversal)
    discovery = build_discovery_result(classification, identity_results=())
    return adapt_source_inventory(classification, discovery)


def _report(inventory: AuditSourceInventory) -> CorpusAuditReport:
    request = CorpusAuditRequest(
        scope=CorpusAuditScope(_CORPUS_ID, _CORPUS_VERSION_ID, ()),
        source_inventory=inventory,
        parse_observations=(),
    )
    return CorpusAuditReport.from_request(request)


def _failure_key(inventory: AuditSourceInventory, ordinal: int) -> str:
    failure = inventory.failures[ordinal]
    retryable = str(failure.retryable).lower()
    return (
        f"{failure.category.value}|{failure.scope.value}|"
        f"{failure.stage.value}|{retryable}"
    )


@pytest.mark.parametrize(
    "category",
    (
        DiscoveryFailureCategory.ROOT_NOT_FOUND,
        DiscoveryFailureCategory.ROOT_INACCESSIBLE,
        DiscoveryFailureCategory.INVALID_ROOT,
    ),
)
def test_request_failure_is_retained_outside_files_seen_denominator(
    category: DiscoveryFailureCategory,
) -> None:
    inventory = _adapt_failures(category)

    assert inventory.discovery_counts.files_seen == 0
    assert len(inventory.failures) == 1
    assert inventory.failures[0].category is category
    assert inventory.failures[0].scope is DiscoveryFailureScope.REQUEST

    report = _report(inventory)
    assert report.populations.source_files_seen == 0
    assert report.populations.source_classified_files == 0
    assert report.populations.source_traversal_failures == 0
    assert report.source.failure_counts == ((_failure_key(inventory, 0), 1),)
    assert report.reconciliation.source_files_reconcile is True
    assert report.overall_status is AuditOverallStatus.PARTIAL
    assert "source_upstream_failures" in report.limitations


def test_source_inventory_rejects_mismatched_discovery_policy() -> None:
    traversal = TraversalInventory(
        corpus_id=_CORPUS_ID,
        root_count=1,
        candidates=(),
        failures=(),
        complete_root_ordinals=(0,),
    )
    classification = classify_source_inventory(traversal)
    discovery = build_discovery_result(classification, identity_results=())
    changed_policy = replace(discovery.policy, policy_version="1.0.1")

    with pytest.raises(AuditContractError) as error:
        adapt_source_inventory(
            classification,
            replace(discovery, policy=changed_policy),
        )

    assert error.value.category is AuditErrorCategory.OBSERVATION_MISMATCH


@pytest.mark.parametrize(
    ("category", "hashed_count", "identity_count"),
    (
        (DiscoveryFailureCategory.SOURCE_READ_FAILED, 0, 0),
        (DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS, 1, 0),
        (DiscoveryFailureCategory.IDENTITY_WRITE_FAILED, 1, 1),
    ),
)
def test_source_inventory_projects_stage_progress_without_private_metadata(
    category: DiscoveryFailureCategory,
    hashed_count: int,
    identity_count: int,
) -> None:
    candidate = TraversalCandidate(
        ordinal=0,
        locator=SourceLocator(
            path=Path("private/source.txt"),
            request_root_ordinal=0,
            relative_path=Path("source.txt"),
        ),
        stat_evidence=FileStatEvidence(
            byte_count=17,
            modified_time_ns=1,
            identity_hint=FileIdentityHint(device_id=2, file_id=3),
        ),
    )
    traversal = TraversalInventory(
        corpus_id=_CORPUS_ID,
        root_count=1,
        candidates=(candidate,),
        failures=(),
        complete_root_ordinals=(0,),
    )
    classification = classify_source_inventory(traversal)
    failure = DiscoveryFailure.for_category(ordinal=0, category=category)
    discovery = build_discovery_result(
        classification,
        identity_results=(failure,),
    )

    inventory = adapt_source_inventory(classification, discovery)

    assert inventory.known_source_byte_counts == (17,)
    assert len(inventory.format_coverage) == 1
    coverage = inventory.format_coverage[0]
    assert coverage.media_type == "text/plain"
    assert coverage.file_count == 1
    assert coverage.total_bytes == 17
    assert coverage.hashed_count == hashed_count
    assert coverage.identity_count == identity_count
    assert "private/source.txt" not in repr(inventory)
