"""Acceptance tests for adapting complete ING-02 source inventories."""

import pytest

from inference_lab.evaluation import (
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


def test_mixed_source_failures_remain_inside_files_seen_denominator() -> None:
    inventory = _adapt_failures(
        DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
        DiscoveryFailureCategory.SOURCE_SYMLINK,
    )

    assert inventory.discovery_counts.files_seen == 2
    assert all(
        failure.scope is DiscoveryFailureScope.SOURCE for failure in inventory.failures
    )

    report = _report(inventory)
    assert report.populations.source_files_seen == 2
    assert report.populations.source_classified_files == 0
    assert report.populations.source_traversal_failures == 2
    assert report.source.failure_counts == (
        (_failure_key(inventory, 0), 1),
        (_failure_key(inventory, 1), 1),
    )
    assert report.reconciliation.source_files_reconcile is True
    assert "source_upstream_failures" in report.limitations
