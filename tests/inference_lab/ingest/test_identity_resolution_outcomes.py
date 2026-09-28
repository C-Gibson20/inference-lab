"""Tests for identity-resolution outcome selection."""

from pathlib import Path

import pytest
from structlog.testing import capture_logs

from inference_lab.ingest import (
    DiscoveryFailure,
)
from inference_lab.provenance import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
    ExternalIdentifierResolutionPolicy,
    ResolutionKind,
    ResolutionReason,
)
from tests.inference_lab.ingest.identity_resolution_fixtures import (
    _AUTHORITATIVE_DOI_POLICY,
    _CORPUS_ID,
    _DOI,
    _FIRST_VERSION_ID,
    _HASH_B,
    _OTHER_DOI,
    _SECOND_DOCUMENT_ID,
    _SECOND_VERSION_ID,
    _Allocator,
    _locator,
    _observation,
    _request,
    _resolve,
    _source_case,
    _Transaction,
    _version,
)


def test_multiple_identifier_documents_preserve_ambiguous_outcome(
    tmp_path: Path,
) -> None:
    source, snapshot, inventory = _source_case(
        tmp_path,
        content_hash=_HASH_B,
        external_identifiers=(_DOI,),
    )
    first = _version(external_identifiers=(_DOI,))
    second = _version(
        _SECOND_VERSION_ID,
        _SECOND_DOCUMENT_ID,
        external_identifiers=(_DOI,),
    )
    transaction = _Transaction(
        versions=(first, second),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID, _SECOND_VERSION_ID})},
    )

    with capture_logs() as logs:
        result = _resolve(
            source,
            snapshot,
            inventory,
            transaction,
            _Allocator(),
            _request(tmp_path),
        )

    assert not isinstance(result, DiscoveryFailure)
    assert result.identity_outcome.kind is ResolutionKind.AMBIGUOUS
    assert result.identity_outcome.reason is ResolutionReason.MULTIPLE_CANDIDATES
    source_logs = [record for record in logs if record["event"].startswith("source_")]
    assert [(record["event"], record["log_level"]) for record in source_logs] == [
        ("source_skipped", "info")
    ]
    assert source_logs[0]["outcome_kind"] == "ambiguous"
    assert source_logs[0]["matched_candidate_count"] == 2
    log_payload = repr(logs)
    assert snapshot.locator.relative_path.name not in log_payload
    assert snapshot.content_hash.digest not in log_payload
    assert _DOI.value not in log_payload


def test_conflicting_continuity_identifiers_preserve_conflict_outcome(
    tmp_path: Path,
) -> None:
    source, snapshot, inventory = _source_case(
        tmp_path,
        content_hash=_HASH_B,
        external_identifiers=(_OTHER_DOI,),
    )
    version = _version(external_identifiers=(_DOI,))
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(_observation(version=version),),
        locators=(_locator(snapshot.locator.path),),
    )

    with capture_logs() as logs:
        result = _resolve(
            source,
            snapshot,
            inventory,
            transaction,
            _Allocator(),
            _request(tmp_path),
        )

    assert not isinstance(result, DiscoveryFailure)
    assert result.identity_outcome.kind is ResolutionKind.CONFLICT
    assert result.identity_outcome.reason is ResolutionReason.CONFLICTING_EVIDENCE
    source_logs = [record for record in logs if record["event"].startswith("source_")]
    assert [(record["event"], record["log_level"]) for record in source_logs] == [
        ("source_skipped", "info")
    ]
    assert source_logs[0]["outcome_kind"] == "conflict"


@pytest.mark.parametrize(
    ("policy", "expected_kind", "expected_reason"),
    [
        (
            CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
            ResolutionKind.AMBIGUOUS,
            ResolutionReason.UNESTABLISHED_IDENTIFIER_AUTHORITY,
        ),
        (
            _AUTHORITATIVE_DOI_POLICY,
            ResolutionKind.NEW_VERSION,
            ResolutionReason.EXTERNAL_IDENTIFIER_MATCH,
        ),
    ],
)
def test_request_policy_controls_unique_external_identifier_match(
    tmp_path: Path,
    policy: ExternalIdentifierResolutionPolicy,
    expected_kind: ResolutionKind,
    expected_reason: ResolutionReason,
) -> None:
    source, snapshot, inventory = _source_case(
        tmp_path,
        content_hash=_HASH_B,
        external_identifiers=(_DOI,),
    )
    version = _version(external_identifiers=(_DOI,))
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
    )

    result = _resolve(
        source,
        snapshot,
        inventory,
        transaction,
        _Allocator(),
        _request(tmp_path, policy=policy),
    )

    assert not isinstance(result, DiscoveryFailure)
    assert result.identity_outcome.kind is expected_kind
    assert result.identity_outcome.reason is expected_reason
