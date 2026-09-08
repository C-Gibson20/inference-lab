"""Tests for identity-resolution corpus and request boundaries."""

from pathlib import Path

import pytest
from structlog.contextvars import get_contextvars

import inference_lab.ingest.identity_resolution as identity_resolution_module
from inference_lab.ingest import (
    DiscoveryFailure,
)
from inference_lab.provenance import (
    CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY,
    ExternalIdentifierAssertion,
    IdentityResolutionOutcome,
    IdentityResolutionRequest,
    ResolutionKind,
)
from tests.inference_lab.ingest.identity_resolution_fixtures import (
    _CORPUS_ID,
    _DOI,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
    _OTHER_CORPUS_ID,
    _PMID,
    _SECOND_SOURCE_ID,
    _SOURCE_ID,
    _Allocator,
    _locator,
    _observation,
    _request,
    _resolve,
    _source_case,
    _Transaction,
    _version,
)


def test_candidate_reads_never_cross_the_selected_corpus(tmp_path: Path) -> None:
    source, snapshot, inventory = _source_case(tmp_path)
    other_version = _version()
    other_observation = _observation(
        corpus_id=_OTHER_CORPUS_ID,
        source_id=_SECOND_SOURCE_ID,
        version=other_version,
    )
    other_locator = _locator(
        snapshot.locator.path,
        corpus_id=_OTHER_CORPUS_ID,
        source_id=_SECOND_SOURCE_ID,
    )
    transaction = _Transaction(
        versions=(other_version,),
        memberships={_OTHER_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(other_observation,),
        locators=(other_locator,),
    )

    result = _resolve(
        source,
        snapshot,
        inventory,
        transaction,
        _Allocator(),
        _request(tmp_path),
    )

    assert not isinstance(result, DiscoveryFailure)
    assert result.identity_outcome.kind is ResolutionKind.NEW_DOCUMENT
    assert transaction.corpus_calls
    assert set(transaction.corpus_calls) == {_CORPUS_ID}


def test_complete_request_and_corpus_context_reach_resolver_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, inventory = _source_case(
        tmp_path,
        external_identifiers=(_DOI,),
    )
    version = _version(external_identifiers=(_DOI,))
    observation = _observation(version=version)
    assertion = ExternalIdentifierAssertion(_SOURCE_ID, _FIRST_VERSION_ID, _PMID)
    transaction = _Transaction(
        versions=(version,),
        memberships={_CORPUS_ID: frozenset({_FIRST_VERSION_ID})},
        observations=(observation,),
        locators=(_locator(snapshot.locator.path),),
        assertions=(assertion,),
    )
    captured_request: IdentityResolutionRequest | None = None
    captured_context: dict[str, object] = {}
    captured_log_context: dict[str, object] = {}
    captured_log_fields: dict[str, object] = {}
    original_resolver = identity_resolution_module.resolve_identity

    def _capturing_resolver(
        request: IdentityResolutionRequest,
        **kwargs: object,
    ) -> IdentityResolutionOutcome:
        nonlocal captured_request
        captured_request = request
        captured_context.update(get_contextvars())
        return original_resolver(request, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        identity_resolution_module,
        "resolve_identity",
        _capturing_resolver,
    )

    def _capturing_log(
        module: str,
        method_name: str,
        event: str,
        **fields: object,
    ) -> None:
        captured_log_context.update(get_contextvars())
        captured_log_fields.update(
            module=module,
            method_name=method_name,
            event=event,
            **fields,
        )

    monkeypatch.setattr(
        identity_resolution_module,
        "emit_discovery_event",
        _capturing_log,
    )

    result = _resolve(
        source,
        snapshot,
        inventory,
        transaction,
        _Allocator(),
        _request(tmp_path),
    )

    assert not isinstance(result, DiscoveryFailure)
    assert captured_request is not None
    assert captured_request.continuity_candidates == (version,)
    assert captured_request.content_hash_candidates == (version,)
    assert captured_request.external_identifier_candidates == (version,)
    assert captured_request.identifier_assertions == (assertion,)
    assert captured_request.external_identifier_policy == (
        CONSERVATIVE_EXTERNAL_IDENTIFIER_POLICY
    )
    assert captured_context["corpus_id"] == _CORPUS_ID.as_str()
    assert captured_context["document_id"] is None
    assert captured_context["corpus_version"] is None
    assert captured_log_context == {
        "corpus_id": _CORPUS_ID.as_str(),
        "corpus_version": None,
        "document_id": _FIRST_DOCUMENT_ID.as_str(),
    }
    assert captured_log_fields["event"] == "source_unchanged"
    assert "corpus_id" not in get_contextvars()
