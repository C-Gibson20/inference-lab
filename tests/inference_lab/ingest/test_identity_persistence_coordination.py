"""Tests for identity-persistence transaction and observability coordination."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from structlog.contextvars import get_contextvars
from structlog.testing import capture_logs

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import IdentityApplicationResult
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _CORPUS_ID,
    _FIRST_DOCUMENT_ID,
    _FIRST_VERSION_ID,
    _HASH,
    _SOURCE_ID,
    _case,
    _persist,
    _Repository,
    _resolved,
    _Transaction,
)


def test_resolution_and_application_share_one_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    resolved = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    expected = IdentityApplicationResult(resolved, created_document_version=True)
    transaction = _Transaction()
    repository = _Repository(transaction)
    resolver = Mock(return_value=resolved)
    application = Mock(return_value=expected)
    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        resolver,
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        application,
    )

    with capture_logs() as logs:
        result = _persist(source, snapshot, request, inventory, repository)

    assert result == expected
    assert resolver.call_args.kwargs["transaction"] is transaction
    assert application.call_args.kwargs["transaction"] is transaction
    assert transaction.exit_exception is None
    assert logs == [
        {
            "pipeline_stage": "ingest",
            "source_ordinal": 0,
            "source_observation_id": _SOURCE_ID.as_str(),
            "outcome_kind": "new_document",
            "outcome_reason": "no_match",
            "created_document_version": True,
            "attempt_count": 1,
            "event": "identity_persistence_completed",
            "log_level": "info",
        }
    ]
    assert "source.txt" not in repr(logs)
    assert _HASH.digest not in repr(logs)


def test_completion_is_emitted_after_commit_with_identity_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    resolved = _resolved(source, snapshot, _FIRST_DOCUMENT_ID, _FIRST_VERSION_ID)
    expected = IdentityApplicationResult(resolved, created_document_version=True)
    transaction = _Transaction()
    repository = _Repository(transaction)
    captured_context: dict[str, object] = {}
    captured_fields: dict[str, object] = {}

    monkeypatch.setattr(
        persistence_module,
        "resolve_discovered_source_identity",
        Mock(return_value=resolved),
    )
    monkeypatch.setattr(
        persistence_module,
        "apply_discovery_identity_outcome",
        Mock(return_value=expected),
    )

    def capture_completion(
        module: str,
        method_name: str,
        event: str,
        **fields: object,
    ) -> None:
        assert not transaction.active
        captured_context.update(get_contextvars())
        captured_fields.update(
            module=module,
            method_name=method_name,
            event=event,
            **fields,
        )

    monkeypatch.setattr(
        persistence_module,
        "emit_discovery_event",
        capture_completion,
    )

    result = _persist(source, snapshot, request, inventory, repository)

    assert result == expected
    assert captured_context == {
        "corpus_id": _CORPUS_ID.as_str(),
        "corpus_version": None,
        "document_id": _FIRST_DOCUMENT_ID.as_str(),
    }
    assert captured_fields["module"] == persistence_module.__name__
    assert captured_fields["method_name"] == "info"
    assert captured_fields["event"] == "identity_persistence_completed"
    assert "corpus_id" not in get_contextvars()
