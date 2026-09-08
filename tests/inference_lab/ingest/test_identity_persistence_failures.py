"""Tests for non-retry persistence failures."""

from pathlib import Path
from unittest.mock import Mock

import pytest

import inference_lab.ingest.identity_persistence as persistence_module
from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFileKind,
)
from tests.inference_lab.ingest.identity_persistence_fixtures import (
    _case,
    _persist,
    _Repository,
    _Transaction,
)


def test_resolution_failure_commits_no_application_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, snapshot, request, inventory = _case(tmp_path)
    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.OBSERVATION_AMBIGUOUS,
        locator=source.candidate.locator,
        file_kind=DiscoveryFileKind.REGULAR_FILE,
        media_type="text/plain",
    )
    transaction = _Transaction()
    repository = _Repository(transaction)
    resolver = Mock(return_value=failure)
    application = Mock()
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

    result = _persist(source, snapshot, request, inventory, repository)

    assert result == failure
    application.assert_not_called()
    assert transaction.exit_exception is None
