"""Tests for closed and privacy-safe discovery failure contracts."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from inference_lab.ingest import (
    DiscoveryFailure,
    DiscoveryFailureCategory,
    DiscoveryFailureScope,
    DiscoveryFailureStage,
    DiscoveryFileKind,
    SourceLocator,
    discovery_failure_policy,
)


def _locator(marker: str = "confidential-failure-name.txt") -> SourceLocator:
    return SourceLocator(
        path=Path("private") / marker,
        request_root_ordinal=0,
        relative_path=Path(marker),
    )


def test_every_failure_category_has_one_fixed_policy() -> None:
    policies = {
        category: discovery_failure_policy(category)
        for category in DiscoveryFailureCategory
    }

    assert set(policies) == set(DiscoveryFailureCategory)
    assert policies[DiscoveryFailureCategory.SOURCE_CHANGED_DURING_READ] == (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.HASHING,
        True,
    )
    assert policies[DiscoveryFailureCategory.UNKNOWN_FORMAT] == (
        DiscoveryFailureScope.SOURCE,
        DiscoveryFailureStage.CLASSIFICATION,
        False,
    )
    assert policies[DiscoveryFailureCategory.REPOSITORY_UNAVAILABLE] == (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.PERSISTENCE,
        True,
    )
    assert policies[DiscoveryFailureCategory.ROOT_INACCESSIBLE] == (
        DiscoveryFailureScope.REQUEST,
        DiscoveryFailureStage.INPUT,
        True,
    )


def test_failure_factory_uses_category_owned_policy() -> None:
    failure = DiscoveryFailure.for_category(
        ordinal=3,
        category=DiscoveryFailureCategory.SOURCE_INACCESSIBLE,
        locator=_locator(),
        file_kind=DiscoveryFileKind.REGULAR_FILE,
    )

    assert failure.scope is DiscoveryFailureScope.SOURCE
    assert failure.stage is DiscoveryFailureStage.TRAVERSAL
    assert failure.retryable is True


def test_failure_rejects_scope_stage_or_retryability_mismatch() -> None:
    with pytest.raises(ValueError, match="scope does not match"):
        DiscoveryFailure(
            ordinal=0,
            category=DiscoveryFailureCategory.INVALID_ROOT,
            scope=DiscoveryFailureScope.SOURCE,
            stage=DiscoveryFailureStage.INPUT,
            retryable=False,
        )
    with pytest.raises(ValueError, match="stage does not match"):
        DiscoveryFailure(
            ordinal=0,
            category=DiscoveryFailureCategory.SOURCE_TOO_LARGE,
            scope=DiscoveryFailureScope.SOURCE,
            stage=DiscoveryFailureStage.TRAVERSAL,
            retryable=False,
        )
    with pytest.raises(ValueError, match="retryability does not match"):
        DiscoveryFailure(
            ordinal=0,
            category=DiscoveryFailureCategory.SOURCE_TOO_LARGE,
            scope=DiscoveryFailureScope.SOURCE,
            stage=DiscoveryFailureStage.HASHING,
            retryable=True,
        )


def test_failure_requires_strict_discriminants_and_ordinal() -> None:
    with pytest.raises(TypeError, match="ordinal must be an integer"):
        DiscoveryFailure.for_category(
            ordinal=True,  # pyright: ignore[reportArgumentType]
            category=DiscoveryFailureCategory.INVALID_ROOT,
        )
    with pytest.raises(TypeError, match="category must be a DiscoveryFailureCategory"):
        discovery_failure_policy("invalid_root")  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match="file_kind must be a DiscoveryFileKind"):
        DiscoveryFailure.for_category(
            ordinal=0,
            category=DiscoveryFailureCategory.INVALID_ROOT,
            file_kind="unknown",  # pyright: ignore[reportArgumentType]
        )


def test_failure_rejects_unbounded_media_type_metadata() -> None:
    with pytest.raises(ValueError, match="canonical lowercase media type"):
        DiscoveryFailure.for_category(
            ordinal=0,
            category=DiscoveryFailureCategory.UNKNOWN_FORMAT,
            media_type="private-file-name",
        )


def test_failure_retains_no_raw_exception_or_payload_field() -> None:
    assert set(DiscoveryFailure.__dataclass_fields__) == {
        "ordinal",
        "category",
        "scope",
        "stage",
        "retryable",
        "locator",
        "file_kind",
        "media_type",
    }


def test_failure_repr_and_errors_do_not_disclose_private_paths() -> None:
    marker = "confidential-failure-name-2917.txt"
    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.SOURCE_READ_FAILED,
        locator=_locator(marker),
    )

    assert marker not in repr(failure)
    with pytest.raises(ValueError) as captured:
        DiscoveryFailure(
            ordinal=0,
            category=DiscoveryFailureCategory.SOURCE_READ_FAILED,
            scope=DiscoveryFailureScope.SOURCE,
            stage=DiscoveryFailureStage.INPUT,
            retryable=True,
            locator=_locator(marker),
        )
    assert marker not in str(captured.value)


def test_failure_is_immutable() -> None:
    failure = DiscoveryFailure.for_category(
        ordinal=0,
        category=DiscoveryFailureCategory.INVALID_ROOT,
    )

    with pytest.raises(FrozenInstanceError):
        failure.retryable = True  # type: ignore[misc]
