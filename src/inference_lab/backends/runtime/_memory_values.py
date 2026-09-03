"""Immutable identities and observations for runtime memory accounting."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from inference_lab._clock import utc_now
from inference_lab._validation import is_canonical_semantic_version
from inference_lab.backends.runtime._validation import (
    validate_non_empty_string,
    validate_non_negative_integer,
)
from inference_lab.profiling._validation import validate_utc_datetime
from inference_lab.profiling.resources import (
    MEASUREMENT_STATUSES,
    MEMORY_RESOURCE_CATEGORIES,
    MeasurementStatus,
    MemoryResourceCategory,
)


class MemoryComponentRole(StrEnum):
    """Bounded role identifying the owner of a memory contribution."""

    GENERATOR = "generator"
    EMBEDDER = "embedder"
    RERANKER = "reranker"
    KV_CACHE = "kv_cache"
    ACTIVATION_WORKSPACE = "activation_workspace"
    LEXICAL_INDEX = "lexical_index"
    DENSE_INDEX = "dense_index"
    METADATA_INDEX = "metadata_index"
    STRUCTURED_INDEX = "structured_index"
    SUMMARY_INDEX = "summary_index"
    RUNTIME_FRAMEWORK = "runtime_framework"
    PINNED_HOST_ALLOCATION = "pinned_host_allocation"
    STORAGE_CACHE = "storage_cache"


class MemorySpace(StrEnum):
    """Physical memory space in which a contribution resides."""

    ACCELERATOR = "accelerator"
    HOST = "host"
    PINNED_HOST = "pinned_host"
    STORAGE_CACHE = "storage_cache"


class ContributionKind(StrEnum):
    """Whether a measurement is an exclusive contribution or a roll-up."""

    EXCLUSIVE = "exclusive"
    ROLL_UP = "roll_up"


class MemoryOverlapPolicy(StrEnum):
    """Policy for preventing exclusive contributions from double counting."""

    DISALLOW = "disallow"


def _validate_safe_identifier(value: str, field_name: str) -> None:
    """Validate a bounded identifier suitable for operational correlation."""
    validate_non_empty_string(value, field_name)
    if len(value) > 128:
        raise ValueError(f"{field_name} must be at most 128 characters")
    if any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
        for character in value
    ):
        raise ValueError(f"{field_name} must contain only safe identifier characters")


@dataclass(frozen=True, slots=True)
class MemoryAdapterIdentity:
    """Stable identity for the adapter and calibration that produced a value."""

    adapter_name: str
    adapter_version: str
    calibration_revision: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the adapter identity's fields."""
        _validate_safe_identifier(self.adapter_name, "adapter_name")
        if not is_canonical_semantic_version(self.adapter_version):
            raise ValueError("adapter_version must be a canonical semantic version")
        _validate_safe_identifier(self.calibration_revision, "calibration_revision")


@dataclass(frozen=True, slots=True)
class MemoryCollectorIdentity:
    """Stable identity for the policy that assembled a memory snapshot."""

    collector_name: str
    collector_version: str
    inventory_revision: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate the collector identity's fields."""
        _validate_safe_identifier(self.collector_name, "collector_name")
        if not is_canonical_semantic_version(self.collector_version):
            raise ValueError("collector_version must be a canonical semantic version")
        _validate_safe_identifier(self.inventory_revision, "inventory_revision")


@dataclass(frozen=True, slots=True)
class ExpectedMemoryContribution:
    """One planned exclusive component/category contribution in a memory space."""

    component_id: str
    component_role: MemoryComponentRole
    category: MemoryResourceCategory
    memory_space: MemorySpace
    device_id: str | None
    adapter: MemoryAdapterIdentity

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate expected contribution types, units, and identity rules."""
        _validate_safe_identifier(self.component_id, "component_id")
        if not isinstance(self.component_role, MemoryComponentRole):
            raise TypeError("component_role must be a MemoryComponentRole")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in MEMORY_RESOURCE_CATEGORIES:
            raise ValueError("category must be a supported memory category")
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if not isinstance(self.adapter, MemoryAdapterIdentity):
            raise TypeError("adapter must be MemoryAdapterIdentity")


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    """One point-in-time, component-attributed memory observation.

    ``value_bytes`` may be retained for an unreliable observation, but an
    unavailable observation never carries a value. Roll-up coverage is
    validated for shape here; graph and reconciliation validation belongs to
    the snapshot accountant.
    """

    measurement_id: str
    component_id: str
    component_role: MemoryComponentRole
    category: MemoryResourceCategory
    memory_space: MemorySpace
    device_id: str | None
    status: MeasurementStatus
    value_bytes: int | None
    adapter: MemoryAdapterIdentity
    contribution_kind: ContributionKind = ContributionKind.EXCLUSIVE
    covers_measurement_ids: tuple[str, ...] = ()
    observed_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate measurement types, units, identity, and status rules."""
        _validate_safe_identifier(self.measurement_id, "measurement_id")
        _validate_safe_identifier(self.component_id, "component_id")
        if not isinstance(self.component_role, MemoryComponentRole):
            raise TypeError("component_role must be a MemoryComponentRole")
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in MEMORY_RESOURCE_CATEGORIES:
            raise ValueError("category must be a supported memory category")
        if not isinstance(self.memory_space, MemorySpace):
            raise TypeError("memory_space must be a MemorySpace")
        if self.device_id is not None:
            _validate_safe_identifier(self.device_id, "device_id")
        if not isinstance(self.status, str):
            raise TypeError("status must be a string")
        if self.status not in MEASUREMENT_STATUSES:
            raise ValueError("status must be a supported measurement status")
        if self.value_bytes is not None:
            validate_non_negative_integer(self.value_bytes, "value_bytes")
        if self.status == "measured" and self.value_bytes is None:
            raise ValueError("measured memory must have a value")
        if self.status == "unavailable" and self.value_bytes is not None:
            raise ValueError("unavailable memory must not have a value")
        if (
            self.memory_space is MemorySpace.ACCELERATOR
            and self.device_id is None
            and self.status == "measured"
        ):
            raise ValueError("measured accelerator memory must identify its device")
        if not isinstance(self.adapter, MemoryAdapterIdentity):
            raise TypeError("adapter must be MemoryAdapterIdentity")
        if not isinstance(self.contribution_kind, ContributionKind):
            raise TypeError("contribution_kind must be a ContributionKind")
        if not isinstance(self.covers_measurement_ids, tuple):
            raise TypeError("covers_measurement_ids must be a tuple")
        for covered_id in self.covers_measurement_ids:
            _validate_safe_identifier(covered_id, "covered measurement ID")
        if len(set(self.covers_measurement_ids)) != len(self.covers_measurement_ids):
            raise ValueError("covers_measurement_ids must not contain duplicates")
        if self.contribution_kind is ContributionKind.EXCLUSIVE:
            if self.covers_measurement_ids:
                raise ValueError("exclusive measurements must not cover contributions")
        elif not self.covers_measurement_ids:
            raise ValueError("roll-up measurements must cover contributions")
        if self.measurement_id in self.covers_measurement_ids:
            raise ValueError("a measurement cannot cover itself")
        if not isinstance(self.observed_at, datetime):
            raise TypeError("observed_at must be a datetime")
        validate_utc_datetime(self.observed_at, "observed_at")


@runtime_checkable
class MemoryAccountingAdapter(Protocol):
    """Optional one-shot capability for component memory observations."""

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Return one immutable point-in-time memory observation set."""
        ...
