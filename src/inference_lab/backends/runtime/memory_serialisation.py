"""Strict deterministic serialisation for runtime memory accounting values."""

import json
from datetime import datetime
from typing import Final, cast

from inference_lab.backends.runtime.memory import (
    SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS,
    BudgetComparison,
    BudgetState,
    Completeness,
    ContributionKind,
    ExpectedMemoryContribution,
    MemoryAdapterIdentity,
    MemoryBudget,
    MemoryCollectorIdentity,
    MemoryComponentRole,
    MemoryMeasurement,
    MemoryOverlapPolicy,
    MemorySnapshot,
    MemorySpace,
)
from inference_lab.profiling.resources import (
    MEASUREMENT_STATUSES,
    MEMORY_RESOURCE_CATEGORIES,
    MeasurementStatus,
    MemoryResourceCategory,
)

JsonObject = dict[str, object]
JsonValue = object

_SNAPSHOT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "collector",
        "expected_contributions",
        "measurements",
        "observed_at",
        "overlap_policy",
        "request_id",
        "run_id",
        "schema_version",
    }
)
_MEASUREMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter",
        "category",
        "component_id",
        "component_role",
        "contribution_kind",
        "covers_measurement_ids",
        "device_id",
        "measurement_id",
        "memory_space",
        "observed_at",
        "status",
        "value_bytes",
    }
)
_ADAPTER_FIELDS: Final[frozenset[str]] = frozenset(
    {"adapter_name", "adapter_version", "calibration_revision"}
)
_COLLECTOR_FIELDS: Final[frozenset[str]] = frozenset(
    {"collector_name", "collector_version", "inventory_revision"}
)
_EXPECTED_CONTRIBUTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "adapter",
        "category",
        "component_id",
        "component_role",
        "device_id",
        "memory_space",
    }
)
_BUDGET_FIELDS: Final[frozenset[str]] = frozenset(
    {"budget_bytes", "device_id", "memory_space"}
)
_COMPARISON_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "budget",
        "completeness",
        "measured_bytes",
        "measured_count",
        "missing_count",
        "provisional_headroom_bytes",
        "state",
        "unavailable_count",
        "unreliable_count",
    }
)


def _json_bytes(payload: JsonObject) -> bytes:
    """Encode a JSON object with the project canonical JSON policy."""
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _object_pairs_without_duplicates(
    pairs: list[tuple[str, JsonValue]],
) -> JsonObject:
    """Reject duplicate object keys before JSON values are interpreted."""
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique")
        result[key] = value
    return result


def _parse_json(data: bytes) -> JsonObject:
    """Decode one strict UTF-8 JSON object."""
    if not isinstance(data, bytes):
        raise TypeError("serialised memory data must be bytes")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_pairs_without_duplicates,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant} is not permitted")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("serialised memory data must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("serialised memory data must contain a JSON object")
    return cast(JsonObject, value)


def _require_exact_fields(
    payload: JsonObject,
    expected: frozenset[str],
    field_name: str,
) -> None:
    """Reject missing and unknown fields in one JSON object."""
    actual = frozenset(payload)
    if actual != expected:
        missing = expected - actual
        detail = "missing fields" if missing else "unknown fields"
        raise ValueError(f"{field_name} contains {detail}")


def _require_string(payload: JsonObject, field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_optional_string(payload: JsonObject, field_name: str) -> str | None:
    value = payload[field_name]
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or null")
    return value


def _require_integer(payload: JsonObject, field_name: str) -> int:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _require_boolean(payload: JsonObject, field_name: str) -> bool:
    value = payload[field_name]
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _require_memory_category(
    payload: JsonObject,
    field_name: str,
) -> MemoryResourceCategory:
    value = _require_string(payload, field_name)
    if value not in MEMORY_RESOURCE_CATEGORIES:
        raise ValueError(f"{field_name} must be a supported memory category")
    return cast(MemoryResourceCategory, value)


def _require_measurement_status(
    payload: JsonObject,
    field_name: str,
) -> MeasurementStatus:
    value = _require_string(payload, field_name)
    if value not in MEASUREMENT_STATUSES:
        raise ValueError(f"{field_name} must be a supported measurement status")
    return cast(MeasurementStatus, value)


def _parse_datetime(payload: JsonObject, field_name: str) -> datetime:
    value = _require_string(payload, field_name)
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO datetime") from error


def _adapter_payload(adapter: MemoryAdapterIdentity) -> JsonObject:
    return {
        "adapter_name": adapter.adapter_name,
        "adapter_version": adapter.adapter_version,
        "calibration_revision": adapter.calibration_revision,
    }


def _parse_adapter(value: JsonValue) -> MemoryAdapterIdentity:
    if not isinstance(value, dict):
        raise TypeError("adapter must be a JSON object")
    payload = cast(JsonObject, value)
    _require_exact_fields(payload, _ADAPTER_FIELDS, "adapter")
    return MemoryAdapterIdentity(
        adapter_name=_require_string(payload, "adapter_name"),
        adapter_version=_require_string(payload, "adapter_version"),
        calibration_revision=_require_string(payload, "calibration_revision"),
    )


def _collector_payload(collector: MemoryCollectorIdentity) -> JsonObject:
    return {
        "collector_name": collector.collector_name,
        "collector_version": collector.collector_version,
        "inventory_revision": collector.inventory_revision,
    }


def _parse_collector(value: JsonValue) -> MemoryCollectorIdentity:
    if not isinstance(value, dict):
        raise TypeError("collector must be a JSON object")
    payload = cast(JsonObject, value)
    _require_exact_fields(payload, _COLLECTOR_FIELDS, "collector")
    return MemoryCollectorIdentity(
        collector_name=_require_string(payload, "collector_name"),
        collector_version=_require_string(payload, "collector_version"),
        inventory_revision=_require_string(payload, "inventory_revision"),
    )


def _expected_contribution_payload(
    expected: ExpectedMemoryContribution,
) -> JsonObject:
    return {
        "adapter": _adapter_payload(expected.adapter),
        "category": expected.category,
        "component_id": expected.component_id,
        "component_role": expected.component_role.value,
        "device_id": expected.device_id,
        "memory_space": expected.memory_space.value,
    }


def _parse_expected_contribution(value: JsonValue) -> ExpectedMemoryContribution:
    if not isinstance(value, dict):
        raise TypeError("expected_contributions must contain JSON objects")
    payload = cast(JsonObject, value)
    _require_exact_fields(
        payload,
        _EXPECTED_CONTRIBUTION_FIELDS,
        "expected contribution",
    )
    try:
        component_role = MemoryComponentRole(_require_string(payload, "component_role"))
        memory_space = MemorySpace(_require_string(payload, "memory_space"))
    except ValueError as error:
        raise ValueError(
            "expected contribution contains an unsupported enum value"
        ) from error
    return ExpectedMemoryContribution(
        component_id=_require_string(payload, "component_id"),
        component_role=component_role,
        category=_require_memory_category(payload, "category"),
        memory_space=memory_space,
        device_id=_require_optional_string(payload, "device_id"),
        adapter=_parse_adapter(payload["adapter"]),
    )


def _measurement_payload(measurement: MemoryMeasurement) -> JsonObject:
    return {
        "adapter": _adapter_payload(measurement.adapter),
        "category": measurement.category,
        "component_id": measurement.component_id,
        "component_role": measurement.component_role.value,
        "contribution_kind": measurement.contribution_kind.value,
        "covers_measurement_ids": list(measurement.covers_measurement_ids),
        "device_id": measurement.device_id,
        "measurement_id": measurement.measurement_id,
        "memory_space": measurement.memory_space.value,
        "observed_at": measurement.observed_at.isoformat(),
        "status": measurement.status,
        "value_bytes": measurement.value_bytes,
    }


def _parse_measurement(value: JsonValue) -> MemoryMeasurement:
    if not isinstance(value, dict):
        raise TypeError("measurements must contain JSON objects")
    payload = cast(JsonObject, value)
    _require_exact_fields(payload, _MEASUREMENT_FIELDS, "measurement")
    coverage = payload["covers_measurement_ids"]
    if not isinstance(coverage, list):
        raise TypeError("covers_measurement_ids must be a JSON array")
    if not all(isinstance(item, str) for item in coverage):
        raise TypeError("covers_measurement_ids must contain strings")
    try:
        component_role = MemoryComponentRole(_require_string(payload, "component_role"))
        contribution_kind = ContributionKind(
            _require_string(payload, "contribution_kind")
        )
        memory_space = MemorySpace(_require_string(payload, "memory_space"))
    except ValueError as error:
        raise ValueError("measurement contains an unsupported enum value") from error
    value_bytes = payload["value_bytes"]
    if value_bytes is not None and (
        isinstance(value_bytes, bool) or not isinstance(value_bytes, int)
    ):
        raise TypeError("value_bytes must be an integer or null")
    status = _require_measurement_status(payload, "status")
    return MemoryMeasurement(
        measurement_id=_require_string(payload, "measurement_id"),
        component_id=_require_string(payload, "component_id"),
        component_role=component_role,
        category=_require_memory_category(payload, "category"),
        memory_space=memory_space,
        device_id=_require_optional_string(payload, "device_id"),
        status=status,
        value_bytes=value_bytes,
        adapter=_parse_adapter(payload["adapter"]),
        contribution_kind=contribution_kind,
        covers_measurement_ids=tuple(cast(list[str], coverage)),
        observed_at=_parse_datetime(payload, "observed_at"),
    )


def serialise_memory_snapshot(snapshot: MemorySnapshot) -> bytes:
    """Serialise a validated snapshot as canonical UTF-8 JSON."""
    if not isinstance(snapshot, MemorySnapshot):
        raise TypeError("snapshot must be a MemorySnapshot")
    payload: JsonObject = {
        "collector": _collector_payload(snapshot.collector),
        "expected_contributions": [
            _expected_contribution_payload(expected)
            for expected in snapshot.expected_contributions
        ],
        "measurements": [
            _measurement_payload(measurement)
            for measurement in sorted(
                snapshot.measurements,
                key=lambda item: item.measurement_id,
            )
        ],
        "observed_at": snapshot.observed_at.isoformat(),
        "overlap_policy": snapshot.overlap_policy.value,
        "request_id": snapshot.request_id,
        "run_id": snapshot.run_id,
        "schema_version": snapshot.schema_version,
    }
    return _json_bytes(payload)


def deserialise_memory_snapshot(data: bytes) -> MemorySnapshot:
    """Strictly deserialise one canonical memory snapshot."""
    payload = _parse_json(data)
    _require_exact_fields(payload, _SNAPSHOT_FIELDS, "snapshot")
    schema_version = _require_string(payload, "schema_version")
    if schema_version not in SUPPORTED_MEMORY_SNAPSHOT_SCHEMA_VERSIONS:
        raise ValueError("unsupported memory snapshot schema version")
    measurements = payload["measurements"]
    if not isinstance(measurements, list):
        raise TypeError("measurements must be a JSON array")
    expected_contributions = payload["expected_contributions"]
    if not isinstance(expected_contributions, list):
        raise TypeError("expected_contributions must be a JSON array")
    try:
        overlap_policy = MemoryOverlapPolicy(_require_string(payload, "overlap_policy"))
    except ValueError as error:
        raise ValueError("snapshot contains an unsupported overlap policy") from error
    return MemorySnapshot(
        observed_at=_parse_datetime(payload, "observed_at"),
        collector=_parse_collector(payload["collector"]),
        expected_contributions=tuple(
            _parse_expected_contribution(item) for item in expected_contributions
        ),
        measurements=tuple(_parse_measurement(item) for item in measurements),
        overlap_policy=overlap_policy,
        run_id=_require_optional_string(payload, "run_id"),
        request_id=_require_optional_string(payload, "request_id"),
        schema_version=schema_version,
    )


def _budget_payload(budget: MemoryBudget) -> JsonObject:
    return {
        "budget_bytes": budget.budget_bytes,
        "device_id": budget.device_id,
        "memory_space": budget.memory_space.value,
    }


def _parse_budget(value: JsonValue) -> MemoryBudget:
    if not isinstance(value, dict):
        raise TypeError("budget must be a JSON object")
    payload = cast(JsonObject, value)
    _require_exact_fields(payload, _BUDGET_FIELDS, "budget")
    try:
        memory_space = MemorySpace(_require_string(payload, "memory_space"))
    except ValueError as error:
        raise ValueError("budget contains an unsupported memory space") from error
    return MemoryBudget(
        memory_space=memory_space,
        device_id=_require_optional_string(payload, "device_id"),
        budget_bytes=_require_integer(payload, "budget_bytes"),
    )


def serialise_budget_comparison(comparison: BudgetComparison) -> bytes:
    """Serialise a validated budget comparison as canonical UTF-8 JSON."""
    if not isinstance(comparison, BudgetComparison):
        raise TypeError("comparison must be a BudgetComparison")
    return _json_bytes(
        {
            "budget": _budget_payload(comparison.budget),
            "completeness": comparison.completeness.value,
            "measured_bytes": comparison.measured_bytes,
            "measured_count": comparison.measured_count,
            "missing_count": comparison.missing_count,
            "provisional_headroom_bytes": comparison.provisional_headroom_bytes,
            "state": comparison.state.value,
            "unavailable_count": comparison.unavailable_count,
            "unreliable_count": comparison.unreliable_count,
        }
    )


def deserialise_budget_comparison(data: bytes) -> BudgetComparison:
    """Strictly deserialise one canonical budget comparison."""
    payload = _parse_json(data)
    _require_exact_fields(payload, _COMPARISON_FIELDS, "budget comparison")
    try:
        state = BudgetState(_require_string(payload, "state"))
        completeness = Completeness(_require_string(payload, "completeness"))
    except ValueError as error:
        raise ValueError(
            "budget comparison contains an unsupported enum value"
        ) from error
    return BudgetComparison(
        state=state,
        budget=_parse_budget(payload["budget"]),
        measured_bytes=_require_integer(payload, "measured_bytes"),
        provisional_headroom_bytes=_require_integer(
            payload,
            "provisional_headroom_bytes",
        ),
        measured_count=_require_integer(payload, "measured_count"),
        unreliable_count=_require_integer(payload, "unreliable_count"),
        unavailable_count=_require_integer(payload, "unavailable_count"),
        missing_count=_require_integer(payload, "missing_count"),
        completeness=completeness,
    )
