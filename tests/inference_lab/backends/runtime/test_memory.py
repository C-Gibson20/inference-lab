from datetime import UTC, datetime

import pytest

import inference_lab.backends.runtime as runtime
from inference_lab.backends.runtime import (
    ContributionKind,
    MemoryAccountingAdapter,
    MemoryAdapterIdentity,
    MemoryCollectorIdentity,
    MemoryComponentRole,
    MemoryMeasurement,
    MemorySpace,
)

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")


def measurement(**overrides: object) -> MemoryMeasurement:
    values: dict[str, object] = {
        "measurement_id": "generator-weights",
        "component_id": "generator",
        "component_role": MemoryComponentRole.GENERATOR,
        "category": "model_weights",
        "memory_space": MemorySpace.ACCELERATOR,
        "device_id": "cuda-0",
        "status": "measured",
        "value_bytes": 1024,
        "adapter": ADAPTER,
        "observed_at": OBSERVED_AT,
    }
    values.update(overrides)
    return MemoryMeasurement(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("role", list(MemoryComponentRole))
def test_all_component_roles_are_bounded(role: MemoryComponentRole) -> None:
    assert measurement(component_role=role).component_role is role


def test_memory_role_has_an_unambiguous_public_name() -> None:
    assert "MemoryComponentRole" in runtime.__all__
    assert "ComponentRole" not in runtime.__all__
    assert not hasattr(runtime, "ComponentRole")


@pytest.mark.parametrize("category", ["gpu_utilisation", "power"])
def test_runtime_rejects_non_memory_profiling_categories(category: str) -> None:
    with pytest.raises(ValueError, match="supported memory category"):
        measurement(category=category)


@pytest.mark.parametrize("space", list(MemorySpace))
def test_all_memory_spaces_are_supported(space: MemorySpace) -> None:
    assert measurement(memory_space=space).memory_space is space


def test_measurement_is_immutable_and_requires_tuple_coverage() -> None:
    record = measurement()

    with pytest.raises((AttributeError, TypeError)):
        record.value_bytes = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be a tuple"):
        measurement(covers_measurement_ids=["child"])  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["1", "1.0", "01.0.0", "1.0.0-01"])
def test_memory_identity_versions_require_canonical_semantic_versions(
    version: str,
) -> None:
    with pytest.raises(ValueError, match="adapter_version"):
        MemoryAdapterIdentity("fake", version, "calibration-1")
    with pytest.raises(ValueError, match="collector_version"):
        MemoryCollectorIdentity("runtime", version, "inventory-1")


def test_memory_identity_versions_accept_semantic_version_extensions() -> None:
    version = "1.2.0-rc.1+local"

    assert (
        MemoryAdapterIdentity("fake", version, "calibration-1").adapter_version
        == version
    )
    assert (
        MemoryCollectorIdentity("runtime", version, "inventory-1").collector_version
        == version
    )


@pytest.mark.parametrize("version", [1, True, None])
def test_memory_identity_versions_require_strings(version: object) -> None:
    with pytest.raises(TypeError, match="adapter_version must be a string"):
        MemoryAdapterIdentity("fake", version, "calibration-1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="collector_version must be a string"):
        MemoryCollectorIdentity(
            "runtime",
            version,  # type: ignore[arg-type]
            "inventory-1",
        )


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_byte_values_are_non_negative_integers(value: object) -> None:
    expected = "must not be negative" if value == -1 else "must be an integer"
    with pytest.raises((TypeError, ValueError), match=expected):
        measurement(value_bytes=value)  # type: ignore[arg-type]


def test_status_presence_rules_are_strict() -> None:
    with pytest.raises(ValueError, match="supported measurement status"):
        measurement(status="unknown")
    with pytest.raises(ValueError, match="measured memory must have a value"):
        measurement(value_bytes=None)
    with pytest.raises(ValueError, match="unavailable memory must not have a value"):
        measurement(status="unavailable", value_bytes=1)
    assert measurement(status="unavailable", value_bytes=None).value_bytes is None
    assert measurement(status="unreliable", value_bytes=1).status == "unreliable"


def test_measured_accelerator_memory_requires_a_device() -> None:
    with pytest.raises(ValueError, match="must identify its device"):
        measurement(device_id=None)

    assert (
        measurement(status="unavailable", value_bytes=None, device_id=None).device_id
        is None
    )


def test_roll_up_requires_unique_non_self_coverage() -> None:
    with pytest.raises(ValueError, match="must cover contributions"):
        measurement(contribution_kind=ContributionKind.ROLL_UP)
    with pytest.raises(ValueError, match="duplicates"):
        measurement(
            contribution_kind=ContributionKind.ROLL_UP,
            covers_measurement_ids=("child", "child"),
        )
    with pytest.raises(ValueError, match="cannot cover itself"):
        measurement(
            contribution_kind=ContributionKind.ROLL_UP,
            covers_measurement_ids=("generator-weights",),
        )


def test_structural_optional_adapter_protocol() -> None:
    class Adapter:
        def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
            return (measurement(),)

    assert isinstance(Adapter(), MemoryAccountingAdapter)
    assert not isinstance(object(), MemoryAccountingAdapter)
