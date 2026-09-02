from datetime import UTC, datetime

import pytest

from inference_lab.backends.runtime import (
    ComponentRole,
    ContributionKind,
    MemoryAccountingAdapter,
    MemoryAdapterIdentity,
    MemoryMeasurement,
    MemorySpace,
)

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")


def measurement(**overrides: object) -> MemoryMeasurement:
    values: dict[str, object] = {
        "measurement_id": "generator-weights",
        "component_id": "generator",
        "component_role": ComponentRole.GENERATOR,
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


@pytest.mark.parametrize("role", list(ComponentRole))
def test_all_component_roles_are_bounded(role: ComponentRole) -> None:
    assert measurement(component_role=role).component_role is role


@pytest.mark.parametrize("space", list(MemorySpace))
def test_all_memory_spaces_are_supported(space: MemorySpace) -> None:
    assert measurement(memory_space=space).memory_space is space


def test_measurement_is_immutable_and_requires_tuple_coverage() -> None:
    record = measurement()

    with pytest.raises((AttributeError, TypeError)):
        record.value_bytes = 2  # type: ignore[misc]
    with pytest.raises(TypeError, match="must be a tuple"):
        measurement(covers_measurement_ids=["child"])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_byte_values_are_non_negative_integers(value: object) -> None:
    expected = "must not be negative" if value == -1 else "must be an integer"
    with pytest.raises((TypeError, ValueError), match=expected):
        measurement(value_bytes=value)  # type: ignore[arg-type]


def test_status_presence_rules_are_strict() -> None:
    with pytest.raises(ValueError, match="measured memory must have a value"):
        measurement(value_bytes=None)
    with pytest.raises(ValueError, match="unavailable memory must not have a value"):
        measurement(status="unavailable", value_bytes=1)
    assert measurement(status="unavailable", value_bytes=None).value_bytes is None
    assert measurement(status="unreliable", value_bytes=1).status == "unreliable"


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
