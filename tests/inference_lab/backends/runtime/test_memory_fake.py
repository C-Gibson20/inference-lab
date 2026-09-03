from datetime import UTC, datetime

import pytest

from inference_lab.backends.runtime import (
    FakeMemoryAccountingAdapter,
    MemoryAdapterIdentity,
    MemoryComponentRole,
    MemoryMeasurement,
    MemorySpace,
)

ADAPTER = MemoryAdapterIdentity("fake", "1.0.0", "calibration-1")
SOURCE_TIME = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
INJECTED_TIME = datetime(2026, 8, 21, 12, 1, tzinfo=UTC)


def make_measurement() -> MemoryMeasurement:
    return MemoryMeasurement(
        measurement_id="weights",
        component_id="generator",
        component_role=MemoryComponentRole.GENERATOR,
        category="model_weights",
        memory_space=MemorySpace.ACCELERATOR,
        device_id="cuda-0",
        status="measured",
        value_bytes=123,
        adapter=ADAPTER,
        observed_at=SOURCE_TIME,
    )


def test_fake_adapter_returns_injected_values_and_clock() -> None:
    adapter = FakeMemoryAccountingAdapter(
        (make_measurement(),),
        clock=lambda: INJECTED_TIME,
    )

    first = adapter.measure_memory()
    second = adapter.measure_memory()

    assert first == second
    assert first is not adapter.measure_memory()
    assert first[0].value_bytes == 123
    assert first[0].observed_at == INJECTED_TIME


def test_fake_adapter_validates_immutable_input_and_clock() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        FakeMemoryAccountingAdapter([make_measurement()])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be callable"):
        FakeMemoryAccountingAdapter((make_measurement(),), clock=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must return a datetime"):
        FakeMemoryAccountingAdapter(
            (make_measurement(),),
            clock=lambda: "not-a-time",  # type: ignore[return-value]
        ).measure_memory()
