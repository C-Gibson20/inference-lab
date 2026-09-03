"""Tests for PyTorch runtime memory measurement."""

from datetime import UTC, datetime

import pytest
import torch

from inference_lab.backends.runtime import (
    MemoryComponentRole,
    MemorySpace,
    PyTorchMemoryCalibrationAdapter,
)

OBSERVED_AT = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class SharedStorageModel(torch.nn.Module):
    """Small model with tied parameters, views, and one independent tensor."""

    def __init__(self) -> None:
        super().__init__()
        base = torch.empty(4, dtype=torch.float32)
        self.primary = torch.nn.Parameter(base)
        self.tied = self.primary
        self.view = torch.nn.Parameter(base[:2])
        self.independent = torch.nn.Parameter(torch.empty(3, dtype=torch.float64))


def test_calibration_counts_unique_parameter_storage_once() -> None:
    adapter = PyTorchMemoryCalibrationAdapter(
        SharedStorageModel(),
        observed_at_factory=lambda: OBSERVED_AT,
    )

    measurements = adapter.measure_memory()
    model_weights = next(
        measurement
        for measurement in measurements
        if measurement.category == "model_weights"
    )

    assert model_weights.value_bytes == 16 + 24
    assert model_weights.memory_space is MemorySpace.HOST
    assert model_weights.device_id == "cpu"
    assert model_weights.observed_at == OBSERVED_AT
    assert model_weights.adapter.adapter_name == "pytorch"
    assert model_weights.adapter.calibration_revision == "unique-storage-nbytes-v1"


def test_calibration_counts_unique_native_kv_storage_once() -> None:
    cache = torch.empty(2, 3, dtype=torch.float32)
    adapter = PyTorchMemoryCalibrationAdapter(
        None,
        kv_cache=(cache, cache[:1]),
        observed_at_factory=lambda: OBSERVED_AT,
    )

    measurements = adapter.measure_memory()
    kv_cache = next(
        measurement
        for measurement in measurements
        if measurement.category == "kv_cache"
    )

    assert kv_cache.value_bytes == 24
    assert kv_cache.component_role is MemoryComponentRole.KV_CACHE


def test_calibration_is_repeatable_and_returns_sorted_measurements() -> None:
    adapter = PyTorchMemoryCalibrationAdapter(
        SharedStorageModel(),
        kv_cache=torch.empty(2, 2),
        observed_at_factory=lambda: OBSERVED_AT,
    )

    first = adapter.measure_memory()
    second = adapter.measure_memory()

    assert first == second
    assert tuple(item.measurement_id for item in first) == tuple(
        sorted(item.measurement_id for item in first)
    )


def test_unsupported_or_missing_storage_is_unavailable() -> None:
    no_model = PyTorchMemoryCalibrationAdapter(
        None,
        observed_at_factory=lambda: OBSERVED_AT,
    )
    meta_model = PyTorchMemoryCalibrationAdapter(
        torch.nn.Linear(2, 2, device="meta"),
        observed_at_factory=lambda: OBSERVED_AT,
    )

    no_model_measurements = no_model.measure_memory()
    meta_measurements = meta_model.measure_memory()

    assert {item.category for item in no_model_measurements} == {
        "kv_cache",
        "model_weights",
    }
    assert {item.category for item in meta_measurements} == {
        "kv_cache",
        "model_weights",
    }
    assert all(item.status == "unavailable" for item in no_model_measurements)
    assert all(item.status == "unavailable" for item in meta_measurements)


def test_calibration_validates_inputs() -> None:
    with pytest.raises(TypeError, match="torch.nn.Module"):
        PyTorchMemoryCalibrationAdapter("model")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MemoryComponentRole"):
        PyTorchMemoryCalibrationAdapter(
            None,
            component_role="generator",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="callable"):
        PyTorchMemoryCalibrationAdapter(None, observed_at_factory=None)  # type: ignore[arg-type]
