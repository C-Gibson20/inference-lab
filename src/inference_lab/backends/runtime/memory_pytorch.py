"""One-shot PyTorch calibration for inspectable tensor-backed allocations."""

from collections.abc import Callable, Iterator
from datetime import datetime

import torch

from inference_lab.backends.runtime.memory import (
    ComponentRole,
    MemoryAccountingAdapter,
    MemoryAdapterIdentity,
    MemoryMeasurement,
    MemorySpace,
)
from inference_lab.profiling.utils import utc_now


def _device_identity(device: torch.device) -> tuple[MemorySpace, str]:
    """Map a PyTorch device to a safe accounting space and device identifier."""
    if device.type == "cpu":
        return MemorySpace.HOST, "cpu"
    device_id = f"{device.type}-{device.index if device.index is not None else 0}"
    return MemorySpace.ACCELERATOR, device_id


def _iter_tensors(value: object) -> Iterator[torch.Tensor]:
    """Yield tensors from supported native cache/container shapes only."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_tensors(child)
    else:
        for attribute in ("key_cache", "value_cache"):
            child = getattr(value, attribute, None)
            if child is not None:
                yield from _iter_tensors(child)


def _unique_storage_bytes(tensors: Iterator[torch.Tensor]) -> dict[str, int]:
    """Count each backing storage once, grouped by safe device identity."""
    seen: set[tuple[str, int]] = set()
    totals: dict[str, int] = {}
    for tensor in tensors:
        if tensor.device.type == "meta":
            raise ValueError("meta tensors have no inspectable storage")
        storage = tensor.untyped_storage()
        device_key = str(tensor.device)
        storage_key = (device_key, storage.data_ptr())
        if storage_key in seen:
            continue
        seen.add(storage_key)
        totals[device_key] = totals.get(device_key, 0) + storage.nbytes()
    return totals


class PyTorchMemoryCalibrationAdapter(MemoryAccountingAdapter):
    """Calibrate exact model and optional native-KV storage once per call.

    This adapter deliberately reports only tensor-backed allocations that
    PyTorch exposes directly. It does not infer allocator overhead, process
    RSS, activation lifetimes, or device-reserved memory.
    """

    def __init__(
        self,
        model: torch.nn.Module | None,
        *,
        component_id: str = "generator",
        component_role: ComponentRole = ComponentRole.GENERATOR,
        kv_cache: object | None = None,
        adapter: MemoryAdapterIdentity | None = None,
        observed_at_factory: Callable[[], datetime] = utc_now,
    ) -> None:
        """Configure an explicit local model and optional inspectable cache."""
        if model is not None and not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        if not isinstance(component_id, str):
            raise TypeError("component_id must be a string")
        if not isinstance(component_role, ComponentRole):
            raise TypeError("component_role must be a ComponentRole")
        if adapter is not None and not isinstance(adapter, MemoryAdapterIdentity):
            raise TypeError("adapter must be a MemoryAdapterIdentity or None")
        if not callable(observed_at_factory):
            raise TypeError("observed_at_factory must be callable")
        self._model = model
        self._component_id = component_id
        self._component_role = component_role
        self._kv_cache = kv_cache
        self._adapter = adapter or MemoryAdapterIdentity(
            adapter_name="pytorch",
            adapter_version=torch.__version__.split("+")[0],
            calibration_revision="unique-storage-nbytes-v1",
        )
        self._observed_at_factory = observed_at_factory

    @property
    def adapter_identity(self) -> MemoryAdapterIdentity:
        """Return the framework and calibration identity used by the adapter."""
        return self._adapter

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Return exact tensor-backed observations and honest fallbacks."""
        observed_at = self._observed_at_factory()
        if not isinstance(observed_at, datetime):
            raise TypeError("observed_at_factory must return a datetime")
        measurements: list[MemoryMeasurement] = []
        measurements.extend(self._measure_model(observed_at))
        measurements.extend(self._measure_kv_cache(observed_at))
        return tuple(
            sorted(measurements, key=lambda measurement: measurement.measurement_id)
        )

    def _measure_model(self, observed_at: datetime) -> tuple[MemoryMeasurement, ...]:
        if self._model is None:
            return (self._unavailable("model-weights", "model_weights", observed_at),)
        try:
            tensors = iter((*self._model.parameters(), *self._model.buffers()))
            totals = _unique_storage_bytes(tensors)
        except (RuntimeError, ValueError):
            return (self._unavailable("model-weights", "model_weights", observed_at),)
        if not totals:
            return (self._unavailable("model-weights", "model_weights", observed_at),)
        return tuple(
            self._measured(
                measurement_id=f"model-weights-{device.replace(':', '-')}",
                category="model_weights",
                component_role=self._component_role,
                memory_space=_device_identity(torch.device(device))[0],
                device_id=_device_identity(torch.device(device))[1],
                value_bytes=value,
                observed_at=observed_at,
            )
            for device, value in sorted(totals.items())
        )

    def _measure_kv_cache(
        self,
        observed_at: datetime,
    ) -> tuple[MemoryMeasurement, ...]:
        if self._kv_cache is None:
            return (self._unavailable("kv-cache", "kv_cache", observed_at),)
        try:
            totals = _unique_storage_bytes(_iter_tensors(self._kv_cache))
        except (RuntimeError, ValueError):
            return (self._unavailable("kv-cache", "kv_cache", observed_at),)
        if not totals:
            return (self._unavailable("kv-cache", "kv_cache", observed_at),)
        return tuple(
            self._measured(
                measurement_id=f"kv-cache-{device.replace(':', '-')}",
                category="kv_cache",
                component_role=ComponentRole.KV_CACHE,
                memory_space=_device_identity(torch.device(device))[0],
                device_id=_device_identity(torch.device(device))[1],
                value_bytes=value,
                observed_at=observed_at,
            )
            for device, value in sorted(totals.items())
        )

    def _measured(
        self,
        *,
        measurement_id: str,
        category: str,
        component_role: ComponentRole,
        memory_space: MemorySpace,
        device_id: str,
        value_bytes: int,
        observed_at: datetime,
    ) -> MemoryMeasurement:
        return MemoryMeasurement(
            measurement_id=measurement_id,
            component_id=self._component_id,
            component_role=component_role,
            category=category,  # type: ignore[arg-type]
            memory_space=memory_space,
            device_id=device_id,
            status="measured",
            value_bytes=value_bytes,
            adapter=self._adapter,
            observed_at=observed_at,
        )

    def _unavailable(
        self,
        measurement_id: str,
        category: str,
        observed_at: datetime,
    ) -> MemoryMeasurement:
        role = (
            self._component_role
            if category == "model_weights"
            else ComponentRole.KV_CACHE
        )
        return MemoryMeasurement(
            measurement_id=measurement_id,
            component_id=self._component_id,
            component_role=role,
            category=category,  # type: ignore[arg-type]
            memory_space=MemorySpace.ACCELERATOR,
            device_id=None,
            status="unavailable",
            value_bytes=None,
            adapter=self._adapter,
            observed_at=observed_at,
        )
