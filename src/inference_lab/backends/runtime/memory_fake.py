"""Deterministic fake memory-accounting adapter for offline tests."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from inference_lab._clock import utc_now
from inference_lab.backends.runtime._memory_values import (
    MemoryAccountingAdapter,
    MemoryMeasurement,
)


class FakeMemoryAccountingAdapter(MemoryAccountingAdapter):
    """Return injected observations stamped by an injected UTC clock."""

    def __init__(
        self,
        measurements: tuple[MemoryMeasurement, ...],
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        """Configure deterministic values and the observation-time factory."""
        if not isinstance(measurements, tuple):
            raise TypeError("measurements must be a tuple")
        if any(not isinstance(item, MemoryMeasurement) for item in measurements):
            raise TypeError("measurements must contain MemoryMeasurement values")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._measurements = measurements
        self._clock = clock

    def measure_memory(self) -> tuple[MemoryMeasurement, ...]:
        """Return fresh immutable observations with the injected timestamp."""
        observed_at = self._clock()
        if not isinstance(observed_at, datetime):
            raise TypeError("clock must return a datetime")
        return tuple(
            replace(measurement, observed_at=observed_at)
            for measurement in self._measurements
        )
