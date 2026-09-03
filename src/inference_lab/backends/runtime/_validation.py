"""Shared primitive validation for runtime value contracts."""

import math


def validate_non_empty_string(value: object, field_name: str) -> None:
    """Require a string containing at least one character."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def validate_integer(value: object, field_name: str) -> None:
    """Require an integer while excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")


def validate_non_negative_integer(value: object, field_name: str) -> None:
    """Require a non-negative integer while excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def validate_finite_number(value: object, field_name: str) -> None:
    """Require a finite integer or floating-point number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
