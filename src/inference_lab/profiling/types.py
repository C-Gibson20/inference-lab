"""Shared type definitions for concrete profiling records."""

from typing import Literal

LogLevel = Literal[
    "debug",
    "info",
    "warning",
    "error",
    "critical",
]
