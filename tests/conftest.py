"""Shared fixtures for cross-domain contract tests."""

import os
from pathlib import Path
from typing import Final, cast

import pytest

_LOCAL_PATH_SHAPE_CASES: Final = (
    ("relative/path", True),
    (Path("relative/path"), True),
    ("/absolute/local/path", True),
    ("~/local/path", True),
    (r"C:\local\path", os.name == "nt"),
    ("C:/local/path", os.name == "nt"),
    ("", False),
    ("   ", False),
    ("invalid\x00path", False),
    ("https://example.test/path", False),
    (Path("https://example.test/path"), False),
    ("https:/example.test/path", False),
    ("s3://private-bucket/path", False),
    ("file:/local-looking/path", False),
    ("mailto:user@example.test", False),
    ("//remote-host/share/path", False),
    (r"\\remote-host\share\path", False),
)
_REJECTED_LOCAL_PATH_SHAPES: Final = tuple(
    value for value, is_local in _LOCAL_PATH_SHAPE_CASES if not is_local
)


@pytest.fixture(params=_LOCAL_PATH_SHAPE_CASES)
def local_path_shape_case(
    request: pytest.FixtureRequest,
) -> tuple[str | Path, bool]:
    """Provide one path-shaped value and its host-specific classification."""
    return cast(tuple[str | Path, bool], request.param)


@pytest.fixture(params=_REJECTED_LOCAL_PATH_SHAPES)
def rejected_local_path_shape(request: pytest.FixtureRequest) -> str | Path:
    """Provide one path value rejected by the shared host policy."""
    return cast(str | Path, request.param)
