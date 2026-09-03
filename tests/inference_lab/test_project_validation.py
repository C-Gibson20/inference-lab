"""Tests for project-wide validation predicates."""

from pathlib import Path

import pytest

from inference_lab._validation import is_local_path_shape


def test_local_path_shape_policy(
    local_path_shape_case: tuple[str | Path, bool],
) -> None:
    value, expected = local_path_shape_case
    assert is_local_path_shape(value) is expected


@pytest.mark.parametrize("value", [None, 1, object()])
def test_local_path_shape_rejects_non_path_values(value: object) -> None:
    assert not is_local_path_shape(value)
