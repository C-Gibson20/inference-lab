"""Tests for project-wide validation predicates."""

from pathlib import Path

import pytest

from inference_lab._validation import (
    is_canonical_media_type,
    is_local_path_shape,
    is_lowercase_slug,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("parser", True),
        ("text-native", True),
        ("a1-b2", True),
        ("", False),
        ("Text-native", False),
        ("text_native", False),
        ("text--native", False),
        ("-text", False),
        ("text-", False),
        (None, False),
        (1, False),
    ],
)
def test_lowercase_slug_policy(value: object, expected: bool) -> None:
    assert is_lowercase_slug(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text/plain", True),
        ("application/vnd.example+json", True),
        ("application/x.custom_format", True),
        ("", False),
        ("text", False),
        ("text/plain; charset=utf-8", False),
        ("Text/Plain", False),
        ("text/plain/extra", False),
        (None, False),
        (1, False),
    ],
)
def test_canonical_media_type_policy(value: object, expected: bool) -> None:
    assert is_canonical_media_type(value) is expected


def test_local_path_shape_policy(
    local_path_shape_case: tuple[str | Path, bool],
) -> None:
    value, expected = local_path_shape_case
    assert is_local_path_shape(value) is expected


@pytest.mark.parametrize("value", [None, 1, object()])
def test_local_path_shape_rejects_non_path_values(value: object) -> None:
    assert not is_local_path_shape(value)
