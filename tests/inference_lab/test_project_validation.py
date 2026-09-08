"""Tests for project-wide validation predicates."""

from pathlib import Path

import pytest

from inference_lab._validation import (
    is_canonical_media_type,
    is_local_path_shape,
    is_lowercase_slug,
    paths_overlap,
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


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Path("state"), Path("state"), True),
        (Path("state"), Path("state/indexes"), True),
        (Path("state/indexes"), Path("state"), True),
        (Path("state/indexes"), Path("state/sources"), False),
    ],
)
def test_path_overlap_policy(first: Path, second: Path, expected: bool) -> None:
    assert paths_overlap(first, second) is expected


@pytest.mark.parametrize(("first", "second"), [("state", Path("state")), (Path(), 1)])
def test_path_overlap_policy_requires_paths(first: object, second: object) -> None:
    with pytest.raises(TypeError, match="must be a Path"):
        paths_overlap(first, second)  # type: ignore[arg-type]


def test_local_path_shape_policy(
    local_path_shape_case: tuple[str | Path, bool],
) -> None:
    value, expected = local_path_shape_case
    assert is_local_path_shape(value) is expected


@pytest.mark.parametrize("value", [None, 1, object()])
def test_local_path_shape_rejects_non_path_values(value: object) -> None:
    assert not is_local_path_shape(value)
