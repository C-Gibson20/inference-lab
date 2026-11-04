"""Tests for project-wide validation predicates."""

from pathlib import Path

import pytest

from inference_lab._validation import (
    is_canonical_media_type,
    is_lowercase_slug,
    is_path_free_identifier,
    paths_overlap,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text-native", True),
        ("Text-native", False),
        ("text--native", False),
    ],
)
def test_lowercase_slug_policy(value: object, expected: bool) -> None:
    assert is_lowercase_slug(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text/plain", True),
        ("application/vnd.example+json", True),
        ("text/plain; charset=utf-8", False),
    ],
)
def test_canonical_media_type_policy(value: object, expected: bool) -> None:
    assert is_canonical_media_type(value) is expected


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Path("state"), Path("state"), True),
        (Path("state"), Path("state/indexes"), True),
        (Path("state/indexes"), Path("state/sources"), False),
    ],
)
def test_path_overlap_policy(first: Path, second: Path, expected: bool) -> None:
    assert paths_overlap(first, second) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", True),
        ("bge-reranker-v2-m3", True),
        ("1.0.0", True),
        ("/mnt/models/embedder", False),
        ("~/models/embedder", False),
        ("C:/models/embedder", False),
        ("C:\\models\\embedder", False),
        ("https://example.invalid/model", False),
        ("file://model", False),
        (3, False),
    ],
)
def test_path_free_identifier_policy(value: object, expected: bool) -> None:
    assert is_path_free_identifier(value) is expected
