"""Tests for shared evaluation validation utilities."""

from inference_lab.evaluation.utils import normalise_for_duplicate_detection


def test_duplicate_detection_normalization_is_explicitly_lossy() -> None:
    assert normalise_for_duplicate_detection("  Ａ\tStraße \n") == "a strasse"
