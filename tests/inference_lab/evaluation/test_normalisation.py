"""Tests for evaluation duplicate-detection normalisation."""

from inference_lab.evaluation.normalisation import normalise_for_duplicate_detection


def test_normalise_for_duplicate_detection_is_lossy_and_deterministic() -> None:
    assert normalise_for_duplicate_detection("  Ａ\tStraße \n") == "a strasse"
