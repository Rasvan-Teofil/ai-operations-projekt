"""Abbildung auf drei Klassen und Chunk-Längen, ohne Modellgewichte."""

import pytest

from src.hf_models import label_maps_for_docs
from src.labels import (
    canonical_label,
    collapse_probabilities,
    content_windows,
    mean_probability_rows,
)


def test_five_class_names_collapse_to_three():
    scores = collapse_probabilities(
        ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        [0.1, 0.2, 0.3, 0.15, 0.25],
    )
    assert scores["negative"] == pytest.approx(0.3)
    assert scores["neutral"] == pytest.approx(0.3)
    assert scores["positive"] == pytest.approx(0.4)


def test_cardiff_style_labels_stay_in_place():
    scores = collapse_probabilities(
        ["negative", "neutral", "positive"],
        [0.2, 0.5, 0.3],
    )
    assert scores["negative"] == pytest.approx(0.2)
    assert scores["neutral"] == pytest.approx(0.5)
    assert scores["positive"] == pytest.approx(0.3)


def test_mean_of_chunks_then_collapse():
    mean = mean_probability_rows([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
    scores = collapse_probabilities(
        ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
        mean,
    )
    assert scores["negative"] == pytest.approx(0.5)
    assert scores["positive"] == pytest.approx(0.5)
    assert scores["neutral"] == pytest.approx(0.0)


def test_content_windows_leave_room_for_special_tokens():
    windows = content_windows(list(range(1000)), max_length=512, special_tokens=2)
    assert windows
    assert all(len(window) <= 510 for window in windows)
    assert sum(len(window) for window in windows) == 1000


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        canonical_label("LABEL_0")


def test_documented_label_maps_match_the_checked_configs():
    maps = label_maps_for_docs()
    assert maps["tabularisai/multilingual-sentiment-analysis"] == [
        "Very Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Very Positive",
    ]
    assert maps["cardiffnlp/twitter-xlm-roberta-base-sentiment"] == [
        "negative",
        "neutral",
        "positive",
    ]
