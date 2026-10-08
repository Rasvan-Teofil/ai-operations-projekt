"""Metriken und Konfusionsmatrix im festen Klassenraum."""

from src.evaluate import confusion_frame


def test_confusion_matrix_uses_all_three_classes():
    frame = confusion_frame(
        ["negative", "positive"],
        ["negative", "positive"],
    )
    assert list(frame.index) == ["negative", "neutral", "positive"]
    assert list(frame.columns) == ["negative", "neutral", "positive"]
    assert frame.loc["negative", "negative"] == 1
    assert frame.loc["positive", "positive"] == 1
    assert frame.loc["neutral", "neutral"] == 0
