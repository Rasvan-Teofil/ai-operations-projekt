"""The handwritten slice stays out of val and test."""

from src.config import LABEL_COLUMN, LABELS_3, LANGUAGE_COLUMN, TEXT_COLUMN
from src.data import load_opinion_supplement, load_work_splits
from tests.test_api import OBVIOUS_NEGATIVES


def test_supplement_rows_are_labelled_and_original():
    frame = load_opinion_supplement()
    assert len(frame) >= 40
    assert set(frame[LABEL_COLUMN]) == set(LABELS_3)
    assert set(frame[LANGUAGE_COLUMN]) == {"de", "en"}
    assert frame[TEXT_COLUMN].is_unique
    for probe in OBVIOUS_NEGATIVES:
        assert probe not in set(frame[TEXT_COLUMN])


def test_supplement_is_appended_to_train_only(monkeypatch):
    monkeypatch.setenv("SENTIMENT_DATA", "dummy")
    train, val, test = load_work_splits()
    texts = set(load_opinion_supplement()[TEXT_COLUMN])
    assert texts <= set(train[TEXT_COLUMN])
    assert texts.isdisjoint(set(val[TEXT_COLUMN]))
    assert texts.isdisjoint(set(test[TEXT_COLUMN]))
