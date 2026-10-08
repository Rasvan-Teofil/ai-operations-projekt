"""Prüft, dass der Split an einer Stelle passiert und wiederholbar ist."""

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE
from src.data import load_dataset, split_dataset


def test_split_is_complete_and_reproducible():
    frame = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(frame)
    x_train_again, x_test_again, _, _ = split_dataset(frame)

    assert list(x_train.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in x_train.columns
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(x_train.index) | set(x_test.index) == set(frame.index)
    assert abs(len(x_test) / len(frame) - TEST_SIZE) < 0.05
    assert x_train.index.equals(x_train_again.index)
    assert x_test.index.equals(x_test_again.index)
