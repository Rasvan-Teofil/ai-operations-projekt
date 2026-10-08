"""Split und Dummy-Datensatz."""

from src.config import LABEL_COLUMN, LABELS_3, LANGUAGE_COLUMN, ORIGIN_COLUMN, TEST_SIZE, TEXT_COLUMN
from src.data import load_dataset, split_dataset


def test_sample_is_marked_as_dummy():
    frame = load_dataset()
    assert set(frame[ORIGIN_COLUMN]) == {"dummy"}
    assert set(frame[LANGUAGE_COLUMN]) == {"de", "en"}
    assert set(frame[LABEL_COLUMN]) == set(LABELS_3)
    assert frame[TEXT_COLUMN].str.strip().ne("").all()


def test_split_is_complete_and_reproducible():
    frame = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(frame)
    x_train_again, x_test_again, _, _ = split_dataset(frame)

    assert list(x_train.columns) == [TEXT_COLUMN, LANGUAGE_COLUMN]
    assert LABEL_COLUMN not in x_train.columns
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert set(x_train.index).isdisjoint(x_test.index)
    assert set(x_train.index) | set(x_test.index) == set(frame.index)
    assert abs(len(x_test) / len(frame) - TEST_SIZE) < 0.05
    assert x_train.index.equals(x_train_again.index)
    assert x_test.index.equals(x_test_again.index)
    assert set(y_train) == set(LABELS_3)
    assert set(y_test) <= set(LABELS_3)
