"""Split-Helfer ohne Netz und ohne die echten Texte."""

import pandas as pd
import pytest

from src.config import LABEL_COLUMN, LABELS_3, LANGUAGE_COLUMN, TEXT_COLUMN
from src.prepare_data import stratified_train_val_test


def _frame(per_stratum: int) -> pd.DataFrame:
    rows = []
    for language in ("de", "en"):
        for label in LABELS_3:
            for index in range(per_stratum):
                rows.append(
                    {
                        TEXT_COLUMN: f"{language}-{label}-{index}",
                        LABEL_COLUMN: label,
                        LANGUAGE_COLUMN: language,
                        "source": "synthetic",
                        "dataset": "synthetic",
                    }
                )
    return pd.DataFrame(rows)


def test_three_way_split_is_stratified_and_reproducible():
    frame = _frame(20)
    train, val, test = stratified_train_val_test(frame, seed=42)
    train_again, val_again, test_again = stratified_train_val_test(frame, seed=42)

    assert len(train) + len(val) + len(test) == len(frame)
    assert set(train[TEXT_COLUMN]).isdisjoint(val[TEXT_COLUMN])
    assert set(train[TEXT_COLUMN]).isdisjoint(test[TEXT_COLUMN])
    assert set(val[TEXT_COLUMN]).isdisjoint(test[TEXT_COLUMN])
    assert set(train[TEXT_COLUMN]) | set(val[TEXT_COLUMN]) | set(test[TEXT_COLUMN]) == set(frame[TEXT_COLUMN])
    assert "_stratum" not in train.columns

    def strata(part: pd.DataFrame) -> set[str]:
        return set(part[LABEL_COLUMN] + "|" + part[LANGUAGE_COLUMN])

    assert strata(train) == strata(val) == strata(test)
    assert len(strata(train)) == 6
    assert abs(len(test) / len(frame) - 0.15) < 0.05
    assert abs(len(val) / len(frame) - 0.15) < 0.05
    assert train[TEXT_COLUMN].tolist() == train_again[TEXT_COLUMN].tolist()
    assert val[TEXT_COLUMN].tolist() == val_again[TEXT_COLUMN].tolist()
    assert test[TEXT_COLUMN].tolist() == test_again[TEXT_COLUMN].tolist()


def test_split_rejects_a_stratum_with_one_row():
    frame = _frame(1)
    with pytest.raises(ValueError, match="stratifizierte Drei-Teilung"):
        stratified_train_val_test(frame)
