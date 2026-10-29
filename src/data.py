"""Loading and splitting the texts.

Two paths, one rule per path:

- ``SENTIMENT_DATA=dummy`` (tests, CI): the hand-written sample,
  stratified 80/20 by label only, seed 42. No validation split.
- Default ``prepared``: the three files from ``python -m src.prepare_data``.
  They are not reshuffled here.

Baseline, LinearSVC, zero-shot and fine-tuning all read ``load_work_splits``.
"""

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALLOWED_LANGUAGES,
    DATASET_CONFIG_EN,
    DATASET_ID_DE,
    DATASET_ID_EN,
    LABEL_COLUMN,
    LABELS_3,
    LANGUAGE_COLUMN,
    PREPARE_TEST_SIZE,
    PREPARE_VAL_SIZE,
    PREPARED_TEST_PATH,
    PREPARED_TRAIN_PATH,
    PREPARED_VAL_PATH,
    RANDOM_SEED,
    SAMPLE_DATA_PATH,
    TEST_SIZE,
    TEXT_COLUMN,
)

# Relative path for logs. The file is dummy text, not a news dataset.
PLACEHOLDER_DATASET = "data/sample/sentiment_sample.csv"


def load_dataset(path: Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Loads the labelled texts.

    PLACEHOLDER: ``data/sample/sentiment_sample.csv`` contains a few
    hand-written DE/EN sentences with ``origin=dummy``. For milestone 1 a
    real labelled news dataset is needed and this function has to point
    to it. Column contract: text, label, language.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    required = [TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset does not contain the expected columns: {missing}")

    frame = frame.copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[LANGUAGE_COLUMN] = frame[LANGUAGE_COLUMN].astype(str).str.strip().str.lower()

    if frame[TEXT_COLUMN].eq("").any():
        raise ValueError("Dataset contains empty texts.")
    if len(frame) < 2:
        raise ValueError("Dataset needs at least two rows for a split.")

    unknown_labels = sorted(set(frame[LABEL_COLUMN]) - set(LABELS_3))
    if unknown_labels:
        raise ValueError(f"Unknown labels: {unknown_labels}. Allowed: {list(LABELS_3)}")
    unknown_languages = sorted(set(frame[LANGUAGE_COLUMN]) - set(ALLOWED_LANGUAGES))
    if unknown_languages:
        raise ValueError(
            f"Unknown languages: {unknown_languages}. Allowed: {sorted(ALLOWED_LANGUAGES)}"
        )

    counts = frame[LABEL_COLUMN].value_counts()
    too_small = [label for label in LABELS_3 if int(counts.get(label, 0)) < 2]
    if too_small:
        raise ValueError(f"Too few examples for a stratified split: {too_small}")
    return frame.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 split on the sentiment label, fixed seed.

    This suits independent example sentences. Real articles probably need
    a different rule: the same story, outlet or date must not appear in
    both train and test. The rule is replaced here, not in the notebook
    and not per model.
    """
    features = frame.loc[:, [TEXT_COLUMN, LANGUAGE_COLUMN]]
    target = frame.loc[:, LABEL_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )


def data_mode() -> str:
    """``dummy`` for tests and CI, otherwise the prepared splits."""
    raw = os.environ.get("SENTIMENT_DATA", "prepared").strip().lower()
    if raw in {"dummy", "sample"}:
        return "dummy"
    if raw in {"", "prepared", "real"}:
        return "prepared"
    raise ValueError(f"Unknown SENTIMENT_DATA: {raw!r}. Allowed: prepared, dummy.")


def dataset_log_params() -> dict:
    """Provenance for MLflow and the comparison table. No text rows."""
    if data_mode() == "dummy":
        return {
            "dataset": PLACEHOLDER_DATASET,
            "dataset_kind": "dummy_placeholder",
            "split_rule": "stratified_label_80_20",
            "random_seed": RANDOM_SEED,
            "test_size": TEST_SIZE,
        }
    return {
        "dataset": f"{DATASET_ID_EN}:{DATASET_CONFIG_EN}+{DATASET_ID_DE}:de",
        "dataset_kind": "financial_news_sentences",
        "split_rule": "stratified_label_language_70_15_15",
        "random_seed": RANDOM_SEED,
        "test_size": PREPARE_TEST_SIZE,
        "val_size": PREPARE_VAL_SIZE,
    }


def load_work_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, val, test. For the dummy sample, val is empty.

    The prepared files are already split. No second random split here,
    otherwise comparison and fine-tuning would no longer use the same rows.
    """
    if data_mode() == "dummy":
        return _dummy_work_splits()
    missing = [
        path.name
        for path in (PREPARED_TRAIN_PATH, PREPARED_VAL_PATH, PREPARED_TEST_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Prepared splits are missing "
            f"({', '.join(missing)}). Please run `python -m src.prepare_data` in the project root. "
            "Tests set SENTIMENT_DATA=dummy and do not need this step."
        )
    return (
        _read_prepared(PREPARED_TRAIN_PATH),
        _read_prepared(PREPARED_VAL_PATH),
        _read_prepared(PREPARED_TEST_PATH),
    )


def _dummy_work_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_dataset(SAMPLE_DATA_PATH)
    x_train, x_test, y_train, y_test = split_dataset(frame)
    train = x_train.copy()
    train[LABEL_COLUMN] = y_train
    test = x_test.copy()
    test[LABEL_COLUMN] = y_test
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    val = train.iloc[0:0].copy()
    return train, val, test


def _read_prepared(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    required = [TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} does not contain the expected columns: {missing}")
    frame = frame.copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[LANGUAGE_COLUMN] = frame[LANGUAGE_COLUMN].astype(str).str.strip().str.lower()
    if frame[TEXT_COLUMN].eq("").any():
        raise ValueError(f"{path.name} contains empty texts.")
    unknown_labels = sorted(set(frame[LABEL_COLUMN]) - set(LABELS_3))
    if unknown_labels:
        raise ValueError(f"{path.name}: unknown labels {unknown_labels}.")
    unknown_languages = sorted(set(frame[LANGUAGE_COLUMN]) - set(ALLOWED_LANGUAGES))
    if unknown_languages:
        raise ValueError(f"{path.name}: unbekannte Sprachen {unknown_languages}.")
    return frame.reset_index(drop=True)
