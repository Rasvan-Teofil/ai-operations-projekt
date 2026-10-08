"""Laden und Aufteilen der Texte.

Zwei Wege, eine Regel je Weg:

- ``SENTIMENT_DATA=dummy`` (Tests, CI): die handgeschriebene Stichprobe,
  stratifiziert 80/20 nur nach Label, Seed 42. Kein Validierungssplit.
- Standard ``prepared``: die drei Dateien aus ``python -m src.prepare_data``.
  Die werden hier nicht noch einmal gemischt.

Baseline, LinearSVC, Zero-Shot und Fine-Tuning lesen ``load_work_splits``.
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

# Relativer Pfad für Logs. Die Datei ist Dummy-Text, kein Nachrichtendatensatz.
PLACEHOLDER_DATASET = "data/sample/sentiment_sample.csv"


def load_dataset(path: Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Lädt die gelabelten Texte.

    PLACEHOLDER: ``data/sample/sentiment_sample.csv`` enthält wenige
    handgeschriebene DE/EN-Sätze mit ``origin=dummy``. Für Meilenstein 1
    muss ein echter gelabelter Nachrichten-Datensatz her und diese
    Funktion darauf zeigen. Spaltenvertrag: text, label, language.
    """
    if not path.exists():
        raise FileNotFoundError(f"Datensatz nicht gefunden: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    required = [TEXT_COLUMN, LABEL_COLUMN, LANGUAGE_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Datensatz enthält nicht die erwarteten Spalten: {missing}")

    frame = frame.copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[LANGUAGE_COLUMN] = frame[LANGUAGE_COLUMN].astype(str).str.strip().str.lower()

    if frame[TEXT_COLUMN].eq("").any():
        raise ValueError("Datensatz enthält leere Texte.")
    if len(frame) < 2:
        raise ValueError("Datensatz braucht mindestens zwei Zeilen für einen Split.")

    unknown_labels = sorted(set(frame[LABEL_COLUMN]) - set(LABELS_3))
    if unknown_labels:
        raise ValueError(f"Unbekannte Labels: {unknown_labels}. Erlaubt: {list(LABELS_3)}")
    unknown_languages = sorted(set(frame[LANGUAGE_COLUMN]) - set(ALLOWED_LANGUAGES))
    if unknown_languages:
        raise ValueError(
            f"Unbekannte Sprachen: {unknown_languages}. Erlaubt: {sorted(ALLOWED_LANGUAGES)}"
        )

    counts = frame[LABEL_COLUMN].value_counts()
    too_small = [label for label in LABELS_3 if int(counts.get(label, 0)) < 2]
    if too_small:
        raise ValueError(f"Zu wenige Beispiele für einen stratifizierten Split: {too_small}")
    return frame.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratifizierter 80/20-Split auf dem Stimmungslabel, fester Seed.

    Das passt zu unabhängigen Beispielsätzen. Echte Artikel brauchen
    vermutlich eine andere Regel: gleicher Vorgang, dasselbe Medium oder
    dasselbe Datum darf nicht in Train und Test zugleich liegen.
    Die Regel wird hier ersetzt, nicht im Notebook und nicht je Modell.
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
    """``dummy`` für Tests und CI, sonst die aufbereiteten Splits."""
    raw = os.environ.get("SENTIMENT_DATA", "prepared").strip().lower()
    if raw in {"dummy", "sample"}:
        return "dummy"
    if raw in {"", "prepared", "real"}:
        return "prepared"
    raise ValueError(f"SENTIMENT_DATA unbekannt: {raw!r}. Erlaubt: prepared, dummy.")


def dataset_log_params() -> dict:
    """Herkunft für MLflow und die Vergleichstabelle. Keine Textzeilen."""
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
    """Train, Val, Test. Bei der Dummy-Stichprobe ist Val leer.

    Die aufbereiteten Dateien sind schon gesplittet. Hier kein zweiter Zufallssplit,
    sonst wären Vergleich und Fine-Tuning nicht mehr dieselben Zeilen.
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
            "Aufbereitete Splits fehlen "
            f"({', '.join(missing)}). Bitte im Projektroot `python -m src.prepare_data` ausführen. "
            "Tests setzen SENTIMENT_DATA=dummy und brauchen diesen Schritt nicht."
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
        raise ValueError(f"{path.name} enthält nicht die erwarteten Spalten: {missing}")
    frame = frame.copy()
    frame[TEXT_COLUMN] = frame[TEXT_COLUMN].astype(str).str.strip()
    frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip().str.lower()
    frame[LANGUAGE_COLUMN] = frame[LANGUAGE_COLUMN].astype(str).str.strip().str.lower()
    if frame[TEXT_COLUMN].eq("").any():
        raise ValueError(f"{path.name} enthält leere Texte.")
    unknown_labels = sorted(set(frame[LABEL_COLUMN]) - set(LABELS_3))
    if unknown_labels:
        raise ValueError(f"{path.name}: unbekannte Labels {unknown_labels}.")
    unknown_languages = sorted(set(frame[LANGUAGE_COLUMN]) - set(ALLOWED_LANGUAGES))
    if unknown_languages:
        raise ValueError(f"{path.name}: unbekannte Sprachen {unknown_languages}.")
    return frame.reset_index(drop=True)
