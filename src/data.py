"""Laden und Aufteilen der Texte.

Die Split-Logik lebt nur hier. Baseline, klassisches Alternativmodell,
Zero-Shot-Modelle und das Fine-Tuning nutzen denselben Split.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ALLOWED_LANGUAGES,
    LABEL_COLUMN,
    LABELS_3,
    LANGUAGE_COLUMN,
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
