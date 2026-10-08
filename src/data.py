"""Laden und Aufteilen der Daten.

Die Split-Logik lebt nur hier. Notebooks, Training und Tests rufen
dieselben Funktionen auf, damit später niemand einen zweiten Split erfindet.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, RANDOM_SEED, TARGET_COLUMN, TARGET_NAMES, TEST_SIZE

# Klar markierter Platzhalter, bis Geschäftsproblem und Datensatz feststehen.
PLACEHOLDER_DATASET = "sklearn.datasets.load_iris"


def load_dataset() -> pd.DataFrame:
    """Lädt den Platzhalter-Datensatz als Tabelle.

    PLACEHOLDER: Iris aus scikit-learn, nicht der echte Projektdatensatz.
    Tauschstelle für M1: diese Funktion auf eine Datei unter ``data/`` umstellen
    und ``FEATURE_COLUMNS`` / ``TARGET_COLUMN`` in ``src/config.py`` anpassen.
    Die Zielwerte sind Klassenindizes passend zu ``TARGET_NAMES``.
    """
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURE_COLUMNS)
    frame[TARGET_COLUMN] = iris.target.astype("int64")

    expected = list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Datensatz enthält nicht die erwarteten Spalten: {missing}")
    if len(frame) < 2:
        raise ValueError("Datensatz braucht mindestens zwei Zeilen für einen Split.")
    unknown = sorted(set(frame[TARGET_COLUMN].unique()) - set(range(len(TARGET_NAMES))))
    if unknown:
        raise ValueError(f"Unbekannte Zielklassen im Platzhalter: {unknown}")
    return frame


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Teilt in Train und Test.

    Platzhalter-Regel: stratifizierter Zufallssplit (80/20, fester Seed).
    Das ist nur gültig, solange die Zeilen unabhängig sind. Zeitreihen,
    gruppierte Kunden oder Leakage brauchen später eine andere Regel –
    dann wird sie hier ersetzt, nicht im Notebook und nicht im Training.
    """
    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame.loc[:, TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )
