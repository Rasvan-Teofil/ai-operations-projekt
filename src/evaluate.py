"""Metriken für den Vergleich von Baseline und einfachem Modell."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Berechnet die Kennzahlen eines Testdurchlaufs.

    macro-F1 behandelt jede Klasse gleich und ist das Auswahlkriterium.
    Accuracy bleibt als leicht lesbare Zusatzkennzahl erhalten. Bei
    ungleichen Klassenanteilen reicht sie allein nicht für die Auswahl.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
