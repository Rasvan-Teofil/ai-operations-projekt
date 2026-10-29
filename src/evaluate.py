"""Shared metrics for the model comparison.

All models are evaluated with these on the same test split and in the
same three-class space.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.config import LABELS_3


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy, macro-F1 and per-class F1.

    Classes missing from the split count as 0, so the keys are the same
    in every run.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(LABELS_3)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    for label, value in zip(labels, per_class, strict=True):
        metrics[f"f1_{label}"] = float(value)
    return metrics


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Konfusionsmatrix, Zeile = wahr, Spalte = vorhergesagt."""
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=list(LABELS_3))
    frame = pd.DataFrame(matrix, index=list(LABELS_3), columns=list(LABELS_3))
    frame.index.name = "true"
    frame.columns.name = "predicted"
    return frame
