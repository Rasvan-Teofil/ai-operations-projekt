"""Bewertet ein Modell und schreibt den Lauf nach MLflow.

Metriken sind für jeden Lauf dieselben: Accuracy, macro-F1, F1 je Klasse,
Inferenzzeit. Die Konfusionsmatrix liegt als CSV-Artefakt im Lauf.
"""

import os

# Vor dem Import: aktuelles MLflow blockiert den lokalen Ordner sonst,
# und der Hinweis auf interne Skills gehört nicht in die Trainingsausgabe.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_DIR
from src.data import dataset_log_params
from src.evaluate import classification_metrics, confusion_frame
from src.interface import SentimentModel


def setup_mlflow() -> None:
    """Lokaler Ordner ``mlruns/``. Aktuelles MLflow verlangt dafür eine Freigabe."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def shared_dataset_params(n_train: int, n_test: int, n_val: int | None = None) -> dict:
    """Herkunft kommt aus ``dataset_log_params``, damit Dummy und Echtlauf nicht vermischt werden."""
    params = dataset_log_params()
    params["n_train"] = int(n_train)
    params["n_test"] = int(n_test)
    if n_val is not None:
        params["n_val"] = int(n_val)
    params["label_space"] = "negative|neutral|positive"
    return params


def evaluate_model(model: SentimentModel, texts: Sequence[str], y_true: Sequence[str]) -> dict:
    """Misst Qualität und Inferenzzeit auf genau diesen Testtexten."""
    texts = [str(text) for text in texts]
    y_true = [str(label) for label in y_true]
    if len(texts) != len(y_true):
        raise ValueError("Texte und Labels sind unterschiedlich lang.")
    started = time.perf_counter()
    predicted = model.predict(texts)
    elapsed = time.perf_counter() - started
    size = model.size_mb()
    return {
        "model": model.name,
        "model_version": model.model_version,
        "role": model.role,
        "metrics": classification_metrics(y_true, predicted),
        "confusion": confusion_frame(y_true, predicted),
        "inference_seconds": float(elapsed),
        "inference_ms_per_text": float(elapsed / len(texts) * 1000.0) if texts else 0.0,
        "size_mb": None if size is None else float(size),
        "n_test": len(texts),
        "params": model.tracking_params(),
        "notes": getattr(model, "notes", ""),
    }


def log_evaluation(row: dict, shared_params: dict) -> None:
    """Ein MLflow-Lauf pro Modell, plus Konfusionsmatrix."""
    metrics = {
        **row["metrics"],
        "inference_seconds": row["inference_seconds"],
        "inference_ms_per_text": row["inference_ms_per_text"],
    }
    if row.get("size_mb") is not None:
        metrics["size_mb"] = row["size_mb"]
    with mlflow.start_run(run_name=row["model"]):
        mlflow.log_params({**shared_params, **row.get("params", {}), "model_version": row["model_version"]})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("role", row["role"])
        mlflow.set_tag("model_name", row["model"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "confusion_matrix.csv"
            row["confusion"].to_csv(path)
            mlflow.log_artifact(str(path))
            artifact = row.get("artifact_path")
            if artifact:
                mlflow.log_artifact(str(artifact), artifact_path="model")


def attach_artifact_paths(row: dict, model: SentimentModel) -> dict:
    path = getattr(model, "artifact_path", None)
    if path is not None and Path(path).exists() and Path(path).is_file():
        row["artifact_path"] = Path(path)
    return row
