"""The default run sticks to the TF-IDF models."""

import sys

import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_DIR
from src.train import train


def test_default_train_does_not_import_transformers():
    before = set(sys.modules)
    rows = train()
    imported = set(sys.modules) - before
    assert {row["model"] for row in rows} == {"tfidf_logreg", "tfidf_linearsvc"}
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.resolve().as_uri())
    logged = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], max_results=20)
    tagged = logged[logged["tags.iteration"] == "1-first-path"]
    assert set(tagged["tags.model_name"]) >= {"tfidf_logreg", "tfidf_linearsvc"}
    assert set(tagged["tags.source"]) >= {"live-training"}
    assert all("f1_macro" in row["metrics"] for row in rows)
    assert all(row["inference_ms_per_text"] >= 0 for row in rows)
    assert not any(name == "transformers" or name.startswith("transformers.") for name in imported)
    assert not any(name == "torch" or name.startswith("torch.") for name in imported)
