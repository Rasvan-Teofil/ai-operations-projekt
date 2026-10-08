"""Baseline trainieren, in MLflow protokollieren, Artefakt speichern.

Vom Projektroot:

    python -m src.train

Es entstehen zwei Läufe: ein Dummy-Modell als Bezugspunkt und eine
logistische Regression als einfaches Modell. Gewonnen hat das Modell
mit dem höchsten macro-F1 auf dem Testdatensatz. Bei Gleichstand
entscheidet die Accuracy, danach die geringere Komplexität.
Das Artefakt unter ``models/`` ist die einzige Schnittstelle zur API.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    FEATURE_COLUMNS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_DIR,
    MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COLUMN,
    TARGET_NAMES,
    TEST_SIZE,
)
from src.data import PLACEHOLDER_DATASET, load_dataset, split_dataset
from src.evaluate import classification_metrics

# Kleinere Zahl = geringere Komplexität. Zählt nur bei metrischem Gleichstand.
COMPLEXITY_RANK = {
    "dummy_most_frequent": 0,
    "logistic_regression": 1,
}


@dataclass
class CandidateResult:
    name: str
    estimator: object
    params: dict[str, str | int | float]
    metrics: dict[str, float]


def build_artifact(estimator, model_name: str) -> dict:
    """Schnürt das, was die API später laden darf – ohne Trainingscode."""
    class_labels = {
        int(class_index): TARGET_NAMES[int(class_index)]
        for class_index in estimator.classes_
    }
    return {
        "model_name": model_name,
        "estimator": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,
        "placeholder_dataset": PLACEHOLDER_DATASET,
    }


def _candidates() -> list[tuple[str, object, dict]]:
    return [
        (
            "dummy_most_frequent",
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            {"model_type": "DummyClassifier", "strategy": "most_frequent"},
        ),
        (
            "logistic_regression",
            LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
            {"model_type": "LogisticRegression", "max_iter": 500},
        ),
    ]


def _selection_key(result: CandidateResult) -> tuple[float, float, int]:
    """Höher ist besser. Komplexität wird nur als letzter Tie-Break abgezogen."""
    return (
        result.metrics["f1_macro"],
        result.metrics["accuracy"],
        -COMPLEXITY_RANK[result.name],
    )


def train() -> CandidateResult:
    """Trainiert die Kandidaten auf dem festen Split und speichert den Sieger."""
    frame = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(frame)

    fitted: list[CandidateResult] = []
    for name, estimator, params in _candidates():
        estimator.fit(x_train, y_train)
        metrics = classification_metrics(y_test, estimator.predict(x_test))
        fitted.append(CandidateResult(name, estimator, params, metrics))

    best = max(fitted, key=_selection_key)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(build_artifact(best.estimator, best.name), MODEL_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    shared_params = {
        "dataset": PLACEHOLDER_DATASET,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "selection_metric": "f1_macro",
    }

    for result in fitted:
        selected = result.name == best.name
        with mlflow.start_run(run_name=result.name):
            mlflow.log_params({**shared_params, **result.params})
            mlflow.log_metrics(result.metrics)
            mlflow.set_tag("selected", "true" if selected else "false")
            role = "baseline" if result.name.startswith("dummy") else "simple_model"
            mlflow.set_tag("role", role)
            _log_model_artifact(build_artifact(result.estimator, result.name))

    _print_summary(fitted, best)
    return best


def _log_model_artifact(artifact: dict) -> None:
    """Legt das joblib-Artefakt im aktiven MLflow-Lauf ab."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / MODEL_PATH.name
        joblib.dump(artifact, path)
        mlflow.log_artifact(str(path), artifact_path="model")


def _print_summary(fitted: list[CandidateResult], best: CandidateResult) -> None:
    print("Vergleich auf dem Testdatensatz (PLACEHOLDER Iris):")
    for result in fitted:
        print(
            f"- {result.name}: "
            f"accuracy={result.metrics['accuracy']:.3f}, "
            f"f1_macro={result.metrics['f1_macro']:.3f}"
        )
    print(
        f"Ausgewählt: {best.name} "
        "(Kriterium: höchstes macro-F1, dann Accuracy, dann geringere Komplexität)"
    )
    print(f"Artefakt: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("MLflow: mlflow ui --backend-store-uri ./mlruns")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
