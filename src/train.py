"""Trainiert die beiden TF-IDF-Modelle und protokolliert sie in MLflow.

Vom Projektroot:

    python -m src.train
    python -m src.train --all

Ohne ``--all`` bleiben Hugging Face und Torch außen vor. ``--all`` ist
der vollständige Vergleich und delegiert an ``src.compare``.
"""

import argparse

from src.config import DEFAULT_MODEL_NAME, LABEL_COLUMN, TEXT_COLUMN
from src.data import data_mode, load_work_splits
from src.sklearn_models import train_sklearn_models
from src.tracking import attach_artifact_paths, evaluate_model, log_evaluation, setup_mlflow, shared_dataset_params


def train() -> list[dict]:
    """Baseline und LinearSVC, derselbe Split, ein MLflow-Experiment."""
    train_frame, val_frame, test_frame = load_work_splits()
    models = train_sklearn_models(
        train_frame[TEXT_COLUMN].astype(str).tolist(),
        train_frame[LABEL_COLUMN].astype(str).tolist(),
    )
    test_texts = test_frame[TEXT_COLUMN].astype(str).tolist()
    test_labels = test_frame[LABEL_COLUMN].astype(str).tolist()
    rows = [attach_artifact_paths(evaluate_model(model, test_texts, test_labels), model) for model in models]

    setup_mlflow()
    shared = shared_dataset_params(len(train_frame), len(test_frame), n_val=len(val_frame))
    for row in rows:
        log_evaluation(row, shared)
    _print_summary(rows)
    return rows


def _print_summary(rows: list[dict]) -> None:
    kind = "Dummy-Stichprobe" if data_mode() == "dummy" else "aufbereiteter Testsplit"
    print(f"TF-IDF-Modelle auf dem Testdatensatz ({kind}):")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"- {row['model']}: accuracy={metrics['accuracy']:.3f}, "
            f"f1_macro={metrics['f1_macro']:.3f}, "
            f"inference_ms_per_text={row['inference_ms_per_text']:.2f}"
        )
    print(f"API-Standard: {DEFAULT_MODEL_NAME} (SENTIMENT_MODEL zum Umschalten).")
    print("Vollständiger Vergleich inklusive Transformer: python -m src.train --all")
    print("MLflow: MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TF-IDF-Baseline trainieren oder alle Modelle vergleichen.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Alle Modelle auf demselben Testsplit vergleichen (lädt Transformer).",
    )
    args = parser.parse_args(argv)
    if args.all:
        from src.compare import main as compare_main

        compare_main()
        return
    train()


if __name__ == "__main__":
    main()
