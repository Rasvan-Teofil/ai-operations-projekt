"""Trains both TF-IDF models and logs them to MLflow.

From the project root:

    python -m src.train
    python -m src.train --all

Without ``--all``, Hugging Face and Torch are not used. ``--all`` is
the full comparison and delegates to ``src.compare``.
"""

import argparse

from src.config import DEFAULT_MODEL_NAME, LABEL_COLUMN, TEXT_COLUMN
from src.data import data_mode, load_work_splits
from src.sklearn_models import train_sklearn_models
from src.tracking import attach_artifact_paths, evaluate_model, log_evaluation, setup_mlflow, shared_dataset_params


def train() -> list[dict]:
    """Baseline and LinearSVC, same split, one MLflow experiment."""
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
    kind = "dummy sample" if data_mode() == "dummy" else "prepared test split"
    print(f"TF-IDF models on the test set ({kind}):")
    for row in rows:
        metrics = row["metrics"]
        print(
            f"- {row['model']}: accuracy={metrics['accuracy']:.3f}, "
            f"f1_macro={metrics['f1_macro']:.3f}, "
            f"inference_ms_per_text={row['inference_ms_per_text']:.2f}"
        )
    print(f"API default: {DEFAULT_MODEL_NAME} (switch with SENTIMENT_MODEL).")
    print("Full comparison including transformers: python -m src.train --all")
    print("MLflow: MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline or compare all models.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all models on the same test split (loads transformers).",
    )
    args = parser.parse_args(argv)
    if args.all:
        from src.compare import main as compare_main

        compare_main()
        return
    train()


if __name__ == "__main__":
    main()
