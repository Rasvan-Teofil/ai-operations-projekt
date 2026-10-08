"""Vergleicht alle Modelle auf demselben Testsplit.

Aufruf vom Projektroot:

    python -m src.compare
    python -m src.train --all

Ohne Übergabe lädt das die beiden TF-IDF-Modelle, die beiden
Zero-Shot-Transformer und – falls vorhanden – das eigene Fine-Tuning.
Das braucht ``requirements-transformer.txt`` und lädt Modellgewichte.
Tests übergeben fertige Modelle und treffen das Netz nicht.
"""

from pathlib import Path

import pandas as pd

from src.config import LABEL_COLUMN, REPORTS_DIR, TEXT_COLUMN
from src.data import dataset_log_params, load_work_splits
from src.interface import SentimentModel
from src.tracking import attach_artifact_paths, evaluate_model, log_evaluation, setup_mlflow, shared_dataset_params

SELECTION_NOTE = """\
Auswahl für Meilenstein 2: Primär gewinnt das höchste macro-F1 auf diesem
gemeinsamen Testsplit (jede Klasse zählt gleich). Liegen zwei Modelle nah
beieinander, entscheiden Inferenzzeit (ms pro Text) und Modellgröße.
Das ist ein fachlicher Trade-off, kein automatisches Deployment: die API
lädt das Modell aus der Variable SENTIMENT_MODEL.
Die gemessene Empfehlung steht in reports/model_comparison.md.
"""


def build_default_models(train_texts: list[str], train_labels: list[str]):
    """Trainiert die sklearn-Modelle und lädt die Transformer. Kann Gewichte herunterladen."""
    from src.hf_models import iter_transformer_models
    from src.sklearn_models import train_sklearn_models

    sklearn_models = train_sklearn_models(train_texts, train_labels)
    transformers, skipped = iter_transformer_models()
    return [*sklearn_models, *transformers], skipped


def run_comparison(
    models: list[SentimentModel] | None = None,
    output_dir: Path | None = None,
    log_mlflow: bool = True,
) -> list[dict]:
    """Bewertet jedes Modell auf dem einen Testsplit und schreibt die Tabelle."""
    train_frame, val_frame, test_frame = load_work_splits()
    test_texts = test_frame[TEXT_COLUMN].astype(str).tolist()
    test_labels = test_frame[LABEL_COLUMN].astype(str).tolist()
    skipped: list[str] = []
    if models is None:
        rows, skipped = _evaluate_default(train_frame, test_texts, test_labels)
    else:
        rows = []
        for model in models:
            row = evaluate_model(model, test_texts, test_labels)
            attach_artifact_paths(row, model)
            rows.append(row)

    destination = output_dir or REPORTS_DIR
    csv_path, markdown_path = write_comparison(
        rows,
        skipped,
        destination,
        len(train_frame),
        len(test_frame),
        n_val=len(val_frame),
    )
    _print_table(csv_path, skipped)

    if log_mlflow:
        setup_mlflow()
        shared = shared_dataset_params(len(train_frame), len(test_frame), n_val=len(val_frame))
        for row in rows:
            log_evaluation(row, shared)
        _log_comparison_artifact(csv_path, markdown_path, shared)
    return rows


def _evaluate_default(train_frame: pd.DataFrame, test_texts: list[str], test_labels: list[str]):
    """Ein Modell nach dem anderen, damit nicht alle Transformer gleichzeitig im Speicher liegen."""
    import gc

    from src.hf_models import finetuned_is_ready, load_finetuned, load_pretrained
    from src.hf_models import PRETRAINED_SPECS
    from src.sklearn_models import train_sklearn_models

    rows = []
    skipped: list[str] = []
    sklearn_models = train_sklearn_models(
        train_frame[TEXT_COLUMN].astype(str).tolist(),
        train_frame[LABEL_COLUMN].astype(str).tolist(),
    )
    for model in sklearn_models:
        print(f"Bewerte {model.name} ...", flush=True)
        rows.append(attach_artifact_paths(evaluate_model(model, test_texts, test_labels), model))
    for name in PRETRAINED_SPECS:
        print(f"Bewerte {name} ...", flush=True)
        model = load_pretrained(name)
        rows.append(attach_artifact_paths(evaluate_model(model, test_texts, test_labels), model))
        del model
        gc.collect()
    if finetuned_is_ready():
        print("Bewerte finetuned ...", flush=True)
        model = load_finetuned()
        rows.append(attach_artifact_paths(evaluate_model(model, test_texts, test_labels), model))
        del model
        gc.collect()
    else:
        skipped.append("finetuned")
    return rows, skipped


def write_comparison(
    rows: list[dict],
    skipped: list[str],
    directory: Path,
    n_train: int,
    n_test: int,
    n_val: int | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    table = _table(rows)
    csv_path = directory / "comparison.csv"
    markdown_path = directory / "comparison.md"
    table.to_csv(csv_path, index=False)
    markdown_path.write_text(
        _markdown(table, rows, skipped, n_train, n_test, n_val),
        encoding="utf-8",
    )
    return csv_path, markdown_path


def _table(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                "model": row["model"],
                "model_version": row["model_version"],
                "role": row["role"],
                "accuracy": row["metrics"]["accuracy"],
                "f1_macro": row["metrics"]["f1_macro"],
                "f1_negative": row["metrics"]["f1_negative"],
                "f1_neutral": row["metrics"]["f1_neutral"],
                "f1_positive": row["metrics"]["f1_positive"],
                "inference_ms_per_text": row["inference_ms_per_text"],
                "size_mb": row.get("size_mb"),
            }
        )
    frame = pd.DataFrame.from_records(records)
    return frame.sort_values(
        ["f1_macro", "inference_ms_per_text"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _markdown(
    table: pd.DataFrame,
    rows: list[dict],
    skipped: list[str],
    n_train: int,
    n_test: int,
    n_val: int | None = None,
) -> str:
    info = dataset_log_params()
    val_text = "—" if n_val is None else str(n_val)
    lines = [
        "# Modellvergleich",
        "",
        f"Datensatz: {info['dataset']} ({info['dataset_kind']}).",
        f"Split: {info['split_rule']}, Seed {info['random_seed']}, "
        f"n_train={n_train}, n_val={val_text}, n_test={n_test}.",
        "Labelraum: negative, neutral, positive.",
        "",
        _markdown_table(table),
        "",
        SELECTION_NOTE.strip(),
        "",
    ]
    if skipped:
        lines.append("Nicht bewertet, weil das Artefakt fehlt: " + ", ".join(skipped) + ".")
        lines.append("Fine-Tuning: `python -m src.finetune` (siehe README, GPU bzw. Colab).")
        lines.append("")
    lines.append("## Kurznoten")
    lines.append("")
    notes = {row["model"]: row.get("notes", "") for row in rows}
    for _, record in table.iterrows():
        note = notes.get(record["model"], "")
        suffix = f": {note}" if note else ""
        lines.append(f"- `{record['model']}` ({record['role']}){suffix}")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for record in table.itertuples(index=False):
        cells = []
        for value in record:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def _print_table(csv_path: Path, skipped: list[str]) -> None:
    print(csv_path.read_text(encoding="utf-8"))
    if skipped:
        print("Übersprungen:", ", ".join(skipped))
    print(f"Tabelle: {csv_path}")


def _log_comparison_artifact(csv_path: Path, markdown_path: Path, shared: dict) -> None:
    import mlflow

    with mlflow.start_run(run_name="comparison"):
        mlflow.log_params(shared)
        mlflow.set_tag("role", "comparison")
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(markdown_path))


def main() -> None:
    print("Vollständiger Vergleich. Transformer-Gewichte werden bei Bedarf geladen.")
    run_comparison()


if __name__ == "__main__":
    main()
