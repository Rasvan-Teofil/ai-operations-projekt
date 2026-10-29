"""Writes the already measured metrics of both iterations to ``./mlruns``.

CPU only, no training and no GPU. The values are read from the committed
reports, not recomputed. A second call does not create duplicate runs.

    python -m src.import_recorded

In the UI the runs are in the store ``./mlruns``:

- Experiment ``article-sentiment``, Tag ``iteration=1-first-path``
- Experiment ``article-sentiment-second-path``, Tag ``iteration=2-second-path``
  (the reconstructed first iteration in the external audit carries
  ``iteration=1-first-path``)
- Experiment ``article-sentiment-comparison`` with both tags, for a
  direct side-by-side view. ``comparable_group`` separates tests that must
  not be ranked against each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow

from src.config import (
    MLFLOW_COMPARISON_EXPERIMENT,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ITERATION,
    MLFLOW_TRACKING_DIR,
    PROJECT_ROOT,
)

SECOND_EXPERIMENT = "article-sentiment-second-path"
SECOND_ITERATION = "2-second-path"
SOURCE = "recorded-report"


def _require(text: str, needle: str, source: str) -> None:
    if needle not in text:
        raise RuntimeError(f"{needle!r} missing in {source}; aborting the import instead of inventing numbers.")


def _parse_markdown_table(text: str, header_prefix: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith(header_prefix)), None)
    if header_idx is None:
        raise RuntimeError(f"Table {header_prefix!r} not found.")
    headers = [cell.strip() for cell in lines[header_idx].strip().strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[header_idx + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(dict(zip(headers, cells, strict=True)))
    if not rows:
        raise RuntimeError(f"Table {header_prefix!r} is empty.")
    return rows


def _f(value: str | float | int) -> float:
    return float(value)


def _metrics(pairs: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in pairs.items()}


def _log_once(
    experiment: str,
    recorded_key: str,
    run_name: str,
    iteration: str,
    params: dict[str, str],
    metrics: dict[str, float],
    extra_tags: dict[str, str],
) -> str:
    mlflow.set_experiment(experiment)
    found = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.recorded_key = '{recorded_key}'",
        max_results=1,
    )
    if found is not None and not found.empty:
        return "skipped"
    tags = {
        "iteration": iteration,
        "source": SOURCE,
        "recorded_key": recorded_key,
        **extra_tags,
    }
    with mlflow.start_run(run_name=run_name, tags=tags):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    return "logged"


def _log_pair(
    native_experiment: str,
    recorded_key: str,
    run_name: str,
    iteration: str,
    params: dict[str, str],
    metrics: dict[str, float],
    extra_tags: dict[str, str],
) -> dict[str, str]:
    """Native experiment plus the shared comparison experiment."""
    native = _log_once(
        native_experiment, recorded_key, run_name, iteration, params, metrics, extra_tags
    )
    shared = _log_once(
        MLFLOW_COMPARISON_EXPERIMENT,
        f"comparison/{recorded_key}",
        run_name,
        iteration,
        params,
        metrics,
        extra_tags,
    )
    return {"native": native, "comparison": shared}


def import_recorded(store: Path | None = None) -> dict[str, int]:
    """Reads the reports and creates the runs in the file store."""
    root = PROJECT_ROOT
    tracking = (store or MLFLOW_TRACKING_DIR).resolve()
    tracking.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking.as_uri())

    counts = {"logged": 0, "skipped": 0}
    first_md = (root / "reports" / "model_comparison.md").read_text(encoding="utf-8")
    _require(first_md, "n_test = 1270", "reports/model_comparison.md")
    _require(first_md, "0.7202", "reports/model_comparison.md")
    for row in _parse_markdown_table(first_md, "| model |"):
        metrics = _metrics(
            {
                "accuracy": _f(row["accuracy"]),
                "f1_macro": _f(row["f1_macro"]),
                "macro_f1": _f(row["f1_macro"]),
                "f1_negative": _f(row["f1_negative"]),
                "f1_neutral": _f(row["f1_neutral"]),
                "f1_positive": _f(row["f1_positive"]),
                "inference_ms_per_text": _f(row["inference_ms_per_text"]),
                "size_mb": _f(row["size_mb"]),
            }
        )
        outcome = _log_pair(
            MLFLOW_EXPERIMENT_NAME,
            f"first-own-test/{row['model']}",
            f"recorded-{row['model']}",
            MLFLOW_ITERATION,
            {
                "model": row["model"],
                "model_version": row["model_version"],
                "role": row["role"],
                "dataset": "financial-sentences-test",
                "n_test": "1270",
                "comparable_group": "first-path-own-test",
                "source_report": "reports/model_comparison.md",
                "rank_against": "only other runs with comparable_group=first-path-own-test",
            },
            metrics,
            {"comparable_group": "first-path-own-test", "phase": "first-path-own-test"},
        )
        for status in outcome.values():
            counts[status] += 1

    second_note = (root / "reports" / "second_path_model_comparison.md").read_text(encoding="utf-8")
    _require(second_note, "1,787", "reports/second_path_model_comparison.md")
    _require(second_note, "0.7187", "reports/second_path_model_comparison.md")
    own = json.loads((root / "reports" / "comparison.json").read_text(encoding="utf-8"))
    for row in own["results"]:
        metrics = _metrics(
            {
                "accuracy": row["accuracy"],
                "f1_macro": row["macro_f1"],
                "macro_f1": row["macro_f1"],
                "f1_negative": row["f1_negative"],
                "f1_neutral": row["f1_neutral"],
                "f1_positive": row["f1_positive"],
                "macro_f1_de": row["macro_f1_de"],
                "macro_f1_en": row["macro_f1_en"],
                "inference_ms_per_text": row["inference_ms_per_text"],
                "model_size_bytes": row["model_size_bytes"],
                "size_mb": row["model_size_bytes"] / (1024 * 1024),
            }
        )
        outcome = _log_pair(
            SECOND_EXPERIMENT,
            f"second-own-test/{row['model']}",
            f"recorded-{row['model']}",
            SECOND_ITERATION,
            {
                "model": row["model"],
                "original_run_id": row["run_id"],
                "dataset": "second-path-own-test",
                "n_test": "1787",
                "selected": own["selected"],
                "comparable_group": "second-path-own-test",
                "source_report": "reports/comparison.json",
                "rank_against": "only other runs with comparable_group=second-path-own-test",
            },
            metrics,
            {"comparable_group": "second-path-own-test", "phase": "v1-own-test"},
        )
        for status in outcome.values():
            counts[status] += 1

    curve_path = root / "reports" / "development" / "curve.csv"
    curve_text = curve_path.read_text(encoding="utf-8")
    _require(curve_text, "R3_deploy", "reports/development/curve.csv")
    _require(curve_text, "0.6728847544039361", "reports/development/curve.csv")
    with curve_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            size_bytes = _f(row["model_size_bytes"])
            macro = _f(row["mean_macro_f1"])
            latency = _f(row["median_latency_ms"])
            metrics = _metrics(
                {
                    "f1_macro": macro,
                    "macro_f1": macro,
                    "sd_macro_f1": _f(row["sd_macro_f1"]),
                    "macro_f1_de": _f(row["mean_macro_f1_de"]),
                    "macro_f1_en": _f(row["mean_macro_f1_en"]),
                    "negative_recall": _f(row["mean_negative_recall"]),
                    "negative_precision": _f(row["mean_negative_precision"]),
                    "inference_ms_per_text": latency,
                    "median_latency_ms": latency,
                    "model_size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                    "html_fixture_exact_match": _f(row["html_fixture_exact_match"]),
                }
            )
            outcome = _log_pair(
                SECOND_EXPERIMENT,
                f"second-source-cv/{row['stage']}",
                f"recorded-{row['stage']}",
                SECOND_ITERATION,
                {
                    "model": row["stage"],
                    "stage": row["stage"],
                    "folds": row["folds"],
                    "dataset": "source-group-cv-inside-train",
                    "comparable_group": "second-path-source-group-cv",
                    "source_report": "reports/development/curve.csv",
                    "original_summary_run_id": row["summary_run_id"],
                    "rank_against": "only other runs with comparable_group=second-path-source-group-cv",
                },
                metrics,
                {"comparable_group": "second-path-source-group-cv", "phase": "development-v2-summary", "stage": row["stage"]},
            )
            for status in outcome.values():
                counts[status] += 1

    branch_md = (root / "reports" / "development" / "branch_comparison.md").read_text(encoding="utf-8")
    _require(branch_md, "0.4524", "reports/development/branch_comparison.md")
    _require(branch_md, "0.7293", "reports/development/branch_comparison.md")
    _require(branch_md, "0.2770", "reports/development/branch_comparison.md")
    branch = json.loads((root / "reports" / "development" / "branch_comparison.json").read_text(encoding="utf-8"))
    for row in branch["results"]:
        iteration = MLFLOW_ITERATION if row["model"] == "colleague_main_default" else SECOND_ITERATION
        size_bytes = _f(row["model_size_bytes"])
        metrics = _metrics(
            {
                "accuracy": row["accuracy"],
                "f1_macro": row["macro_f1"],
                "macro_f1": row["macro_f1"],
                "macro_f1_en": row["macro_f1_en"],
                "macro_f1_de": row["macro_f1_de"],
                "f1_negative": row["f1_negative"],
                "f1_neutral": row["f1_neutral"],
                "f1_positive": row["f1_positive"],
                "inference_ms_per_text": row["inference_ms_per_text"],
                "latency_p50_ms": row["latency_p50_ms"],
                "latency_p95_ms": row["latency_p95_ms"],
                "model_size_bytes": size_bytes,
                "size_mb": size_bytes / (1024 * 1024),
            }
        )
        outcome = _log_pair(
            SECOND_EXPERIMENT,
            f"sentfin-external/{row['model']}",
            f"recorded-{row['model']}",
            iteration,
            {
                "model": row["model"],
                "original_run_id": row["run_id"],
                "dataset": "sentfin-unanimous-headlines-plus-mt-de",
                "comparable_group": "sentfin-external",
                "source_report": "reports/development/branch_comparison.json",
                "benchmark_sha256": branch["benchmark_sha256"],
                "native_en_delta_v2_minus_peer": f"{branch['native_en_delta_v2_minus_peer']:.4f}",
                "rank_against": "only other runs with comparable_group=sentfin-external",
            },
            metrics,
            {"comparable_group": "sentfin-external", "phase": "final-branch-comparison"},
        )
        for status in outcome.values():
            counts[status] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Import the measured metrics of both iterations into MLflow.")
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help="MLflow file store folder. Default: ./mlruns in the project root.",
    )
    args = parser.parse_args(argv)
    counts = import_recorded(args.store)
    print(
        f"MLflow-Import: {counts['logged']} neu, {counts['skipped']} schon vorhanden. "
        "UI: MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000"
    )
    print(f"Experimente: {MLFLOW_EXPERIMENT_NAME}, {SECOND_EXPERIMENT}, {MLFLOW_COMPARISON_EXPERIMENT}")


if __name__ == "__main__":
    main()
