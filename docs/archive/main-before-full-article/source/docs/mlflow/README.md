# MLflow evidence

Screenshots of the local MLflow UI (MLflow 3.17.0, store `./mlruns`), taken on 2026-10-30.

The runs were created with `python -m src.import_recorded`. This imports the metrics that are already committed under `reports/` (no retraining). Live iteration-1 runs come from `python -m src.train`.

| File | What it shows |
|---|---|
| `01_comparison_runs.png` | Experiment `article-sentiment-comparison`: all 18 runs of both iterations, with `iteration` and `comparable_group` tags |
| `02_sentfin_macro_f1_chart.png` | Same experiment, filter `tags.comparable_group = 'sentfin-external'`: macro-F1 EN/DE on the shared SEntFiN audit. Iteration 1 (yellow, `recorded-colleague_main_default`): 0.45 / 0.43. Iteration 2 v2 (red): 0.73 / 0.66. v1 (black): 0.74 / 0.65 |
| `03_sentfin_compare_runs.png` | Compare view of the three SEntFiN runs: params, metrics, tags `iteration=1-first-path` vs. `iteration=2-second-path` |

Notes:

- `colleague_main_default` is the reconstructed first-iteration LinearSVC (pinned commit `a840bb7`), see `reports/development/branch_comparison.md`.
- Only runs with the same `comparable_group` are directly comparable. Own test sets of the two iterations are different data.

Reproduce (Git Bash, project root):

```bash
source .venv/Scripts/activate
python -m src.import_recorded
export MLFLOW_ALLOW_FILE_STORE=true
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open experiment `article-sentiment-comparison` and filter by `tags.comparable_group`.
