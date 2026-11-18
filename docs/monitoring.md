# Monitoring (minimal prototype)

Status: prototype on `daily-refinement`, not on `main`. Local only, no external services.
Code: `app/monitoring.py`, tests: `tests/test_monitoring.py`.

## What is logged

Logging is off by default. Set `PREDICTION_LOG_PATH` (e.g. `logs/predictions.jsonl`) to switch it on.
Each successful `/predict` call appends one JSON line:

| Field | Meaning |
|---|---|
| `ts` | UTC timestamp (seconds) |
| `model_name`, `model_version` | which model answered |
| `source` | `text` or `url` |
| `text_chars` | length of the classified text |
| `latency_ms` | model time only (no scraping) |
| `label`, `scores` | predicted label and class scores |

Failed predictions are not logged. A failed write is logged as an error; the API still answers.

## Privacy

- The raw text and the URL are never written.
- The log holds metadata and model output only, so it can be shared with the team without exposing article content or user input.
- The file stays on the machine that runs the API. Delete it to reset.

## Reading the log

```bash
python -m app.monitoring logs/predictions.jsonl           # label share per model version
python -m app.monitoring logs/predictions.jsonl --daily   # per UTC day and version: n, label share, median latency
python -m app.monitoring logs/predictions.jsonl --drift   # label share vs. test split, drift flag
python -m app.monitoring logs/predictions.jsonl --drift --fail-on-drift   # same, exit code 1 if any version is flagged
```

How to read `--daily`:

- `n` per day shows usage. A sudden drop to 0 for a version usually means a deployment or loading problem, not drift.
- `label_share` per day is the main signal. Stable shares are expected for a stable news mix.
- `median_latency_ms` is `null` when no latency was logged. A jump after a model change points to the heavier model (e.g. transformer vs. TF-IDF).

## Drift rule

- Reference: label share of the test split the models were evaluated on (`reference_label_share()`).
- Distance: total variation distance (half the summed absolute share difference, 0 = same, 1 = disjoint).
- Flag: `drift = true` when the distance is above 0.15.
- Versions with fewer than 30 predictions get `drift = null`, because the share is too noisy to judge.
- This only detects a shift in predicted labels (prediction drift). It does not measure accuracy, since live requests have no gold labels.

## Retraining trigger (first step built)

`--drift --fail-on-drift` exits with code 1 when a version is flagged, so a daily cron job or CI step can stop and alert. Steps 1 to 4 below are still manual.


1. Drift flag on for one version on 3 consecutive days with at least 30 predictions per day.
2. Then: label a small sample of recent inputs by hand (texts must be collected separately with consent, since the log keeps none).
3. If macro-F1 on that sample drops clearly below the test result, retrain with the new sample added and compare in MLflow against the current version, the same way run 1 and run 2 were compared.
4. Promote the new version only if it wins on the test split and on the new sample.

## Limits

- No dashboard. Alerting is only the exit code of `--fail-on-drift`; nothing runs it on a schedule yet.
- Topic drift without label drift goes unnoticed. A cheap next signal would be the `text_chars` distribution.
- The 0.15 threshold and the 30-prediction minimum are first guesses, not tuned on real traffic.
