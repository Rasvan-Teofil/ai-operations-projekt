"""Minimal prediction log for monitoring (off by default).

Set ``PREDICTION_LOG_PATH`` to a file path to append one JSON line per
successful ``/predict`` call. The line holds metadata only: timestamp,
model name and version, source type, text length, model latency, label
and scores.
The raw text and the URL are never written.

Read it back with ``summarize`` to see the label distribution, which is
the first drift signal we track before the mock deployment.
``drift_report`` compares that distribution with a reference (by default
the label share of the test split) and flags a version when the total
variation distance passes a threshold.
``daily_summary`` groups the log by UTC day and model version with count,
label share and median latency.
"""

import json
import os
from collections import Counter
from statistics import median
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH_ENV = "PREDICTION_LOG_PATH"


def log_path() -> Path | None:
    value = os.environ.get(LOG_PATH_ENV, "").strip()
    return Path(value) if value else None


def log_prediction(
    *,
    model_name: str,
    model_version: str,
    source: str,
    text_chars: int,
    label: str,
    scores: dict[str, float],
    latency_ms: float | None = None,
) -> bool:
    """Appends one record. Returns False when logging is disabled."""
    path = log_path()
    if path is None:
        return False
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_version": model_version,
        "source": source,
        "text_chars": text_chars,
        "latency_ms": None if latency_ms is None else round(latency_ms, 1),
        "label": label,
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return True


def summarize(path: Path) -> dict:
    """Counts and label shares per model version."""
    counts: dict[str, Counter] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            counts.setdefault(record["model_version"], Counter())[record["label"]] += 1
    summary = {}
    for version, counter in counts.items():
        total = sum(counter.values())
        summary[version] = {
            "n": total,
            "label_share": {k: round(v / total, 4) for k, v in sorted(counter.items())},
        }
    return summary


def daily_summary(path: Path) -> dict:
    """Per UTC day and model version: n, label share, median latency (ms)."""
    labels: dict[str, dict[str, Counter]] = {}
    latencies: dict[str, dict[str, list[float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            day = record["ts"][:10]
            version = record["model_version"]
            labels.setdefault(day, {}).setdefault(version, Counter())[record["label"]] += 1
            bucket = latencies.setdefault(day, {}).setdefault(version, [])
            if record.get("latency_ms") is not None:
                bucket.append(float(record["latency_ms"]))
    summary: dict[str, dict] = {}
    for day in sorted(labels):
        summary[day] = {}
        for version, counter in sorted(labels[day].items()):
            total = sum(counter.values())
            values = latencies[day][version]
            summary[day][version] = {
                "n": total,
                "label_share": {k: round(v / total, 4) for k, v in sorted(counter.items())},
                "median_latency_ms": round(median(values), 1) if values else None,
            }
    return summary


DRIFT_THRESHOLD = 0.15


def reference_label_share() -> dict[str, float]:
    """Label share of the test split the models were evaluated on."""
    from src.data import load_work_splits

    _, _, test = load_work_splits()
    shares = test["label"].value_counts(normalize=True)
    return {k: round(float(v), 4) for k, v in sorted(shares.items())}


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Half the summed absolute share difference (0 = same, 1 = disjoint)."""
    labels = set(a) | set(b)
    return round(0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in labels), 4)


def drift_report(
    path: Path,
    reference: dict[str, float] | None = None,
    threshold: float = DRIFT_THRESHOLD,
    min_n: int = 30,
) -> dict:
    """Per model version: n, label share, distance to reference, flag.

    Versions with fewer than ``min_n`` predictions get ``drift=None``
    because the share is too noisy to judge.
    """
    reference = reference if reference is not None else reference_label_share()
    report = {"reference": reference, "threshold": threshold, "versions": {}}
    for version, entry in summarize(path).items():
        distance = total_variation(entry["label_share"], reference)
        enough = entry["n"] >= min_n
        report["versions"][version] = {
            **entry,
            "tv_distance": distance,
            "drift": (distance > threshold) if enough else None,
        }
    return report


if __name__ == "__main__":
    import sys

    flags = {"--drift", "--daily"}
    args = [a for a in sys.argv[1:] if a not in flags]
    target = Path(args[0]) if args else log_path()
    if target is None:
        raise SystemExit(f"Pass a log file or set {LOG_PATH_ENV}.")
    if "--drift" in sys.argv:
        result = drift_report(target)
    elif "--daily" in sys.argv:
        result = daily_summary(target)
    else:
        result = summarize(target)
    print(json.dumps(result, indent=2))
