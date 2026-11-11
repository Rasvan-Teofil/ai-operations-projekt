"""Minimal prediction log for monitoring (off by default).

Set ``PREDICTION_LOG_PATH`` to a file path to append one JSON line per
successful ``/predict`` call. The line holds metadata only: timestamp,
model name and version, source type, text length, model latency, label
and scores.
The raw text and the URL are never written.

Read it back with ``summarize`` to see the label distribution, which is
the first drift signal we track before the mock deployment.
"""

import json
import os
from collections import Counter
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


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else log_path()
    if target is None:
        raise SystemExit(f"Pass a log file or set {LOG_PATH_ENV}.")
    print(json.dumps(summarize(target), indent=2))
