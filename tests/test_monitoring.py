"""Prediction log: off by default, metadata only, summary per version."""

import json

from app.monitoring import LOG_PATH_ENV, log_prediction, summarize


def _log(label):
    return log_prediction(
        model_name="tfidf_logreg",
        model_version="v1",
        source="text",
        text_chars=42,
        label=label,
        scores={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
    )


def test_disabled_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(LOG_PATH_ENV, raising=False)
    assert _log("positive") is False
    assert list(tmp_path.iterdir()) == []


def test_writes_metadata_only_and_summarizes(monkeypatch, tmp_path):
    path = tmp_path / "logs" / "predictions.jsonl"
    monkeypatch.setenv(LOG_PATH_ENV, str(path))
    assert _log("positive") and _log("positive") and _log("negative")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3
    assert "text" not in records[0] and "url" not in records[0]
    assert records[0]["text_chars"] == 42
    summary = summarize(path)
    assert summary["v1"]["n"] == 3
    assert summary["v1"]["label_share"] == {"negative": 0.3333, "positive": 0.6667}


def test_api_logs_prediction(monkeypatch, tmp_path):
    import os

    from fastapi.testclient import TestClient

    path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv(LOG_PATH_ENV, str(path))
    monkeypatch.setenv("SENTIMENT_MODEL", "tfidf_logreg")
    from src.config import MODELS_DIR

    if not (MODELS_DIR / "tfidf_logreg.joblib").exists():
        from src.train import train

        train()
    from app.main import app

    secret = "The clinic reported fewer infections after the new treatment."
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": secret})
    assert response.status_code == 200
    content = path.read_text()
    assert secret not in content
    record = json.loads(content.splitlines()[0])
    assert record["source"] == "text"
    assert record["latency_ms"] >= 0
    assert record["label"] == response.json()["label"]
    assert os.environ[LOG_PATH_ENV] == str(path)


def test_drift_report_flags_shifted_share(monkeypatch, tmp_path):
    from app.monitoring import drift_report, total_variation

    path = tmp_path / "predictions.jsonl"
    monkeypatch.setenv(LOG_PATH_ENV, str(path))
    for _ in range(30):
        _log("negative")
    for _ in range(10):
        _log("positive")
    reference = {"negative": 0.3333, "neutral": 0.3334, "positive": 0.3333}
    assert total_variation(reference, reference) == 0.0
    report = drift_report(path, reference=reference)
    entry = report["versions"]["v1"]
    assert entry["n"] == 40
    assert entry["tv_distance"] == 0.4167
    assert entry["drift"] is True
    small = drift_report(path, reference=reference, min_n=100)
    assert small["versions"]["v1"]["drift"] is None


def test_reference_share_from_test_split():
    from app.monitoring import reference_label_share

    shares = reference_label_share()
    assert set(shares) <= {"negative", "neutral", "positive"}
    assert abs(sum(shares.values()) - 1.0) < 0.01
