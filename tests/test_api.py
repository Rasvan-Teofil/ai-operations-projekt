"""Gültige und ungültige Anfragen. Kein Transformer, kein Netz."""

import os

import pytest
from fastapi.testclient import TestClient

from src.config import LABELS_3
from src.scraping import EmptyArticleError, FetchError


@pytest.fixture(scope="module")
def client():
    previous = os.environ.get("SENTIMENT_MODEL")
    os.environ["SENTIMENT_MODEL"] = "tfidf_logreg"
    from src.config import MODELS_DIR

    if not (MODELS_DIR / "tfidf_logreg.joblib").exists():
        from src.train import main

        main()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    if previous is None:
        os.environ.pop("SENTIMENT_MODEL", None)
    else:
        os.environ["SENTIMENT_MODEL"] = previous


def test_health_reports_the_baseline(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["model_name"] == "tfidf_logreg"
    assert body["model_version"] == "tfidf-logreg-v1"


def test_predict_valid_text(client):
    response = client.post(
        "/predict",
        json={"text": "The clinic reported fewer infections after the new treatment."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in LABELS_3
    assert set(body["scores"]) == set(LABELS_3)
    assert abs(sum(body["scores"].values()) - 1) < 1e-4
    assert body["model_version"] == "tfidf-logreg-v1"


def test_predict_rejects_text_and_url_together(client):
    response = client.post(
        "/predict",
        json={"text": "Hallo", "url": "https://example.com/a"},
    )
    assert response.status_code == 422


def test_predict_rejects_empty_body(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_url_uses_the_fetcher(client, monkeypatch):
    monkeypatch.setattr(
        "app.main.fetch_article_text",
        lambda url: "Freiwillige haben genug Geld gesammelt um die Bibliothek wieder zu öffnen.",
    )
    response = client.post("/predict", json={"url": "https://example.com/news/bibliothek"})
    assert response.status_code == 200
    assert response.json()["label"] in LABELS_3


def test_predict_fetch_failure(client, monkeypatch):
    def boom(url):
        raise FetchError("Abruf der URL ist fehlgeschlagen.")

    monkeypatch.setattr("app.main.fetch_article_text", boom)
    response = client.post("/predict", json={"url": "https://example.com/missing"})
    assert response.status_code == 502


def test_predict_empty_article(client, monkeypatch):
    def empty(url):
        raise EmptyArticleError("Aus der Seite ließ sich kein Artikeltext lesen.")

    monkeypatch.setattr("app.main.fetch_article_text", empty)
    response = client.post("/predict", json={"url": "https://example.com/empty"})
    assert response.status_code == 422
