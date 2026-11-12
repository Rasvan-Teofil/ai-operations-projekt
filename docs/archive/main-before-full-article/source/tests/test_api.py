"""Valid and invalid requests. No transformer, no network."""

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
        from src.train import train

        train()

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
        raise FetchError("Fetching the URL failed.")

    monkeypatch.setattr("app.main.fetch_article_text", boom)
    response = client.post("/predict", json={"url": "https://example.com/missing"})
    assert response.status_code == 502


def test_demo_page_returns_html(client):
    for path in ("/", "/demo"):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Article sentiment" in response.text
        assert 'id="predict-form"' in response.text
        assert 'id="scraped-excerpt"' in response.text
        assert "/static/demo.js" in response.text
    script = client.get("/static/demo.js")
    assert script.status_code == 200
    assert 'fetch("/predict"' in script.text
    assert "scraped_excerpt" in script.text
    style = client.get("/static/demo.css")
    assert style.status_code == 200
    assert "text/css" in style.headers["content-type"]


# Strings the demo got wrong on tfidf-linearsvc-v1. They are not copied
# verbatim into data/sample/opinion_supplement.csv.
OBVIOUS_NEGATIVES = (
    "I dont like that",
    "I don't like that",
    "This is terrible",
    "Das ist absolut schlecht",
    "The company reported a sharp drop in quarterly revenue and warned of further losses.",
    "Das Unternehmen meldete einen starken Umsatzrückgang und warnte vor weiteren Verlusten.",
)


def _ensure_linearsvc_v2() -> None:
    """CI trains before pytest. Retrain if this checkout has no v2 artefact yet."""
    import joblib

    from src.config import MODELS_DIR

    path = MODELS_DIR / "tfidf_linearsvc.joblib"
    if path.exists() and joblib.load(path).get("model_version") == "tfidf-linearsvc-v2":
        return
    from src.train import train

    train()


def test_predict_obvious_negatives_on_the_demo_model(client):
    """The browser demo posts to /predict and serves tfidf_linearsvc, not the CI logreg."""
    _ensure_linearsvc_v2()
    from src.registry import load_model

    previous = (
        client.app.state.model,
        client.app.state.model_name,
        client.app.state.model_version,
    )
    model = load_model("tfidf_linearsvc")
    client.app.state.model = model
    client.app.state.model_name = model.name
    client.app.state.model_version = model.model_version
    try:
        for text in OBVIOUS_NEGATIVES:
            response = client.post("/predict", json={"text": text})
            assert response.status_code == 200, text
            body = response.json()
            assert body["model_version"] == "tfidf-linearsvc-v2"
            assert body["label"] == "negative", (text, body)
            assert body["scores"]["negative"] > body["scores"]["neutral"]
            assert body["scores"]["negative"] > body["scores"]["positive"]
            assert body["scraped_excerpt"] is None
    finally:
        (
            client.app.state.model,
            client.app.state.model_name,
            client.app.state.model_version,
        ) = previous


def test_predict_url_returns_a_short_excerpt(client, monkeypatch):
    article = (
        "The company reported a sharp drop in quarterly revenue and warned of further losses. "
        "Analysts said the warning covers the next two quarters. "
        "The board meets on Thursday to discuss the outlook. "
        "This closing sentence must stay out of the preview."
    )
    monkeypatch.setattr("app.main.fetch_article_text", lambda url: article)
    response = client.post("/predict", json={"url": "https://example.com/news/losses"})
    assert response.status_code == 200
    body = response.json()
    excerpt = body["scraped_excerpt"]
    assert isinstance(excerpt, str)
    assert excerpt.startswith("The company reported a sharp drop")
    assert "next two quarters" in excerpt
    assert "closing sentence" not in excerpt
    assert len(excerpt) <= 400
    assert set(body["scores"]) == set(LABELS_3)


def test_predict_empty_article(client, monkeypatch):
    def empty(url):
        raise EmptyArticleError("No article text could be read from the page.")

    monkeypatch.setattr("app.main.fetch_article_text", empty)
    response = client.post("/predict", json={"url": "https://example.com/empty"})
    assert response.status_code == 422
