"""Gültige und ungültige Anfragen an den lokalen Inferenzdienst."""

import pytest
from fastapi.testclient import TestClient

from src.config import MODEL_PATH, TARGET_NAMES


@pytest.fixture(scope="module")
def client():
    # CI trainiert vorher. Lokal darf pytest auch ohne vorherigen Lauf starten.
    if not MODEL_PATH.exists():
        from src.train import main

        main()

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_predict_valid_request(client):
    response = client.post(
        "/predict",
        json={
            "sepal_length_cm": 5.1,
            "sepal_width_cm": 3.5,
            "petal_length_cm": 1.4,
            "petal_width_cm": 0.2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in TARGET_NAMES
    assert body["model_name"]


def test_predict_invalid_request(client):
    response = client.post(
        "/predict",
        json={"sepal_length_cm": "keine-zahl"},
    )
    assert response.status_code == 422
