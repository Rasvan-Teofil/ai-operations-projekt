"""FastAPI-Dienst für die lokale Inferenz.

Start vom Projektroot, nachdem ein Artefakt existiert:

    uvicorn app.main:app --reload

Das Modell wird beim Start geladen. Fehlt das Artefakt, bleibt der
Prozess trotzdem oben: ``/health`` meldet das, ``/predict`` antwortet
mit HTTP 503 statt mit einem Traceback.
"""

import logging
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from src.config import MODEL_PATH

logger = logging.getLogger("ai_operations.api")


def read_artifact(path=MODEL_PATH) -> tuple[dict | None, str | None]:
    """Lädt das gespeicherte Artefakt.

    Rückgabe: ``(artefakt, None)`` oder ``(None, fehlercode)``.
    Fehlercodes: ``missing``, ``unreadable``, ``invalid``.
    """
    if not path.exists():
        return None, "missing"
    try:
        artifact = joblib.load(path)
    except Exception:
        logger.exception("Modellartefakt konnte nicht gelesen werden: %s", path)
        return None, "unreadable"
    if not isinstance(artifact, dict) or "estimator" not in artifact:
        logger.error("Modellartefakt hat nicht die erwartete Struktur: %s", path)
        return None, "invalid"
    feature_columns = artifact.get("feature_columns")
    class_labels = artifact.get("class_labels")
    if not feature_columns or not isinstance(class_labels, dict):
        return None, "invalid"
    return artifact, None


def _error_detail(load_error: str | None) -> str:
    if load_error == "missing":
        return (
            "Kein Modellartefakt gefunden. "
            "Bitte im Projektroot `python -m src.train` ausführen."
        )
    return (
        "Modellartefakt ist beschädigt oder unvollständig. "
        "Bitte `python -m src.train` erneut ausführen."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact, load_error = read_artifact()
    app.state.artifact = artifact
    app.state.load_error = load_error
    if load_error:
        logger.warning("API startet ohne Modell (%s). Pfad: %s", load_error, MODEL_PATH)
    else:
        logger.info("Modell geladen: %s", artifact.get("model_name"))
    yield


app = FastAPI(
    title="AI Operations – Inferenz (Platzhalter)",
    summary="Lokaler Vorhersagedienst. Der Datenvertrag ist noch der Iris-Platzhalter.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.artifact is not None
    return HealthResponse(status="ok" if loaded else "model_missing", model_loaded=loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail=_error_detail(request.app.state.load_error))

    features = payload.model_dump()
    columns = list(artifact["feature_columns"])
    missing = [column for column in columns if column not in features]
    if missing:
        raise HTTPException(
            status_code=500,
            detail="Artefakt und Eingabevertrag passen nicht zusammen. Training und Schema prüfen.",
        )

    try:
        row = pd.DataFrame([{column: features[column] for column in columns}])
        predicted = int(artifact["estimator"].predict(row)[0])
        label = artifact["class_labels"][predicted]
    except Exception as exc:
        logger.exception("Vorhersage fehlgeschlagen")
        raise HTTPException(status_code=500, detail="Vorhersage fehlgeschlagen.") from exc

    return PredictResponse(prediction=str(label), model_name=str(artifact["model_name"]))
