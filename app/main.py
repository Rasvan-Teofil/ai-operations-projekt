"""FastAPI service for local inference.

Start from the project root after training:

    python -m src.train
    uvicorn app.main:app --reload

The default is ``tfidf_linearsvc`` (best macro-F1 in the documented comparison,
small and fast). Another model:

    SENTIMENT_MODEL=finetuned uvicorn app.main:app --reload

CI sets ``SENTIMENT_MODEL=tfidf_logreg`` and loads no transformers.
If the artefact is missing, the process stays up: ``/health`` reports it,
``/predict`` answers with HTTP 503.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.schemas import HealthResponse, PredictRequest, PredictResponse
from src.config import selected_model_name
from src.labels import label_from_scores
from src.scraping import EmptyArticleError, FetchError, RobotsDenied, fetch_article_text

logger = logging.getLogger("ai_operations.api")


def load_predictor():
    """Loads the configured model. Transformers only if explicitly selected."""
    name = selected_model_name()
    try:
        from src.registry import load_model

        return load_model(name), name, None
    except FileNotFoundError as exc:
        logger.warning("Model not loaded: %s", exc)
        return None, name, "missing"
    except Exception:
        logger.exception("Model %s could not be loaded", name)
        return None, name, "unavailable"


def _missing_detail(load_error: str | None, model_name: str) -> str:
    if load_error == "missing":
        return (
            f"Model '{model_name}' is not on disk. "
            "Please run `python -m src.train` in the project root."
        )
    return (
        f"Model '{model_name}' could not be loaded. "
        "Check the name, or install requirements-transformer.txt for transformers."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, name, load_error = load_predictor()
    app.state.model = model
    app.state.model_name = model.name if model is not None else name
    app.state.model_version = model.model_version if model is not None else None
    app.state.load_error = load_error
    if load_error:
        logger.warning("API starting without a model (%s): %s", load_error, name)
    else:
        logger.info("Model loaded: %s (%s)", model.name, model.model_version)
    yield


app = FastAPI(
    title="Artikel-Sentiment",
    summary="Sentiment of a news text or a single article URL.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.model is not None
    return HealthResponse(
        status="ok" if loaded else "model_missing",
        model_loaded=loaded,
        model_name=request.app.state.model_name,
        model_version=request.app.state.model_version,
    )


def _text_from_request(payload: PredictRequest) -> str:
    if payload.text is not None:
        return payload.text.strip()
    try:
        return fetch_article_text(str(payload.url))
    except RobotsDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except EmptyArticleError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=_missing_detail(request.app.state.load_error, request.app.state.model_name),
        )
    text = _text_from_request(payload)
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text to classify.")
    try:
        scores = model.predict_proba([text])[0]
        label = label_from_scores(scores)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
    return PredictResponse(label=label, scores=scores, model_version=model.model_version)
