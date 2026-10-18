import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator
from src.config import LABELS, MODELS
from src.fetch import FetchError, article_text


def load_predictor():
    from src.inference import Predictor
    override = os.getenv('SENTIMENT_ARTIFACT')
    if override:
        path = Path(override)
    else:
        selection = json.loads((MODELS / 'selected.json').read_text())
        if selection['model'] not in ('frozen', 'finetuned', 'centroid', 'majority'):
            raise ValueError('Unknown selected model')
        path = MODELS / selection['model']
    return Predictor(path)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: StrictStr | None = None
    url: StrictStr | None = None

    @model_validator(mode='after')
    def exactly_one(self):
        if len(self.model_fields_set) != 1:
            raise ValueError('Provide exactly one of text or url; omit the other field')
        value = self.text if self.text is not None else self.url
        if not value or not value.strip() or len(value) > 20000:
            raise ValueError('Input must contain 1–20000 non-blank characters')
        if self.url is not None:
            from urllib.parse import urlsplit
            try:
                u = urlsplit(self.url)
                if (u.scheme not in ('http','https') or not u.hostname or u.username or u.password
                    or any(c.isspace() for c in self.url) or u.port not in (None,80,443)):
                    raise ValueError('Exactly one http(s) URL is required')
            except (ValueError, UnicodeError) as e:
                raise ValueError('Exactly one http(s) URL is required') from e
        return self


class Scores(BaseModel):
    negative: float
    neutral: float
    positive: float


class Prediction(BaseModel):
    label: Literal['negative','neutral','positive']
    scores: Scores
    model_version: str


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.predictor = load_predictor()
        except Exception:
            logging.getLogger(__name__).exception('Local artifact could not be loaded')
            app.state.predictor = None
        yield
    app = FastAPI(title='Article sentiment — second path', lifespan=lifespan)
    app.state.predictor = None
    inference_lock = threading.Lock()

    @app.get('/health')
    def health():
        p = app.state.predictor
        return {'status': 'ok' if p else 'unavailable', 'model_loaded': p is not None,
                'model_version': p.version if p else None}

    @app.post('/predict', response_model=Prediction)
    def predict(payload: PredictRequest):
        p = app.state.predictor
        if p is None:
            raise HTTPException(503, 'Local model artifact unavailable; run the documented training pipeline')
        try:
            text = payload.text if payload.text is not None else article_text(payload.url)
        except FetchError as e:
            raise HTTPException(e.status, e.detail) from e
        try:
            with inference_lock:
                probabilities = p.predict_proba([text])[0]
            return {'label': LABELS[int(probabilities.argmax())],
                    'scores': dict(zip(LABELS, map(float, probabilities))), 'model_version': p.version}
        except Exception as e:
            logging.getLogger(__name__).exception('Local inference failed')
            raise HTTPException(503, 'Local model inference unavailable') from e
    return app

app = create_app()
