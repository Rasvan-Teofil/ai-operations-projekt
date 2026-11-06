import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator
from src.config import LABELS, MODELS
from src.fetch import FetchError, article_text

ModelID = Literal['qwen3_4b', 'minilm_v2', 'linearsvc_v2']
MODEL_NAMES = {'qwen3_4b': 'Qwen3 4B · local zero-shot',
               'minilm_v2': 'MiniLM v2 · fine-tuned',
               'linearsvc_v2': 'LinearSVC v2 · reconstructed'}


def candidate_paths():
    return {'qwen3_4b': MODELS / 'local_llm', 'minilm_v2': MODELS / 'development/release',
            'linearsvc_v2': MODELS / 'local_llm_reference'}


def load_named_predictor(name):
    from src.serving import load_artifact
    return load_artifact(candidate_paths()[name])


def load_predictor():
    from src.serving import load_artifact
    import torch
    torch.set_num_threads(4)  # Same CPU thread bound as the measured comparison.
    override = os.getenv('SENTIMENT_ARTIFACT')
    if override:
        path = Path(override)
    else:
        selection_file = MODELS / 'serving_selected.json'
        if not selection_file.exists():
            selection_file = MODELS / 'selected.json'
        selection = json.loads(selection_file.read_text())
        if selection['model'] not in ('frozen', 'finetuned', 'centroid', 'majority', 'development/release',
                                      'local_llm', 'local_llm_reference'):
            raise ValueError('Unknown selected model')
        path = MODELS / selection['model']
    predictor = load_artifact(path)
    if not override and predictor.version != selection['model_version']:
        raise ValueError('Selected version differs from the local artifact')
    return predictor


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
    scraped_excerpt: str | None = None


class Highlight(BaseModel):
    start: int
    end: int
    text: str
    label: Literal['negative', 'neutral', 'positive']
    scores: Scores


class Analysis(Prediction):
    text: str
    sentences: list[Highlight]
    total_sentences: int
    highlighting_limited: bool


def create_app(queue_timeout=None):
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.predictor = load_predictor()
        except Exception:
            logging.getLogger(__name__).exception('Local artifact could not be loaded')
            app.state.predictor = None
        app.state.default_id = None
        for name, path in candidate_paths().items():
            try:
                if app.state.predictor and json.loads((path / 'artifact.json').read_text())['model_version'] == app.state.predictor.version:
                    app.state.predictors[name] = app.state.predictor
                    app.state.default_id = name
            except (OSError, ValueError, KeyError):
                pass
        yield
    app = FastAPI(title='Article sentiment — second path', lifespan=lifespan)
    app.state.predictor = None
    app.state.predictors = {}
    app.state.default_id = None
    static = Path(__file__).resolve().parent / 'static'
    app.mount('/static', StaticFiles(directory=static), name='static')
    licensing = Path(__file__).resolve().parents[1] / 'licenses'
    app.mount('/licenses/notices', StaticFiles(directory=licensing / 'notices'), name='license-notices')
    inference_lock = threading.BoundedSemaphore(1)
    wait_limit = float(os.getenv('SENTIMENT_QUEUE_TIMEOUT', '2.0')) if queue_timeout is None else queue_timeout
    if wait_limit < 0:
        raise ValueError('Queue timeout must be nonnegative')

    @app.get('/health')
    def health(model: ModelID | None = None):
        p = app.state.predictor if model is None else app.state.predictors.get(model)
        available = p is not None or (model is not None and (candidate_paths()[model] / 'artifact.json').exists())
        return {'status': 'ok' if p else 'not_loaded' if available else 'unavailable', 'model_loaded': p is not None,
                'model_available': available, 'model_id': model or app.state.default_id,
                'model_version': p.version if p else None,
                'model_kind': getattr(p, 'meta', {}).get('kind') if p else None,
                'score_method': getattr(p, 'meta', {}).get('score_method', 'Normalized model scores; uncalibrated') if p else None}

    @app.get('/models')
    def models():
        records = []
        for name, path in candidate_paths().items():
            try:
                meta = json.loads((path / 'artifact.json').read_text())
                version, available = meta['model_version'], True
            except (OSError, ValueError, KeyError):
                version, available = None, False
            records.append({'id': name, 'name': MODEL_NAMES[name], 'model_version': version,
                            'available': available, 'loaded': name in app.state.predictors,
                            'is_default': name == app.state.default_id})
        return {'default': app.state.default_id, 'models': records}

    @app.get('/', include_in_schema=False)
    @app.get('/demo', include_in_schema=False)
    def demo():
        return FileResponse(static / 'index.html')

    @app.get('/licenses', include_in_schema=False)
    def licenses_page():
        return FileResponse(static / 'licenses.html')

    @app.get('/licenses/inventory')
    def license_inventory():
        # Same reviewed inventory as Git; no network or model load for this route.
        return json.loads((licensing / 'inventory.json').read_text(encoding='utf-8'))

    @app.get('/comparison', include_in_schema=False)
    def comparison():
        from src.config import REPORTS
        path = REPORTS / 'local_llm/comparison.json'
        if not path.exists():
            raise HTTPException(503, 'Comparison has not been completed')
        return json.loads(path.read_text())

    @app.get('/comparison.svg', include_in_schema=False)
    def comparison_chart():
        from src.config import REPORTS
        path = REPORTS / 'local_llm/comparison.svg'
        if not path.exists():
            raise HTTPException(503, 'Comparison has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    def input_text(payload, p):
        try:
            if payload.text is not None:
                text = payload.text
            elif getattr(p, 'meta', {}).get('improved_article', False):
                text = article_text(payload.url, improved=True)
            else:
                text = article_text(payload.url)
        except FetchError as e:
            raise HTTPException(e.status, e.detail) from e
        return text

    def predictions(p, texts):
        try:
            if not inference_lock.acquire(timeout=wait_limit):
                raise HTTPException(503, 'Inference queue busy; retry later', headers={'Retry-After': '1'})
            try:
                matrix = p.predict_proba(texts)
            finally:
                inference_lock.release()
            if len(matrix) != len(texts):
                raise ValueError('Invalid model result count')
            result = []
            for row in matrix:
                values = list(map(float, row))
                if (len(values) != 3 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values)
                        or not math.isclose(sum(values), 1., abs_tol=1e-4)):
                    raise ValueError('Invalid model score distribution')
                result.append({'label': LABELS[max(range(3), key=lambda i: values[i])],
                               'scores': dict(zip(LABELS, values)), 'model_version': p.version})
            return result
        except HTTPException:
            raise
        except Exception as e:
            logging.getLogger(__name__).exception('Local inference failed')
            raise HTTPException(503, 'Local model inference unavailable') from e

    def available_predictor(model):
        if model is None:
            if app.state.predictor is not None:
                return app.state.predictor
            raise HTTPException(503, 'Local model artifact unavailable; run the documented artifact setup')
        if not inference_lock.acquire(timeout=wait_limit):
            raise HTTPException(503, 'Inference queue busy; retry later', headers={'Retry-After': '1'})
        try:
            if model not in app.state.predictors:
                app.state.predictors[model] = load_named_predictor(model)
            return app.state.predictors[model]
        except Exception as e:
            logging.getLogger(__name__).exception('Requested local model could not be loaded')
            raise HTTPException(503, 'Requested local model artifact unavailable') from e
        finally:
            inference_lock.release()

    @app.post('/predict', response_model=Prediction)
    def predict(payload: PredictRequest, model: ModelID | None = None):
        p = available_predictor(model)
        text = input_text(payload, p)
        return {**predictions(p, [text])[0],
                'scraped_excerpt': text[:400] if payload.url is not None else None}

    @app.post('/analyze', response_model=Analysis)
    def analyze(payload: PredictRequest, model: ModelID | None = None):
        from src.highlighting import MAX_SENTENCES, sentence_spans
        p = available_predictor(model)
        text = input_text(payload, p)
        spans = sentence_spans(text)
        selected = spans[:MAX_SENTENCES]
        result = predictions(p, [text, *[text[start:end] for start, end in selected]])
        highlights = [{'start': start, 'end': end, 'text': text[start:end],
                       'label': row['label'], 'scores': row['scores']}
                      for (start, end), row in zip(selected, result[1:])]
        return {**result[0], 'text': text, 'sentences': highlights,
                'total_sentences': len(spans), 'highlighting_limited': len(spans) > MAX_SENTENCES,
                'scraped_excerpt': text[:400] if payload.url is not None else None}
    return app

app = create_app()
