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
from app.schemas import PredictRequest, Scores, Prediction, Highlight, Analysis
from src.config import LABELS, MODELS
from src.fetch import FetchError, article_text
from src.article import ARTICLE_POLICY, ArticleError

ModelID = Literal['qwen3_4b', 'minilm_v2', 'linearsvc_v2', 'frozen_news_v3',
                  'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3',
                  'frozen_news_v4', 'minilm_fp32_news_v4', 'minilm_bf16_news_v4', 'linearsvc_news_v4',
                  'qwen_news_v5', 'qwen_fewshot_v5', 'qwen_lora_v5',
                  'article_frozen_v1', 'article_minilm_v1', 'article_qwen_zero_v1', 'article_qwen_lora_v1']
ARTICLE_MODELS = {'article_frozen_v1': 'frozen128', 'article_minilm_v1': 'minilm128',
                  'article_qwen_zero_v1': 'qwen-zero', 'article_qwen_lora_v1': 'qwen-lora'}
MODEL_NAMES = {'qwen3_4b': 'Qwen3 4B · financial rubric · historical zero-shot',
               'minilm_v2': 'MiniLM · fine-tuned BF16 · legacy financial data',
               'linearsvc_v2': 'TF-IDF + LinearSVC · legacy financial data',
               'frozen_news_v3': 'Frozen MiniLM · AI news pilot',
               'minilm_fp32_news_v3': 'MiniLM v1 recipe · AI news pilot',
               'minilm_bf16_news_v3': 'MiniLM v2 recipe · AI news pilot',
               'linearsvc_news_v3': 'LinearSVC · AI news pilot',
               'frozen_news_v4': 'Frozen MiniLM · expanded AI news',
               'minilm_fp32_news_v4': 'MiniLM FP32 · expanded AI news',
               'minilm_bf16_news_v4': 'MiniLM BF16 · expanded AI news',
               'linearsvc_news_v4': 'LinearSVC · expanded AI news',
               'qwen_news_v5': 'Qwen · matched news rubric',
               'qwen_fewshot_v5': 'Qwen · six Train examples',
               'qwen_lora_v5': 'Qwen · own LoRA fine-tuning',
               'article_frozen_v1': 'Full article · frozen MiniLM + trained head',
               'article_minilm_v1': 'Full article · own MiniLM fine-tuning',
               'article_qwen_zero_v1': 'Full article · Qwen news rubric',
               'article_qwen_lora_v1': 'Full article · own Qwen LoRA'}


def candidate_paths():
    return {'qwen3_4b': MODELS / 'local_llm', 'minilm_v2': MODELS / 'development/release',
            'linearsvc_v2': MODELS / 'local_llm_reference',
            **{name: MODELS / 'common_data_v3' / name for name in
               ['frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3']},
            **{name: MODELS / 'expanded_news_v4/final' / name for name in
               ['frozen_news_v4', 'minilm_fp32_news_v4', 'minilm_bf16_news_v4', 'linearsvc_news_v4']},
            **{name: MODELS / 'llm_adaptation_v5/final' / name for name in
               ['qwen_news_v5', 'qwen_fewshot_v5', 'qwen_lora_v5']},
            **{name: MODELS / 'next_stage/final' / folder for name, folder in ARTICLE_MODELS.items()}}


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
                                      'local_llm', 'local_llm_reference',
                                      *(str(Path('next_stage/final') / folder) for folder in ARTICLE_MODELS.values())):
            raise ValueError('Unknown selected model')
        path = MODELS / selection['model']
    predictor = load_artifact(path)
    if not override and predictor.version != selection['model_version']:
        raise ValueError('Selected version differs from the local artifact')
    return predictor



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
    app = FastAPI(title='Article sentiment', lifespan=lifespan)
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
                meta, version, available = {}, None, False
            if meta.get('article_overflow') == 'reject':
                context = f"Complete article up to {meta['max_article_tokens']} article tokens; overflow rejected, no truncation."
            elif meta.get('kind') == 'local_llm':
                context = f"Up to {meta.get('max_article_tokens', 'unknown')} article tokens; beginning/end retained above limit; middle omitted."
            elif meta.get('max_length'):
                context = f"{meta['max_length']}-token chunks; {meta.get('aggregation', 'uniform')} aggregation."
            else:
                context = 'See the recorded artifact policy.'
            records.append({'id': name, 'name': MODEL_NAMES[name], 'model_version': version,
                            'available': available, 'loaded': name in app.state.predictors,
                            'input_policy': ARTICLE_POLICY, 'context_policy': context,
                            'evaluated_scope': ('Native DE/EN/JA full Wikinews articles; AI-labelled · Codex-reviewed, one annotator; uncalibrated scores.'
                                                if name in ARTICLE_MODELS else 'Historical short texts/headlines/leads; full-article and Japanese quality not established.'),
                            'comparison_group': 'full-article-v1' if name in ARTICLE_MODELS else 'historical',
                            'is_default': name == app.state.default_id,
                            'experimental': name in ARTICLE_MODELS or name.endswith(('_news_v3', '_news_v4')) or name in
                                ('qwen_news_v5', 'qwen_fewshot_v5', 'qwen_lora_v5')})
        return {'default': app.state.default_id, 'models': records,
                'article_research_choice': 'article_qwen_lora_v1', 'article_fast_alternative': 'article_minilm_v1',
                'v5_research_choice': 'qwen_lora_v5', 'v5_fast_alternative': 'minilm_bf16_news_v4'}

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

    @app.get('/project/{document}', include_in_schema=False)
    def project_document(document: str):
        root = Path(__file__).resolve().parents[1]
        documents = {'readme': 'README.md', 'next-stage': 'docs/next-stage.md',
                     'archive': 'docs/archive/project-history-through-v5.md',
                     'protocol': 'reports/next_stage/protocol.json',
                     'progress': 'reports/next_stage/progress.json',
                     'final-report': 'reports/next_stage/comparison-v1.md',
                     'model-cards': 'reports/next_stage/model-cards-v1.md',
                     'reproduction': 'docs/reproduce-full-article.md',
                     'structure': 'docs/repository-structure.md'}
        if document not in documents:
            raise HTTPException(404, 'Unknown project document')
        path = root / documents[document]
        if not path.is_file():
            raise HTTPException(503, 'Project document unavailable')
        return FileResponse(path, media_type='application/json' if document in ('protocol', 'progress') else 'text/plain; charset=utf-8')

    @app.get('/comparison/full-article-development', include_in_schema=False)
    def full_article_development():
        path = Path(__file__).resolve().parents[1] / 'reports/next_stage/development-v1.json'
        if not path.exists():
            raise HTTPException(503, 'Full-article development has not completed')
        return json.loads(path.read_text())

    @app.get('/comparison/full-article-development.svg', include_in_schema=False)
    def full_article_development_chart():
        path = Path(__file__).resolve().parents[1] / 'reports/next_stage/development-summary.svg'
        if not path.exists():
            raise HTTPException(503, 'Full-article development chart unavailable')
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/comparison/full-article', include_in_schema=False)
    def full_article_comparison():
        from src.config import REPORTS
        path = REPORTS / 'next_stage/comparison-v1.json'
        if not path.exists():
            raise HTTPException(503, 'Full-article final comparison has not completed')
        return json.loads(path.read_text())

    @app.get('/comparison/full-article.svg', include_in_schema=False)
    def full_article_comparison_chart():
        from src.config import REPORTS
        path = REPORTS / 'next_stage/comparison-v1.svg'
        if not path.exists():
            raise HTTPException(503, 'Full-article final comparison chart unavailable')
        return FileResponse(path, media_type='image/svg+xml')

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

    @app.get('/comparison/news-v3', include_in_schema=False)
    def news_comparison():
        from src.config import REPORTS
        path = REPORTS / 'common_data_v3/comparison.json'
        if not path.exists():
            raise HTTPException(503, 'AI-labelled news pilot has not been completed')
        return json.loads(path.read_text())

    @app.get('/comparison/news-v3.svg', include_in_schema=False)
    def news_comparison_chart():
        from src.config import REPORTS
        path = REPORTS / 'common_data_v3/comparison.svg'
        if not path.exists():
            raise HTTPException(503, 'AI-labelled news pilot has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/comparison/news-v4', include_in_schema=False)
    def expanded_comparison():
        from src.config import REPORTS
        path = REPORTS / 'expanded_news_v4/comparison.json'
        if not path.exists():
            raise HTTPException(503, 'Expanded news comparison has not been completed')
        return json.loads(path.read_text())

    @app.get('/comparison/news-v4.svg', include_in_schema=False)
    def expanded_chart():
        from src.config import REPORTS
        path = REPORTS / 'expanded_news_v4/comparison.svg'
        if not path.exists():
            raise HTTPException(503, 'Expanded news comparison has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/comparison/news-v4-learning.svg', include_in_schema=False)
    def expanded_learning_chart():
        from src.config import REPORTS
        path = REPORTS / 'expanded_news_v4/learning_curve.svg'
        if not path.exists():
            raise HTTPException(503, 'Training-only learning curve has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/comparison/news-v5', include_in_schema=False)
    def llm_adaptation_comparison():
        from src.config import REPORTS
        path = REPORTS / 'llm_adaptation_v5/comparison.json'
        if not path.exists():
            raise HTTPException(503, 'Matched-task LLM comparison has not been completed')
        return json.loads(path.read_text())

    @app.get('/comparison/news-v5.svg', include_in_schema=False)
    def llm_adaptation_chart():
        from src.config import REPORTS
        path = REPORTS / 'llm_adaptation_v5/comparison.svg'
        if not path.exists():
            raise HTTPException(503, 'Matched-task LLM comparison has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    @app.get('/comparison/news-v5-development.svg', include_in_schema=False)
    def llm_adaptation_development_chart():
        from src.config import REPORTS
        path = REPORTS / 'llm_adaptation_v5/development.svg'
        if not path.exists():
            raise HTTPException(503, 'Train-only LLM development comparison has not been completed')
        return FileResponse(path, media_type='image/svg+xml')

    def input_text(payload):
        try:
            if payload.text is not None:
                text = payload.text
            else:
                text = article_text(payload.url, policy=ARTICLE_POLICY)
        except FetchError as e:
            raise HTTPException(e.status, e.detail, headers={'X-Article-Error': e.code}) from e
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
        except ArticleError as e:
            raise HTTPException(422, str(e), headers={'X-Article-Error': e.code}) from e
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
        text = input_text(payload)
        return {**predictions(p, [text])[0],
                'input_policy': ARTICLE_POLICY if payload.url is not None else 'text-verbatim-v1',
                'scraped_excerpt': text[:400] if payload.url is not None else None}

    @app.post('/analyze', response_model=Analysis)
    def analyze(payload: PredictRequest, model: ModelID | None = None):
        from src.highlighting import MAX_SENTENCES, sentence_spans
        p = available_predictor(model)
        text = input_text(payload)
        spans = sentence_spans(text)
        selected = spans[:MAX_SENTENCES]
        result = predictions(p, [text, *[text[start:end] for start, end in selected]])
        highlights = [{'start': start, 'end': end, 'text': text[start:end],
                       'label': row['label'], 'scores': row['scores']}
                      for (start, end), row in zip(selected, result[1:])]
        return {**result[0], 'text': text, 'sentences': highlights,
                'input_policy': ARTICLE_POLICY if payload.url is not None else 'text-verbatim-v1',
                'total_sentences': len(spans), 'highlighting_limited': len(spans) > MAX_SENTENCES,
                'scraped_excerpt': text[:400] if payload.url is not None else None}
    return app

app = create_app()
