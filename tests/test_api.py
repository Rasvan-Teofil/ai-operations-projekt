import socket
import numpy as np
import pytest
from fastapi.testclient import TestClient
import app.main as api
from src.fetch import FetchError


class Stub:
    version = 'test-local-v1'
    def predict_proba(self, texts):
        return np.array([[.1,.2,.7] for _ in texts])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network access in unit test')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, 'load_predictor', lambda: Stub())
    with TestClient(api.create_app()) as c:
        yield c


@pytest.mark.parametrize('text', ['Das Unternehmen meldet Rekordumsatz.', 'Profits increased strongly.'])
def test_text_and_health(client, text):
    assert client.get('/health').json()['model_loaded']
    r = client.post('/predict', json={'text': text})
    assert r.status_code == 200
    d = r.json()
    assert d['label'] == 'positive'
    assert set(d['scores']) == {'negative','neutral','positive'}
    assert sum(d['scores'].values()) == pytest.approx(1)
    assert d['model_version'] == 'test-local-v1'


@pytest.mark.parametrize('payload', [{}, {'text':'x','url':'https://example.com'},
    {'text':'x','url':None}, {'text':None}, {'text':''}, {'text':'  '}, {'text':12},
    {'url':[]}, {'text':'x','other':1}, {'url':'ftp://example.com'},
    {'url':'https://example.com https://other.com'}, {'url':'https://user:pass@example.com'},
    {'url':'http://example.com:9000'}, {'text':'x'*20001}])
def test_invalid(client, payload):
    assert client.post('/predict', json=payload).status_code == 422


def test_url(client, monkeypatch):
    monkeypatch.setattr(api, 'article_text', lambda url, **kwargs: 'Die Umsätze steigen deutlich und übertreffen die Erwartungen.')
    assert client.post('/predict', json={'url':'https://example.com/article'}).status_code == 200


@pytest.mark.parametrize('status', [422,403,502])
def test_fetch_errors(client, monkeypatch, status):
    def fail(url, **kwargs):
        raise FetchError(status, 'controlled failure')
    monkeypatch.setattr(api, 'article_text', fail)
    assert client.post('/predict', json={'url':'https://example.com/article'}).status_code == status


def test_missing_model(monkeypatch):
    def missing():
        raise FileNotFoundError('model')
    monkeypatch.setattr(api, 'load_predictor', missing)
    with TestClient(api.create_app()) as c:
        assert not c.get('/health').json()['model_loaded']
        assert c.post('/predict', json={'text':'hello'}).status_code == 503


def test_inference_unavailable(client, monkeypatch):
    def broken(self, texts):
        raise RuntimeError('device unavailable')
    monkeypatch.setattr(Stub, 'predict_proba', broken)
    assert client.post('/predict', json={'text':'hello'}).status_code == 503


@pytest.mark.parametrize('values', [[float('nan'),.5,.5],[-.1,.5,.6],[.2,.2,.2],[.5,.5]])
def test_invalid_model_distributions_are_503(client,monkeypatch,values):
    monkeypatch.setattr(Stub,'predict_proba',lambda self,texts: np.array([values]))
    assert client.post('/predict',json={'text':'Valid text'}).status_code==503


def test_queue_timeout_is_bounded_and_recovers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    entered=threading.Event()
    release=threading.Event()
    class Slow(Stub):
        def predict_proba(self,texts):
            entered.set()
            assert release.wait(5), 'Test did not release held inference'
            return super().predict_proba(texts)
    monkeypatch.setattr(api,'load_predictor',lambda: Slow())
    with TestClient(api.create_app(queue_timeout=.01)) as c:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(c.post,'/predict',json={'text':'First request'})
            assert entered.wait(2)
            try:
                second=c.post('/predict',json={'text':'Second request'})
                assert second.status_code==503
                assert second.headers['retry-after']=='1'
            finally:
                release.set()
            assert first.result().status_code==200
        assert c.post('/predict',json={'text':'Recovered'}).status_code==200


def test_url_extraction_policy_is_shared_by_model_profiles(client,monkeypatch):
    monkeypatch.setattr(Stub,'meta',{'improved_article':True},raising=False)
    calls=[]
    def fetch(url, **kwargs):
        calls.append(kwargs)
        return 'The company reports record profits and raises its forecast.'
    monkeypatch.setattr(api,'article_text',fetch)
    assert client.post('/predict',json={'url':'https://example.com/news'}).status_code==200
    assert calls == [{'policy': 'full-article-v5'}]


def test_same_url_input_and_policy_reach_different_models(client, monkeypatch):
    received = []
    class Recording(Stub):
        meta = {'improved_article': False}
        def predict_proba(self, texts):
            received.append(texts)
            return super().predict_proba(texts)
    other = Recording()
    monkeypatch.setattr(api, 'load_named_predictor', lambda name: other)
    monkeypatch.setattr(Stub, 'predict_proba', lambda self, texts: received.append(texts) or np.array([[.1, .2, .7]]))
    calls = []
    def fetch(url, **kwargs):
        calls.append(kwargs)
        return 'Headline\n\nThe complete article has an important middle paragraph.'
    monkeypatch.setattr(api, 'article_text', fetch)
    for endpoint in ['/predict', '/predict?model=minilm_v2']:
        response = client.post(endpoint, json={'url': 'https://example.com/news'})
        assert response.status_code == 200
        assert response.json()['input_policy'] == 'full-article-v5'
    assert calls == [{'policy': 'full-article-v5'}] * 2
    assert received[0] == received[1]


def test_access_page_returns_diagnostic_without_model_inference(client, monkeypatch):
    import src.fetch as fetch
    monkeypatch.setattr(fetch, 'public_target', lambda url: (fetch.urlsplit(url), 'example.com', 443, '1.1.1.1'))
    monkeypatch.setattr(fetch, 'request', lambda *args: (404, 'text/plain', '') if args[-1] == '/robots.txt'
                        else (200, 'text/html', '<title>Access denied</title><main><p>' + 'A publisher access challenge. ' * 8 + '</p></main>'))
    def forbidden(*args):
        raise AssertionError('An access page must never be classified')
    monkeypatch.setattr(Stub, 'predict_proba', forbidden)
    response = client.post('/predict', json={'url': 'https://example.com/news'})
    assert response.status_code == 422
    assert response.headers['X-Article-Error'] == 'access_page'


def test_upstream_403_stays_502_but_is_distinguishable(client, monkeypatch):
    import src.fetch as fetch
    monkeypatch.setattr(fetch, 'public_target', lambda url: (fetch.urlsplit(url), 'example.com', 443, '1.1.1.1'))
    monkeypatch.setattr(fetch, 'request', lambda *args: (404, 'text/plain', '') if args[-1] == '/robots.txt' else (403, 'text/html', ''))
    response = client.post('/predict', json={'url': 'https://example.com/news'})
    assert response.status_code == 502
    assert response.json()['detail'] == 'Article returned HTTP 403'
    assert response.headers['X-Article-Error'] == 'upstream_forbidden'


def test_demo_and_static_assets_are_available_without_network(client):
    for path in ['/', '/demo']:
        response = client.get(path)
        assert response.status_code == 200
        assert 'Article sentiment' in response.text
    assert client.get('/static/app.js').status_code == 200
    assert client.get('/static/style.css').status_code == 200


def test_current_local_project_documents_are_reviewable_without_github_or_inference(client, monkeypatch):
    def forbidden(*args):
        raise AssertionError('Project documents must not predict')
    monkeypatch.setattr(Stub, 'predict_proba', forbidden)
    for document in ['readme', 'next-stage', 'archive', 'protocol']:
        assert client.get('/project/' + document).status_code == 200
    assert client.get('/project/protocol').json()['primary_metric']['languages'] == ['de', 'en']
    assert client.get('/project/unknown').status_code == 404
    assert client.get('/project/%2e%2e%2f.env').status_code == 404


def test_url_excerpt_and_text_null(client, monkeypatch):
    monkeypatch.setattr(api, 'article_text', lambda url, **kwargs: 'Article body. ' * 100)
    response = client.post('/predict', json={'url':'https://example.com/article'})
    assert response.status_code == 200
    assert len(response.json()['scraped_excerpt']) == 400
    assert client.post('/predict', json={'text':'Some text'}).json()['scraped_excerpt'] is None


def test_comparison_missing_is_explicit(client, monkeypatch, tmp_path):
    import src.config
    monkeypatch.setattr(src.config, 'REPORTS', tmp_path)
    assert client.get('/comparison').status_code == 503
    assert client.get('/comparison.svg').status_code == 503
    assert client.get('/comparison/news-v3').status_code == 503
    assert client.get('/comparison/news-v3.svg').status_code == 503
    assert client.get('/comparison/news-v4').status_code == 503
    assert client.get('/comparison/news-v4.svg').status_code == 503
    assert client.get('/comparison/news-v4-learning.svg').status_code == 503
    assert client.get('/comparison/news-v5').status_code == 503
    assert client.get('/comparison/news-v5.svg').status_code == 503
    assert client.get('/comparison/news-v5-development.svg').status_code == 503
    assert client.get('/comparison/full-article').status_code == 503
    assert client.get('/comparison/full-article.svg').status_code == 503


def test_full_article_evidence_routes_do_not_predict(client, monkeypatch, tmp_path):
    import json
    import src.config
    target = tmp_path / 'next_stage'; target.mkdir()
    (target / 'comparison-v1.json').write_text(json.dumps({'test_rows': 691, 'selected_on_development': 'qwen-lora'}))
    (target / 'comparison-v1.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    monkeypatch.setattr(src.config, 'REPORTS', tmp_path)
    monkeypatch.setattr(Stub, 'predict_proba', lambda *args: pytest.fail('Evidence must not invoke inference'))
    assert client.get('/comparison/full-article').json()['selected_on_development'] == 'qwen-lora'
    assert client.get('/comparison/full-article.svg').headers['content-type'].startswith('image/svg+xml')


def test_full_article_model_discovery_has_accurate_context_and_scope(client, monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(api, 'MODELS', tmp_path)
    path=tmp_path/'next_stage/final/qwen-lora';path.mkdir(parents=True)
    (path/'artifact.json').write_text(json.dumps({'kind':'local_llm','model_version':'article-qwen-lora-v1',
        'max_article_tokens':4096,'article_overflow':'reject'}))
    records=client.get('/models').json()
    model=next(x for x in records['models'] if x['id']=='article_qwen_lora_v1')
    assert model['available'] and model['comparison_group']=='full-article-v1'
    assert 'overflow rejected' in model['context_policy'] and 'middle omitted' not in model['context_policy']
    assert 'DE/EN/JA' in model['evaluated_scope']
    assert records['article_research_choice']=='article_qwen_lora_v1'
    assert client.post('/predict?model=not-a-model',json={'text':'Test'}).status_code==422


@pytest.mark.parametrize('folder,route,curve', [('expanded_news_v4', 'news-v4', 'learning'),
                                              ('llm_adaptation_v5', 'news-v5', 'development')])
def test_expansion_comparison_routes_serve_saved_evidence_without_inference(client, monkeypatch, tmp_path, folder, route, curve):
    import json
    import src.config
    report = tmp_path / folder; report.mkdir()
    (report / 'comparison.json').write_text(json.dumps({'test_rows': 227, 'results': []}))
    for name in ['comparison.svg', 'learning_curve.svg' if curve == 'learning' else 'development.svg']:
        (report / name).write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    monkeypatch.setattr(src.config, 'REPORTS', tmp_path)
    def forbidden(*args):
        raise AssertionError('Saved comparison route must not predict')
    monkeypatch.setattr(Stub, 'predict_proba', forbidden)
    assert client.get('/comparison/' + route).json()['test_rows'] == 227
    for path in ['/comparison/' + route + '.svg', '/comparison/' + route + '-' + curve + '.svg']:
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('image/svg+xml')


def test_serving_selection_loads_same_artifact_and_version(monkeypatch, tmp_path):
    import json
    import src.serving
    calls = []
    monkeypatch.setattr(api, 'MODELS', tmp_path)
    monkeypatch.delenv('SENTIMENT_ARTIFACT', raising=False)
    (tmp_path / 'selected.json').write_text(json.dumps({'model':'development/release', 'model_version':'historical'}))
    (tmp_path / 'serving_selected.json').write_text(json.dumps({'model':'local_llm', 'model_version':Stub.version}))
    def loader(path):
        calls.append(path)
        return Stub()
    monkeypatch.setattr(src.serving, 'load_artifact', loader)
    assert api.load_predictor().version == Stub.version
    assert calls == [tmp_path / 'local_llm']
    (tmp_path / 'serving_selected.json').write_text(json.dumps({'model':'local_llm', 'model_version':'wrong'}))
    with pytest.raises(ValueError, match='version'):
        api.load_predictor()


def test_model_context_overflow_returns_422_and_recovers(client, monkeypatch):
    from src.article import ArticleError
    original = Stub.predict_proba
    def too_long(self, texts):
        raise ArticleError('article_context_exceeded', 'Full-text token limit exceeded; no text was truncated')
    monkeypatch.setattr(Stub, 'predict_proba', too_long)
    response=client.post('/predict',json={'text':'A synthetically overlong model input.'})
    assert response.status_code==422
    assert response.headers['x-article-error']=='article_context_exceeded'
    monkeypatch.setattr(Stub,'predict_proba',original)
    assert client.post('/predict',json={'text':'A valid shorter input.'}).status_code==200
