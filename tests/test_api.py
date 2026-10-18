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
    monkeypatch.setattr(api, 'article_text', lambda url: 'Die Umsätze steigen deutlich und übertreffen die Erwartungen.')
    assert client.post('/predict', json={'url':'https://example.com/article'}).status_code == 200


@pytest.mark.parametrize('status', [422,403,502])
def test_fetch_errors(client, monkeypatch, status):
    def fail(url):
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
    import threading,time
    entered=threading.Event()
    class Slow(Stub):
        def predict_proba(self,texts):
            entered.set()
            time.sleep(.15)
            return super().predict_proba(texts)
    monkeypatch.setattr(api,'load_predictor',lambda: Slow())
    with TestClient(api.create_app(queue_timeout=.01)) as c:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(c.post,'/predict',json={'text':'First request'})
            assert entered.wait(2)
            second=c.post('/predict',json={'text':'Second request'})
            assert second.status_code==503
            assert second.headers['retry-after']=='1'
            assert first.result().status_code==200
        assert c.post('/predict',json={'text':'Recovered'}).status_code==200


def test_article_profile_uses_improved_extractor(client,monkeypatch):
    monkeypatch.setattr(Stub,'meta',{'improved_article':True},raising=False)
    calls=[]
    def fetch(url,improved=False):
        calls.append(improved)
        return 'The company reports record profits and raises its forecast.'
    monkeypatch.setattr(api,'article_text',fetch)
    assert client.post('/predict',json={'url':'https://example.com/news'}).status_code==200
    assert calls==[True]
