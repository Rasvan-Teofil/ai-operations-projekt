import socket

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as api
from src.fetch import FetchError
from src.highlighting import MAX_SENTENCES, sentence_spans


class SentenceStub:
    version = 'sentence-test-v1'
    meta = {'kind': 'test'}

    def predict_proba(self, texts):
        result = []
        for text in texts:
            index = 0 if 'Losses' in text else 2 if 'Profits' in text else 1
            row = [.05, .05, .05]
            row[index] = .9
            result.append(row)
        return np.asarray(result)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network access in unit test')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, 'load_predictor', lambda: SentenceStub())
    with TestClient(api.create_app()) as client:
        yield client


def test_de_en_boundaries_preserve_decimal_abbreviation_and_unicode():
    text = ' 📰 Dr. Müller reports 3.5% growth.\n Acme Inc. publishes results!\n\nNo final punctuation  '
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == [
        '📰 Dr. Müller reports 3.5% growth.', 'Acme Inc. publishes results!', 'No final punctuation']
    assert all(0 <= a < b <= len(text) for a, b in spans)
    assert all(previous[1] <= current[0] for previous, current in zip(spans, spans[1:]))


def test_analysis_retains_original_and_classifies_each_sentence(client):
    text = '📰 Profits increased.  Losses widened.\nThe report is due Tuesday.'
    response = client.post('/analyze', json={'text': text})
    assert response.status_code == 200
    data = response.json()
    assert data['text'] == text
    assert data['total_sentences'] == 3
    assert data['highlighting_limited'] is False
    assert [row['label'] for row in data['sentences']] == ['positive', 'negative', 'neutral']
    for row in data['sentences']:
        assert text[row['start']:row['end']] == row['text']
        assert set(row['scores']) == {'negative', 'neutral', 'positive'}
    assert data['model_version'] == 'sentence-test-v1'


def test_highlight_limit_keeps_unscored_remainder_visible(client):
    text = ' '.join(f'Update number {i}.' for i in range(MAX_SENTENCES + 2))
    data = client.post('/analyze', json={'text': text}).json()
    assert data['text'] == text
    assert data['total_sentences'] == MAX_SENTENCES + 2
    assert len(data['sentences']) == MAX_SENTENCES
    assert data['highlighting_limited'] is True
    assert data['sentences'][-1]['end'] < len(text)


def test_analysis_url_and_validation(client, monkeypatch):
    monkeypatch.setattr(api, 'article_text', lambda url: 'Profits increased. Losses widened.')
    assert client.post('/analyze', json={'url': 'https://example.com/news'}).status_code == 200
    for payload in [{}, {'text': ''}, {'text': 'x', 'url': 'https://example.com'}, {'text': 3}]:
        assert client.post('/analyze', json=payload).status_code == 422


@pytest.mark.parametrize('status', [422, 403, 502])
def test_analysis_fetch_errors(client, monkeypatch, status):
    def fail(url):
        raise FetchError(status, 'controlled')
    monkeypatch.setattr(api, 'article_text', fail)
    assert client.post('/analyze', json={'url': 'https://example.com/news'}).status_code == status


def test_analysis_failed_sentence_inference_is_503(client, monkeypatch):
    monkeypatch.setattr(SentenceStub, 'predict_proba', lambda self, texts: np.array([[.2, .3, .5]]))
    assert client.post('/analyze', json={'text': 'One sentence. Another sentence.'}).status_code == 503


def test_model_choice_is_cached_versioned_and_does_not_change_default(client, monkeypatch):
    calls = []
    def load(name):
        calls.append(name)
        predictor = SentenceStub()
        predictor.version = 'chosen-' + name
        return predictor
    monkeypatch.setattr(api, 'load_named_predictor', load)
    for endpoint in ['/predict', '/analyze']:
        data = client.post(endpoint + '?model=minilm_v2', json={'text': 'Profits increased.'}).json()
        assert data['model_version'] == 'chosen-minilm_v2'
    assert calls == ['minilm_v2']
    assert client.get('/health?model=minilm_v2').json()['model_loaded'] is True
    assert client.post('/predict', json={'text': 'Profits increased.'}).json()['model_version'] == 'sentence-test-v1'
    models = client.get('/models').json()['models']
    assert set(row['id'] for row in models) == {'minilm_v2', 'linearsvc_v2', 'qwen3_4b',
        'frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3'}
    assert {row['id'] for row in models if row['experimental']} == {
        'frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3'}


@pytest.mark.parametrize('model', ['frozen_news_v3', 'minilm_fp32_news_v3', 'minilm_bf16_news_v3', 'linearsvc_news_v3'])
def test_experimental_model_choice_preserves_prediction_contract(client, monkeypatch, model):
    def load(name):
        predictor = SentenceStub()
        predictor.version = 'ai-pilot-' + name
        return predictor
    monkeypatch.setattr(api, 'load_named_predictor', load)
    data = client.post('/predict?model=' + model, json={'text': 'Profits increased.'}).json()
    assert data['model_version'] == 'ai-pilot-' + model
    assert set(data['scores']) == {'negative', 'neutral', 'positive'}
    assert client.post('/predict', json={'text': 'Profits increased.'}).json()['model_version'] == 'sentence-test-v1'


def test_unknown_or_unavailable_requested_model(client, monkeypatch):
    assert client.post('/predict?model=unknown', json={'text': 'Valid text'}).status_code == 422
    assert client.post('/analyze?model=../weights', json={'text': 'Valid text'}).status_code == 422
    def missing(name):
        raise FileNotFoundError('Not installed')
    monkeypatch.setattr(api, 'load_named_predictor', missing)
    for endpoint in ['/predict', '/analyze']:
        assert client.post(endpoint + '?model=linearsvc_v2', json={'text': 'Valid text'}).status_code == 503
