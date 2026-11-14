"""Public source notices must work without weights or external services."""
import hashlib
import json
import socket
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

import app.main as api

LICENCES = Path(__file__).resolve().parents[1] / 'licenses'


@pytest.fixture
def client(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('Licence routes must not use the network')

    def no_weights():
        raise FileNotFoundError('Controlled missing artifact')

    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(socket, 'getaddrinfo', no_network)
    monkeypatch.setattr(api, 'load_predictor', no_weights)
    with TestClient(api.create_app()) as client:
        yield client


def test_inventory_is_same_as_git_and_available_without_weights(client):
    assert client.get('/health').json()['model_loaded'] is False
    response = client.get('/licenses/inventory')
    assert response.status_code == 200
    assert response.json() == json.loads((LICENCES / 'inventory.json').read_text())
    entries = response.json()['entries']
    assert set(api.MODEL_NAMES) <= {e['id'] for e in entries if e['group'] == 'current'}
    assert {e['id'] for e in entries if e['group'] == 'candidate'} == {'gerntsa', 'sentivent'}
    for entry in entries:
        for key in ('source_url', 'license_url'):
            assert urlsplit(entry[key]).scheme == 'https'
        assert all(entry[k] for k in ('scope', 'status', 'revision', 'attribution'))


def test_website_links_and_clearly_separates_candidate_sources(client):
    assert 'href="/licenses"' in client.get('/').text
    response = client.get('/licenses')
    assert response.status_code == 200
    assert 'text/html' in response.headers['content-type']
    assert 'Proposed sources · not used yet' in response.text
    assert 'non-commercial restrictions' in response.text
    assert '/licenses/inventory' in response.text
    assert client.get('/static/licenses.js').status_code == 200


def test_restricted_data_is_not_presented_as_permissive(client):
    entries = {e['id']: e for e in client.get('/licenses/inventory').json()['entries']}
    assert 'non-commercial' in entries['kenpache']['license']
    assert entries['phrasebank']['license'] == 'CC-BY-NC-SA-3.0'
    assert 'source news explicitly excluded' in entries['sentivent']['license']
    assert 'publisher' in entries['sentfin']['status']
    assert 'Project-reviewed references' in entries['wikinews']['status']
    assert 'AI-labelled' in entries['wikinews']['original_annotation_status']
    assert entries['codex_annotations']['group'] == 'current'


def test_project_review_is_accessible_without_weights_and_preserves_timing(client):
    review = client.get('/project/data-review').json()
    assert review['reviewed_on'] == '2026-11-14'
    assert review['reviewer_role'] == 'project_owner'
    assert review['review_timing'].startswith('After')
    assert review['reference_labels_changed'] is False
    assert review['final_test_rerun'] is False
    assert review['independent_blind_annotation_claimed'] is False
    assert 'Daten und Originalquellen' in client.get('/project/data-sources').text


def test_all_notices_serve_exact_verified_bytes(client):
    manifest = json.loads((LICENCES / 'notice_manifest.json').read_text())
    tracked_names = {row['file'] for row in manifest}
    assert tracked_names == {p.name for p in (LICENCES / 'notices').iterdir() if p.is_file()}
    for record in manifest:
        response = client.get('/licenses/notices/' + record['file'])
        assert response.status_code == 200
        assert response.content == (LICENCES / 'notices' / record['file']).read_bytes()
        assert hashlib.sha256(response.content).hexdigest() == record['sha256']
        assert len(response.content) == record['bytes']
    for entry in json.loads((LICENCES / 'inventory.json').read_text())['entries']:
        assert set(entry['notices']) <= tracked_names


@pytest.mark.parametrize('path', [
    '/licenses/notices/not-found.txt',
    '/licenses/notices/%2e%2e/inventory.json',
    '/licenses/notices/%2e%2e/%2e%2e/app/main.py',
])
def test_notice_mount_does_not_expose_other_project_files(client, path):
    assert client.get(path).status_code == 404
