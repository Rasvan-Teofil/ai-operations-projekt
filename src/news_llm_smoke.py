"""Synthetic live API/adapter regression controls, never fresh-Test inference."""
import argparse
from datetime import datetime, timezone
import os
import socket
import subprocess
import sys
import time

import httpx
from mlflow import MlflowClient
import numpy as np

from src.config import ROOT, LABELS
from src.news_llm_experiment import OUT, read_json, write_json
from src.tracking import tracking


def check(base):
    records, regression = [], []
    reference = read_json(OUT / 'legacy_regression_reference.json')['cases']
    def record(name, response, expected=200):
        assert response.status_code == expected, (name, response.status_code, response.text[:300])
        records.append({'case': name, 'status': response.status_code, 'expected': expected})

    with httpx.Client(base_url=base, timeout=90, trust_env=False) as client:
        health = client.get('/health'); record('health', health)
        assert health.json()['model_loaded'] and health.json()['model_id'] == 'qwen3_4b'
        models = client.get('/models'); record('models', models)
        available = models.json()['models']; assert len(available) == 14 and all(r['available'] for r in available)

        def regress(stage):
            maximum = 0.
            for i, item in enumerate(reference):
                response = client.post('/predict', params={'model': 'qwen3_4b'}, json={'text': item['text']})
                record(f'legacy_{stage}_{i}', response)
                actual, expected = response.json(), item['response']
                assert actual['model_version'] == expected['model_version']
                x = np.array([actual['scores'][label] for label in LABELS])
                y = np.array([expected['scores'][label] for label in LABELS])
                assert np.allclose(x, y, atol=1e-6, rtol=1e-6), (stage, i, x, y)
                maximum = max(maximum, float(np.max(np.abs(x - y))))
            regression.append({'stage': stage, 'synthetic_controls': len(reference), 'max_absolute_score_delta': maximum})
        regress('before_adapter_load')
        samples = {'de': '📰 Die Firma meldet Rekordgewinne. Ein Brand zerstörte die Fabrik. Die Konferenz beginnt am Dienstag.',
                   'en': 'The team won the championship. The factory closed after heavy losses. The report will be published on Tuesday.'}
        for model in available:
            for lang, text in samples.items():
                response = client.post('/analyze', params={'model': model['id']}, json={'text': text})
                record(model['id'] + '_' + lang + '_highlights', response)
                value = response.json()
                assert value['model_version'] == model['model_version'] and value['text'] == text
                assert value['label'] in LABELS and set(value['scores']) == set(LABELS)
                assert abs(sum(value['scores'].values()) - 1) < 1e-4
                assert len(value['sentences']) == 3
                for row in value['sentences']:
                    assert text[row['start']:row['end']] == row['text'] and row['label'] in LABELS
                    assert set(row['scores']) == set(LABELS)
                records[-1]['model_version'] = value['model_version']
        regress('after_adapter_and_prompt_switches')
        for name, payload, expected in [
            ('valid_url', {'url': 'https://www.bbc.com/news/articles/c687z8127302o'}, 200),
            ('missing', {}, 422), ('both', {'text': 'hello', 'url': 'https://example.com'}, 422),
            ('blank', {'text': ' '}, 422), ('wrong_type', {'text': 42}, 422),
            ('private_address', {'url': 'http://127.0.0.1/'}, 422),
            ('non_article', {'url': 'https://www.bbc.com/robots.txt'}, 422),
            ('robots_denial', {'url': 'https://www.google.com/search?q=sentiment'}, 403),
            ('dns_failure', {'url': 'https://second-path-no-such-host.invalid/news'}, 502)]:
            response = client.post('/predict', params={'model': 'qwen_news_v5'}, json=payload)
            record(name, response, expected)
            if expected == 200:
                assert response.json()['scraped_excerpt']
                records[-1].update(excerpt_present=True, model_version=response.json()['model_version'])
        for path in ['/', '/docs', '/comparison/news-v5', '/comparison/news-v5.svg',
                     '/comparison/news-v5-development.svg', '/licenses', '/licenses/inventory']:
            record(path, client.get(path))
        inventory = client.get('/licenses/inventory').json()['entries']
        for notice in {n for r in inventory for n in r['notices']}:
            assert client.get('/licenses/notices/' + notice).status_code == 200
        assert client.get('/models').json()['default'] == 'qwen3_4b'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    missing = ROOT / 'models/__absent_llm_adaptation_smoke__'; assert not missing.exists()
    env = {**os.environ, 'SENTIMENT_ARTIFACT': str(missing), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    with (ROOT / 'news_v5_missing_api.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port), '--workers', '1'],
                                 cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5, trust_env=False) as client:
                for _ in range(100):
                    if child.poll() is not None: raise RuntimeError('Controlled missing-artifact API stopped')
                    try:
                        response = client.get('/health')
                        if response.status_code == 200: break
                    except httpx.HTTPError: pass
                    time.sleep(.2)
                else: raise RuntimeError('Missing-artifact API not ready')
                assert not response.json()['model_loaded']
                record('isolated_http_missing_artifact', client.post('/predict', json={'text': 'A valid synthetic control.'}), 503)
        finally:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
    tracking(); store = MlflowClient()
    runs = read_json(OUT / 'cv.json')['runs'] + read_json(OUT / 'comparison.json')['results']
    required = {'accuracy', 'macro_f1', 'f1_negative', 'f1_neutral', 'f1_positive', 'inference_ms_per_text', 'model_size_bytes'}
    for r in runs:
        run = store.get_run(r['run_id'])
        assert run.info.status == 'FINISHED' and required <= set(run.data.metrics)
        assert run.data.tags['phase'] == 'wikinews-llm-adaptation-v5' and run.data.tags['human_gold'] == 'false'
        assert 'confusion_matrix.json' in {a.path for a in store.list_artifacts(r['run_id'])}
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'api': base, 'all_passed': True,
        'scope': 'Synthetic functional and before/after legacy-score regression controls; public URL and controlled errors. No held-out Test reuse or accuracy claim.',
        'cases': records, 'legacy_regression': regression, 'models_checked': len(available),
        'license_inventory_entries': len(inventory), 'finished_new_mlflow_runs_checked': len(runs),
        'run_ids': [r['run_id'] for r in runs], 'default_preserved': 'qwen3_4b'}
    write_json(OUT / 'live_verification.json', result)
    print(f'{len(records)} live checks, {len(runs)} new finished runs, and legacy adapter isolation verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args(); check(args.api)
