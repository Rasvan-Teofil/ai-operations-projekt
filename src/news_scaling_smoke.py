"""Live functional checks on synthetic controls, never on held-out Test inputs."""
import argparse
from datetime import datetime, timezone
import json
import os
import socket
import subprocess
import sys
import time

import httpx
from mlflow import MlflowClient

from src.config import ROOT, LABELS
from src.news_scaling import OUT, read_json, write_json
from src.tracking import tracking


def check(base):
    records = []
    def record(name, response, expected=200, **detail):
        assert response.status_code == expected, (name, response.status_code, response.text[:300])
        records.append({'case': name, 'status': response.status_code, 'expected': expected, **detail})

    with httpx.Client(base_url=base, timeout=90, trust_env=False) as client:
        health = client.get('/health'); record('health', health)
        assert health.json()['model_loaded'] and health.json()['model_id'] == 'qwen3_4b'
        models = client.get('/models'); record('models', models)
        available = models.json()['models']; assert len(available) >= 11 and all(r['available'] for r in available)
        samples = {'de': 'Die Firma meldet Rekordgewinne. Ein Brand zerstörte die Fabrik. Die Konferenz beginnt am Dienstag.',
                   'en': 'The team won the championship. The factory closed after heavy losses. The report will be published on Tuesday.'}
        for model in available:
            for lang, text in samples.items():
                response = client.post('/analyze', params={'model': model['id']}, json={'text': text})
                record(model['id'] + '_' + lang + '_highlight', response)
                result = response.json()
                assert result['model_version'] == model['model_version'] and result['text'] == text
                assert result['label'] in LABELS and set(result['scores']) == set(LABELS)
                assert abs(sum(result['scores'].values()) - 1) < 1e-4
                assert len(result['sentences']) == 3
                for row in result['sentences']:
                    assert text[row['start']:row['end']] == row['text']
                    assert row['label'] in LABELS and set(row['scores']) == set(LABELS)
                records[-1]['model_version'] = result['model_version']
        for name, payload, status in [
            ('valid_url', {'url': 'https://www.bbc.com/news/articles/c687z8127302o'}, 200),
            ('missing', {}, 422), ('both', {'text': 'hello', 'url': 'https://example.com'}, 422),
            ('blank', {'text': '   '}, 422), ('wrong_type', {'text': 42}, 422),
            ('private_address', {'url': 'http://127.0.0.1/'}, 422),
            ('non_article', {'url': 'https://www.bbc.com/robots.txt'}, 422),
            ('robots_denial', {'url': 'https://www.google.com/search?q=sentiment'}, 403),
            ('dns_failure', {'url': 'https://second-path-no-such-host.invalid/news'}, 502)]:
            response = client.post('/predict', params={'model': 'minilm_bf16_news_v4'}, json=payload)
            record(name, response, status)
            if status == 200:
                value = response.json()
                assert value['scraped_excerpt'] and set(value['scores']) == set(LABELS)
                records[-1].update(model_version=value['model_version'], excerpt_present=True)
        for path in ['/', '/docs', '/comparison/news-v4', '/comparison/news-v4.svg',
                     '/comparison/news-v4-learning.svg', '/licenses', '/licenses/inventory']:
            record(path, client.get(path))
        inventory = client.get('/licenses/inventory').json()['entries']
        for entry in inventory:
            for notice in entry['notices']:
                response = client.get('/licenses/notices/' + notice)
                assert response.status_code == 200
        assert client.get('/models').json()['default'] == 'qwen3_4b'
    # Actual separate HTTP process with a deliberately absent artifact, never a
    # second loaded Qwen. Terminate only the child started here.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    missing_path = ROOT / 'models/__absent_scaling_smoke__'
    assert not missing_path.exists()
    env = {**os.environ, 'SENTIMENT_ARTIFACT': str(missing_path), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    with (ROOT / 'news_v4_missing_api.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(port), '--workers', '1'],
                                 cwd=ROOT, env=env, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=5, trust_env=False) as client:
                for _ in range(100):
                    if child.poll() is not None: raise RuntimeError('Controlled missing-artifact API stopped')
                    try:
                        health = client.get('/health')
                        if health.status_code == 200: break
                    except httpx.HTTPError: pass
                    time.sleep(.2)
                else: raise RuntimeError('Missing-artifact API not ready')
                assert not health.json()['model_loaded']
                record('isolated_http_missing_artifact', client.post('/predict', json={'text': 'A valid control sentence.'}), 503)
        finally:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=5)
    tracking(); store = MlflowClient(); cv = read_json(OUT / 'cv.json')['runs']
    final = read_json(OUT / 'comparison.json')['results']
    required = {'accuracy', 'macro_f1', 'f1_negative', 'f1_neutral', 'f1_positive', 'inference_ms_per_text', 'model_size_bytes'}
    run_ids = []
    for r in cv + final:
        run = store.get_run(r['run_id'])
        assert run.info.status == 'FINISHED' and required <= set(run.data.metrics)
        assert run.data.tags['phase'] == 'wikinews-ai-v4-expanded' and run.data.tags['human_gold'] == 'false'
        assert 'confusion_matrix.json' in {a.path for a in store.list_artifacts(r['run_id'])}
        run_ids.append(r['run_id'])
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'api': base,
        'scope': 'Synthetic functional controls, public URL and controlled HTTP failures; no Test predictions repeated or accuracy claim.',
        'cases': records, 'all_passed': True, 'mlflow_finished_runs_checked': len(run_ids), 'run_ids': run_ids,
        'models_checked': len(available), 'license_inventory_entries': len(inventory), 'default_preserved': 'qwen3_4b'}
    write_json(OUT / 'live_verification.json', result)
    print(f'{len(records)} live checks and {len(run_ids)} finished MLflow runs verified; no Test rerun.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--api', default='http://127.0.0.1:8000')
    args = parser.parse_args(); check(args.api)
