import hashlib
import json
import mlflow
from src.config import ROOT, DATA, EXPERIMENT


def tracking():
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    mlflow.set_experiment(EXPERIMENT)


def fingerprint(path):
    h = hashlib.sha256()
    for f in sorted(path.rglob('*')):
        if f.is_file():
            h.update(str(f.relative_to(path)).encode())
            with f.open('rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    h.update(block)
    return h.hexdigest()


def verify_data():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for split, record in manifest['splits'].items():
        if hashlib.sha256((DATA / f'{split}.csv').read_bytes()).hexdigest() != record['sha256']:
            raise RuntimeError(f'{split} split changed since preparation')
    return manifest
