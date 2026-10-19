import hashlib
import json
import os
from contextlib import contextmanager
import mlflow
from second_path.config import ROOT, DATA, EXPERIMENT, ITERATION


def tracking():
    """Gleicher Datei-Store wie der erste Durchlauf (``./mlruns``), eigenes Experiment."""
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    store = ROOT / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)


@contextmanager
def iteration_run(*args, **kwargs):
    """MLflow-Lauf mit Tag ``iteration=2-second-path``."""
    tags = dict(kwargs.pop("tags", None) or {})
    tags["iteration"] = ITERATION
    kwargs["tags"] = tags
    with mlflow.start_run(*args, **kwargs) as run:
        mlflow.set_tag("iteration", ITERATION)
        yield run


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
