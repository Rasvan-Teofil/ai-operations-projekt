"""A single terminal test evaluation, after every candidate has been frozen."""
import json
import time
import platform
import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.config import *
from src.inference import Predictor
from src.tracking import tracking, verify_data, fingerprint


def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise SystemExit('Evaluation protocol requires CUDA')
    manifest = verify_data()
    tracking()
    candidates = ['frozen', 'finetuned', 'centroid', 'majority']
    records = {}
    for name in candidates:
        path = MODELS / name
        meta = json.loads((path / 'artifact.json').read_text())
        run = mlflow.get_run(meta['run_id'])
        if fingerprint(path) != run.data.params['artifact_sha256']:
            raise RuntimeError(f'{name} artifact differs from trained artifact')
        records[name] = meta
    # Exclusive creation prevents accidentally probing the test again after seeing scores.
    lock = REPORTS / 'evaluation.lock'
    with lock.open('x') as f:
        json.dump({'status': 'started', 'test_sha256': manifest['splits']['test']['sha256'],
                   'candidates': records}, f, indent=2)
    test = pd.read_csv(DATA / 'test.csv')
    truth = [LABELS.index(v) for v in test.label]
    rows = []
    for name in candidates:
        path = MODELS / name
        predictor = Predictor(path, device='cuda')
        for _ in range(20):
            predictor.predict_proba(['Die Firma meldet ihre Quartalszahlen.'])
        # One prediction per test text, warm CUDA, batch size one, tokenizer + transfer included.
        preds, elapsed = [], 0.
        for text in test.text:
            torch.cuda.synchronize()
            start = time.perf_counter()
            p = predictor.predict_proba([text])[0]
            torch.cuda.synchronize()
            elapsed += time.perf_counter()-start
            preds.append(int(p.argmax()))
        per_class = f1_score(truth, preds, labels=[0,1,2], average=None, zero_division=0)
        metrics = {'accuracy': accuracy_score(truth, preds),
                   'macro_f1': f1_score(truth, preds, labels=[0,1,2], average='macro', zero_division=0),
                   'inference_ms_per_text': 1000*elapsed/len(test),
                   'model_size_bytes': sum(f.stat().st_size for f in path.rglob('*') if f.is_file()),
                   **{f'f1_{label}': float(per_class[i]) for i,label in enumerate(LABELS)}}
        cm = pd.DataFrame(confusion_matrix(truth, preds, labels=[0,1,2]), index=LABELS, columns=LABELS)
        cm.index.name = 'actual/predicted'
        cm.to_csv(REPORTS / f'{name}_confusion_matrix.csv')
        by_language = {}
        for lang in ['de', 'en']:
            ix = np.flatnonzero(test.language.to_numpy() == lang)
            by_language[f'macro_f1_{lang}'] = f1_score(np.array(truth)[ix], np.array(preds)[ix], labels=[0,1,2], average='macro', zero_division=0)
        with mlflow.start_run(run_id=records[name]['run_id']):
            mlflow.log_metrics({**metrics, **by_language})
            mlflow.log_params({'test_rows': len(test), 'latency_batch_size': 1, 'warmup_texts': 20,
                               'evaluation_device': torch.cuda.get_device_name(), 'platform': platform.platform()})
            mlflow.log_artifact(str(REPORTS / f'{name}_confusion_matrix.csv'))
            mlflow.log_dict(metrics, 'test_metrics.json')
        rows.append({'model': name, 'run_id': records[name]['run_id'], **metrics, **by_language})
        print(name, metrics, flush=True)
        del predictor
        torch.cuda.empty_cache()
    # Predeclared close margin: <= 0.005 absolute macro-F1; prefer latency, then bytes.
    best = max(r['macro_f1'] for r in rows)
    close = [r for r in rows if best-r['macro_f1'] <= .005]
    selected = min(close, key=lambda r: (r['inference_ms_per_text'], r['model_size_bytes']))
    (MODELS / 'selected.json').write_text(json.dumps({'model': selected['model'], 'model_version': records[selected['model']]['model_version']}, indent=2))
    result = {'test_sha256': manifest['splits']['test']['sha256'], 'selection_margin': .005,
              'selected': selected['model'], 'results': rows}
    (REPORTS / 'comparison.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(rows).to_csv(REPORTS / 'comparison.csv', index=False)
    with lock.open('w') as f:
        json.dump({'status': 'complete', **result}, f, indent=2)
    print('Selected:', selected['model'])

if __name__ == '__main__':
    main()
