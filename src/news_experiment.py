"""Separate AI-annotated Wikinews pilot. Never overwrites historical experiments."""
import argparse
import csv
import gc
import hashlib
import json
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from src.config import ROOT, MODELS, LABELS, ENCODER, ENCODER_REVISION
from src.news_data import LOCAL, OUT, SEED, digest, normalized
from src.inference import SentimentNet, save_model
from src.serving import load_artifact
from src.tracking import tracking, fingerprint

ART = MODELS / 'common_data_v3'
NAMES = {'frozen_news_v3': 'Frozen MiniLM · AI news pilot',
         'minilm_fp32_news_v3': 'MiniLM FP32 · v1 recipe / new AI data',
         'minilm_bf16_news_v3': 'MiniLM BF16 · v2 deployment / new AI data',
         'linearsvc_news_v3': 'LinearSVC · new AI news data',
         'qwen3_4b': 'Qwen3 4B · unchanged zero-shot'}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def groups(rows):
    """Conservative linked-page, shared-source and near-duplicate components."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        parent[find(j)] = find(i)

    keys = {}
    for i, row in enumerate(rows):
        for key in ['page:' + row['title_key'], 'text:' + normalized(row['text'])] + [
                'page:' + k for k in row['linked_articles']] + ['source:' + k for k in row['source_keys']]:
            if key in keys:
                join(i, keys[key])
            else:
                keys[key] = i
    vectors = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 5)).fit_transform([r['text'] for r in rows])
    similarities = (vectors @ vectors.T).toarray()
    for i in range(len(rows)):
        for j in range(i):
            if similarities[i, j] >= .85:
                join(i, j)
    return [rows[find(i)]['id'] for i in range(len(rows))]


def prepare():
    if (OUT / 'split_manifest.json').exists():
        raise RuntimeError('Splits already frozen; refusing to overwrite')
    source = read_json(OUT / 'source_manifest.json')
    assert digest((LOCAL / 'articles.jsonl').read_bytes()) == source['sha256']
    rows = [json.loads(x) for x in (LOCAL / 'articles.jsonl').read_text().splitlines()]
    with (OUT / 'annotations.csv').open() as stream:
        annotation_rows = list(csv.DictReader(stream))
    labels = {r['id']: r for r in annotation_rows}
    if len(labels) != len(annotation_rows) or set(labels) != {r['id'] for r in rows}:
        raise ValueError('Every source item needs exactly one reviewed annotation')
    accepted, exclusions = [], []
    for row in rows:
        a = labels[row['id']]
        if a['label'] not in LABELS + ['exclude'] or not a['reason'].strip():
            raise ValueError('Invalid annotation or missing decision rationale')
        if a['label'] == 'exclude':
            exclusions.append({'id': row['id'], 'reason': a['reason']})
        else:
            accepted.append({**row, 'label': a['label'], 'annotation_source': 'AI-labelled / Codex-reviewed'})
    group_ids = groups(accepted)
    df = pd.DataFrame(accepted)
    df['group'] = group_ids
    strata = df.language + ':' + df.label
    cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    folds = list(cv.split(df, strata, df['group']))
    test_ix, val_ix = folds[0][1], folds[1][1]
    membership = np.full(len(df), 'train', dtype=object)
    membership[test_ix], membership[val_ix] = 'test', 'val'
    manifest = {'data_version': source['data_version'], 'seed': SEED,
        'label_source': 'Single AI annotator: Codex assistant in this chat; exact model revision unavailable',
        'review': 'Same assistant reviewed consistency without candidate predictions; no human adjudication',
        'annotation_sha256': digest((OUT / 'annotations.csv').read_bytes()),
        'source_sha256': source['sha256'], 'exclusions': exclusions,
        'group_policy': 'Linked DE/EN pages, same external source URL, normalized duplicate or char-TFIDF cosine >= .85; residual event overlap possible',
        'splits': {}, 'membership': []}
    for split in ['train', 'val', 'test']:
        part = df[membership == split]
        if len(part) == 0:
            raise ValueError('Empty split')
        counts = (part.language + ':' + part.label).value_counts().to_dict()
        if any(counts.get(lang + ':' + label, 0) < (5 if split == 'train' else 1)
               for lang in ['de', 'en'] for label in LABELS):
            raise ValueError(f'Insufficient cell coverage in fixed split: {split}, {counts}')
        raw = part.to_csv(index=False).encode()
        (LOCAL / f'{split}.csv').write_bytes(raw)
        manifest['splits'][split] = {'rows': len(part), 'sha256': digest(raw), 'language_label_counts': counts}
        manifest['membership'].extend({'id': r.id, 'group': r.group, 'split': split, 'text_sha256': r.text_sha256}
                                      for r in part.itertuples())
    assert not (set(df.iloc[test_ix]['group']) & set(df.iloc[val_ix]['group']))
    write_json(OUT / 'split_manifest.json', manifest)
    print(json.dumps(manifest['splits'], indent=2), flush=True)


def verified():
    manifest = read_json(OUT / 'split_manifest.json')
    if read_json(OUT / 'overlap_audit.json')['matched_ids']:
        raise RuntimeError('Known project inputs overlap with the new pilot')
    for split, info in manifest['splits'].items():
        if digest((LOCAL / f'{split}.csv').read_bytes()) != info['sha256']:
            raise RuntimeError('Split changed: ' + split)
    if digest((OUT / 'annotations.csv').read_bytes()) != manifest['annotation_sha256']:
        raise RuntimeError('Annotations changed after split freezing')
    return manifest


def gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 required; refusing CPU training/evaluation')
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tracking()


def protocol_hash():
    return digest((OUT / 'protocol.json').read_bytes())


def tags(recipe):
    return {'phase': 'wikinews-ai-v3-pilot', 'data_version': 'wikinews-ai-v3-pilot',
        'comparable_group': 'wikinews-ai-v3-heldout', 'recipe': recipe,
        'label_source': 'AI-labelled / Codex-reviewed', 'human_gold': 'false',
        'metric_interpretation': 'Agreement with single-assistant annotations, not human-verified accuracy',
        'deployment_eligible': 'false'}


def train():
    gpu()
    manifest = verified()
    if (OUT / 'trained.json').exists() or (OUT / 'evaluation.lock').exists() or ART.exists():
        raise RuntimeError('Pilot training already started/completed; preserve artifacts')
    ART.mkdir(parents=True)
    df = pd.read_csv(LOCAL / 'train.csv')  # Val/Test labels never read during fitting.
    y = np.array([LABELS.index(x) for x in df.label])
    text = df.text.tolist()
    records = []
    common = {'seed': SEED, 'train_rows': len(df), 'test_sha256': manifest['splits']['test']['sha256'],
        'annotation_sha256': manifest['annotation_sha256'], 'protocol_sha256': protocol_hash(),
        'val_policy': 'Never opened; fixed final epoch, no candidate selection',
        'data_version': manifest['data_version'], 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'cpu_threads': 4,
        'label_source': 'AI-labelled / Codex-reviewed'}
    import subprocess
    common['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    def record(name, path, run_id):
        sha = fingerprint(path)
        mlflow.log_param('artifact_sha256', sha)
        mlflow.log_artifact(str(path / 'artifact.json'), 'model_metadata')
        records.append({'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                        'run_id': run_id, 'artifact_sha256': sha})
        write_json(OUT / 'training_progress.json', records)

    # Independent implementation of the colleague-style C=1 balanced word/bigram recipe.
    name = 'linearsvc_news_v3'
    with mlflow.start_run(run_name='AI news v3 · LinearSVC') as run:
        mlflow.set_tags(tags(name)); mlflow.log_params({**common, 'C': 1., 'ngram_range': '(1,2)',
            'min_df': 1, 'class_weight': 'balanced', 'max_iter': 2000, 'backend': 'sklearn CPU on GPU tower'})
        started = time.perf_counter()
        estimator = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                                  LinearSVC(C=1., class_weight='balanced', max_iter=2000, random_state=SEED))
        estimator.fit(text, df.label.tolist())
        import joblib, sklearn
        path = ART / name; path.mkdir()
        version = 'wikinews-ai-v3-linearsvc'
        joblib.dump({'estimator': estimator, 'model_version': version}, path / 'tfidf_linearsvc.joblib')
        write_json(path / 'artifact.json', {'kind': 'reference_sklearn', 'model_version': version,
            'labels': LABELS, 'sklearn': sklearn.__version__, 'improved_article': True,
            'data_version': manifest['data_version'], 'label_source': 'AI-labelled / Codex-reviewed'})
        mlflow.log_metric('training_seconds', time.perf_counter() - started)
        record(name, path, run.info.run_id)

    # Original pinned Hub weights only: no legacy Kenpache fine-tune initialization.
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION, local_files_only=True)
    net = SentimentNet(AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION, local_files_only=True)).cuda()
    tokens = tokenizer(text, truncation=True, max_length=128, padding=False)

    def batch(indices):
        return tokenizer.pad([{k: v[int(i)] for k, v in tokens.items()} for i in indices],
                             padding=True, return_tensors='pt').to('cuda')

    name = 'frozen_news_v3'
    with mlflow.start_run(run_name='AI news v3 · Frozen GPU baseline') as run:
        mlflow.set_tags(tags(name)); mlflow.log_params({**common, 'encoder': ENCODER,
            'encoder_revision': ENCODER_REVISION, 'initialization': 'Clean original pretrained Hub snapshot',
            'max_length': 128, 'encoder_frozen': True, 'head': 'Balanced C=1 logistic regression', 'batch_size': 64})
        started = time.perf_counter(); net.eval(); vectors = []
        with torch.inference_mode():
            for start in range(0, len(df), 64):
                vectors.append(net.features(batch(range(start, min(start + 64, len(df))))).cpu().numpy())
        classifier = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=SEED)
        classifier.fit(np.concatenate(vectors), y)
        with torch.no_grad():
            net.head.weight.copy_(torch.tensor(classifier.coef_, dtype=torch.float32, device='cuda'))
            net.head.bias.copy_(torch.tensor(classifier.intercept_, dtype=torch.float32, device='cuda'))
        meta = {'kind': 'frozen', 'model_version': 'wikinews-ai-v3-frozen', 'run_id': run.info.run_id,
            'precision': 'fp32', 'aggregation': 'uniform', 'improved_article': True,
            'data_version': manifest['data_version'], 'label_source': 'AI-labelled / Codex-reviewed'}
        path = ART / name; save_model(path, net, tokenizer, meta)
        torch.cuda.synchronize(); mlflow.log_metric('training_seconds', time.perf_counter() - started)
        record(name, path, run.info.run_id)

    name = 'minilm_fp32_news_v3'
    with mlflow.start_run(run_name='AI news v3 · Own GPU fine-tune / v1 recipe') as run:
        mlflow.set_tags(tags(name)); mlflow.log_params({**common, 'encoder': ENCODER,
            'encoder_revision': ENCODER_REVISION, 'initialization': 'New frozen baseline head and clean encoder',
            'max_length': 128, 'epochs': 3, 'batch_size': 32, 'encoder_lr': 2e-5, 'head_lr': 2e-4,
            'weight_decay': .01, 'warmup_fraction': .1, 'class_weight': 'Inverse class frequency',
            'training_precision': 'BF16 autocast, FP32 master weights', 'serving_precision': 'fp32',
            'checkpoint_rule': 'Fixed final epoch 3; no Val/Test model choice'})
        started = time.perf_counter(); net.train()
        optimizer = torch.optim.AdamW([{'params': net.encoder.parameters(), 'lr': 2e-5},
            {'params': net.head.parameters(), 'lr': 2e-4}], weight_decay=.01)
        steps = 3 * int(np.ceil(len(df) / 32))
        schedule = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
        weights = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32, device='cuda')
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        for epoch in range(3):
            order = np.random.default_rng(SEED + epoch).permutation(len(df)); total = 0.; seen = 0
            for start in range(0, len(df), 32):
                ix = order[start:start + 32]; optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = criterion(net(batch(ix)).float(), torch.tensor(y[ix], device='cuda'))
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                optimizer.step(); schedule.step(); total += loss.item() * len(ix); seen += len(ix)
            mlflow.log_metric('train_loss', total / seen, step=epoch + 1)
            mlflow.log_metric('train_rows_seen', seen, step=epoch + 1)
            print(f'fine-tune epoch {epoch + 1}/3: full Train {seen} rows, loss {total / seen:.4f}', flush=True)
        net.eval(); meta = {**meta, 'kind': 'finetuned', 'model_version': 'wikinews-ai-v3-minilm-fp32',
                            'run_id': run.info.run_id}
        path = ART / name; save_model(path, net, tokenizer, meta)
        torch.cuda.synchronize(); mlflow.log_metric('training_seconds', time.perf_counter() - started)
        record(name, path, run.info.run_id)
        fine_run_id = run.info.run_id
    del optimizer, schedule; gc.collect(); torch.cuda.empty_cache()

    name = 'minilm_bf16_news_v3'
    with mlflow.start_run(run_name='AI news v3 · v2 BF16 deployment derivation') as run:
        mlflow.set_tags(tags(name)); mlflow.log_params({**common, 'derived_from_run': fine_run_id,
            'operation': 'Cast same newly trained weights to BF16; no second independent fit',
            'serving_precision': 'bf16', 'aggregation': 'uniform', 'max_length': 128})
        started = time.perf_counter(); net.to(dtype=torch.bfloat16)
        meta = {**meta, 'model_version': 'wikinews-ai-v3-minilm-bf16', 'precision': 'bf16',
                'run_id': run.info.run_id, 'derived_from_run': fine_run_id}
        path = ART / name; save_model(path, net, tokenizer, meta)
        torch.cuda.synchronize(); mlflow.log_metric('artifact_conversion_seconds', time.perf_counter() - started)
        record(name, path, run.info.run_id)
    write_json(OUT / 'trained.json', {'protocol_sha256': protocol_hash(), 'models': records})
    print('All new classifier artifacts frozen. No Val or Test predictions made.', flush=True)


def scores(y, pred):
    result = {'accuracy': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0))}
    result.update({'f1_' + label: float(x) for label, x in zip(LABELS,
        f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0))})
    return result


def evaluate(resume=False):
    gpu(); manifest = verified(); trained = read_json(OUT / 'trained.json')
    if protocol_hash() != trained['protocol_sha256']:
        raise RuntimeError('Protocol changed after fitting')
    records = trained['models'] + [{'model': 'qwen3_4b', 'name': NAMES['qwen3_4b'],
        'path': 'models/local_llm', 'run_id': None}]
    for record in records:
        if record.get('artifact_sha256') and fingerprint(ROOT / record['path']) != record['artifact_sha256']:
            raise RuntimeError('Classifier artifact changed')
    lock = OUT / 'evaluation.lock'
    if lock.exists():
        old = read_json(lock)
        if old['status'] == 'complete' or not resume:
            raise RuntimeError('Test evaluation already started/completed; no repeated probing')
        if old['protocol_sha256'] != protocol_hash():
            raise RuntimeError('Protocol changed during interrupted evaluation')
    else:
        write_json(lock, {'status': 'started', 'protocol_sha256': protocol_hash(),
                         'test_sha256': manifest['splits']['test']['sha256']})
    test = pd.read_csv(LOCAL / 'test.csv')
    y = np.array([LABELS.index(x) for x in test.label]); results = []
    completed_path = OUT / 'evaluation_progress.json'
    completed = read_json(completed_path) if completed_path.exists() else []
    for record in records:
        name = record['model']
        if any(x['model'] == name for x in completed):
            results.append(next(x for x in completed if x['model'] == name)); continue
        p = load_artifact(ROOT / record['path'])
        for _ in range(20):
            p.predict_proba(['Die Firma veröffentlicht ihren Bericht.'])
        predictions = []
        checkpoint = LOCAL / f'{name}_test_predictions.jsonl'
        if checkpoint.exists():
            predictions = [json.loads(x) for x in checkpoint.read_text().splitlines()]
        for i, row in enumerate(test.itertuples()):
            if i < len(predictions):
                assert predictions[i]['id'] == row.id
                continue
            torch.cuda.synchronize(); started = time.perf_counter()
            distribution = np.asarray(p.predict_proba([row.text])[0], dtype=float)
            torch.cuda.synchronize(); elapsed = 1000 * (time.perf_counter() - started)
            if distribution.shape != (3,) or not np.isfinite(distribution).all() or not np.isclose(distribution.sum(), 1):
                raise ValueError('Invalid probability scores')
            item = {'id': row.id, 'scores': distribution.tolist(), 'ms': elapsed}
            with checkpoint.open('a') as stream:
                stream.write(json.dumps(item) + '\n')
            predictions.append(item)
        pred = np.array([np.argmax(x['scores']) for x in predictions])
        metrics = scores(y, pred)
        for lang in ['de', 'en']:
            ix = test.language.to_numpy() == lang
            metrics.update({k + '_' + lang: v for k, v in scores(y[ix], pred[ix]).items()})
        metrics['balanced_language_macro_f1'] = (metrics['macro_f1_de'] + metrics['macro_f1_en']) / 2
        metrics.update(inference_ms_per_text=float(np.mean([x['ms'] for x in predictions])),
            latency_p50_ms=float(np.median([x['ms'] for x in predictions])),
            latency_p95_ms=float(np.percentile([x['ms'] for x in predictions], 95)),
            model_size_bytes=sum(f.stat().st_size for f in (ROOT / record['path']).rglob('*') if f.is_file() and '.cache' not in f.parts))
        matrices = {'all': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
        for lang in ['de', 'en']:
            ix = test.language.to_numpy() == lang
            matrices[lang] = confusion_matrix(y[ix], pred[ix], labels=[0, 1, 2]).tolist()
        if record['run_id']:
            context = mlflow.start_run(run_id=record['run_id'])
        else:
            context = mlflow.start_run(run_name='AI news v3 · Unchanged Qwen zero-shot')
        with context as run:
            mlflow.set_tags(tags(name))
            mlflow.log_params({'test_rows': len(test), 'test_sha256': manifest['splits']['test']['sha256'],
                'protocol_sha256': protocol_hash(), 'warmup_calls': 20, 'latency_batch_size': 1,
                'model_version': p.version, 'evaluation_device': torch.cuda.get_device_name(),
                'role': 'unchanged zero-shot comparator' if name == 'qwen3_4b' else 'new AI-data classifier'})
            mlflow.log_metrics(metrics); mlflow.log_dict(matrices, 'confusion_matrix.json')
            mlflow.log_dict({'label_source': 'AI-labelled / Codex-reviewed', 'metrics': metrics}, 'test_metrics.json')
            mlflow.log_artifact(str(OUT / 'protocol.json')); mlflow.log_artifact(str(OUT / 'split_manifest.json'))
            run_id = run.info.run_id
        result = {**record, 'run_id': run_id, 'model_version': p.version, **metrics, 'confusion_matrix': matrices}
        results.append(result); completed.append(result); write_json(completed_path, completed)
        print(name, metrics, flush=True)
        del p; gc.collect(); torch.cuda.empty_cache()
    best = max(x['balanced_language_macro_f1'] for x in results)
    close = [x for x in results if best - x['balanced_language_macro_f1'] <= .005]
    leader = min(close, key=lambda x: (x['inference_ms_per_text'], x['model_size_bytes']))
    report = {'data_version': manifest['data_version'], 'label_status': 'AI-labelled · Codex-reviewed',
        'annotation_method': manifest['label_source'], 'test_rows': len(test),
        'test_counts': manifest['splits']['test']['language_label_counts'],
        'test_sha256': manifest['splits']['test']['sha256'], 'pilot_leader': leader['model'],
        'serving_default_changed': False, 'results': results,
        'interpretation': 'Agreement with single-assistant annotations on a small short-news pilot. Not a human-gold benchmark or evidence of superiority to human annotators.',
        'selection_rule': 'Mean DE/EN macro-F1; within .005 prefer latency then size; descriptive pilot leader only'}
    write_json(OUT / 'comparison.json', report)
    write_json(lock, {'status': 'complete', 'protocol_sha256': protocol_hash(),
                     'test_sha256': report['test_sha256'], 'candidate_count': len(results)})
    print('Pilot completed. Existing serving selection preserved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'train': train()
    else: evaluate(args.resume)
