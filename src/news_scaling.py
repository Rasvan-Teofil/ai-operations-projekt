"""Frozen data-expansion experiment. No API imports this training module."""
import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import ROOT, LABELS, ENCODER, ENCODER_REVISION
from src.inference import SentimentNet, save_model
from src.news_data import digest, normalized
from src.news_experiment import groups, read_json, write_json, scores
from src.serving import load_artifact
from src.tracking import fingerprint, tracking

LOCAL = ROOT / 'data/expanded_news_v4'
OUT = (ROOT / os.getenv('SENTIMENT_SCALING_REPORTS', 'reports/expanded_news_v4')).resolve()
ART = (ROOT / os.getenv('SENTIMENT_SCALING_MODELS', 'models/expanded_news_v4')).resolve()
OLD = ROOT / 'reports/common_data_v3'
SEED, SPLIT_SEED = 20261107, 20261108
NAMES = {'frozen_news_v4': 'Expanded frozen GPU baseline',
    'minilm_fp32_news_v4': 'Expanded own fine-tune · FP32',
    'minilm_bf16_news_v4': 'Same expanded fine-tune · BF16',
    'linearsvc_news_v4': 'Expanded LinearSVC',
    'frozen_news_v3': 'Original 244 · frozen baseline',
    'minilm_fp32_news_v3': 'Original 244 · own fine-tune FP32',
    'linearsvc_news_v3': 'Original 244 · LinearSVC',
    'qwen3_4b': 'Unchanged Qwen3 4B · zero-shot'}


def protocol_hash():
    return digest((OUT / 'protocol.json').read_bytes())


def prepare(restore=False):
    previous = read_json(OUT / 'split_manifest.json') if (OUT / 'split_manifest.json').exists() else None
    if previous and not restore:
        raise RuntimeError('Expansion split already frozen')
    source = read_json(OUT / 'source_manifest.json')
    if digest((LOCAL / 'articles.jsonl').read_bytes()) != source['sha256']:
        raise RuntimeError('Source inputs changed')
    rows = [json.loads(x) for x in (LOCAL / 'articles.jsonl').read_text().splitlines()]
    with (OUT / 'annotations.csv').open() as stream:
        annotations = list(csv.DictReader(stream))
    indexed = {x['id']: x for x in annotations}
    if len(indexed) != len(annotations) or set(indexed) != {x['id'] for x in rows}:
        raise ValueError('Exactly one reviewed annotation required per new input')
    accepted, exclusions = [], []
    for row in rows:
        a = indexed[row['id']]
        if (a['text_sha256'] != row['text_sha256'] or digest(row['text'].encode()) != row['text_sha256']
                or a['label'] not in LABELS + ['exclude'] or not a['reason'].strip()
                or a['review_status'] != 'Codex-reviewed'):
            raise ValueError('Invalid annotation provenance: ' + row['id'])
        if a['label'] == 'exclude':
            exclusions.append({'id': row['id'], 'reason': a['reason']})
        else:
            accepted.append({**row, 'label': a['label'], 'origin': 'new',
                             'annotation_source': 'AI-labelled / Codex-reviewed'})
    old_manifest = read_json(OLD / 'split_manifest.json')
    old_train_ids = {x['id'] for x in old_manifest['membership'] if x['split'] == 'train'}
    old_all = [json.loads(x) for x in (ROOT / 'data/common_data_v3/articles.jsonl').read_text().splitlines()]
    if old_train_ids & {r['id'] for r in rows}:
        raise ValueError('New collection repeats historical training IDs')
    with (OLD / 'annotations.csv').open() as stream:
        old_annotations = {x['id']: x for x in csv.DictReader(stream)}
    carried = [{**r, 'label': old_annotations[r['id']]['label'], 'assigned_role': 'train',
                'origin': 'carried-v3-train', 'annotation_source': 'AI-labelled / Codex-reviewed'}
               for r in old_all if r['id'] in old_train_ids]
    if len(carried) != 244 or len(old_train_ids) != 244:
        raise ValueError('Original 244 Train control changed')
    # All historical inputs participate in the leakage audit, including excluded rows.
    combined = old_all + accepted
    component_ids = groups(combined)
    old_groups = set(component_ids[:len(old_all)])
    for row, group in zip(accepted, component_ids[len(old_all):]):
        if group in old_groups:
            raise ValueError('New input connects to historical input: ' + row['id'])
        row['group'] = group
    old_map = {r['id']: g for r, g in zip(old_all, component_ids)}
    for row in carried:
        row['group'] = old_map[row['id']]
    df = pd.DataFrame(carried + accepted)
    fit = df[df.assigned_role == 'train'].copy()
    test = df[df.assigned_role == 'test'].copy()
    if set(fit['group']) & set(test['group']):
        raise ValueError('Train/Test source component overlap')
    forbidden = {r['id'] for r in old_all} - old_train_ids
    if set(fit.id) & forbidden or set(test.id) & {r['id'] for r in old_all}:
        raise ValueError('Historical Val/Test/exclusions leaked')
    # Existing non-Wikinews project inputs: exact normalized text only. The source
    # group audit above handles related Wikinews pages, not just exact copies.
    known = set()
    scanned = []
    for directory in ['data/processed', 'data/local_llm', 'data/development', 'data/benchmark']:
        for path in (ROOT / directory).glob('*.csv'):
            try:
                table = pd.read_csv(path)
                columns = [c for c in ['text', 'headline', 'sentence'] if c in table]
                for c in columns:
                    known.update(normalized(str(x)) for x in table[c].dropna())
                if columns: scanned.append(str(path.relative_to(ROOT)))
            except (pd.errors.ParserError, UnicodeDecodeError):
                continue
    matches = [r['id'] for r in accepted if normalized(r['text']) in known]
    if matches:
        raise ValueError('New inputs duplicate other project text: ' + str(matches))
    manifest = {'data_version': source['data_version'], 'sampling_seed': SPLIT_SEED,
        'training_seed': SEED, 'label_source': 'Single AI annotator: Codex assistant in this chat',
        'human_gold': False, 'label_status': 'AI-labelled · Codex-reviewed',
        'source_sha256': source['sha256'], 'annotation_sha256': digest((OUT / 'annotations.csv').read_bytes()),
        'old_annotation_sha256': old_manifest['annotation_sha256'], 'old_split_sha256': digest((OLD / 'split_manifest.json').read_bytes()),
        'protocol_sha256': protocol_hash(), 'exclusions': exclusions,
        'historical_protected_rows': len(forbidden), 'splits': {}, 'membership': []}
    for name, part in [('train', fit), ('test', test)]:
        counts = (part.language + ':' + part.label).value_counts().to_dict()
        if any(counts.get(lang + ':' + label, 0) < (5 if name == 'train' else 1)
               for lang in ['de', 'en'] for label in LABELS):
            raise ValueError('Insufficient coverage in fixed roles: ' + str(counts))
        raw = part.to_csv(index=False).encode()
        target = LOCAL / f'{name}.csv'
        if target.exists() and target.read_bytes() != raw:
            raise RuntimeError('Refusing to change existing split')
        target.write_bytes(raw)
        manifest['splits'][name] = {'rows': len(part), 'sha256': digest(raw), 'language_label_counts': counts,
                                  'origin_counts': part.origin.value_counts().to_dict()}
        manifest['membership'].extend({'id': r.id, 'group': r.group, 'split': name,
            'origin': r.origin, 'text_sha256': r.text_sha256} for r in part.itertuples())
    if previous and manifest != previous:
        raise RuntimeError('Restoration differs from frozen manifest')
    if not previous:
        write_json(OUT / 'split_manifest.json', manifest)
    write_json(OUT / 'overlap_audit.json', {'matched_ids': matches, 'scanned_files': scanned,
        'historical_rows_in_group_audit': len(old_all), 'old_train_carried': len(carried),
        'protected_old_rows': len(forbidden), 'train_test_component_overlap': [],
        'limits': 'Exact project text and conservative linked/source/near-duplicate groups; residual event and pretraining overlap unknown.'})
    print(json.dumps(manifest['splits'], indent=2), flush=True)


def verified():
    m = read_json(OUT / 'split_manifest.json')
    for name, value in m['splits'].items():
        if digest((LOCAL / f'{name}.csv').read_bytes()) != value['sha256']:
            raise RuntimeError('Split changed: ' + name)
    for path, expected in [(OUT / 'annotations.csv', m['annotation_sha256']),
                           (OLD / 'annotations.csv', m['old_annotation_sha256']),
                           (OLD / 'split_manifest.json', m['old_split_sha256'])]:
        if digest(path.read_bytes()) != expected:
            raise RuntimeError('Frozen provenance changed: ' + str(path))
    if protocol_hash() != m['protocol_sha256']:
        raise RuntimeError('Protocol changed after split freezing')
    return m


def gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Visible CUDA GPU with BF16 required')
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tracking()


def params(m):
    return {'seed': SEED, 'sampling_seed': SPLIT_SEED, 'protocol_sha256': protocol_hash(),
        'annotation_sha256': m['annotation_sha256'], 'original_annotation_sha256': m['old_annotation_sha256'],
        'train_sha256': m['splits']['train']['sha256'], 'test_sha256': m['splits']['test']['sha256'],
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cpu_threads': 4, 'label_source': 'AI-labelled / Codex-reviewed',
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'val_policy': 'Original Val untouched; final fixed recipe, diagnostic CV only'}


def tags(stage, group):
    return {'phase': 'wikinews-ai-v4-expanded', 'stage': stage, 'comparable_group': group,
        'label_source': 'AI-labelled / Codex-reviewed', 'human_gold': 'false',
        'metric_interpretation': 'Agreement with one AI reference annotator', 'deployment_eligible': 'false'}


def nested_subsets(fit, seed):
    """Sample groups once, yielding nested fits without opening development labels."""
    ordered = np.array(sorted(fit['group'].unique()), dtype=object)
    np.random.default_rng(seed).shuffle(ordered)
    for fraction in [1 / 3, 2 / 3, 1.]:
        chosen = set(ordered[:math.ceil(len(ordered) * fraction)])
        part = fit[fit['group'].isin(chosen)].copy()
        if set(part.label) != set(LABELS):
            raise ValueError('Fixed nested subset lacks a class; do not search a different seed')
        yield fraction, part


def metrics(y, pred, languages):
    result = scores(y, pred)
    for label_ix, label in enumerate(LABELS):
        mask = y == label_ix
        result['recall_' + label] = float(np.mean(pred[mask] == label_ix)) if mask.any() else 0.
    for lang in ['de', 'en']:
        mask = languages == lang
        result.update({k + '_' + lang: v for k, v in scores(y[mask], pred[mask]).items()})
    result['balanced_language_macro_f1'] = (result['macro_f1_de'] + result['macro_f1_en']) / 2
    return result


def size(path):
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file() and '.cache' not in f.parts)


def predict_rows(path, df, checkpoint):
    p = load_artifact(path)
    for _ in range(20):
        p.predict_proba(['Die Firma veröffentlicht ihren Bericht.'])
    predictions = [json.loads(x) for x in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    if len(predictions) > len(df):
        raise RuntimeError('Checkpoint longer than fixed evaluation set')
    for i, row in enumerate(df.itertuples()):
        if i < len(predictions):
            if predictions[i]['id'] != row.id:
                raise RuntimeError('Checkpoint row differs')
            continue
        torch.cuda.synchronize(); start = time.perf_counter()
        probs = np.asarray(p.predict_proba([row.text])[0], dtype=float)
        torch.cuda.synchronize(); elapsed = (time.perf_counter() - start) * 1000
        if probs.shape != (3,) or not np.isfinite(probs).all() or not np.isclose(probs.sum(), 1):
            raise ValueError('Invalid scores')
        value = {'id': row.id, 'scores': probs.tolist(), 'ms': elapsed}
        with checkpoint.open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        predictions.append(value)
    y = np.array([LABELS.index(x) for x in df.label])
    pred = np.array([np.argmax(x['scores']) for x in predictions])
    values = metrics(y, pred, df.language.to_numpy())
    values.update(inference_ms_per_text=float(np.mean([x['ms'] for x in predictions])),
        latency_p50_ms=float(np.median([x['ms'] for x in predictions])),
        latency_p95_ms=float(np.percentile([x['ms'] for x in predictions], 95)), model_size_bytes=size(path))
    matrices = {'all': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
    for lang in ['de', 'en']:
        mask = df.language.to_numpy() == lang
        matrices[lang] = confusion_matrix(y[mask], pred[mask], labels=[0, 1, 2]).tolist()
    version, meta = p.version, p.meta
    del p; gc.collect(); torch.cuda.empty_cache()
    return values, matrices, version, meta


def log_evaluation(path, df, checkpoint):
    values, matrices, version, meta = predict_rows(path, df, checkpoint)
    mlflow.log_metrics(values); mlflow.log_dict(matrices, 'confusion_matrix.json')
    mlflow.log_dict(meta, 'evaluated_artifact_metadata.json')
    mlflow.log_param('model_version', version)
    return {**values, 'confusion_matrix': matrices, 'model_version': version}


def fit_pair(df, destination, common, stage, group, evaluate_df=None, cv_info=None):
    """One frozen head, then exactly one all-layer fine-tune; fresh Hub base per call."""
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=ENCODER_REVISION, local_files_only=True)
    net = SentimentNet(AutoModel.from_pretrained(ENCODER, revision=ENCODER_REVISION, local_files_only=True)).cuda()
    y = np.array([LABELS.index(x) for x in df.label])
    tokens = tokenizer(df.text.tolist(), truncation=True, max_length=128, padding=False)

    def batch(indices):
        return tokenizer.pad([{k: v[int(i)] for k, v in tokens.items()} for i in indices],
                             padding=True, return_tensors='pt').to('cuda')

    records = []
    prefix = 'Expanded final' if stage == 'final-test' else f'CV fold{cv_info["fold"]} size{cv_info["size"]}'
    for kind, name in [('frozen', 'frozen_news_v4'), ('finetuned', 'minilm_fp32_news_v4')]:
        path = destination / name
        if path.exists():
            raise RuntimeError('Refusing to overwrite trained artifact: ' + str(path))
        with mlflow.start_run(run_name=prefix + ' · ' + kind) as run:
            mlflow.set_tags(tags(stage, group))
            mlflow.log_params({**common, **(cv_info or {}), 'train_rows': len(df),
                'encoder': ENCODER, 'encoder_revision': ENCODER_REVISION,
                'initialization': 'Clean original Hub encoder + new balanced frozen head' if kind == 'finetuned' else 'Clean original Hub encoder',
                'max_length': 128, 'head_C': 1., 'head_class_weight': 'balanced', 'encoder_frozen': kind == 'frozen'})
            started = time.perf_counter()
            if kind == 'frozen':
                net.eval(); vectors = []
                with torch.inference_mode():
                    for start in range(0, len(df), 64):
                        vectors.append(net.features(batch(range(start, min(start + 64, len(df))))).cpu().numpy())
                head = LogisticRegression(C=1., class_weight='balanced', max_iter=2000, random_state=SEED)
                head.fit(np.concatenate(vectors), y)
                with torch.no_grad():
                    net.head.weight.copy_(torch.tensor(head.coef_, dtype=torch.float32, device='cuda'))
                    net.head.bias.copy_(torch.tensor(head.intercept_, dtype=torch.float32, device='cuda'))
                mlflow.log_params({'head_max_iter': 2000, 'feature_batch_size': 64})
            else:
                mlflow.log_params({'epochs': 3, 'batch_size': 32, 'encoder_lr': 2e-5, 'head_lr': 2e-4,
                    'weight_decay': .01, 'warmup_fraction': .1, 'class_weight': 'Inverse class frequency',
                    'training_precision': 'BF16 autocast, FP32 master', 'serving_precision': 'fp32',
                    'checkpoint_rule': 'Fixed final epoch3; no Val/Test selection'})
                net.train()
                optimizer = torch.optim.AdamW([{'params': net.encoder.parameters(), 'lr': 2e-5},
                    {'params': net.head.parameters(), 'lr': 2e-4}], weight_decay=.01)
                steps = 3 * math.ceil(len(df) / 32)
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
                    print(prefix, f'epoch{epoch + 1}/3 full Train={seen} loss={total / seen:.4f}', flush=True)
                del optimizer, schedule
            net.eval()
            version = f'wikinews-ai-v4-{kind}-fp32' + (f'-cv{cv_info["fold"]}-{cv_info["size"]}' if cv_info else '')
            meta = {'kind': kind, 'model_version': version, 'run_id': run.info.run_id, 'precision': 'fp32',
                'aggregation': 'uniform', 'max_length': 128, 'improved_article': True,
                'data_version': 'wikinews-ai-v4-expanded', 'label_source': 'AI-labelled / Codex-reviewed'}
            save_model(path, net, tokenizer, meta)
            torch.cuda.synchronize(); mlflow.log_metric('training_seconds', time.perf_counter() - started)
            sha = fingerprint(path); mlflow.log_param('artifact_sha256', sha)
            mlflow.log_artifact(str(path / 'artifact.json'), 'model_metadata')
            record = {'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                      'run_id': run.info.run_id, 'artifact_sha256': sha, **(cv_info or {})}
            if evaluate_df is not None:
                record.update(log_evaluation(path, evaluate_df, destination / (kind + '_dev_predictions.jsonl')))
            records.append(record)
    if stage == 'final-test':
        name = 'minilm_bf16_news_v4'; path = destination / name
        with mlflow.start_run(run_name='Expanded final · BF16 deployment derivation') as run:
            mlflow.set_tags(tags(stage, group))
            mlflow.log_params({**common, 'train_rows': len(df), 'derived_from_run': records[-1]['run_id'],
                'operation': 'Cast same expanded-fit weights, no independent second fit', 'serving_precision': 'bf16'})
            started = time.perf_counter(); net.to(dtype=torch.bfloat16)
            meta.update(precision='bf16', model_version='wikinews-ai-v4-finetuned-bf16',
                        derived_from_run=records[-1]['run_id'], run_id=run.info.run_id)
            save_model(path, net, tokenizer, meta)
            torch.cuda.synchronize(); mlflow.log_metric('artifact_conversion_seconds', time.perf_counter() - started)
            sha = fingerprint(path); mlflow.log_param('artifact_sha256', sha)
            mlflow.log_artifact(str(path / 'artifact.json'), 'model_metadata')
            records.append({'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                            'run_id': run.info.run_id, 'artifact_sha256': sha})
    del net, tokenizer; gc.collect(); torch.cuda.empty_cache()
    return records


def cv():
    gpu(); m = verified()
    if (OUT / 'cv.json').exists() or (ART / 'cv').exists():
        raise RuntimeError('CV already started/completed; preserve evidence')
    train = pd.read_csv(LOCAL / 'train.csv')
    folds = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SPLIT_SEED)
    records, membership = [], []
    # Persist all fold and subset memberships before fitting any candidate.
    jobs = []
    for fold, (fit_ix, dev_ix) in enumerate(folds.split(train, train.language + ':' + train.label, train['group']), 1):
        fit, dev = train.iloc[fit_ix], train.iloc[dev_ix]
        for step, (fraction, part) in enumerate(nested_subsets(fit, SPLIT_SEED + fold), 1):
            if set(part['group']) & set(dev['group']):
                raise ValueError('CV group leakage')
            info = {'fold': fold, 'size': step, 'fraction': fraction, 'fit_rows': len(part), 'dev_rows': len(dev),
                    'fit_ids_sha256': digest('\n'.join(part.id).encode()), 'dev_ids_sha256': digest('\n'.join(dev.id).encode())}
            membership.append({**info, 'fit_ids': part.id.tolist(), 'dev_ids': dev.id.tolist()})
            jobs.append((info, part, dev))
    write_json(OUT / 'cv_membership.json', membership)
    for info, fit, dev in jobs:
        path = ART / 'cv' / f'fold{info["fold"]}-size{info["size"]}'
        path.mkdir(parents=True)
        records.extend(fit_pair(fit, path, params(m), 'cv', f'wikinews-v4-cv-fold{info["fold"]}', dev, info))
        write_json(OUT / 'cv_progress.json', records)
    write_json(OUT / 'cv.json', {'protocol_sha256': protocol_hash(), 'train_only': True,
        'selection': 'Diagnostic learning curve; fixed final recipe unchanged', 'runs': records})
    print('18 training-only CV runs complete; fresh Test unopened.', flush=True)


def train():
    gpu(); m = verified()
    if not (OUT / 'cv.json').exists():
        raise RuntimeError('Run the frozen diagnostic CV first')
    if (OUT / 'trained.json').exists() or (OUT / 'evaluation.lock').exists() or (ART / 'final').exists():
        raise RuntimeError('Final training already started/completed')
    destination = ART / 'final'; destination.mkdir(parents=True)
    df = pd.read_csv(LOCAL / 'train.csv')  # No Test or original Val read here.
    common = params(m)
    records = fit_pair(df, destination, common, 'final-test', 'wikinews-v4-fresh-test')
    name = 'linearsvc_news_v4'; path = destination / name; path.mkdir()
    with mlflow.start_run(run_name='Expanded final · LinearSVC') as run:
        mlflow.set_tags(tags('final-test', 'wikinews-v4-fresh-test'))
        mlflow.log_params({**common, 'train_rows': len(df), 'C': 1., 'ngram_range': '(1,2)',
            'min_df': 1, 'class_weight': 'balanced', 'max_iter': 2000, 'backend': 'sklearn CPU on GPU tower'})
        started = time.perf_counter()
        estimator = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LinearSVC(C=1., class_weight='balanced', max_iter=2000, random_state=SEED))
        estimator.fit(df.text.tolist(), df.label.tolist())
        import joblib, sklearn
        version = 'wikinews-ai-v4-linearsvc'
        joblib.dump({'estimator': estimator, 'model_version': version}, path / 'tfidf_linearsvc.joblib')
        write_json(path / 'artifact.json', {'kind': 'reference_sklearn', 'model_version': version,
            'labels': LABELS, 'sklearn': sklearn.__version__, 'improved_article': True,
            'data_version': m['data_version'], 'label_source': 'AI-labelled / Codex-reviewed'})
        mlflow.log_metric('training_seconds', time.perf_counter() - started)
        sha = fingerprint(path); mlflow.log_param('artifact_sha256', sha)
        records.append({'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                        'run_id': run.info.run_id, 'artifact_sha256': sha})
    # Pin unchanged controls before opening the fresh Test, with new evaluation runs.
    old = read_json(OLD / 'trained.json')['models']
    for name in ['frozen_news_v3', 'minilm_fp32_news_v3', 'linearsvc_news_v3']:
        item = next(r for r in old if r['model'] == name)
        if fingerprint(ROOT / item['path']) != item['artifact_sha256']:
            raise RuntimeError('Original control artifact changed')
        records.append({**item, 'name': NAMES[name], 'original_run_id': item['run_id'], 'run_id': None})
    records.append({'model': 'qwen3_4b', 'name': NAMES['qwen3_4b'], 'path': 'models/local_llm',
                    'run_id': None, 'artifact_sha256': fingerprint(ROOT / 'models/local_llm')})
    write_json(OUT / 'trained.json', {'protocol_sha256': protocol_hash(), 'models': records})
    print('All eight final conditions pinned; no fresh Test predictions made.', flush=True)


def evaluate(resume=False):
    gpu(); m = verified(); trained = read_json(OUT / 'trained.json')
    if trained['protocol_sha256'] != protocol_hash():
        raise RuntimeError('Training protocol changed')
    for r in trained['models']:
        if fingerprint(ROOT / r['path']) != r['artifact_sha256']:
            raise RuntimeError('Frozen artifact changed: ' + r['model'])
    lock = OUT / 'evaluation.lock'
    if lock.exists():
        old = read_json(lock)
        if old['status'] == 'complete' or not resume or old['protocol_sha256'] != protocol_hash():
            raise RuntimeError('Fresh Test already started/completed; no repeated probing')
    else:
        write_json(lock, {'status': 'started', 'protocol_sha256': protocol_hash(),
                         'test_sha256': m['splits']['test']['sha256']})
    test = pd.read_csv(LOCAL / 'test.csv')
    progress = OUT / 'evaluation_progress.json'
    results = read_json(progress) if progress.exists() else []
    for record in trained['models']:
        if any(r['model'] == record['model'] for r in results):
            continue
        context = mlflow.start_run(run_id=record['run_id']) if record['run_id'] else mlflow.start_run(run_name='Fresh v4 test · ' + record['name'])
        with context as run:
            mlflow.set_tags(tags('final-test', 'wikinews-v4-fresh-test'))
            mlflow.log_params({'test_rows': len(test), 'test_sha256': m['splits']['test']['sha256'],
                'protocol_sha256': protocol_hash(), 'warmup_calls': 20, 'latency_batch_size': 1,
                'evaluation_device': torch.cuda.get_device_name(), 'evaluation_cpu_threads': 4,
                'evaluation_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'annotation_source_sha256': m['annotation_sha256'], 'artifact_sha256': record['artifact_sha256'],
                'evaluation_seed': SEED, 'original_run_id': record.get('original_run_id', 'new expanded fit' if record['model'] != 'qwen3_4b' else 'unchanged zero-shot')})
            measured = log_evaluation(ROOT / record['path'], test, LOCAL / (record['model'] + '_test_predictions.jsonl'))
            mlflow.log_artifact(str(OUT / 'protocol.json')); mlflow.log_artifact(str(OUT / 'split_manifest.json'))
            run_id = run.info.run_id
        value = {**record, **measured, 'run_id': run_id}
        results.append(value); write_json(progress, results)
        print(record['model'], measured, flush=True)
    best = max(r['balanced_language_macro_f1'] for r in results)
    leader = min((r for r in results if best - r['balanced_language_macro_f1'] <= .005),
                 key=lambda r: (r['inference_ms_per_text'], r['model_size_bytes']))
    report = {'data_version': m['data_version'], 'label_status': m['label_status'], 'human_gold': False,
        'test_rows': len(test), 'test_counts': m['splits']['test']['language_label_counts'],
        'test_sha256': m['splits']['test']['sha256'], 'descriptive_leader': leader['model'],
        'serving_default_changed': False, 'results': results,
        'interpretation': 'Same fresh random Test for original and expanded artifacts. AI annotation agreement; quantity and enrichment both changed. Not comparable numerically to old Test scores.'}
    write_json(OUT / 'comparison.json', report)
    write_json(lock, {'status': 'complete', 'protocol_sha256': protocol_hash(),
        'test_sha256': m['splits']['test']['sha256'], 'candidate_count': len(results)})
    print('Fresh Test complete exactly once per candidate; default preserved.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'cv', 'train', 'evaluate'])
    parser.add_argument('--restore', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.restore)
    elif args.action == 'cv': cv()
    elif args.action == 'train': train()
    else: evaluate(args.resume)
