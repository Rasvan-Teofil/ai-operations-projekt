"""Fresh, locked common benchmark. No model/prompt search on evaluation data."""
import argparse
import ast
import gc
import hashlib
import io
import json
import platform
import shutil
import time
import urllib.request
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import ROOT, MODELS, REPORTS, DATA, LABELS
from src.local_llm import file_manifest
from src.prepare import normalized
from src.serving import load_artifact
from src.tracking import tracking

OUT = REPORTS / 'local_llm'
LOCAL = ROOT / 'data' / 'local_llm'
SEED = 20261106
PATHS = {'minilm_v2': MODELS / 'development/release',
         'linearsvc_v2': MODELS / 'local_llm_reference', 'qwen3_4b': MODELS / 'local_llm'}
DISPLAY = {'minilm_v2': 'MiniLM v2 (fine-tuned)', 'linearsvc_v2': 'LinearSVC v2 (reconstructed)',
           'qwen3_4b': 'Qwen3 4B (local, zero-shot)'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 required')
    torch.set_num_threads(4)
    torch.manual_seed(SEED)


def artifact_info(path):
    files = file_manifest(path)
    return {'sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'files': files, 'size_bytes': sum((path / f).stat().st_size for f in files)}


def prepare(reference):
    require_gpu()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if (OUT / 'benchmark_manifest.json').exists():
        raise RuntimeError('Benchmark already frozen; use the existing data or an explicitly separate reproduction directory')
    from src.branch_benchmark import URL, REV, TRANSLATOR, TRANS_REV
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    raw = urllib.request.urlopen(URL, timeout=30).read()
    old = json.loads((REPORTS / 'development/benchmark_manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest() != old['source_sha256']:
        raise RuntimeError('Pinned SEntFiN source content changed')
    previous = json.loads((OUT / 'historical_exclusions.json').read_text())
    if previous['source_benchmark_sha256'] != old['sha256']:
        raise RuntimeError('Historical exclusions differ from the original benchmark')
    excluded_ids = set(previous['paired_ids'])
    excluded_hashes = set(previous['normalized_text_sha256'])
    excluded_texts = set()
    excluded_sources = {}
    split_paths = [*DATA.glob('*.csv'), *(reference / 'data/processed').glob('*.csv'),
                   reference / 'data/sample/opinion_supplement.csv']
    for p in split_paths:
        frame = pd.read_csv(p)
        if 'text' in frame:
            excluded_texts.update(frame.text.map(normalized))
            excluded_sources[str(p.relative_to(ROOT.parent))] = digest(p)
    rows = []
    for _, row in pd.read_csv(io.BytesIO(raw)).iterrows():
        try:
            decisions = ast.literal_eval(row['Decisions'])
        except (SyntaxError, ValueError):
            continue
        if not isinstance(decisions, dict) or not decisions:
            continue
        labels = {str(v).lower().strip() for v in decisions.values()}
        text = str(row['Title']).strip()
        pair = 'sentfin-' + str(row['S No.'])
        if (len(labels) == 1 and labels.issubset(LABELS) and 20 <= len(text) <= 20000
                and pair not in excluded_ids and normalized(text) not in excluded_texts
                and hashlib.sha256(normalized(text).encode()).hexdigest() not in excluded_hashes):
            rows.append({'paired_id': pair, 'text': text, 'label': next(iter(labels)),
                         'normalized': normalized(text)})
    eligible = pd.DataFrame(rows).drop_duplicates('normalized')
    selected = (eligible.groupby('label', group_keys=False).sample(n=200, random_state=SEED)
                .sort_values('paired_id').reset_index(drop=True))
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR, revision=TRANS_REV)
    translator = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR, revision=TRANS_REV).cuda().eval()
    translations = []
    with torch.inference_mode():
        for start in range(0, len(selected), 24):
            texts = selected.text.iloc[start:start + 24].tolist()
            if any(len(tokenizer.encode(t)) > 256 for t in texts):
                raise RuntimeError('Unexpected source truncation; do not silently alter the protocol')
            tokens = tokenizer(texts, padding=True, return_tensors='pt').to('cuda')
            output = translator.generate(**tokens, num_beams=4, max_new_tokens=128, do_sample=False)
            if any(int(ids[-1]) != tokenizer.pad_token_id and int(ids[-1]) != tokenizer.eos_token_id
                   for ids in output):
                raise RuntimeError('Translation reached the generation cap')
            translations.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
            print('Translated', len(translations), '/', len(selected), flush=True)
    en = selected[['paired_id', 'text', 'label']].copy()
    en['language'] = 'en'
    de = en.copy()
    de['language'] = 'de'
    de['text'] = translations
    if not all(t.strip() for t in translations):
        raise RuntimeError('Empty translation')
    frame = pd.concat([en, de], ignore_index=True)
    frame['id'] = frame.paired_id + ':' + frame.language
    frame.to_csv(LOCAL / 'shared.csv', index=False)
    artifacts = {name: artifact_info(path) for name, path in PATHS.items()}
    manifest = {'protocol_sha256': digest(OUT / 'protocol.json'), 'seed': SEED,
                'dataset': 'pyRis/SEntFiN', 'revision': REV, 'source_url': URL,
                'source_sha256': hashlib.sha256(raw).hexdigest(), 'license': old['license'],
                'selection': '200 per class, fixed seed, disjoint from old benchmark and normalized project data',
                'eligible_counts': eligible.label.value_counts().to_dict(),
                'native_en_rows': len(en), 'de_mt_rows': len(de), 'independent_pairs': len(en),
                'selected_ids': en.paired_id.tolist(), 'excluded_previous_pairs': len(excluded_ids),
                'excluded_data_hashes': excluded_sources, 'sha256': digest(LOCAL / 'shared.csv'),
                'translator': TRANSLATOR, 'translator_revision': TRANS_REV,
                'translator_license': 'CC-BY-4.0', 'translation': 'CUDA fp32 beam4 max_new_tokens128; no source truncations',
                'artifacts': artifacts,
                'inference_source_hashes': {f: digest(ROOT / f) for f in
                                           ['src/local_llm.py', 'src/serving.py', 'src/inference.py']},
                'limitations': json.loads((OUT / 'protocol.json').read_text())['limitations']}
    (OUT / 'benchmark_manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(MODELS / 'local_llm_reference/provenance.json', OUT / 'reference_provenance.json')
    print('Fresh common benchmark frozen; no predictions made.', flush=True)


def scores(truth, probabilities):
    pred = probabilities.argmax(-1)
    return {'accuracy': float(accuracy_score(truth, pred)),
            'macro_f1': float(f1_score(truth, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            **{f'f1_{label}': float(v) for label, v in zip(LABELS, f1_score(
                truth, pred, labels=[0, 1, 2], average=None, zero_division=0))},
            'negative_recall': float(recall_score(truth == 0, pred == 0, zero_division=0)),
            'negative_precision': float(precision_score(truth == 0, pred == 0, zero_division=0))}


def select(records):
    best = max(r['macro_f1_en'] for r in records)
    close = [r for r in records if best - r['macro_f1_en'] <= .005]
    return min(close, key=lambda r: (r['inference_ms_per_text'], r['model_size_bytes']))


def audit():
    require_gpu()
    manifest = json.loads((OUT / 'benchmark_manifest.json').read_text())
    if digest(LOCAL / 'shared.csv') != manifest['sha256'] or digest(OUT / 'protocol.json') != manifest['protocol_sha256']:
        raise RuntimeError('Frozen data or protocol changed')
    for source, expected in manifest['inference_source_hashes'].items():
        if digest(ROOT / source) != expected:
            raise RuntimeError('Frozen inference source changed: ' + source)
    for name, path in PATHS.items():
        if artifact_info(path) != manifest['artifacts'][name]:
            raise RuntimeError('Frozen artifact changed: ' + name)
    with (LOCAL / 'evaluation.lock').open('x') as stream:
        json.dump({'status': 'started', 'benchmark_sha256': manifest['sha256']}, stream)
    tracking()
    frame = pd.read_csv(LOCAL / 'shared.csv')
    truth = np.array([LABELS.index(v) for v in frame.label])
    records = []
    for name, path in PATHS.items():
        with mlflow.start_run(run_name='local-llm-common-' + name,
                             tags={'phase': 'local-llm-comparison', 'candidate': name,
                                   'comparable_group': 'fresh-sentfin-20261106',
                                   'source': 'measured-live', 'selection_metric': 'macro_f1_en'}) as run:
            mlflow.log_params({'benchmark_sha256': manifest['sha256'], 'seed': SEED,
                'model_version': json.loads((path / 'artifact.json').read_text())['model_version'],
                'artifact_sha256': manifest['artifacts'][name]['sha256'], 'batch_size': 1,
                'warmup_texts': 10, 'native_en_rows': 600, 'de_mt_rows': 600,
                'reference_commit': json.loads((OUT / 'protocol.json').read_text())['reference_commit'],
                'backend': 'CPU' if name == 'linearsvc_v2' else 'CUDA BF16',
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'platform': platform.platform(), 'protocol_sha256': manifest['protocol_sha256'],
                'training': 'reconstructed balanced LinearSVC with supplement' if name == 'linearsvc_v2'
                            else 'zero-shot; no fine-tuning' if name == 'qwen3_4b' else 'unchanged R3 fine-tuned release'})
            mlflow.log_dict(json.loads((path / 'artifact.json').read_text()), 'artifact_metadata.json')
            if name == 'linearsvc_v2':
                mlflow.log_dict(json.loads((OUT / 'reference_provenance.json').read_text()), 'reference_provenance.json')
            start = time.perf_counter()
            predictor = load_artifact(path)
            load_ms = 1000 * (time.perf_counter() - start)
            for _ in range(10):
                predictor.predict_proba(['The company announces its quarterly reporting date.'])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            probabilities, durations = [], []
            for index, text in enumerate(frame.text):
                torch.cuda.synchronize()
                start = time.perf_counter()
                probabilities.append(predictor.predict_proba([text])[0])
                torch.cuda.synchronize()
                durations.append(1000 * (time.perf_counter() - start))
                if (index + 1) % 200 == 0:
                    print(name, index + 1, '/', len(frame), flush=True)
            probabilities = np.stack(probabilities)
            if (probabilities.shape != (len(frame), 3) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1, atol=1e-4)):
                raise RuntimeError('Invalid probability matrix')
            np.savez(LOCAL / (name + '.npz'), probs=probabilities, durations=durations, ids=frame.id.to_numpy(dtype=str))
            metrics = scores(truth, probabilities)
            metrics.update(inference_ms_per_text=float(np.mean(durations)), latency_p50_ms=float(np.median(durations)),
                           latency_p95_ms=float(np.percentile(durations, 95)), load_ms=load_ms,
                           model_size_bytes=manifest['artifacts'][name]['size_bytes'],
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if name != 'linearsvc_v2' else 0)
            for language in ['en', 'de']:
                ix = frame.language.to_numpy() == language
                metrics.update({key + '_' + language: value for key, value in scores(truth[ix], probabilities[ix]).items()})
            for language in ['all', 'en', 'de']:
                ix = np.ones(len(frame), dtype=bool) if language == 'all' else frame.language.to_numpy() == language
                cm = confusion_matrix(truth[ix], probabilities[ix].argmax(-1), labels=[0, 1, 2])
                target = OUT / f'{name}_confusion_matrix_{language}.csv'
                pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(target)
                mlflow.log_artifact(str(target), 'confusion_matrix')
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(OUT / 'protocol.json'))
            mlflow.log_artifact(str(OUT / 'benchmark_manifest.json'))
            record = {'model': name, 'model_version': predictor.version, 'run_id': run.info.run_id, **metrics}
            (OUT / (name + '_metrics.json')).write_text(json.dumps(record, indent=2))
            records.append(record)
            print(name, 'EN macro-F1', metrics['macro_f1_en'], 'DE-MT', metrics['macro_f1_de'], flush=True)
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
    selected = select(records)
    result = {'benchmark_sha256': manifest['sha256'], 'primary_metric': 'macro_f1_en',
              'selected': selected['model'], 'selection_margin': .005, 'results': records}
    en = np.flatnonzero(frame.language.to_numpy() == 'en')
    selected_probs = np.load(LOCAL / (selected['model'] + '.npz'))['probs']
    intervals = {}
    for record in records:
        if record['model'] == selected['model']:
            continue
        other = np.load(LOCAL / (record['model'] + '.npz'))['probs']
        rng = np.random.default_rng(SEED)
        delta = []
        for _ in range(1000):
            sample = rng.choice(en, len(en), replace=True)
            delta.append(scores(truth[sample], selected_probs[sample])['macro_f1'] - scores(truth[sample], other[sample])['macro_f1'])
        intervals[record['model']] = list(map(float, np.percentile(delta, [2.5, 97.5])))
    result['selected_minus_other_native_en_paired_bootstrap_95_percentile'] = intervals
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(records).to_csv(OUT / 'comparison.csv', index=False)
    (LOCAL / 'evaluation.lock').write_text(json.dumps({'status': 'complete', 'selected': selected['model'],
                                                     'benchmark_sha256': manifest['sha256']}))
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads((OUT / 'comparison.json').read_text())
    records = result['results']
    colors = ['#2563eb', '#64748b', '#0d9488']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    x = np.arange(len(records))
    for offset, language, color in [(-.18, 'en', '#2563eb'), (.18, 'de', '#f59e0b')]:
        bars = axes[0].bar(x + offset, [r['macro_f1_' + language] for r in records], .34,
                           label='Native EN gold' if language == 'en' else 'DE translation (diagnostic)', color=color)
        axes[0].bar_label(bars, fmt='%.3f', fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Macro-F1')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[0].set_title('Same fresh paired headlines')
    for ax, key, title, ylabel, divisor in [(axes[1], 'inference_ms_per_text', 'Warm batch-one inference', 'Milliseconds / text (log scale)', 1),
                                           (axes[2], 'model_size_bytes', 'Local serving artifact', 'MiB (log scale)', 2**20)]:
        bars = ax.bar(x, [r[key] / divisor for r in records], color=colors)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
        ax.set_yscale('log')
        ax.set_ylim(top=max(r[key] / divisor for r in records) * 3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    for ax in axes:
        ax.set_xticks(x, ['MiniLM v2', 'LinearSVC v2', 'Qwen3 4B'], rotation=15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Local LLM comparison — 600 native EN headlines + 600 DE translations\nFresh 2026-11-06 benchmark; different from historical tests', fontsize=13)
    for suffix in ['svg', 'png']:
        fig.savefig(OUT / ('comparison.' + suffix), dpi=180)
    plt.close(fig)
    lines = ['# Local LLM comparison (measured)', '',
             'Primary selection: native English macro-F1; close margin 0.005, then latency and artifact size.', '',
             '| Model | EN macro-F1 | DE-MT macro-F1 | Pooled accuracy | ms/text | MiB | MLflow run |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in records:
        lines.append(f"| {DISPLAY[r['model']]} | {r['macro_f1_en']:.4f} | {r['macro_f1_de']:.4f} | {r['accuracy']:.4f} | {r['inference_ms_per_text']:.3f} | {r['model_size_bytes']/2**20:.2f} | `{r['run_id']}` |")
    lines += ['', 'Selected: **' + DISPLAY[result['selected']] + '**.', '',
              'The German texts are machine translations with inherited labels, not native German gold. These are 600 independent paired headlines, not 1,200 independent articles. No prompt/model settings were changed using this benchmark. Public pretraining contamination and semantic overlap remain unknown.', '',
              'The LinearSVC is reconstructed from the pinned current-main v2 recipe, not the original supplied weights. See `reference_provenance.json` for source, dataset, supplement and environment hashes. Its old financial-test score 0.7330 and our earlier tests are not ranked against this benchmark.', '',
              '![Measured accuracy, latency and size](comparison.svg)', '']
    (OUT / 'comparison.md').write_text('\n'.join(lines))


def activate():
    result = json.loads((OUT / 'comparison.json').read_text())
    lock = json.loads((LOCAL / 'evaluation.lock').read_text())
    if lock['status'] != 'complete' or lock['benchmark_sha256'] != result['benchmark_sha256']:
        raise RuntimeError('Completed evaluation required')
    chosen = select(result['results'])
    manifest = json.loads((OUT / 'benchmark_manifest.json').read_text())
    if artifact_info(PATHS[chosen['model']]) != manifest['artifacts'][chosen['model']]:
        raise RuntimeError('Serving artifact differs from the evaluated artifact')
    selection = {'model': str(PATHS[chosen['model']].relative_to(MODELS)), 'candidate': chosen['model'],
                 'model_version': chosen['model_version'], 'benchmark_sha256': result['benchmark_sha256'],
                 'run_id': chosen['run_id'], 'selection_metric': 'macro_f1_en'}
    (MODELS / 'serving_selected.json').write_text(json.dumps(selection, indent=2))
    (OUT / 'deployment.json').write_text(json.dumps(selection, indent=2))
    tracking()
    client = mlflow.MlflowClient()
    client.set_tag(chosen['run_id'], 'deployment', 'selected-local-website')
    print('Activated', chosen['model'], '; historical models/selected.json preserved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'audit', 'report', 'activate'])
    parser.add_argument('--reference-checkout', type=Path, default=ROOT.parent / 'colleague-v2-reference')
    parser.add_argument('--reports-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    args = parser.parse_args()
    if args.reports_dir:
        target = args.reports_dir.resolve()
        target.mkdir(parents=True, exist_ok=True)
        if target != OUT:
            for filename in ['protocol.json', 'historical_exclusions.json']:
                if not (target / filename).exists():
                    shutil.copyfile(OUT / filename, target / filename)
        OUT = target
    if args.data_dir:
        LOCAL = args.data_dir.resolve()
    if args.action == 'prepare':
        prepare(args.reference_checkout.resolve())
    else:
        globals()[args.action]()
