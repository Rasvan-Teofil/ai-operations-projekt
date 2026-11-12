"""Matched-task LLM adaptation. Separate fits and one locked fresh-Test audit."""
import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import ROOT, LABELS
from src.local_llm import MODEL, REVISION
from src.news_data import digest, normalized
from src.news_experiment import groups, read_json, write_json
from src.news_llm_data import LOCAL, historical
from src.news_llm_profiles import PROMPT, examples, demonstrations
from src.news_scaling import predict_rows, size
from src.tracking import fingerprint, tracking

OUT = (ROOT / os.getenv('SENTIMENT_LLM_ADAPT_REPORTS', 'reports/llm_adaptation_v5')).resolve()
ART = (ROOT / os.getenv('SENTIMENT_LLM_ADAPT_MODELS', 'models/llm_adaptation_v5')).resolve()
BASE = ROOT / 'models/local_llm'
SEED = 20261109
QWEN = ['qwen3_4b', 'qwen_news_v5', 'qwen_fewshot_v5', 'qwen_lora_v5']
NAMES = {'qwen3_4b': 'Qwen · original financial prompt', 'qwen_news_v5': 'Qwen · news rubric zero-shot',
    'qwen_fewshot_v5': 'Qwen · news rubric + six examples', 'qwen_lora_v5': 'Qwen · own LoRA on same Train',
    'frozen_news_v4': 'Frozen MiniLM · 692 Train', 'minilm_bf16_news_v4': 'MiniLM BF16 · 692 Train',
    'linearsvc_news_v4': 'LinearSVC · 692 Train'}


def protocol_hash():
    return digest((OUT / 'protocol.json').read_bytes())


def prepare(restore=False):
    previous = read_json(OUT / 'split_manifest.json') if (OUT / 'split_manifest.json').exists() else None
    if previous and not restore: raise RuntimeError('Matched-task inputs already frozen')
    protocol = read_json(OUT / 'protocol.json')
    if protocol['prompt_sha256'] != digest(PROMPT.encode()): raise RuntimeError('Prompt differs from frozen rubric')
    source = read_json(OUT / 'source_manifest.json'); raw = (LOCAL / 'articles.jsonl').read_bytes()
    if digest(raw) != source['sha256']: raise RuntimeError('New source bytes changed')
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    with (OUT / 'annotations.csv').open() as stream: ledger = list(csv.DictReader(stream))
    indexed = {a['id']: a for a in ledger}
    if len(indexed) != len(ledger) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Exactly one annotation required per fresh input')
    accepted, exclusions = [], []
    for r in rows:
        a = indexed[r['id']]
        if (a['text_sha256'] != r['text_sha256'] or digest(r['text'].encode()) != r['text_sha256']
                or a['label'] not in LABELS + ['exclude'] or not a['reason'].strip()
                or a['review_status'] != 'Codex-reviewed'):
            raise ValueError('Invalid annotation provenance')
        if a['label'] == 'exclude': exclusions.append({'id': r['id'], 'reason': a['reason']})
        else: accepted.append({**r, 'label': a['label'], 'annotation_source': 'AI-labelled / Codex-reviewed'})
    old = historical(); all_groups = groups(old + accepted); protected = set(all_groups[:len(old)])
    for r, group in zip(accepted, all_groups[len(old):]):
        if group in protected: raise ValueError('Fresh Test connects to protected historical input')
        r['group'] = group
    known, scanned = set(), []
    for directory in ['data/processed', 'data/benchmark', 'data/local_llm']:
        for path in (ROOT / directory).glob('*.csv'):
            table = pd.read_csv(path)
            for col in ['text', 'headline', 'sentence']:
                if col in table: known.update(normalized(str(x)) for x in table[col].dropna())
            scanned.append(str(path.relative_to(ROOT)))
    matches = [r['id'] for r in accepted if normalized(r['text']) in known or normalized(r['text'].split('\n')[0]) in known]
    if matches: raise ValueError('Known project/benchmark overlap')
    train_bytes = (ROOT / 'data/expanded_news_v4/train.csv').read_bytes()
    old_manifest = read_json(ROOT / 'reports/expanded_news_v4/split_manifest.json')
    if digest(train_bytes) != old_manifest['splits']['train']['sha256']: raise RuntimeError('Existing692 Train changed')
    train = pd.read_csv(ROOT / 'data/expanded_news_v4/train.csv'); test = pd.DataFrame(accepted)
    if set(train.id) & set(test.id) or set(train['group']) & set(test['group']): raise ValueError('Train/Test overlap')
    test_bytes = test.to_csv(index=False).encode()
    manifest = {'data_version': source['data_version'], 'seed': SEED, 'label_status': 'AI-labelled · Codex-reviewed',
        'human_gold': False, 'annotation_sha256': digest((OUT / 'annotations.csv').read_bytes()),
        'protocol_sha256': protocol_hash(), 'source_sha256': source['sha256'],
        'original_train_annotation_sha256': old_manifest['annotation_sha256'],
        'original_train_split_sha256': digest((ROOT / 'reports/expanded_news_v4/split_manifest.json').read_bytes()),
        'exclusions': exclusions, 'historical_protected_inputs': len(old), 'splits': {}, 'membership': []}
    for name, part, raw in [('train', train, train_bytes), ('test', test, test_bytes)]:
        counts = (part.language + ':' + part.label).value_counts().to_dict()
        if any(counts.get(lang + ':' + label, 0) < 1 for lang in ['de', 'en'] for label in LABELS):
            raise ValueError('Fixed sample lacks a language/class; do not alter seed after labels')
        target = LOCAL / (name + '.csv')
        if target.exists() and target.read_bytes() != raw: raise RuntimeError('Existing split differs')
        target.write_bytes(raw)
        manifest['splits'][name] = {'rows': len(part), 'sha256': digest(raw), 'language_label_counts': counts}
        manifest['membership'].extend({'id': r.id, 'group': r.group, 'split': name, 'text_sha256': r.text_sha256}
                                      for r in part.itertuples())
    if previous and previous != manifest: raise RuntimeError('Restoration differs from fixed manifest')
    if not previous: write_json(OUT / 'split_manifest.json', manifest)
    write_json(OUT / 'overlap_audit.json', {'matched_ids': matches, 'scanned_files': scanned,
        'protected_historical_inputs': len(old), 'linked_source_near_duplicate_groups_disjoint': True,
        'limits': 'Conservative observed input/event groups; residual semantic event and public pretraining overlap unknown'})
    print(json.dumps(manifest['splits'], indent=2), flush=True)


def verified():
    m = read_json(OUT / 'split_manifest.json')
    if protocol_hash() != m['protocol_sha256'] or digest(PROMPT.encode()) != read_json(OUT / 'protocol.json')['prompt_sha256']:
        raise RuntimeError('Frozen protocol/prompt changed')
    for name, value in m['splits'].items():
        if digest((LOCAL / (name + '.csv')).read_bytes()) != value['sha256']: raise RuntimeError('Split changed: ' + name)
    if digest((OUT / 'annotations.csv').read_bytes()) != m['annotation_sha256']: raise RuntimeError('Fresh annotations changed')
    if digest((ROOT / 'data/expanded_news_v4/train.csv').read_bytes()) != m['splits']['train']['sha256']:
        raise RuntimeError('Original Train changed')
    if digest((ROOT / 'reports/expanded_news_v4/split_manifest.json').read_bytes()) != m['original_train_split_sha256']:
        raise RuntimeError('Original split provenance changed')
    return m


def gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required')
    torch.set_num_threads(4); random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); tracking()


def common(m):
    return {'seed': SEED, 'train_sha256': m['splits']['train']['sha256'], 'test_sha256': m['splits']['test']['sha256'],
        'annotation_sha256': m['annotation_sha256'], 'protocol_sha256': protocol_hash(),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'cpu_threads': 4, 'model_id': MODEL, 'revision': REVISION,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}


def tags(stage, group):
    return {'phase': 'wikinews-llm-adaptation-v5', 'stage': stage, 'comparable_group': group,
        'label_source': 'AI-labelled / Codex-reviewed', 'human_gold': 'false', 'deployment_eligible': 'false',
        'reproduction': os.getenv('SENTIMENT_LLM_ADAPT_REPRODUCTION', 'original-measurement')}


def make_profile(path, mode, fit=None, seed=SEED, adapter=None):
    if path.exists(): raise RuntimeError('Profile already exists: ' + str(path))
    meta = read_json(BASE / 'artifact.json')
    meta.update(base_artifact='models/local_llm', system_prompt=PROMPT, max_article_tokens=3072,
        model_version='qwen3-4b-news-v5-' + mode + '-bf16', experiment='wikinews-llm-adaptation-v5',
        prompt_sha256=digest(PROMPT.encode()), adaptation=mode, label_source='AI-labelled / Codex-reviewed')
    if mode == 'fewshot':
        chosen = examples(fit, seed)
        meta.update(demonstrations=demonstrations(chosen), example_seed=seed,
                    example_provenance=[{k: r[k] for k in ['id', 'language', 'label', 'text_sha256']} for r in chosen])
    if adapter: meta['adapter_path'] = str(adapter.relative_to(ROOT))
    write_json(path / 'artifact.json', meta)
    return path


def profile_params(path):
    meta = read_json(path / 'artifact.json')
    return {'system_prompt_sha256': digest(meta['system_prompt'].encode()),
        'adaptation': meta.get('adaptation', 'unchanged financial zero-shot'),
        'demonstration_count': len(meta.get('demonstrations', [])) // 2,
        'max_article_tokens': meta['max_article_tokens'], 'score_method': meta['score_method'],
        'adapter': meta.get('adapter_path', 'none'),
        'example_ids_sha256': digest('\n'.join(r['id'] for r in meta.get('example_provenance', [])).encode())}


def full_size(path):
    meta = read_json(path / 'artifact.json'); incremental = size(path)
    if meta.get('adapter_path'):
        adapter = ROOT / meta['adapter_path']
        if not adapter.is_relative_to(path): incremental += size(adapter)
    standalone = incremental + (size(ROOT / meta['base_artifact']) if meta.get('base_artifact') else 0)
    return standalone, incremental


def artifact_hash(path):
    meta = read_json(path / 'artifact.json')
    parts = [fingerprint(path)]
    for key in ['base_artifact', 'adapter_path']:
        if meta.get(key) and not (ROOT / meta[key]).is_relative_to(path): parts.append(fingerprint(ROOT / meta[key]))
    return digest('\n'.join(parts).encode())


def measure(path, df, checkpoint):
    result, matrices, version, meta = predict_rows(path, df, checkpoint)
    total, incremental = full_size(path)
    result.update(model_size_bytes=total, model_incremental_storage_bytes=incremental)
    mlflow.log_metrics(result); mlflow.log_dict(matrices, 'confusion_matrix.json')
    mlflow.log_dict({k: v for k, v in meta.items() if k != 'demonstrations'}, 'evaluated_artifact_metadata.json')
    mlflow.log_param('model_version', version)
    return {**result, 'confusion_matrix': matrices, 'model_version': version}


def lora_model():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(BASE / 'weights', local_files_only=True)
    net = AutoModelForCausalLM.from_pretrained(BASE / 'weights', local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    net.config.use_cache = False
    net = get_peft_model(net, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=.05,
                                       target_modules=['q_proj', 'v_proj'], bias='none'))
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    net.enable_input_require_grads(); net.train()
    return tokenizer, net


def training_tokens(tokenizer, fit):
    tokens = []
    for text in fit.text:
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': PROMPT},
            {'role': 'user', 'content': 'Classify this article:\n' + text}], tokenize=False, add_generation_prompt=True)
        row = tokenizer(prompt, return_tensors='pt')
        if row.input_ids.shape[1] > 1024: raise RuntimeError('Full input exceeds fixed training resource guard; do not truncate silently')
        tokens.append(row)
    return tokens


def probe():
    gpu()
    if (OUT / 'feasibility.json').exists(): raise RuntimeError('Engineering probe already recorded')
    fit = pd.read_csv(ROOT / 'data/expanded_news_v4/train.csv')
    tokenizer, net = lora_model(); tokens = training_tokens(tokenizer, fit)
    index = int(np.argmax([r.input_ids.shape[1] for r in tokens]))
    choices = [tokenizer.encode(x, add_special_tokens=False)[0] for x in ['A', 'B', 'C']]
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    output = net(**tokens[index].to('cuda'), use_cache=False, logits_to_keep=1).logits[0, -1, choices].float()
    loss = torch.nn.functional.cross_entropy(output[None], torch.tensor([LABELS.index(fit.iloc[index].label)], device='cuda'))
    loss.backward(); torch.cuda.synchronize()
    report = {'engineering_only': True, 'retained_training_fit': False, 'probe_input_id': fit.iloc[index].id,
        'sequence_tokens': tokens[index].input_ids.shape[1], 'all_train_rows_fit_without_truncation': len(tokens),
        'forward_backward_seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'protocol_sha256': protocol_hash(),
        'purpose': 'Discarded one-batch GPU capacity check on Train, not an accuracy/model-selection run'}
    write_json(OUT / 'feasibility.json', report); print(json.dumps(report, indent=2), flush=True)
    del loss, output, tokens, net, tokenizer; gc.collect(); torch.cuda.empty_cache()


def fit_adapter(fit, path):
    if path.exists(): raise RuntimeError('Adapter already exists')
    tokenizer, net = lora_model(); tokens = training_tokens(tokenizer, fit)
    choices = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ['A', 'B', 'C']]
    y = np.array([LABELS.index(x) for x in fit.label]); weights = len(y) / (3 * np.bincount(y, minlength=3))
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    steps = 3 * math.ceil(len(fit) / 16)
    schedule = get_linear_schedule_with_warmup(optimizer, int(.1 * steps), steps)
    mlflow.log_params({'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': .05, 'target_modules': 'q_proj,v_proj',
        'epochs': 3, 'microbatch_size': 1, 'gradient_accumulation': 16, 'learning_rate': 1e-4,
        'weight_decay': .01, 'warmup_fraction': .1, 'gradient_clip': 1., 'gradient_checkpointing': 'non-reentrant',
        'loss': 'inverse-class weighted conditional next-token A/B/C cross entropy', 'training_precision': 'BF16 base, FP32 adapters/Adam states',
        'train_rows': len(fit), 'max_train_sequence_tokens': max(t.input_ids.shape[1] for t in tokens),
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'checkpoint_rule': 'Fixed final epoch3, no Val/Test model selection'})
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    for epoch in range(3):
        order = np.random.default_rng(SEED + epoch).permutation(len(fit)); total, seen = 0., 0
        for start in range(0, len(fit), 16):
            chunk = order[start:start + 16]; optimizer.zero_grad(set_to_none=True)
            for i in chunk:
                logits = net(**tokens[int(i)].to('cuda'), use_cache=False, logits_to_keep=1).logits[0, -1, choices].float()
                loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([y[i]], device='cuda')) * float(weights[y[i]])
                (loss / len(chunk)).backward(); total += float(loss.detach()); seen += 1
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.)
            optimizer.step(); schedule.step()
            if seen % 160 == 0: print(f'LoRA epoch{epoch + 1} {seen}/{len(fit)} rows', flush=True)
        mlflow.log_metric('train_loss', total / seen, step=epoch + 1); mlflow.log_metric('train_rows_seen', seen, step=epoch + 1)
        print(f'LoRA epoch{epoch + 1}/3 full fit={seen} loss={total / seen:.4f}', flush=True)
    net.eval(); net.save_pretrained(path, safe_serialization=True)
    torch.cuda.synchronize(); mlflow.log_metrics({'training_seconds': time.perf_counter() - started,
        'peak_training_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_training_reserved_bytes': torch.cuda.max_memory_reserved()})
    del net, tokenizer, tokens, optimizer, schedule, logits, loss; gc.collect(); torch.cuda.empty_cache()


def cv():
    gpu(); m = verified()
    if not (OUT / 'feasibility.json').exists(): raise RuntimeError('Run Train-only memory probe first')
    if (OUT / 'cv.json').exists() or (ART / 'cv').exists(): raise RuntimeError('CV already started/completed')
    train = pd.read_csv(LOCAL / 'train.csv')
    membership = [r for r in read_json(ROOT / 'reports/expanded_news_v4/cv_membership.json') if r['size'] == 3]
    small = [r for r in read_json(ROOT / 'reports/expanded_news_v4/cv.json')['runs'] if r['size'] == 3]
    records, audit = [], []
    for member in membership:
        fold = member['fold']; fit = train[train.id.isin(member['fit_ids'])]; dev = train[train.id.isin(member['dev_ids'])]
        assert fit.id.tolist() == member['fit_ids'] and dev.id.tolist() == member['dev_ids']
        assert not (set(fit['group']) & set(dev['group']))
        folder = ART / 'cv' / f'fold{fold}'; folder.mkdir(parents=True)
        profile_paths = {'qwen3_4b': BASE,
            'qwen_news_v5': make_profile(folder / 'qwen_news_v5', 'news'),
            'qwen_fewshot_v5': make_profile(folder / 'qwen_fewshot_v5', 'fewshot', fit, SEED + fold)}
        example_ids = [r['id'] for r in read_json(profile_paths['qwen_fewshot_v5'] / 'artifact.json')['example_provenance']]
        assert set(example_ids) <= set(fit.id) and not set(example_ids) & set(dev.id)
        audit.append({**member, 'fewshot_example_ids': example_ids})
        write_json(OUT / 'cv_membership.json', audit)
        for name in QWEN:
            with mlflow.start_run(run_name=f'LLM adaptation v5 · fold{fold} · {name}') as run:
                mlflow.set_tags(tags('cv', f'wikinews-v4-cv-fold{fold}'))
                mlflow.log_params({**common(m), 'fold': fold, 'fit_rows': len(fit), 'dev_rows': len(dev),
                    'fit_ids_sha256': member['fit_ids_sha256'], 'dev_ids_sha256': member['dev_ids_sha256'],
                    'warmup_calls': 20, 'latency_batch_size': 1})
                if name == 'qwen_lora_v5':
                    adapter = folder / 'adapter'; fit_adapter(fit, adapter)
                    path = make_profile(folder / name, 'lora', adapter=adapter)
                else: path = profile_paths[name]
                mlflow.log_params(profile_params(path)); sha = artifact_hash(path); mlflow.log_param('artifact_sha256', sha)
                measured = measure(path, dev, folder / (name + '_dev_predictions.jsonl'))
                records.append({'model': name, 'name': NAMES[name], 'fold': fold, 'run_id': run.info.run_id,
                    'path': str(path.relative_to(ROOT)), 'artifact_sha256': sha, **measured})
                write_json(OUT / 'cv_progress.json', records)
                print('CV', fold, name, measured['balanced_language_macro_f1'], flush=True)
    means = {name: float(np.mean([r['balanced_language_macro_f1'] for r in records if r['model'] == name])) for name in QWEN}
    write_json(OUT / 'cv.json', {'train_only': True, 'protocol_sha256': protocol_hash(), 'runs': records,
        'llm_mean_language_f1': means, 'reused_small_model_runs': small,
        'selection': 'Diagnostic fixed conditions; no parameter/epoch/prompt search. All seven final conditions retained.'})
    print('Twelve LLM CV conditions complete. Fresh Test has not been predicted.', flush=True)


def train():
    gpu(); m = verified()
    if not (OUT / 'cv.json').exists(): raise RuntimeError('Complete Train-only CV first')
    if (ART / 'final').exists() or (OUT / 'trained.json').exists() or (OUT / 'evaluation.lock').exists():
        raise RuntimeError('Final preparation already started/completed')
    fit = pd.read_csv(LOCAL / 'train.csv'); folder = ART / 'final'; folder.mkdir(parents=True)
    records = []
    for name, mode in [('qwen_news_v5', 'news'), ('qwen_fewshot_v5', 'fewshot')]:
        path = make_profile(folder / name, mode, fit, SEED)
        records.append({'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                        'run_id': None, 'artifact_sha256': artifact_hash(path),
                        'example_provenance': read_json(path / 'artifact.json').get('example_provenance', [])})
    with mlflow.start_run(run_name='LLM adaptation v5 · full692 LoRA') as run:
        mlflow.set_tags(tags('final-test', 'wikinews-v5-fresh-test')); mlflow.log_params(common(m))
        adapter = folder / 'adapter'; fit_adapter(fit, adapter)
        name = 'qwen_lora_v5'; path = make_profile(folder / name, 'lora', adapter=adapter)
        mlflow.log_params(profile_params(path)); sha = artifact_hash(path); mlflow.log_param('artifact_sha256', sha)
        records.append({'model': name, 'name': NAMES[name], 'path': str(path.relative_to(ROOT)),
                        'run_id': run.info.run_id, 'artifact_sha256': sha})
    records.append({'model': 'qwen3_4b', 'name': NAMES['qwen3_4b'], 'path': 'models/local_llm',
                    'run_id': None, 'artifact_sha256': artifact_hash(BASE)})
    old = read_json(ROOT / 'reports/expanded_news_v4/trained.json')['models']
    for name in ['frozen_news_v4', 'minilm_bf16_news_v4', 'linearsvc_news_v4']:
        r = next(r for r in old if r['model'] == name)
        if fingerprint(ROOT / r['path']) != r['artifact_sha256']: raise RuntimeError('Existing small-model control changed')
        records.append({**r, 'name': NAMES[name], 'original_run_id': r['run_id'], 'run_id': None,
                        'artifact_sha256': artifact_hash(ROOT / r['path'])})
    write_json(OUT / 'trained.json', {'protocol_sha256': protocol_hash(), 'models': records,
        'test_unpredicted': True, 'full_train_rows': len(fit)})
    print('All seven final conditions/dependencies pinned, fresh Test unpredicted.', flush=True)


def evaluate(resume=False):
    gpu(); m = verified(); trained = read_json(OUT / 'trained.json')
    if trained['protocol_sha256'] != protocol_hash(): raise RuntimeError('Protocol changed after fitting')
    lock = OUT / 'evaluation.lock'
    if lock.exists():
        r = read_json(lock)
        if r['status'] == 'complete' or not resume or r['protocol_sha256'] != protocol_hash():
            raise RuntimeError('Fresh Test already started/completed; no repeated probing')
    for r in trained['models']:
        if artifact_hash(ROOT / r['path']) != r['artifact_sha256']: raise RuntimeError('Frozen artifact/dependency changed')
    if not lock.exists(): write_json(lock, {'status': 'started', 'protocol_sha256': protocol_hash(), 'test_sha256': m['splits']['test']['sha256']})
    test = pd.read_csv(LOCAL / 'test.csv'); progress = OUT / 'evaluation_progress.json'
    results = read_json(progress) if progress.exists() else []
    for r in trained['models']:
        if any(x['model'] == r['model'] for x in results): continue
        context = mlflow.start_run(run_id=r['run_id']) if r['run_id'] else mlflow.start_run(run_name='Fresh v5 test · ' + r['name'])
        with context as run:
            mlflow.set_tags(tags('final-test', 'wikinews-v5-fresh-test'))
            evaluation_params = common(m)
            if r['run_id']: evaluation_params.pop('git_commit')  # Preserve the fit commit on the resumed LoRA run.
            if r['model'] not in QWEN:
                from src.config import ENCODER, ENCODER_REVISION
                evaluation_params['model_id'] = ENCODER if r['model'] != 'linearsvc_news_v4' else 'TFIDF + LinearSVC'
                evaluation_params['revision'] = ENCODER_REVISION if r['model'] != 'linearsvc_news_v4' else 'sklearn1.8.0 C1 balanced'
            mlflow.log_params({**evaluation_params, 'test_rows': len(test), 'warmup_calls': 20, 'latency_batch_size': 1,
                'artifact_sha256': r['artifact_sha256'], 'original_run_id': r.get('original_run_id', 'new LLM condition'),
                'evaluation_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
            if r['model'] in QWEN: mlflow.log_params(profile_params(ROOT / r['path']))
            measured = measure(ROOT / r['path'], test, LOCAL / (r['model'] + '_test_predictions.jsonl'))
            mlflow.log_artifact(str(OUT / 'protocol.json')); mlflow.log_artifact(str(OUT / 'split_manifest.json'))
            results.append({**r, **measured, 'run_id': run.info.run_id})
        write_json(progress, results)
        print('FINAL', r['model'], measured['balanced_language_macro_f1'], measured['recall_positive'], flush=True)
    best = max(r['balanced_language_macro_f1'] for r in results)
    winner = min([r for r in results if best - r['balanced_language_macro_f1'] <= .005],
                 key=lambda r: (r['inference_ms_per_text'], r['model_size_bytes']))
    write_json(OUT / 'comparison.json', {'data_version': m['data_version'], 'label_status': m['label_status'],
        'human_gold': False, 'test_rows': len(test), 'test_counts': m['splits']['test']['language_label_counts'],
        'test_sha256': m['splits']['test']['sha256'], 'descriptive_leader': winner['model'], 'results': results,
        'serving_default_changed': False, 'interpretation': 'One fresh random Test, matched task/Train adaptation opportunities; AI reference agreement, not human gold or compute-matched architecture proof.'})
    write_json(lock, {'status': 'complete', 'protocol_sha256': protocol_hash(),
        'test_sha256': m['splits']['test']['sha256'], 'candidate_count': len(results)})
    print('Final audit complete once per candidate. No winner was preassigned.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'probe', 'cv', 'train', 'evaluate'])
    parser.add_argument('--restore', action='store_true'); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.restore)
    elif args.action == 'probe': probe()
    elif args.action == 'cv': cv()
    elif args.action == 'train': train()
    else: evaluate(args.resume)
