"""Offline sequential Qwen-LoRA factorial runner. Never loads final-Test text."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .design import FACTORS, REFERENCE, configurations, contrasts, validate_membership

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/optimizer_ablation'
LOCAL = ROOT / 'runtime/optimizer_ablation'
ART = ROOT / 'models/optimizer_ablation'
PROTOCOL = OUT / 'protocol-v1.json'
EXPERIMENT = 'article-optimizer-ablation-v1'
MODULE = 'src.studies.optimizer_ablation'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hash():
    return hashlib.sha256('\n'.join(p.name + ':' + sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()


def write(path, value, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable:
        with path.open('x') as stream:
            stream.write(text)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text)
        temp.replace(path)


def initialize():
    import pandas as pd
    from src.next_stage_experiment import TRAIN, BASE, verified, code_hash
    from src.tracking import fingerprint
    verified()
    table = pd.read_csv(TRAIN)
    membership = json.loads((ROOT / 'reports/next_stage/experiment-membership-v1.json').read_text())
    member = next(m for m in membership if m['fold'] == 0)
    validate_membership(table[['id', 'group', 'language', 'label']].to_dict('records'), member)
    protocol = {
        'version': 'optimizer-ablation-v1', 'created_at': now(), 'status_at_freeze': 'No study fit or development prediction started',
        'purpose': 'Exploratory post-Test follow-up, not a replacement for the frozen confirmatory comparison',
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'study_source_sha256': source_hash(), 'frozen_runtime_sha256': code_hash(),
        'train_path': str(TRAIN.relative_to(ROOT)), 'train_sha256': sha(TRAIN),
        'original_protocol_sha256': sha(ROOT / 'reports/next_stage/development-protocol-v1.json'),
        'base_artifact_sha256': fingerprint(BASE),
        'membership_source_sha256': sha(ROOT / 'reports/next_stage/experiment-membership-v1.json'),
        'fold': 0, 'fit_ids': member['fit_ids'], 'dev_ids': member['dev_ids'],
        'fit_rows': len(member['fit_ids']), 'dev_rows': len(member['dev_ids']),
        'support': {role: table[table.id.isin(member[key])].groupby(['language', 'label']).size().to_dict()
                    for role, key in [('fit', 'fit_ids'), ('development', 'dev_ids')]},
        'factors': FACTORS, 'reference': REFERENCE, 'runs': configurations(), 'expected_runs': 32,
        'fixed': {'model': 'Qwen LoRA, original pretrained base', 'optimizer': 'torch.optim.AdamW',
                  'betas': [0.9, 0.999], 'eps': 1e-8, 'epochs': 3, 'accumulation_articles': 16,
                  'lora_rank': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'target_modules': ['q_proj', 'v_proj'],
                  'precision': 'BF16 base, FP32 adapter/AdamW states', 'scheduler': 'linear decay with optional linear warmup',
                  'loss': 'fit-only inverse-class weighted article A/B/C cross entropy',
                  'context': 'complete articles, at most 4096 article/4608 prompt tokens, reject overflow',
                  'checkpoint': 'final epoch only, no early stopping or development checkpoint selection'},
        'metrics': {'primary': '(macro_F1_DE + macro_F1_EN) / 2, all three classes',
                    'secondary': ['macro_F1_JA', 'macro_F1_DE', 'macro_F1_EN', 'macro_f1', 'development_nll'],
                    'diagnostics': ['gradient norm before clipping', 'clip fraction', 'actual learning rate',
                                    'adapter update L2 norm', 'training loss', 'time', 'CUDA peak memory']},
        'analysis': 'Within each seed average 8 matched high-minus-low contrasts per factor; average across two seeds; report per-seed effects and six two-factor difference-of-differences. No p-values.',
        'selection': 'No deployment selection. Current serving model and original Test study remain unchanged.',
        'limits': ['Only fold 0 of the original Train-only grouped split; no full cross-validation.',
                   'Two seeds do not establish broad uncertainty; the same 81 development articles are reused.',
                   'Post-Test exploratory study; prior project results were already known.',
                   'Weight-decay contrast concerns AdamW regularization, not AdamW versus another optimizer.',
                   'Measured changes cannot by themselves establish a causal learning mechanism.'],
        'mlflow_experiment': EXPERIMENT,
        'dependencies': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'peft', 'accelerate', 'mlflow', 'numpy', 'pandas']},
    }
    # JSON keys for support are readable language/label combinations, not tuples.
    protocol['support'] = {role: {'/'.join(k): int(v) for k, v in support.items()} for role, support in protocol['support'].items()}
    write(PROTOCOL, protocol, immutable=True)
    print(f'Frozen 32-run protocol: {protocol["fit_rows"]} fit / {protocol["dev_rows"]} development; no Test access.', flush=True)


def verify(check_base=False):
    from src.next_stage_experiment import TRAIN, BASE, verified, code_hash
    from src.tracking import fingerprint
    protocol = json.loads(PROTOCOL.read_text())
    verified()
    if any(importlib.metadata.version(name) != version for name, version in protocol['dependencies'].items()):
        raise RuntimeError('Frozen dependency versions changed')
    if (protocol['study_source_sha256'] != source_hash() or protocol['frozen_runtime_sha256'] != code_hash()
            or protocol['train_sha256'] != sha(TRAIN)
            or protocol['membership_source_sha256'] != sha(ROOT / 'reports/next_stage/experiment-membership-v1.json')
            or protocol['runs'] != configurations()):
        raise RuntimeError('Frozen study code, configuration or Train input changed')
    if check_base and protocol['base_artifact_sha256'] != fingerprint(BASE):
        raise RuntimeError('Base artifact changed')
    return protocol


def evaluate(net, tokenizer, dev, choices):
    import numpy as np
    import torch
    from src.config import LABELS
    from src.next_stage_experiment import qwen_tokens, metrics
    net.eval()
    records, predicted, losses = [], [], []
    with torch.inference_mode():
        for row in dev.itertuples():
            tokens = qwen_tokens(tokenizer, row.text).to('cuda')
            logits = net(**tokens, use_cache=False, logits_to_keep=1).logits[0, -1, choices].float()
            scores = torch.softmax(logits, dim=-1).cpu().tolist()
            prediction = int(np.argmax(scores))
            target = LABELS.index(row.label)
            losses.append(float(torch.nn.functional.cross_entropy(logits[None], torch.tensor([target], device='cuda'))))
            predicted.append(prediction)
            records.append(dict(id=row.id, text_sha256=row.text_sha256, language=row.language,
                                reference=row.label, prediction=LABELS[prediction], scores=scores))
    return dict(metrics(dev, predicted), development_nll=float(np.mean(losses))), records


def one(run_id):
    import random
    import mlflow
    import numpy as np
    import pandas as pd
    import torch
    from transformers import get_linear_schedule_with_warmup
    from src.config import LABELS
    from src.next_stage_experiment import TRAIN, lora, qwen_tokens, gpu
    from src.tracking import fingerprint
    protocol = verify(check_base=True)
    config = next(row for row in protocol['runs'] if row['id'] == run_id)
    result_path = OUT / 'runs' / f'{run_id}.json'
    folder = ART / run_id
    if result_path.exists() or folder.exists():
        raise RuntimeError('Existing run result/checkpoint requires explicit recovery audit; never overwrite')
    evidence = gpu()
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 1024**3:
        raise RuntimeError('Less than 12 GiB free CUDA memory; refusing concurrent GPU work')
    random.seed(config['seed']); np.random.seed(config['seed'])
    torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
    table = pd.read_csv(TRAIN)
    validate_membership(table[['id', 'group', 'language', 'label']].to_dict('records'), protocol)
    fit = table[table.id.isin(protocol['fit_ids'])].reset_index(drop=True)
    dev = table[table.id.isin(protocol['dev_ids'])].reset_index(drop=True)
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    mlflow.set_experiment(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=False)
    with mlflow.start_run(run_name=f'{run_id} · seed {config["seed"]}') as run:
        mlflow.set_tags({'phase': 'optimizer-ablation-v1', 'stage': 'post-test-train-only-development',
                         'test_used': 'false', 'serving_selection_changed': 'false', 'fold': '0',
                         'model_family': 'Qwen-LoRA', 'label_status': 'Project-reviewed 2026-11-14; original annotation provenance retained',
                         'protocol_sha256': sha(PROTOCOL), 'source_sha256': source_hash(),
                         'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         'base_artifact_sha256': protocol['base_artifact_sha256'], 'train_sha256': protocol['train_sha256']})
        mlflow.log_params(dict(config, gradient_clip='disabled' if config['gradient_clip'] is None else config['gradient_clip'],
                               fit_rows=len(fit), dev_rows=len(dev), epochs=3, batch_articles=16,
                               optimizer='AdamW', beta1=0.9, beta2=0.999, eps=1e-8))
        mlflow.log_artifact(str(PROTOCOL), artifact_path='protocol')
        started = time.perf_counter()
        tokenizer, net = lora()
        tokens = [qwen_tokens(tokenizer, text) for text in fit.text]
        choices = [tokenizer.encode(c, add_special_tokens=False)[0] for c in 'ABC']
        if len(set(choices)) != 3 or any(len(tokenizer.encode(c, add_special_tokens=False)) != 1 for c in 'ABC'):
            raise RuntimeError('A/B/C must be distinct single tokens')
        trainable = [p for p in net.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in trainable):
            raise RuntimeError('Expected FP32 LoRA parameters')
        labels = np.array([LABELS.index(v) for v in fit.label])
        weights = len(fit) / (3 * np.bincount(labels, minlength=3))
        optimizer = torch.optim.AdamW(trainable, lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                      betas=(0.9, 0.999), eps=1e-8)
        steps = 3 * math.ceil(len(fit) / 16)
        warmup_steps = int(config['warmup_fraction'] * steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
        mlflow.log_params(dict(optimizer_steps=steps, warmup_steps=warmup_steps, trainable_parameters=sum(p.numel() for p in trainable)))
        load_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); training_start = time.perf_counter()
        telemetry = []
        for epoch in range(3):
            net.train()
            order = np.random.default_rng(config['seed'] + epoch).permutation(len(fit))
            for start in range(0, len(fit), 16):
                batch = order[start:start + 16]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                step_started = time.perf_counter()
                for i in batch:
                    logits = net(**tokens[int(i)].to('cuda'), use_cache=False, logits_to_keep=1).logits[0, -1, choices].float()
                    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([labels[i]], device='cuda')) * float(weights[labels[i]])
                    (loss / len(batch)).backward()
                    batch_loss += float(loss.detach()) / len(batch)
                threshold = config['gradient_clip']
                # Infinity measures the same global norm without scaling finite gradients.
                norm = float(torch.nn.utils.clip_grad_norm_(trainable, math.inf if threshold is None else threshold, error_if_nonfinite=True))
                before = [p.detach().clone() for p in trainable]
                learning_rate = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                update = float(torch.sqrt(sum((p.detach() - old).square().sum() for p, old in zip(trainable, before))))
                del before
                torch.cuda.synchronize()
                event = dict(step=len(telemetry) + 1, epoch=epoch + 1, train_loss=batch_loss, gradient_norm=norm,
                             clipped=int(threshold is not None and norm > threshold), learning_rate=learning_rate,
                             adapter_update_l2=update, step_seconds=time.perf_counter() - step_started)
                telemetry.append(event)
                mlflow.log_metrics({k: v for k, v in event.items() if k not in ('step', 'epoch')}, step=event['step'])
                write(LOCAL / 'current-step.json', dict(run_id=run_id, mlflow_run_id=run.info.run_id, updated_at=now(), total_steps=steps, **event))
                print(json.dumps(dict(run_id=run_id, **event)), flush=True)
        training_seconds = time.perf_counter() - training_start
        peak = torch.cuda.max_memory_allocated()
        net.eval(); net.save_pretrained(folder / 'adapter', safe_serialization=True)
        write(folder / 'config.json', config, immutable=True)
        write(OUT / 'telemetry' / f'{run_id}.json', telemetry, immutable=True)
        measured, predictions = evaluate(net, tokenizer, dev, choices)
        prediction_path = OUT / 'predictions' / f'{run_id}.json'
        write(prediction_path, predictions, immutable=True)
        measured.update(training_seconds=training_seconds, loading_seconds=load_seconds, peak_training_allocated_bytes=peak,
                        mean_gradient_norm=float(np.mean([t['gradient_norm'] for t in telemetry])),
                        max_gradient_norm=max(t['gradient_norm'] for t in telemetry),
                        clip_fraction=float(np.mean([t['clipped'] for t in telemetry])),
                        mean_adapter_update_l2=float(np.mean([t['adapter_update_l2'] for t in telemetry])),
                        final_train_loss=float(np.mean([t['train_loss'] for t in telemetry if t['epoch'] == 3])))
        result = dict(config=config, completed_at=now(), mlflow_run_id=run.info.run_id, metrics=measured,
                      protocol_sha256=sha(PROTOCOL), adapter_sha256=fingerprint(folder),
                      predictions_sha256=sha(prediction_path), telemetry_sha256=sha(OUT / 'telemetry' / f'{run_id}.json'), gpu=evidence)
        mlflow.log_metrics({k: v for k, v in measured.items() if isinstance(v, (float, int))})
        for path in (prediction_path, OUT / 'telemetry' / f'{run_id}.json'):
            mlflow.log_artifact(str(path), artifact_path=path.parent.name)
        mlflow.log_dict(result, 'result.json')
        write(result_path, result, immutable=True)
        print(f'COMPLETED {run_id}: primary F1={measured["primary_score"]:.4f}', flush=True)


def summarize(state='running'):
    results = [json.loads(p.read_text()) for p in sorted((OUT / 'runs').glob('run-*.json'))]
    summary = dict(updated_at=now(), state=state, expected_runs=32, completed_runs=len(results),
                   mlflow_experiment=EXPERIMENT, protocol_sha256=sha(PROTOCOL),
                   effects={metric: contrasts(results, metric) for metric in ['primary_score', 'macro_f1_de', 'macro_f1_en', 'macro_f1_ja', 'development_nll']},
                   runs=results)
    write(OUT / 'summary-v1.json', summary)
    lines = ['# Optimizer ablation — exploratory development results', '',
             f'Status: **{state}**, {len(results)}/32 runs complete. Updated: {summary["updated_at"]}.', '',
             '159 fit / 81 development articles, original Train fold 0; no final-Test inference. Two seeds.',
             'Current serving model and frozen Test results remain unchanged.', '',
             '| Run | Seed | Clip | LR | AdamW decay | Warmup | DE/EN F1 | JA F1 | Clipped steps |',
             '|---|---|---|---|---|---|---|---|---|']
    for row in results:
        c, m = row['config'], row['metrics']
        lines.append(f'| {c["id"]} | {c["seed"]} | {c["gradient_clip"]} | {c["learning_rate"]} | {c["weight_decay"]} | {c["warmup_fraction"]} | {m["primary_score"]:.4f} | {m["macro_f1_ja"]:.4f} | {m["clip_fraction"]:.1%} |')
    effect = summary['effects']['primary_score']
    if effect['status'] == 'complete':
        lines += ['', '## Matched factor contrasts', '',
                  'High minus low, absolute F1 percentage points, averaged over matched settings and both seeds.', '',
                  '| Factor | Mean effect (pp) | Seed 1 (pp) | Seed 2 (pp) |', '|---|---|---|---|']
        for name, info in effect['main_effects'].items():
            per_seed = [r['main_effects'][name]['high_minus_low'] * 100 for r in effect['per_seed'].values()]
            lines.append(f'| {name} | {info["high_minus_low"] * 100:+.2f} | {per_seed[0]:+.2f} | {per_seed[1]:+.2f} |')
        lines += ['', 'Two-factor interactions and all per-seed paired differences are in `summary-v1.json`.']
    lines += ['', 'These are descriptive effects for this fixed split and parameter range. Shared articles and two seeds',
              'do not support independent-sample significance claims or a proven causal mechanism.',
              'AdamW itself is fixed; only its weight decay varies. No deployment selection is made.',
              'Training telemetry can support hypotheses about clipping/stability; it does not establish why F1 changed.']
    (OUT / 'results-v1.md').write_text('\n'.join(lines) + '\n')
    return summary


def worker():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Held by the supervisor throughout; a fresh child releases CUDA between runs.
    with (ROOT / 'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = verify(check_base=True)
        write(LOCAL / 'worker.json', dict(pid=os.getpid(), state='running', started_at=now(), protocol_sha256=sha(PROTOCOL)))
        try:
            for config in protocol['runs']:
                result_path = OUT / 'runs' / f'{config["id"]}.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if result['config'] != config or result['protocol_sha256'] != sha(PROTOCOL):
                        raise RuntimeError('Completed result identity mismatch')
                    for kind in ('predictions', 'telemetry'):
                        if result[f'{kind}_sha256'] != sha(OUT / kind / f'{config["id"]}.json'):
                            raise RuntimeError('Saved evidence changed')
                    continue
                if (LOCAL / 'STOP').exists():
                    summarize('stopped')
                    write(LOCAL / 'worker.json', dict(pid=os.getpid(), state='stopped', updated_at=now()))
                    return
                verify()
                write(LOCAL / 'worker.json', dict(pid=os.getpid(), state='running', current_run=config['id'], updated_at=now()))
                summarize('running')
                subprocess.run([sys.executable, '-m', MODULE, 'one', '--run-id', config['id']], cwd=ROOT, check=True)
                summarize('running')
            summarize('complete')
            write(LOCAL / 'worker.json', dict(pid=os.getpid(), state='complete', completed_at=now()))
        except BaseException as exc:
            traceback.print_exc()
            summarize('failed')
            write(LOCAL / 'worker.json', dict(pid=os.getpid(), state='failed', failed_at=now(), error=str(exc)))
            raise


def launch():
    LOCAL.mkdir(parents=True, exist_ok=True)
    # Don't create a competing supervisor. Worker also takes an exclusive GPU lock.
    with (ROOT / 'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (LOCAL / 'worker.log').open('a') as log:
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        process = subprocess.Popen([sys.executable, '-m', MODULE, 'worker'], cwd=ROOT, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(LOCAL / 'launch.json', dict(pid=process.pid, launched_at=now(), log=str((LOCAL / 'worker.log').relative_to(ROOT))))
    print(json.dumps(dict(pid=process.pid, log=str(LOCAL / 'worker.log'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'launch', 'worker', 'one', 'status', 'summarize'])
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'one':
        if not args.run_id:
            parser.error('--run-id is required')
        one(args.run_id)
    elif args.action == 'status':
        for path in [LOCAL / 'worker.json', LOCAL / 'current-step.json', OUT / 'summary-v1.json']:
            if path.exists():
                data = json.loads(path.read_text())
                print(path.name, json.dumps({k: v for k, v in data.items() if k not in ('runs', 'effects')}, ensure_ascii=False))
    else:
        {'initialize': initialize, 'launch': launch, 'worker': worker, 'summarize': summarize}[args.action]()
