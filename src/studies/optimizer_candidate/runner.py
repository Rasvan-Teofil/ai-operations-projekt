"""Freeze, fit and smoke-check an exploratory full-Train candidate; never score Test."""
import argparse
from collections import defaultdict
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

from src.studies.optimizer_ablation.design import FACTORS, SEED, configurations
from src.studies.optimizer_ablation.runner import sha, write

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/optimizer_candidate'
LOCAL = ROOT / 'runtime/optimizer_candidate'
ART = ROOT / 'models/optimizer_candidate/v1'
PROTOCOL = OUT / 'protocol-v1.json'
MODULE = 'src.studies.optimizer_candidate'
VERSION = 'article-qwen-lora-optimizer-v1'
EXPERIMENT = 'article-optimizer-candidate-v1'
SMOKE_TEXTS = {
    'de': 'Die Rettungskräfte haben alle Vermissten lebend gefunden. Die Geretteten sind wieder bei ihren Familien.',
    'en': 'The council published the date and location of its next routine meeting. The agenda will follow next week.',
    'ja': '工場で火災が発生し、複数の作業員が負傷した。生産は停止している。',
}


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hash():
    return hashlib.sha256('\n'.join(p.name + ':' + sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()


def choose_recipe(results):
    """Use both seeds equally; break a primary-score tie by simpler regularization."""
    expected = {r['id']: r for r in configurations()}
    if len(results) != 32 or {r['config']['id'] for r in results} != set(expected):
        raise ValueError('Selection requires all 32 unique factorial results')
    groups = defaultdict(list)
    for result in results:
        config = result['config']
        if config != expected[config['id']] or not math.isfinite(result['metrics']['primary_score']):
            raise ValueError('Changed factorial condition or nonfinite score')
        groups[tuple(config[k] for k in FACTORS)].append(result['metrics']['primary_score'])
    ranked = []
    for key, scores in groups.items():
        if len(scores) != 2:
            raise ValueError('Each condition needs both seeds')
        ranked.append(dict(recipe=dict(zip(FACTORS, key)), mean_development_primary=sum(scores)/2))
    ranked.sort(key=lambda r: (-r['mean_development_primary'], r['recipe']['weight_decay'],
                              r['recipe']['warmup_fraction'], r['recipe']['learning_rate'],
                              math.inf if r['recipe']['gradient_clip'] is None else r['recipe']['gradient_clip']))
    return ranked[0], ranked


def verify_files(root, records):
    for name, expected in records.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise RuntimeError('Frozen input or baseline changed: ' + name)


def initialize():
    import pandas as pd
    from src.next_stage_experiment import TRAIN, BASE
    from src.studies.optimizer_ablation.runner import verify as verify_ablation
    from src.tracking import fingerprint
    ablation = verify_ablation(check_base=True)
    summary_path = ROOT / 'reports/optimizer_ablation/summary-v1.json'
    summary = json.loads(summary_path.read_text())
    if (summary['state'] != 'complete' or summary['completed_runs'] != 32
            or summary['protocol_sha256'] != sha(ROOT/'reports/optimizer_ablation/protocol-v1.json')):
        raise RuntimeError('Complete and verify the source ablation first')
    selected, ranking = choose_recipe(summary['runs'])
    table = pd.read_csv(TRAIN)
    if len(table) != 240 or table.id.duplicated().any():
        raise RuntimeError('Expected the unchanged 240-row Train set')
    support = table.groupby(['language', 'label']).size()
    if len(support) != 9:
        raise RuntimeError('Every language/class must be represented')
    selection_path = ROOT / 'models/serving_selected.json'
    baseline = json.loads(selection_path.read_text())
    profile = ROOT / 'models' / baseline['model'] / 'artifact.json'
    baseline_meta = json.loads(profile.read_text())
    paths = [TRAIN, selection_path, profile, ROOT / 'reports/next_stage/final-test-evaluation.lock',
             ROOT / 'reports/optimizer_ablation/protocol-v1.json', summary_path,
             ROOT / 'reports/optimizer_ablation/validation-v1.json']
    protocol = dict(version='optimizer-candidate-v1', created_at=now(), status_at_freeze='Before full-Train fit',
        purpose='Exploratory full-Train model candidate, selected after the original Test study; no new Test evaluation',
        model_version=VERSION, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=source_hash(), source_ablation_sha256=ablation['study_source_sha256'],
        files_sha256={str(p.relative_to(ROOT)): sha(p) for p in paths},
        base_artifact_sha256=fingerprint(BASE), baseline_adapter_sha256=fingerprint(ROOT / baseline_meta['adapter_path']),
        baseline_adapter_path=baseline_meta['adapter_path'], baseline_version=baseline['model_version'],
        selected_recipe=selected['recipe'], selection_development_mean=selected['mean_development_primary'],
        selection_rule='Highest mean DE/EN development macro-F1 across both ablation seeds; exact ties prefer lower weight decay, then lower warmup, lower learning rate, finite clipping. Seed is fixed independently of scores.',
        ranking=ranking, seed=SEED, train_rows=len(table), train_ids_sha256=hashlib.sha256('\n'.join(table.id).encode()).hexdigest(),
        language_class_support={'/'.join(k): int(v) for k, v in support.items()},
        fixed=dict(epochs=3, accumulation_articles=16, optimizer_steps=45, optimizer='AdamW', betas=[0.9, 0.999], eps=1e-8,
            lora_rank=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'],
            precision='BF16 base, FP32 adapters and AdamW states', scheduler='linear decay',
            loss='inverse-class weighted article A/B/C cross entropy; weights computed from all 240 Train rows',
            context='Full articles, <=4096 article/4608 prompt tokens; reject overflow',
            checkpoint='Final epoch only; no scoring or checkpoint selection during fitting',
            initialization='Fresh original pretrained base plus fresh LoRA; no continuation or averaging of ablation adapters'),
        evaluation=dict(final_test_used=False, independent_quality_score=None,
            former_development_rows_now_in_fit=True, smoke_only=True, smoke_texts=SMOKE_TEXTS,
            warning='The 0.7742 source development score is not a measured score for this full-Train artifact.'),
        serving_selection_changed=False, mlflow_experiment=EXPERIMENT,
        dependencies=ablation['dependencies'])
    write(PROTOCOL, protocol, immutable=True)
    print(json.dumps({'recipe': selected['recipe'], 'seed': SEED, 'full_train_rows': 240, 'steps': 45}), flush=True)


def verify(check_artifacts=False):
    from src.studies.optimizer_ablation.runner import verify as verify_ablation
    from src.tracking import fingerprint
    from src.next_stage_experiment import BASE
    protocol = json.loads(PROTOCOL.read_text())
    ablation = verify_ablation()
    verify_files(ROOT, protocol['files_sha256'])
    summary = json.loads((ROOT/'reports/optimizer_ablation/summary-v1.json').read_text())
    selected, _ = choose_recipe(summary['runs'])
    if protocol['selected_recipe'] != selected['recipe'] or protocol['seed'] != SEED:
        raise RuntimeError('Declared recipe or fixed seed changed')
    if protocol['source_sha256'] != source_hash() or protocol['source_ablation_sha256'] != ablation['study_source_sha256']:
        raise RuntimeError('Frozen candidate or source-study implementation changed')
    if any(importlib.metadata.version(k) != v for k, v in protocol['dependencies'].items()):
        raise RuntimeError('Dependencies changed')
    if check_artifacts and (protocol['base_artifact_sha256'] != fingerprint(BASE)
            or protocol['baseline_adapter_sha256'] != fingerprint(ROOT / protocol['baseline_adapter_path'])):
        raise RuntimeError('Base or current API adapter changed')
    return protocol


def train():
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            _fit()
        except BaseException as exc:
            write(LOCAL / 'status.json', dict(state='failed', updated_at=now(), error=str(exc)))
            raise


def _fit():
    import random
    import mlflow
    import numpy as np
    import pandas as pd
    import torch
    from transformers import get_linear_schedule_with_warmup
    from src.config import LABELS
    from src.next_stage_experiment import TRAIN, BASE, lora, qwen_tokens, qwen_profile, gpu, clean
    from src.tracking import fingerprint
    protocol = verify(check_artifacts=True)
    if ART.exists() or (OUT / 'result-v1.json').exists():
        raise RuntimeError('Candidate already exists or is partial; never overwrite or automatically retry')
    gpu()
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError('At least 12 GiB free CUDA memory required')
    recipe = protocol['selected_recipe']; seed = protocol['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    table = pd.read_csv(TRAIN)
    assert len(table) == protocol['train_rows']
    assert hashlib.sha256('\n'.join(table.id).encode()).hexdigest() == protocol['train_ids_sha256']
    ART.mkdir(parents=True, exist_ok=False)
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=VERSION + ' · full Train') as run:
        mlflow.set_tags(dict(stage='post-test-full-train-candidate', test_used='false', serving_selection_changed='false',
            independent_quality_evaluated='false', protocol_sha256=sha(PROTOCOL), source_sha256=source_hash(),
            source_study='article-optimizer-ablation-v1', model_version=VERSION))
        mlflow.log_params(dict(recipe, seed=seed, train_rows=len(table), **protocol['fixed']))
        mlflow.log_artifact(str(PROTOCOL), artifact_path='protocol')
        write(LOCAL / 'status.json', dict(state='training', pid=os.getpid(), mlflow_run_id=run.info.run_id, started_at=now()))
        tokenizer, net = lora()
        tokens = [qwen_tokens(tokenizer, text) for text in table.text]
        choices = [tokenizer.encode(c, add_special_tokens=False)[0] for c in 'ABC']
        if len(set(choices)) != 3 or any(len(tokenizer.encode(c, add_special_tokens=False)) != 1 for c in 'ABC'):
            raise RuntimeError('A/B/C must be distinct single tokens')
        params = [p for p in net.parameters() if p.requires_grad]
        if any(p.dtype != torch.float32 for p in params):
            raise RuntimeError('Expected FP32 adapters')
        labels = np.array([LABELS.index(v) for v in table.label])
        weights = len(table) / (3 * np.bincount(labels, minlength=3))
        optimizer = torch.optim.AdamW(params, lr=recipe['learning_rate'], weight_decay=recipe['weight_decay'], betas=(0.9, 0.999), eps=1e-8)
        steps = 3 * math.ceil(len(table)/16)
        assert steps == protocol['fixed']['optimizer_steps']
        scheduler = get_linear_schedule_with_warmup(optimizer, int(recipe['warmup_fraction']*steps), steps)
        telemetry = []; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
        for epoch in range(3):
            net.train(); order = np.random.default_rng(seed+epoch).permutation(len(table))
            for start in range(0, len(table), 16):
                batch = order[start:start+16]; optimizer.zero_grad(set_to_none=True); batch_loss = 0.0
                for i in batch:
                    logits = net(**tokens[int(i)].to('cuda'), use_cache=False, logits_to_keep=1).logits[0, -1, choices].float()
                    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([labels[i]], device='cuda')) * float(weights[labels[i]])
                    (loss/len(batch)).backward(); batch_loss += float(loss.detach())/len(batch)
                norm = float(torch.nn.utils.clip_grad_norm_(params, recipe['gradient_clip'], error_if_nonfinite=True))
                before = [p.detach().clone() for p in params]; rate = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                update = float(torch.sqrt(sum((p.detach()-old).square().sum() for p, old in zip(params, before))))
                del before
                torch.cuda.synchronize()
                event = dict(step=len(telemetry)+1, epoch=epoch+1, train_loss=batch_loss, gradient_norm=norm,
                    clipped=int(norm > recipe['gradient_clip']), learning_rate=rate, adapter_update_l2=update)
                telemetry.append(event)
                mlflow.log_metrics({k:v for k,v in event.items() if k not in ('step','epoch')}, step=event['step'])
                write(LOCAL / 'current-step.json', dict(updated_at=now(), total_steps=steps, **event))
                print(json.dumps(event), flush=True)
        torch.cuda.synchronize(); seconds = time.perf_counter()-started; peak = torch.cuda.max_memory_allocated()
        net.eval(); net.save_pretrained(ART / 'adapter', safe_serialization=True)
        qwen_profile(ART / 'profile', ART / 'adapter')
        profile_path = ART / 'profile/artifact.json'
        meta = json.loads(profile_path.read_text())
        meta.update(model_version=VERSION, label_status='Project-reviewed 2026-11-14; original annotation provenance retained',
            candidate_status='exploratory-full-train-no-independent-evaluation', candidate_protocol_sha256=sha(PROTOCOL),
            train_rows=len(table), independent_quality_score=None)
        write(profile_path, meta)
        write(OUT / 'telemetry-v1.json', telemetry, immutable=True)
        verify(check_artifacts=True)
        measured = dict(training_seconds=seconds, optimizer_steps=steps, train_rows=len(table),
            peak_training_allocated_bytes=peak, clip_fraction=float(np.mean([t['clipped'] for t in telemetry])),
            final_train_loss=float(np.mean([t['train_loss'] for t in telemetry if t['epoch']==3])))
        result = dict(version=VERSION, completed_at=now(), mlflow_run_id=run.info.run_id, mlflow_experiment=EXPERIMENT,
            protocol_sha256=sha(PROTOCOL), recipe=recipe, seed=seed, metrics=measured,
            model_path=str((ART/'profile').relative_to(ROOT)), adapter_path=str((ART/'adapter').relative_to(ROOT)),
            adapter_sha256=fingerprint(ART/'adapter'), profile_sha256=sha(profile_path),
            telemetry_sha256=sha(OUT/'telemetry-v1.json'), base_artifact_sha256=protocol['base_artifact_sha256'],
            independently_evaluated=False, quality_score=None, final_test_used=False, serving_selection_changed=False)
        mlflow.log_metrics(measured)
        mlflow.log_artifact(str(OUT/'telemetry-v1.json'), artifact_path='training')
        mlflow.log_artifact(str(profile_path), artifact_path='profile')
        mlflow.log_dict(result, 'result.json')
        write(OUT / 'result-v1.json', result, immutable=True)
        write(LOCAL / 'status.json', dict(state='complete', completed_at=now(), mlflow_run_id=run.info.run_id))
        print('Completed candidate: '+VERSION, flush=True)
        del net, tokenizer, tokens, optimizer, scheduler, logits, loss; clean()


def smoke():
    import numpy as np
    from src.config import LABELS
    from src.serving import load_artifact
    from src.tracking import fingerprint
    verify(check_artifacts=True)
    result = json.loads((OUT/'result-v1.json').read_text())
    if result['adapter_sha256'] != fingerprint(ART/'adapter') or result['profile_sha256'] != sha(ART/'profile/artifact.json'):
        raise RuntimeError('Candidate artifact changed')
    if (OUT/'smoke-v1.json').exists():
        raise RuntimeError('Smoke evidence already exists; do not overwrite')
    with (ROOT/'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        predictor = load_artifact(ART/'profile')
        records = []
        for language, text in SMOKE_TEXTS.items():
            scores = predictor.predict_proba([text])[0]
            if not (np.isfinite(scores).all() and (scores>=0).all() and np.isclose(scores.sum(), 1, atol=1e-6)):
                raise RuntimeError('Invalid synthetic smoke scores')
            records.append(dict(language=language, text=text, label=LABELS[int(np.argmax(scores))], scores=[float(x) for x in scores]))
        write(OUT/'smoke-v1.json', dict(checked_at=now(), model_version=predictor.version, engineering_only=True,
            independent_quality_evaluation=False, synthetic_examples=True, results=records), immutable=True)
        verify(check_artifacts=True)
        print(json.dumps(records, ensure_ascii=False), flush=True)


def launch():
    verify()
    if ART.exists() or (OUT/'result-v1.json').exists():
        raise RuntimeError('Candidate already exists or is partial')
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (ROOT/'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    with (LOCAL/'worker.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', MODULE, 'train'], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(LOCAL/'launch.json', dict(pid=process.pid, launched_at=now()))
    print(json.dumps(dict(pid=process.pid, log=str(LOCAL/'worker.log'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize','launch','train','smoke','verify','status'])
    args = parser.parse_args()
    if args.action == 'status':
        for path in [LOCAL/'status.json', LOCAL/'current-step.json', OUT/'result-v1.json']:
            if path.exists(): print(path.name, path.read_text().strip())
    else:
        {'initialize':initialize,'launch':launch,'train':train,'smoke':smoke,'verify':verify}[args.action]()
