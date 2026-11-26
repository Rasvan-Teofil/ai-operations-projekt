"""Paired grouped Train-only validation; isolated clipping control, no final Test."""
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

from .design import RECIPES, SEEDS, REUSE, configurations, validate_folds, paired_analysis
from src.studies.optimizer_ablation.design import validate_membership
from src.studies.optimizer_ablation.runner import sha, write, evaluate

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/optimizer_validation'
LOCAL = ROOT / 'runtime/optimizer_validation'
ART = ROOT / 'models/optimizer_validation'
PROTOCOL = OUT / 'protocol-v1.json'
EXPERIMENT = 'article-optimizer-validation-v1'
MODULE = 'src.studies.optimizer_validation'


def now(): return datetime.now(timezone.utc).isoformat()


def source_hash():
    return hashlib.sha256('\n'.join(p.name+':'+sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()


def initialize():
    import pandas as pd
    from src.next_stage_experiment import TRAIN, code_hash
    from src.studies.optimizer_ablation.runner import verify as source_verify
    from src.studies.optimizer_candidate.runner import verify as candidate_verify
    from src.tracking import fingerprint
    original=source_verify(check_base=True); candidate=candidate_verify(check_artifacts=True)
    table=pd.read_csv(TRAIN)
    folds=json.loads((ROOT/'reports/next_stage/experiment-membership-v1.json').read_text())
    validate_folds(table[['id','group','language','label']].to_dict('records'),folds)
    inputs=[TRAIN, ROOT/'reports/next_stage/experiment-membership-v1.json',
            ROOT/'reports/next_stage/final-test-evaluation.lock',ROOT/'models/serving_selected.json',
            ROOT/'reports/optimizer_candidate/result-v1.json',ROOT/'reports/optimizer_ablation/summary-v1.json']
    reuse=[]
    for (fold,seed,recipe),old_id in REUSE.items():
        path=ROOT/f'reports/optimizer_ablation/runs/{old_id}.json';r=json.loads(path.read_text())
        if r['config']['seed']!=seed or any(r['config'][k]!=v for k,v in RECIPES[recipe].items()):
            raise RuntimeError('Reuse configuration differs')
        if fold!=0 or original['fit_ids']!=folds[0]['fit_ids'] or original['dev_ids']!=folds[0]['dev_ids']:
            raise RuntimeError('Reuse membership differs')
        evidence=[path,ROOT/f'reports/optimizer_ablation/predictions/{old_id}.json',ROOT/f'reports/optimizer_ablation/telemetry/{old_id}.json']
        inputs.extend(evidence)
        reuse.append(dict(config_id=f'f{fold}-s{seed}-{recipe}',source_id=old_id,original_mlflow_run_id=r['mlflow_run_id']))
    protocol=dict(version='optimizer-validation-v1',created_at=now(),status_at_freeze='Before any new validation fit; six exact prior fold-0 fits deliberately reused',
        purpose='Exploratory post-selection Train-only paired recipe and clipping validation',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256=source_hash(),frozen_runtime_sha256=code_hash(),ablation_source_sha256=original['study_source_sha256'],
        files_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},base_artifact_sha256=original['base_artifact_sha256'],
        baseline_adapter_path=candidate['baseline_adapter_path'],baseline_adapter_sha256=candidate['baseline_adapter_sha256'],
        candidate_adapter_path='models/optimizer_candidate/v1/adapter',
        candidate_adapter_sha256=fingerprint(ROOT/'models/optimizer_candidate/v1/adapter'),
        recipes=RECIPES,seeds=list(SEEDS),folds=folds,runs=configurations(),expected_records=27,expected_new_fits=21,reuse=reuse,
        train_rows=len(table),group_count=int(table.group.nunique()),
        fixed=original['fixed'],dependencies=original['dependencies'],mlflow_experiment=EXPERIMENT,
        primary='Pool all three held-out folds per seed and recipe, then mean(DE macro-F1, EN macro-F1); average three fixed seeds',
        comparisons=dict(recipe='candidate minus baseline: joint recipe effect; both have clipping=1',clipping='candidate minus candidate_no_clip: identical LR, decay, warmup, seed, fit/dev membership; only clipping differs'),
        analysis=dict(bootstrap_samples=2000,bootstrap_seed=20261126,unit='semantic group, paired across all arms and seeds',interval='2.5/97.5 percentile, conditional exploratory interval',
                      sensitivity='Recompute pooled scores excluding selection fold 0; folds 1+2 only',
                      recipe_gate='Mean primary gain >=1pp, no negative seed primary delta, no mean language loss >1pp',
                      clipping_support='Mean primary clipping delta>0, each seed primary clipping delta>0, conditional group-bootstrap lower primary bound>0'),
        limits=['Recipe selected on fold 0; post-selection CV is not independent Test evidence.',
                'Shared Train articles and three seeds do not constitute independent samples; bootstrap does not account for selection or all training uncertainty.',
                'Six reused fits are linked to original runs, not counted as new training.',
                'Validation measures recipes fitted on 159/160/161 rows, not the saved 240-row artifact directly.',
                'No final-Test inference, no tuning after seeing validation, no automatic serving promotion.'],
        final_test_used=False,serving_selection_changed=False)
    write(PROTOCOL,protocol,immutable=True)
    print('Frozen 27-condition validation: 6 exact reuses +21 new fits; no Test inference',flush=True)


def verify(check_base=False):
    from src.next_stage_experiment import code_hash,BASE
    from src.studies.optimizer_ablation.runner import verify as original_verify
    from src.tracking import fingerprint
    p=json.loads(PROTOCOL.read_text());original_verify()
    if p['source_sha256']!=source_hash() or p['frozen_runtime_sha256']!=code_hash() or p['runs']!=configurations():
        raise RuntimeError('Frozen validation source or configuration changed')
    for name,value in p['files_sha256'].items():
        if sha(ROOT/name)!=value:raise RuntimeError('Frozen validation input changed: '+name)
    if any(importlib.metadata.version(k)!=v for k,v in p['dependencies'].items()):raise RuntimeError('Dependencies changed')
    if check_base:
        if fingerprint(BASE)!=p['base_artifact_sha256']:raise RuntimeError('Base changed')
        for kind in ('baseline','candidate'):
            if fingerprint(ROOT/p[kind+'_adapter_path'])!=p[kind+'_adapter_sha256']:raise RuntimeError('Saved model adapter changed')
    return p


def reuse_one(config,protocol):
    source=next(r for r in protocol['reuse'] if r['config_id']==config['id']);old_id=source['source_id']
    old=json.loads((ROOT/f'reports/optimizer_ablation/runs/{old_id}.json').read_text())
    for kind in ('predictions','telemetry'):
        src=ROOT/f'reports/optimizer_ablation/{kind}/{old_id}.json'
        if sha(src)!=old[kind+'_sha256']:raise RuntimeError('Reused evidence hash mismatch')
        target=OUT/kind/(config['id']+'.json');write(target,json.loads(src.read_text()),immutable=True)
        if sha(target)!=sha(src):raise RuntimeError('Reused evidence is not byte-identical')
    member=next(m for m in protocol['folds'] if m['fold']==config['fold'])
    result={k:v for k,v in old.items() if k!='gpu'}
    result.update(config=config,protocol_sha256=sha(PROTOCOL),evidence_origin='reused-exact-existing-fit',
        original_run_id=old_id,source_result_sha256=sha(ROOT/f'reports/optimizer_ablation/runs/{old_id}.json'),
        source_protocol_sha256=old['protocol_sha256'],fit_ids=member['fit_ids'],dev_ids=member['dev_ids'])
    write(OUT/'runs'/(config['id']+'.json'),result,immutable=True)


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
    member = next(m for m in protocol['folds'] if m['fold'] == config['fold'])
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
    validate_membership(table[['id', 'group', 'language', 'label']].to_dict('records'), member)
    fit = table[table.id.isin(member['fit_ids'])].reset_index(drop=True)
    dev = table[table.id.isin(member['dev_ids'])].reset_index(drop=True)
    mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
    mlflow.set_experiment(EXPERIMENT)
    folder.mkdir(parents=True, exist_ok=False)
    with mlflow.start_run(run_name=f'{run_id} · seed {config["seed"]}') as run:
        mlflow.set_tags({'phase': 'optimizer-validation-v1', 'stage': 'post-test-train-only-development',
                         'test_used': 'false', 'serving_selection_changed': 'false', 'fold': str(config['fold']), 'recipe': config['recipe'],
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
                      predictions_sha256=sha(prediction_path), telemetry_sha256=sha(OUT / 'telemetry' / f'{run_id}.json'), evidence_origin='new-fit', fit_ids=member['fit_ids'], dev_ids=member['dev_ids'])
        mlflow.log_metrics({k: v for k, v in measured.items() if isinstance(v, (float, int))})
        for path in (prediction_path, OUT / 'telemetry' / f'{run_id}.json'):
            mlflow.log_artifact(str(path), artifact_path=path.parent.name)
        mlflow.log_dict(result, 'result.json')
        write(result_path, result, immutable=True)
        print(f'COMPLETED {run_id}: primary F1={measured["primary_score"]:.4f}', flush=True)


def collect(state='running'):
    import pandas as pd
    from src.next_stage_experiment import TRAIN
    protocol=verify();table=pd.read_csv(TRAIN);results=[];predictions={}
    for config in protocol['runs']:
        p=OUT/'runs'/(config['id']+'.json')
        if not p.exists():continue
        r=json.loads(p.read_text())
        if r['config']!=config or r['protocol_sha256']!=sha(PROTOCOL):raise RuntimeError('Result identity differs')
        member=next(m for m in protocol['folds'] if m['fold']==config['fold'])
        records=json.loads((OUT/'predictions'/(config['id']+'.json')).read_text())
        if len(records)!=len(member['dev_ids']) or {x['id'] for x in records}!=set(member['dev_ids']):raise RuntimeError('Held-out prediction coverage differs')
        source=table.set_index('id')
        for row in records:
            ref=source.loc[row['id']]
            if (row['reference'],row['language'],row['text_sha256'])!=(ref.label,ref.language,ref.text_sha256):raise RuntimeError('Prediction reference differs')
        for kind in ('predictions','telemetry'):
            if r[kind+'_sha256']!=sha(OUT/kind/(config['id']+'.json')):raise RuntimeError('Evidence changed')
        from .design import score
        scored=score(records)
        if any(abs(scored[k]-r['metrics'][k])>1e-12 for k in scored):raise RuntimeError('Saved metrics do not reproduce')
        results.append(r);predictions[config['id']]=records
    analysis=paired_analysis(results,predictions,table.id.tolist(),dict(zip(table.id,table.group)))
    summary=dict(version='optimizer-validation-v1',updated_at=now(),state=state,completed_records=len(results),expected_records=27,
        new_fits=sum(r['evidence_origin']=='new-fit' for r in results),reused_fits=sum(r['evidence_origin']!='new-fit' for r in results),
        protocol_sha256=sha(PROTOCOL),analysis=analysis,runs=results,final_test_used=False,serving_selection_changed=False)
    write(OUT/'summary-v1.json',summary)
    return summary


def restore_api():
    import socket
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',8000))==0:return
    if not (LOCAL/'api-paused.json').exists():return
    with (LOCAL/'api-restored.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000','--workers','1'],cwd=ROOT,
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(LOCAL/'api-restored.json',dict(pid=p.pid,started_at=now()))


def worker():
    LOCAL.mkdir(parents=True,exist_ok=True)
    with (ROOT/'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            protocol=verify(check_base=True)
            for reused in protocol['reuse']:
                config=next(c for c in protocol['runs'] if c['id']==reused['config_id'])
                if not (OUT/'runs'/(config['id']+'.json')).exists():reuse_one(config,protocol)
            collect()
            for config in protocol['runs']:
                path=OUT/'runs'/(config['id']+'.json')
                if path.exists():collect();continue
                if (LOCAL/'STOP').exists():
                    collect('stopped');write(LOCAL/'worker.json',dict(state='stopped',updated_at=now()));return
                write(LOCAL/'worker.json',dict(state='running',pid=os.getpid(),current_run=config['id'],updated_at=now()))
                if any(r['config_id']==config['id'] for r in protocol['reuse']):reuse_one(config,protocol)
                else:subprocess.run([sys.executable,'-m',MODULE,'one','--run-id',config['id']],cwd=ROOT,check=True)
                collect()
            collect('complete')
            write(LOCAL/'worker.json',dict(state='complete',completed_at=now()))
        except BaseException as exc:
            traceback.print_exc();write(LOCAL/'worker.json',dict(state='failed',updated_at=now(),error=str(exc)));raise
        finally:restore_api()


def launch():
    LOCAL.mkdir(parents=True,exist_ok=True);verify(check_base=True)
    with (ROOT/'runtime/gpu-training.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (LOCAL/'worker.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-m',MODULE,'worker'],cwd=ROOT,
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    write(LOCAL/'launch.json',dict(pid=p.pid,launched_at=now()));print('Worker launched',p.pid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['initialize','launch','worker','one','collect','verify'])
    parser.add_argument('--run-id');args=parser.parse_args()
    if args.action=='one':
        if not args.run_id:parser.error('--run-id is required')
        one(args.run_id)
    else:{'initialize':initialize,'launch':launch,'worker':worker,'collect':collect,'verify':verify}[args.action]()
