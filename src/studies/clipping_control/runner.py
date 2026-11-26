"""Finish missing clipping-only controls, preserving every original fit."""
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

from .design import configurations, pairs, validate_pair, analyze_cv, METRICS
from src.studies.optimizer_ablation.design import validate_membership
from src.studies.optimizer_ablation.runner import sha, write, evaluate

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/clipping_control'
LOCAL = ROOT / 'runtime/clipping_control'
ART = ROOT / 'models/clipping_control'
PROTOCOL = OUT / 'protocol-v1.json'
EXPERIMENT = 'article-gradient-clipping-controls-v1'
MODULE = 'src.studies.clipping_control'


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hash():
    return hashlib.sha256('\n'.join(p.name + ':' + sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))).encode()).hexdigest()


def initialize():
    from src.studies.optimizer_validation.runner import verify as verify_previous
    previous = verify_previous(check_base=True)
    paths = {ROOT / name for name in previous['files_sha256']}
    paths.add(ROOT / 'scripts/check_clipping_control.py')
    for study in ('optimizer_ablation', 'optimizer_validation'):
        paths.update((ROOT / 'src/studies' / study).glob('*.py'))
        for name in ('protocol-v1.json', 'summary-v1.json', 'verification-v1.json', 'engineering-amendment-v1.json'):
            p = ROOT / 'reports' / study / name
            if p.exists():
                paths.add(p)
    for pair in pairs():
        for arm in ('with_clipping', 'without_clipping'):
            for name in pair[arm].values():
                path = ROOT / name
                if path.exists():
                    paths.add(path)
                elif 'reports/clipping_control/' not in name:
                    raise RuntimeError('Missing pre-existing fit evidence: ' + name)
    protocol = dict(version='gradient-clipping-controls-v1', created_at=now(),
        status_at_freeze='Before any new clipping-control fit; all previous optimizer-study results are already known',
        purpose='Complete every missing without-clipping partner in the two existing scored optimizer studies',
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=source_hash(), frozen_runtime_sha256=previous['frozen_runtime_sha256'],
        files_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
        base_artifact_sha256=previous['base_artifact_sha256'],
        baseline_adapter_path=previous['baseline_adapter_path'], baseline_adapter_sha256=previous['baseline_adapter_sha256'],
        candidate_adapter_path=previous['candidate_adapter_path'], candidate_adapter_sha256=previous['candidate_adapter_sha256'],
        dependencies=previous['dependencies'], fixed=previous['fixed'], folds=previous['folds'],
        seeds=previous['seeds'], runs=configurations(), pairs=pairs(), train_rows=previous['train_rows'],
        unique_pairs=30, expected_unique_fits=60, existing_unique_fits=53, expected_new_fits=7,
        comparisons=dict(sign='WITH clipping minus WITHOUT clipping; positive values favor clipping',
            factorial='All 16 existing matched pairs, same fold 0, eight LR/decay/warmup settings and two seeds',
            cv='Separately for baseline and candidate: pool all three held-out folds before F1 for each seed, then average three fixed seeds',
            overlap='Four pairs are shared by the factorial and CV views; no combined grand mean or independent-sample count',
            primary='Equal mean of DE macro-F1 and EN macro-F1; report DE, EN and JA separately',
            sensitivity='Pool only folds 1+2, excluding the fold used for recipe selection',
            uncertainty='2000 paired semantic-group bootstrap draws, RNG 20261126; 2.5/97.5 percentiles conditional on the same fixed fits and seeds',
            clipping_support='Positive mean primary delta AND positive primary delta in all three seeds AND conditional lower primary bound >0'),
        scope='Scored Qwen-LoRA optimizer-study fits only. Full-Train deployment adapters have no held-out development set and are not counted as CV experiments. No other model family or historical study is retrained.',
        initialization='Fresh original pretrained base and fresh LoRA per fit; identical implementation/order to original validation; only clipping threshold is disabled',
        limits=['Exploratory follow-up after all previous results were known, not a preregistered independent confirmation.',
                'Same small Train set; folds and seeds are dependent. Bootstrap does not correct selection or all training uncertainty.',
                'No new hyperparameter search; no final-Test inference; no accuracy claim for the saved full-Train adapters.',
                'No automatic serving change and no inference that clipping must help.'],
        mlflow_experiment=EXPERIMENT, final_test_used=False, serving_selection_changed=False)
    write(PROTOCOL, protocol, immutable=True)
    print('Frozen: 30 distinct clipping pairs, 53 existing fits +7 missing controls', flush=True)


def verify(check_base=False):
    from src.studies.optimizer_validation.runner import verify as verify_previous
    from src.next_stage_experiment import code_hash, BASE
    from src.tracking import fingerprint
    verify_previous()
    p = json.loads(PROTOCOL.read_text())
    if p['source_sha256'] != source_hash() or p['frozen_runtime_sha256'] != code_hash():
        raise RuntimeError('Frozen clipping-control source changed')
    if p['runs'] != configurations() or p['pairs'] != pairs():
        raise RuntimeError('Frozen clipping-control design changed')
    for name, value in p['files_sha256'].items():
        if sha(ROOT / name) != value:
            raise RuntimeError('Frozen clipping-control input changed: ' + name)
    if any(importlib.metadata.version(k) != v for k, v in p['dependencies'].items()):
        raise RuntimeError('Dependencies changed')
    if check_base:
        if fingerprint(BASE) != p['base_artifact_sha256']:
            raise RuntimeError('Base changed')
        for kind in ('baseline', 'candidate'):
            if fingerprint(ROOT / p[kind + '_adapter_path']) != p[kind + '_adapter_sha256']:
                raise RuntimeError('Saved serving adapter changed')
    return p


def load_evidence(ref, protocol, allow_missing=False):
    import numpy as np
    from src.studies.optimizer_validation.design import score
    path = ROOT / ref['runs']
    if not path.exists():
        if allow_missing and ref['runs'].startswith('reports/clipping_control/'):
            return None
        raise RuntimeError('Missing result: ' + str(path))
    r = json.loads(path.read_text())
    config = r['config']
    if ref['runs'].startswith('reports/clipping_control/'):
        if config not in protocol['runs'] or r['protocol_sha256'] != sha(PROTOCOL):
            raise RuntimeError('New control identity differs')
    records = json.loads((ROOT / ref['predictions']).read_text())
    events = json.loads((ROOT / ref['telemetry']).read_text())
    for kind in ('predictions', 'telemetry'):
        if sha(ROOT / ref[kind]) != r[kind + '_sha256']:
            raise RuntimeError('Changed evidence: ' + ref[kind])
    member = next(m for m in protocol['folds'] if m['fold'] == config.get('fold', 0))
    if len(records) != len(member['dev_ids']) or {x['id'] for x in records} != set(member['dev_ids']):
        raise RuntimeError('Wrong held-out coverage')
    for row in records:
        scores = row['scores']
        if (len(scores) != 3 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
                or abs(sum(scores) - 1) > 1e-5
                or ('negative', 'neutral', 'positive')[int(np.argmax(scores))] != row['prediction']):
            raise RuntimeError('Invalid prediction scores')
    computed = score(records)
    if any(abs(computed[k] - r['metrics'][k]) > 1e-12 for k in computed):
        raise RuntimeError('Prediction metrics differ')
    steps = 3 * math.ceil(len(member['fit_ids']) / 16)
    warmup = int(config['warmup_fraction'] * steps)
    if len(events) != steps or [e['step'] for e in events] != list(range(1, steps + 1)):
        raise RuntimeError('Incomplete optimizer telemetry')
    for event in events:
        if any(not math.isfinite(event[k]) for k in ('train_loss', 'gradient_norm', 'learning_rate', 'adapter_update_l2')):
            raise RuntimeError('Nonfinite training telemetry')
        expected_clip = int(config['gradient_clip'] is not None and event['gradient_norm'] > config['gradient_clip'])
        s = event['step'] - 1
        rate = config['learning_rate'] * (s / max(1, warmup) if s < warmup else (steps - s) / max(1, steps - warmup))
        if event['clipped'] != expected_clip or not math.isclose(event['learning_rate'], rate, rel_tol=1e-10, abs_tol=1e-12):
            raise RuntimeError('Clipping flag or learning-rate schedule differs')
    return r, records, events


def result_relative(run_id):
    return Path('reports/clipping_control/runs') / (run_id + '.json')


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
    pair = next(p for p in protocol['pairs'] if p['without_clipping']['runs'] == str(result_relative(run_id)))
    paired = load_evidence(pair['with_clipping'], protocol)[0]
    result_path = OUT / 'runs' / f'{run_id}.json'
    folder = ART / run_id
    if result_path.exists() or folder.exists():
        raise RuntimeError('Existing run result/checkpoint requires explicit recovery audit; never overwrite')
    gpu()
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
        mlflow.set_tags({'phase': 'gradient-clipping-controls-v1', 'paired_with_clipping_run_id': paired['mlflow_run_id'],
                         'comparison': 'Identical recipe, split and seed; only clipping disabled', 'stage': 'post-test-train-only-development',
                         'test_used': 'false', 'serving_selection_changed': 'false', 'fold': str(config['fold']), 'recipe': config['recipe'],
                         'model_family': 'Qwen-LoRA', 'label_status': 'Project-reviewed 2026-11-14; original annotation provenance retained',
                         'protocol_sha256': sha(PROTOCOL), 'source_sha256': source_hash(),
                         'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                         'base_artifact_sha256': protocol['base_artifact_sha256'], 'train_sha256': protocol['files_sha256'][str(TRAIN.relative_to(ROOT))]})
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
        validate_pair(pair, paired, result)
        result['paired_comparison'] = dict(with_clipping_run_id=paired['mlflow_run_id'],
            with_minus_without={m: paired['metrics'][m] - measured[m] for m in METRICS})
        mlflow.log_metrics(dict(paired_with_clipping_primary_score=paired['metrics']['primary_score'],
            paired_without_clipping_primary_score=measured['primary_score'],
            with_minus_without_primary_pp=100*(paired['metrics']['primary_score']-measured['primary_score'])))
        mlflow.log_dict(result, 'result.json')
        write(result_path, result, immutable=True)
        print(f'COMPLETED {run_id}: primary F1={measured["primary_score"]:.4f}', flush=True)


def collect(state='running'):
    import pandas as pd
    from src.next_stage_experiment import TRAIN
    protocol = verify()
    table = pd.read_csv(TRAIN).set_index('id')
    rows = []; cv_predictions = {}; run_ids = set(); missing = []
    for pair in protocol['pairs']:
        on = load_evidence(pair['with_clipping'], protocol)
        off = load_evidence(pair['without_clipping'], protocol, allow_missing=True)
        for bundle in (on, off):
            if bundle is None:
                continue
            r, records, _ = bundle
            run_ids.add(r['mlflow_run_id'])
            for record in records:
                t = table.loc[record['id']]
                if (record['reference'], record['language'], record['text_sha256']) != (t.label, t.language, t.text_sha256):
                    raise RuntimeError('Reference metadata differs')
        row = dict(pair, status='complete' if off else 'pending', with_mlflow_run_id=on[0]['mlflow_run_id'],
                   with_metrics={m: on[0]['metrics'][m] for m in METRICS})
        if off:
            validate_pair(pair, on[0], off[0])
            row.update(without_mlflow_run_id=off[0]['mlflow_run_id'],
                without_metrics={m: off[0]['metrics'][m] for m in METRICS},
                with_minus_without={m: on[0]['metrics'][m] - off[0]['metrics'][m] for m in METRICS})
        else:
            missing.append(pair['without_clipping']['runs'])
        if pair['cv_recipe']:
            for arm, bundle in (('with', on), ('without', off)):
                if bundle:
                    cv_predictions[pair['cv_recipe'], pair['config']['seed'], pair['config']['fold'], arm] = bundle[1]
        rows.append(row)
    analysis = analyze_cv(cv_predictions, table.group.to_dict())
    done = sum((OUT / 'runs' / (c['id'] + '.json')).exists() for c in protocol['runs'])
    summary = dict(version='gradient-clipping-controls-v1', updated_at=now(), state=state,
        completed_new_fits=done, expected_new_fits=7, completed_pairs=sum(r['status'] == 'complete' for r in rows),
        expected_pairs=30, distinct_successful_mlflow_run_ids=len(run_ids), missing=missing,
        protocol_sha256=sha(PROTOCOL), pairs=rows, cv_analysis=analysis,
        final_test_used=False, serving_selection_changed=False)
    write(OUT / 'summary-v1.json', summary)
    render_report(summary)
    return summary


def render_report(summary):
    import csv
    import io
    lines = ['# Gradient Clipping: vollständige Gegenläufe', '',
        f'**{summary["completed_new_fits"]}/7 neue Fits, {summary["completed_pairs"]}/30 vollständige Paare.**', '',
        'Jeder Lauf mit Clipping erhält einen ansonsten identischen Gegenlauf ohne Clipping. '
        'Lernrate, AdamW Weight Decay, Warmup, Seed, Datenaufteilung, LoRA und Trainingsdauer bleiben gleich. '
        'Vorhandene exakt passende Fits werden wiederverwendet, keine Messung wird erfunden oder überschrieben.', '',
        'Die erste 32-Fit-Serie enthält bereits 16 Paare. Die spätere CV umfasst zwei Rezepte mit je neun Paaren; '
        'vier Paare überlappen die erste Serie. Insgesamt sind es 30 verschiedene Paare / 60 Fits: '
        '53 vorhandene und sieben neue. Fehlend waren nur sieben Ohne-Clipping-Fits des bisherigen Rezepts.', '',
        '**Vorzeichen überall: mit minus ohne Clipping. Positive Werte sprechen für Clipping.** '
        'F1-Unterschiede stehen in absoluten Prozentpunkten.', '',
        '## Gleiche Rezepte, jeweils mit und ohne Clipping', '',
        '| Rezept | Lernrate | Weight Decay | Warmup | Clipping-Stufen |',
        '|---|---:|---:|---:|---|',
        '| Bisheriges Rezept | 0.0001 | 0.01 | 0.1 | 1.0 / aus |',
        '| Neues Rezept | 0.00005 | 0 | 0 | 1.0 / aus |', '',
        'Drei gruppierte Train-Folds und drei feste Seeds (20261110/11/12). '
        'Pro Seed werden die drei zurückgehaltenen Folds auf allen 240 Artikeln gepoolt, dann Macro-F1 berechnet. '
        'Hauptwert ist das gleichgewichtete DE/EN-Mittel; Japanisch wird separat ausgewiesen. '
        'Kein Mittel über die beiden Rezepte als vermeintlich allgemeiner Clipping-Effekt.', '']
    analysis = summary['cv_analysis']
    if analysis['status'] != 'complete':
        lines += ['**Die vollständige CV-Auswertung steht noch aus. Es gibt noch kein neues Gesamtergebnis.**', '']
    else:
        lines += ['## Vollständige gepaarte CV-Ergebnisse', '',
            '| Rezept | DE/EN mit | DE/EN ohne | Δ DE/EN [95%-Intervall] (pp) | Δ DE | Δ EN | Δ JA |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name, values in analysis['recipes'].items():
            delta = values['mean_with_minus_without']; lo, hi = values['conditional_group_bootstrap_95_intervals']['primary_score']
            lines.append(f'| {name} | {values["mean_oof"]["with"]["primary_score"]:.4f} | {values["mean_oof"]["without"]["primary_score"]:.4f} '
                f'| {100*delta["primary_score"]:+.2f} [{100*lo:+.2f}; {100*hi:+.2f}] '
                + ''.join(f'| {100*delta["macro_f1_"+lang]:+.2f} ' for lang in ('de', 'en', 'ja')) + '|')
        lines += ['', '| Rezept | Seed | Δ DE/EN (pp) |', '|---|---:|---:|']
        for name, values in analysis['recipes'].items():
            for seed, result in values['per_seed'].items():
                lines.append(f'| {name} | {seed} | {100*result["with_minus_without"]["primary_score"]:+.2f} |')
        lines += ['', '### Ohne den früheren Auswahlfold 0', '',
            '| Rezept | Δ DE/EN (pp) | Δ DE | Δ EN | Δ JA |', '|---|---:|---:|---:|---:|']
        for name, values in analysis['recipes'].items():
            d = values['excluding_selection_fold_mean_with_minus_without']
            lines.append(f'| {name} ' + ''.join(f'| {100*d[m]:+.2f} ' for m in METRICS) + '|')
        lines += ['', 'Die feste Regel für einen positiven Clipping-Nachweis verlangt einen positiven mittleren Hauptwert, '
            'positive Hauptwertunterschiede in allen drei Seeds und eine positive untere bedingte Bootstrap-Grenze.', '']
        for name, values in analysis['recipes'].items():
            lines.append(f'- {name}: Regel ' + ('erfüllt.' if values['positive_clipping_evidence_under_fixed_rule'] else 'nicht erfüllt.'))
        lines += ['', 'Ein nicht erfüllter positiver Nachweis beweist keine Wirkungslosigkeit oder allgemeine Schädlichkeit. '
            'Der API-Standard wird durch diese explorative Studie nicht automatisch geändert.', '']
    lines += ['## Alle ursprünglichen Faktor-Paare', '',
        'Diese 16 Paare waren vor dem neuen Auftrag bereits abgeschlossen. Nur Fold 0, zwei Seeds; '
        'nicht mit der CV zu unabhängigen Stichproben zusammenzählen.', '',
        '| Seed | LR | Decay | Warmup | Δ DE/EN (pp) |', '|---:|---:|---:|---:|---:|']
    for row in summary['pairs']:
        if row['factorial']:
            c = row['config']
            lines.append(f'| {c["seed"]} | {c["learning_rate"]:g} | {c["weight_decay"]:g} | {c["warmup_fraction"]:g} '
                         f'| {100*row["with_minus_without"]["primary_score"]:+.2f} |')
    lines += ['', '## Nachweise, Betrieb und Grenzen', '',
        '- [Vor Trainingsbeginn eingefrorenes Protokoll](protocol-v1.json), [aktuelle vollständige Zuordnung](summary-v1.json), [Paarliste als CSV](pairs-v1.csv).',
        '- [Erste 32 Fits](../optimizer_ablation/README.md), [vorherige Drei-Rezept-Validierung](../optimizer_validation/README.md). Deren Ergebnisse bleiben unverändert.',
        '- Neues MLflow-Experiment: `article-gradient-clipping-controls-v1`; jeder neue Run verlinkt seinen ursprünglichen Clipping-Partner. '
        '[Öffentliches Original-MLflow](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments).',
        '- Nach bereits bekannten Ergebnissen geplante explorative Erweiterung. Keine unabhängige Bestätigung; kleine, wiederverwendete Train-Stichprobe. '
        '2000 gepaarte Gruppen-Bootstrap-Ziehungen beschreiben bedingte Unsicherheit, keine Auswahlkorrektur und nicht alle Trainingsschwankungen.',
        '- Volltrain-Adapter ohne zurückgehaltene Daten sind keine CV-Paare. Kein erneuter Zugriff auf den 691er Abschlusstest. '
        'Keine Änderung vorhandener Gewichte oder der Serving-Auswahl.',
        '- Fits laufen einzeln mit Trainingssperre und frischem Prozess. Die Inferenz-API kann dafür vorübergehend pausieren und wird anschließend wieder gestartet. '
        'MLflow und dessen öffentlicher Lesezugang bleiben verfügbar. Rechner muss eingeschaltet bleiben.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    stream = io.StringIO()
    fields = ['pair_id', 'fold', 'seed', 'learning_rate', 'weight_decay', 'warmup_fraction', 'factorial', 'cv_recipe', 'status',
              'with_mlflow_run_id', 'without_mlflow_run_id', 'with_minus_without_primary_pp']
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for row in summary['pairs']:
        writer.writerow(dict(pair_id=row['id'], **{k: row['config'][k] for k in fields[1:6]},
            factorial=row['factorial'], cv_recipe=row['cv_recipe'] or '', status=row['status'],
            with_mlflow_run_id=row['with_mlflow_run_id'], without_mlflow_run_id=row.get('without_mlflow_run_id', ''),
            with_minus_without_primary_pp=100*row['with_minus_without']['primary_score'] if row['status'] == 'complete' else ''))
    (OUT / 'pairs-v1.csv').write_text(stream.getvalue())


def pause_api():
    """Only stop the explicitly inventoried inference process, after taking the lock."""
    import signal
    request = LOCAL / 'api-pause-request.json'
    if not request.exists():
        return
    pid = int(json.loads(request.read_text())['pid'])
    cmdline = Path(f'/proc/{pid}/cmdline')
    if not cmdline.exists():
        raise RuntimeError('Inventoried API process no longer exists; inspect before pausing')
    args = cmdline.read_bytes().decode().split('\0')
    if not all(s in args for s in ('uvicorn', 'app.main:app', '--port', '8000')):
        raise RuntimeError('API PID no longer matches the inference server')
    write(LOCAL / 'api-paused.json', dict(pid=pid, paused_at=now(), reason='Exclusive sequential clipping-control training'))
    os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        if not cmdline.exists() or not cmdline.read_bytes():
            return
        time.sleep(0.25)
    raise RuntimeError('API did not stop gracefully; refusing force-kill')


def restore_api():
    import socket
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', 8000)) == 0:
            return
    if not (LOCAL / 'api-paused.json').exists():
        return
    with (LOCAL / 'api-restored.log').open('a') as f:
        p = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000', '--workers', '1'],
            cwd=ROOT, env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'), stdin=subprocess.DEVNULL,
            stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write(LOCAL / 'api-restored.json', dict(pid=p.pid, started_at=now()))


def worker():
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protocol = verify(check_base=True)
            collect()
            if any(not (OUT / 'runs' / (c['id'] + '.json')).exists() for c in protocol['runs']):
                pause_api()
            for config in protocol['runs']:
                if (OUT / 'runs' / (config['id'] + '.json')).exists():
                    collect()
                    continue
                if (LOCAL / 'STOP').exists():
                    collect('stopped'); write(LOCAL / 'worker.json', dict(state='stopped', updated_at=now())); return
                write(LOCAL / 'worker.json', dict(state='running', pid=os.getpid(), current_run=config['id'], updated_at=now()))
                subprocess.run([sys.executable, '-m', MODULE, 'one', '--run-id', config['id']], cwd=ROOT, check=True)
                collect()
            summary = collect('complete')
            if summary['completed_pairs'] != 30 or summary['distinct_successful_mlflow_run_ids'] != 60:
                raise RuntimeError('Incomplete or duplicate final clipping coverage')
            subprocess.run([sys.executable, 'scripts/check_clipping_control.py', '--write'], cwd=ROOT, check=True)
            import mlflow
            mlflow.set_tracking_uri(f'sqlite:///{ROOT / "mlflow.db"}')
            for config in protocol['runs']:
                result = json.loads((OUT / 'runs' / (config['id'] + '.json')).read_text())
                for name in ('README.md', 'summary-v1.json', 'pairs-v1.csv', 'verification-v1.json'):
                    mlflow.MlflowClient().log_artifact(result['mlflow_run_id'], str(OUT / name), artifact_path='clipping-comparison-v1')
            write(LOCAL / 'worker.json', dict(state='complete', completed_at=now()))
        except BaseException as exc:
            traceback.print_exc()
            write(LOCAL / 'worker.json', dict(state='failed', updated_at=now(), error=str(exc)))
            raise
        finally:
            restore_api()


def launch():
    LOCAL.mkdir(parents=True, exist_ok=True)
    verify(check_base=True)
    with (ROOT / 'runtime/gpu-training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (LOCAL / 'worker.log').open('a') as f:
        p = subprocess.Popen([sys.executable, '-m', MODULE, 'worker'], cwd=ROOT,
            env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    write(LOCAL / 'launch.json', dict(pid=p.pid, launched_at=now()))
    print('Worker launched', p.pid, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'launch', 'worker', 'one', 'collect', 'verify'])
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'one':
        if not args.run_id:
            parser.error('--run-id is required')
        one(args.run_id)
    else:
        {'initialize': initialize, 'launch': launch, 'worker': worker, 'collect': collect, 'verify': verify}[args.action]()
