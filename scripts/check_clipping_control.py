"""Independent audit of all matched clipping fits; no model inference."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def audit(partial=False):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from mlflow import MlflowClient
    from src.studies.clipping_control.runner import verify, sha
    from src.tracking import fingerprint
    from src.next_stage_experiment import TRAIN

    protocol = verify(check_base=True)
    train = pd.read_csv(TRAIN).set_index('id')
    client = MlflowClient(tracking_uri=f'sqlite:///{ROOT / "mlflow.db"}')
    labels = ['negative', 'neutral', 'positive']
    pooled = {}; seen = set(); complete_pairs = 0
    for pair in protocol['pairs']:
        both = True
        for arm, field in (('with', 'with_clipping'), ('without', 'without_clipping')):
            ref = pair[field]; path = ROOT / ref['runs']
            if not path.exists():
                assert partial and ref['runs'].startswith('reports/clipping_control/')
                both = False
                continue
            r = json.loads(path.read_text()); c = r['config']
            assert r['mlflow_run_id'] not in seen
            seen.add(r['mlflow_run_id'])
            assert c.get('fold', 0) == pair['config']['fold'] and c['seed'] == pair['config']['seed']
            for k in ('learning_rate', 'weight_decay', 'warmup_fraction'):
                assert c[k] == pair['config'][k]
            assert c['gradient_clip'] == (1.0 if arm == 'with' else None)
            member = next(m for m in protocol['folds'] if m['fold'] == c.get('fold', 0))
            fit, dev = set(member['fit_ids']), set(member['dev_ids'])
            assert not fit & dev and fit | dev == set(train.index)
            assert not set(train.loc[list(fit), 'group']) & set(train.loc[list(dev), 'group'])
            for kind in ('predictions', 'telemetry'):
                assert sha(ROOT / ref[kind]) == r[kind + '_sha256']
            records = json.loads((ROOT / ref['predictions']).read_text())
            assert len(records) == len(dev) and {x['id'] for x in records} == dev
            for x in records:
                t = train.loc[x['id']]
                assert (x['reference'], x['language'], x['text_sha256']) == (t.label, t.language, t.text_sha256)
                assert len(x['scores']) == 3 and all(math.isfinite(s) and 0 <= s <= 1 for s in x['scores'])
                assert abs(sum(x['scores']) - 1) < 1e-5 and labels[int(np.argmax(x['scores']))] == x['prediction']
            y = [x['reference'] for x in records]; pred = [x['prediction'] for x in records]
            values = dict(accuracy=float(accuracy_score(y, pred)),
                          macro_f1=float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)))
            assert confusion_matrix(y, pred, labels=labels).tolist() == r['metrics']['confusion_matrix']
            for lang in ('de', 'en', 'ja'):
                sub = [x for x in records if x['language'] == lang]
                values['macro_f1_' + lang] = float(f1_score([x['reference'] for x in sub], [x['prediction'] for x in sub], labels=labels, average='macro', zero_division=0))
            values['primary_score'] = (values['macro_f1_de'] + values['macro_f1_en']) / 2
            for k, v in values.items():
                assert abs(v - r['metrics'][k]) < 1e-12
            events = json.loads((ROOT / ref['telemetry']).read_text())
            steps = 3 * math.ceil(len(fit) / 16); warmup = int(c['warmup_fraction'] * steps)
            assert len(events) == steps and [e['step'] for e in events] == list(range(1, steps + 1))
            for e in events:
                assert all(math.isfinite(e[k]) for k in ('gradient_norm', 'adapter_update_l2', 'learning_rate', 'train_loss'))
                assert e['clipped'] == int(arm == 'with' and e['gradient_norm'] > 1.0)
                s = e['step'] - 1
                rate = c['learning_rate'] * (s / max(1, warmup) if s < warmup else (steps - s) / max(1, steps - warmup))
                assert math.isclose(e['learning_rate'], rate, rel_tol=1e-10, abs_tol=1e-12)
            study = Path(ref['runs']).parts[1]
            adapter = ROOT / 'models' / study / path.stem
            assert fingerprint(adapter) == r['adapter_sha256']
            run = client.get_run(r['mlflow_run_id'])
            assert run.info.status == 'FINISHED'
            for name in ('gradient_norm', 'clipped', 'learning_rate', 'adapter_update_l2', 'train_loss'):
                history = client.get_metric_history(r['mlflow_run_id'], name)
                assert len(history) == steps
                assert {h.step: h.value for h in history} == {e['step']: e[name] for e in events}
            for k, v in r['metrics'].items():
                if isinstance(v, (int, float)):
                    assert math.isclose(run.data.metrics[k], v, rel_tol=1e-10, abs_tol=1e-12)
            for k in ('learning_rate', 'weight_decay', 'warmup_fraction', 'seed'):
                assert float(run.data.params[k]) == c[k]
            assert run.data.params['gradient_clip'] == ('1.0' if arm == 'with' else 'disabled')
            if study == 'clipping_control':
                assert r['protocol_sha256'] == sha(ROOT / 'reports/clipping_control/protocol-v1.json')
                partner = json.loads((ROOT / pair['with_clipping']['runs']).read_text())
                assert run.data.tags['paired_with_clipping_run_id'] == partner['mlflow_run_id']
                assert abs(run.data.metrics['with_minus_without_primary_pp'] - 100 * (partner['metrics']['primary_score'] - r['metrics']['primary_score'])) < 1e-12
            if pair['cv_recipe']:
                pooled.setdefault((pair['cv_recipe'], c['seed'], arm), []).extend(records)
        complete_pairs += int(both)
    report = dict(status='complete' if len(seen) == 60 else 'partial', checked_unique_fits=len(seen),
        complete_pairs=complete_pairs, expected_pairs=30, expected_unique_fits=60,
        protocol_sha256=sha(ROOT / 'reports/clipping_control/protocol-v1.json'),
        hashes_group_membership_scores_predictions_telemetry_adapters_and_mlflow_verified=True,
        final_test_inference_performed=False, serving_selection_changed=False)
    if len(seen) != 60:
        return report
    summary = json.loads((ROOT / 'reports/clipping_control/summary-v1.json').read_text())
    analysis = summary['cv_analysis']
    assert analysis['status'] == 'complete' and summary['completed_new_fits'] == 7
    group_names = sorted(train.group.unique()); gi = {g: i for i, g in enumerate(group_names)}
    draws = np.random.default_rng(20261126).multinomial(len(group_names), np.full(len(group_names), 1 / len(group_names)), size=2000)
    for recipe in ('baseline', 'candidate'):
        prepared = {}; measured = {}
        for seed in protocol['seeds']:
            measured[seed] = {}
            for arm in ('with', 'without'):
                rows = pooled[recipe, seed, arm]
                assert len(rows) == len(train) and {r['id'] for r in rows} == set(train.index)
                scores = {}
                for lang in ('de', 'en', 'ja'):
                    sub = sorted((r for r in rows if r['language'] == lang), key=lambda x: x['id'])
                    y, p, g = ([r['reference'] for r in sub], [r['prediction'] for r in sub], [gi[train.loc[r['id'], 'group']] for r in sub])
                    prepared[seed, arm, lang] = y, p, g
                    scores['macro_f1_' + lang] = float(f1_score(y, p, labels=labels, average='macro', zero_division=0))
                scores['primary_score'] = (scores['macro_f1_de'] + scores['macro_f1_en']) / 2
                measured[seed][arm] = scores
                for k, v in scores.items():
                    assert abs(v - analysis['recipes'][recipe]['per_seed'][str(seed)]['oof'][arm][k]) < 1e-12
        target = analysis['recipes'][recipe]
        for metric in target['mean_with_minus_without']:
            delta = float(np.mean([m['with'][metric] - m['without'][metric] for m in measured.values()]))
            assert abs(delta - target['mean_with_minus_without'][metric]) < 1e-12
        samples = {k: [] for k in target['mean_with_minus_without']}
        for draw in draws:
            values = {}
            for arm in ('with', 'without'):
                values[arm] = {}
                for lang in ('de', 'en', 'ja'):
                    parts = []
                    for seed in protocol['seeds']:
                        y, p, g = prepared[seed, arm, lang]
                        parts.append(f1_score(y, p, labels=labels, average='macro', zero_division=0, sample_weight=draw[g]))
                    values[arm]['macro_f1_' + lang] = float(np.mean(parts))
                values[arm]['primary_score'] = (values[arm]['macro_f1_de'] + values[arm]['macro_f1_en']) / 2
            for k in samples:
                samples[k].append(values['with'][k] - values['without'][k])
        for k, vals in samples.items():
            assert np.allclose(np.quantile(vals, [.025, .975]), target['conditional_group_bootstrap_95_intervals'][k], atol=1e-12)
    report['pooled_oof_and_2000_bootstrap_draws_per_recipe_independently_reproduced_with_sklearn_weights'] = True
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true'); parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); report = audit(args.partial)
    if args.write:
        assert report['status'] == 'complete'
        path = ROOT / 'reports/clipping_control/verification-v1.json'
        with path.open('x') as f:
            f.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))
