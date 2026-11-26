"""Clipping-only pairs, complete coverage and honest grouped inference, without GPU."""
import ast
import copy
import json
from pathlib import Path

import pytest

from src.studies.clipping_control.design import (
    METRICS, PARAMETERS, SEEDS, analyze_cv, configurations, pairs, validate_pair,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_optimizer_conditions_have_exactly_one_clipping_partner():
    rows = pairs(); new = configurations()
    assert len(rows) == len({r['id'] for r in rows}) == 30
    assert sum(r['factorial'] for r in rows) == 16
    assert sum(bool(r['cv_recipe']) for r in rows) == 18
    assert sum(r['factorial'] and bool(r['cv_recipe']) for r in rows) == 4
    paths = [r[arm]['runs'] for r in rows for arm in ('with_clipping', 'without_clipping')]
    assert len(set(paths)) == 60
    assert sum(p.startswith('reports/clipping_control/') for p in paths) == 7
    for c in new:
        assert c['gradient_clip'] is None and c['recipe'] == 'baseline_no_clip'
        assert (c['learning_rate'], c['weight_decay'], c['warmup_fraction']) == (1e-4, .01, .1)
        assert paths.count(f'reports/clipping_control/runs/{c["id"]}.json') == 1
    for recipe in ('baseline', 'candidate'):
        assert {(r['config']['fold'], r['config']['seed']) for r in rows if r['cv_recipe'] == recipe} == {
            (f, s) for f in range(3) for s in SEEDS}


def test_pair_validation_rejects_other_parameter_changes_or_duplicate_fit():
    pair = pairs()[0]
    on = dict(config=pair['config'], mlflow_run_id='on')
    off = dict(config=dict(pair['config'], gradient_clip=None), mlflow_run_id='off')
    validate_pair(pair, on, off)
    for key in (*PARAMETERS, 'seed', 'fold'):
        bad = copy.deepcopy(off); bad['config'][key] += 1
        with pytest.raises(ValueError, match='Non-matching'):
            validate_pair(pair, on, bad)
    bad = dict(off, mlflow_run_id='on')
    with pytest.raises(ValueError, match='distinct fits'):
        validate_pair(pair, on, bad)


def fixture():
    predictions = {}; groups = {}
    for recipe in ('baseline', 'candidate'):
        for seed in SEEDS:
            for fold in range(3):
                for arm in ('with', 'without'):
                    records = []
                    for lang in ('de', 'en', 'ja'):
                        for label in ('negative', 'neutral', 'positive'):
                            key = f'{fold}-{lang}-{label}'; groups[key] = key
                            pred = 'neutral' if recipe == 'baseline' and arm == 'without' and label == 'positive' else label
                            records.append(dict(id=key, language=lang, reference=label, prediction=pred, text_sha256=key))
                    predictions[recipe, seed, fold, arm] = records
    return predictions, groups


def test_cv_separates_recipes_and_uses_with_minus_without_sign():
    predictions, groups = fixture()
    result = analyze_cv(predictions, groups, bootstrap_samples=100)
    assert result['status'] == 'complete'
    baseline = result['recipes']['baseline']; candidate = result['recipes']['candidate']
    assert baseline['mean_with_minus_without']['primary_score'] > 0
    assert baseline['positive_clipping_evidence_under_fixed_rule']
    assert all(candidate['mean_with_minus_without'][m] == 0 for m in METRICS)
    assert candidate['conditional_group_bootstrap_95_intervals']['primary_score'] == [0, 0]
    assert not candidate['positive_clipping_evidence_under_fixed_rule']
    assert not result['independent_test_evidence'] and not result['serving_selection_changed']


def test_incomplete_or_unpaired_oof_predictions_never_produce_a_final_claim():
    predictions, groups = fixture()
    incomplete = copy.deepcopy(predictions); incomplete.pop(next(iter(incomplete)))
    assert analyze_cv(incomplete, groups)['status'] == 'pending'
    bad = copy.deepcopy(predictions); bad[next(iter(bad))].pop()
    with pytest.raises(ValueError, match='OOF coverage'):
        analyze_cv(bad, groups)
    bad = copy.deepcopy(predictions); bad[next(iter(bad))][0]['text_sha256'] = 'different'
    with pytest.raises(ValueError, match='metadata differs'):
        analyze_cv(bad, groups)


def test_training_loop_is_identical_to_frozen_validation_implementation():
    def loop(path):
        tree = ast.parse((ROOT / path).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'one')
        return next(n for n in ast.walk(function) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'epoch')
    assert ast.dump(loop('src/studies/clipping_control/runner.py')) == ast.dump(loop('src/studies/optimizer_validation/runner.py'))


def test_real_run_setup_logs_original_clipping_partner_before_model_load(monkeypatch, tmp_path):
    import mlflow
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from src.studies.clipping_control import runner
    from src import next_stage_experiment as experiment
    protocol = json.loads((ROOT / 'reports/optimizer_validation/protocol-v1.json').read_text())
    protocol.update(runs=configurations(), pairs=pairs())
    config = protocol['runs'][0]; member = next(m for m in protocol['folds'] if m['fold'] == config['fold'])
    path = tmp_path / 'train.csv'
    pd.DataFrame([dict(id=i, group=i, language=('de', 'en', 'ja')[n % 3],
        label=('negative', 'neutral', 'positive')[(n // 3) % 3], text='synthetic', text_sha256='synthetic')
        for n, i in enumerate(member['fit_ids'] + member['dev_ids'])]).to_csv(path, index=False)
    protocol['files_sha256']['train.csv'] = runner.sha(path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path); monkeypatch.setattr(runner, 'OUT', tmp_path / 'reports')
    monkeypatch.setattr(runner, 'ART', tmp_path / 'models')
    monkeypatch.setattr(runner, 'PROTOCOL', ROOT / 'reports/optimizer_validation/protocol-v1.json')
    monkeypatch.setattr(runner, 'verify', lambda **kw: protocol)
    monkeypatch.setattr(runner, 'load_evidence', lambda *a, **kw: (dict(mlflow_run_id='original-clipped'), [], []))
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **kw: 'synthetic-code-commit')
    monkeypatch.setattr(runner, 'validate_membership', lambda *a: None)
    monkeypatch.setattr(experiment, 'TRAIN', path); monkeypatch.setattr(experiment, 'gpu', lambda: 'synthetic')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (16 * 1024**3, 16 * 1024**3))
    for name in ('set_tracking_uri', 'set_experiment', 'log_params', 'log_artifact'):
        monkeypatch.setattr(mlflow, name, lambda *a, **kw: None)
    tags = {}; monkeypatch.setattr(mlflow, 'set_tags', lambda value: tags.update(value))
    class Context:
        def __enter__(self):
            return SimpleNamespace(info=SimpleNamespace(run_id='synthetic'))
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(mlflow, 'start_run', lambda **kw: Context())
    def stop_before_load():
        raise RuntimeError('intentional stop before model load')
    monkeypatch.setattr(experiment, 'lora', stop_before_load)
    with pytest.raises(RuntimeError, match='intentional stop before model load'):
        runner.one(config['id'])
    assert tags['paired_with_clipping_run_id'] == 'original-clipped'
    assert tags['train_sha256'] == protocol['files_sha256']['train.csv']
