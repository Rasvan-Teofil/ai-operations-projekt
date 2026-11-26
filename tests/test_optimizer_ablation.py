"""No weights, GPU, raw data or tracking server required."""
from collections import Counter
from itertools import product

import pytest

from src.studies.optimizer_ablation.design import (
    FACTORS, REFERENCE, SEED, configurations, contrasts, validate_membership,
)
from src.studies.optimizer_ablation.runner import write


def test_complete_balanced_factorial_twice_with_reference_and_no_aliasing():
    rows = configurations()
    assert len(rows) == len({r['id'] for r in rows}) == 32
    expected = set(product(*FACTORS.values()))
    for seed in (SEED, SEED + 1):
        block = [r for r in rows if r['seed'] == seed]
        assert {tuple(r[k] for k in FACTORS) for r in block} == expected
        assert {k: block[0][k] for k in FACTORS} == REFERENCE
        for factor in FACTORS:
            assert sorted(Counter(r[factor] for r in block).values()) == [8, 8]
    rows[0]['learning_rate'] = 42
    assert configurations()[0]['learning_rate'] == REFERENCE['learning_rate']


def test_matched_contrasts_recover_known_main_effects_and_interaction():
    results = []
    for config in configurations():
        x = [int(config[k] == FACTORS[k][1]) for k in FACTORS]
        # Add a seed offset: matching must remove it from every contrast.
        score = 0.2 + (config['seed'] - SEED) * 0.03 + 0.04*x[0] + 0.02*x[1] - 0.01*x[2] + 0.06*x[0]*x[3]
        results.append(dict(config=config, metrics={'primary_score': score}))
    effect = contrasts(results)
    assert effect['status'] == 'complete'
    expected = dict(gradient_clip=0.07, learning_rate=0.02, weight_decay=-0.01, warmup_fraction=0.03)
    for name, value in expected.items():
        assert effect['main_effects'][name]['high_minus_low'] == pytest.approx(value)
        assert effect['main_effects'][name]['pairs'] == 16
        for seed in effect['per_seed'].values():
            assert seed['main_effects'][name]['high_minus_low'] == pytest.approx(value)
    assert effect['two_way_difference_of_differences']['gradient_clip × warmup_fraction'] == pytest.approx(0.06)
    assert contrasts(results[:-1])['status'] == 'pending'
    results[-1]['config'] = dict(results[-2]['config'])
    with pytest.raises(ValueError, match='Duplicate'):
        contrasts(results)


def membership_fixture():
    rows = [dict(id=f'{part}-{lang}-{label}', group=f'g-{part}-{lang}-{label}', language=lang, label=label)
            for part in ['fit', 'dev'] for lang in ['de', 'en', 'ja'] for label in ['negative', 'neutral', 'positive']]
    member = dict(fit_ids=[r['id'] for r in rows[:9]], dev_ids=[r['id'] for r in rows[9:]])
    return rows, member


def test_fit_development_guard_rejects_group_leakage_and_bad_membership():
    rows, member = membership_fixture()
    validate_membership(rows, member)
    rows[-1]['group'] = rows[0]['group']
    with pytest.raises(ValueError, match='groups overlap'):
        validate_membership(rows, member)
    rows, member = membership_fixture()
    member['dev_ids'].append(member['fit_ids'][0])
    with pytest.raises(ValueError, match='partition'):
        validate_membership(rows, member)
    rows, member = membership_fixture()
    rows[-1]['language'] = 'en'
    with pytest.raises(ValueError, match='support'):
        validate_membership(rows, member)


def test_evidence_is_immutable_and_nonfinite_metrics_cannot_be_written(tmp_path):
    path = tmp_path / 'result.json'
    write(path, {'metric': 0.5}, immutable=True)
    with pytest.raises(FileExistsError):
        write(path, {'metric': 1.0}, immutable=True)
    with pytest.raises(ValueError):
        write(tmp_path / 'bad.json', {'metric': float('nan')}, immutable=True)
    assert not (tmp_path / 'bad.json').exists()


def test_audited_reboot_retry_needs_new_boot_unchanged_protocol_and_armed_state():
    from scripts.resume_optimizer_ablation import may_resume
    state = dict(state='armed', failed_boot_id='old', protocol_sha256='frozen')
    assert not may_resume(state, 'old', 'frozen')
    assert not may_resume(state, 'new', 'changed')
    assert may_resume(state, 'new', 'frozen')
    assert not may_resume(dict(state, state='consumed'), 'new', 'frozen')
