"""Selection and provenance guards; no GPU, weights, model fitting or Test access."""
import hashlib

import pytest

from src.studies.optimizer_ablation.design import configurations
from src.studies.optimizer_candidate.runner import choose_recipe, verify_files


def selection_fixture():
    rows = []
    for config in configurations():
        preferred = config['gradient_clip'] == 1.0 and config['learning_rate'] == 5e-5 and config['warmup_fraction'] == 0
        # Same two-seed mean for both decay settings. Single-seed peaks must not win.
        score = (0.75 if preferred else 0.5) + (0.1 if config['seed'] == 20261110 else -0.1)
        rows.append(dict(config=config, metrics=dict(primary_score=score)))
    return rows


def test_recipe_selection_uses_both_seeds_and_breaks_decay_tie_without_ja():
    rows = selection_fixture()
    outlier = next(r for r in rows if r['config']['gradient_clip'] is None and r['config']['seed'] == 20261110)
    outlier['metrics']['primary_score'] = 0.95  # Highest individual fit, lower paired mean.
    selected, ranked = choose_recipe(rows)
    assert len(ranked) == 16
    assert selected['mean_development_primary'] == pytest.approx(0.75)
    assert selected['recipe'] == dict(gradient_clip=1.0, learning_rate=5e-5, weight_decay=0, warmup_fraction=0)


def test_selection_rejects_missing_duplicate_or_modified_conditions():
    rows = selection_fixture()
    with pytest.raises(ValueError, match='32 unique'):
        choose_recipe(rows[:-1])
    with pytest.raises(ValueError, match='32 unique'):
        choose_recipe(rows[:-1] + [rows[0]])
    rows[0]['config'] = dict(rows[0]['config'], learning_rate=1.0)
    with pytest.raises(ValueError, match='Changed factorial'):
        choose_recipe(rows)


def test_frozen_inputs_and_default_selection_cannot_change(tmp_path):
    path = tmp_path / 'serving_selected.json'
    path.write_text('original')
    records = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_files(tmp_path, records)
    path.write_text('replaced')
    with pytest.raises(RuntimeError, match='baseline changed'):
        verify_files(tmp_path, records)
    with pytest.raises(RuntimeError, match='baseline changed'):
        verify_files(tmp_path, {'../outside': 'untrusted'})
