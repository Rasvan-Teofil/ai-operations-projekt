import json
import numpy as np
import pandas as pd
import pytest

from src import news_scaling as scaling
from src.news_data import digest


def test_learning_curve_subsets_are_nested_and_keep_source_groups_together():
    rows = [{'id': f'{g}-{label}', 'group': f'source-{g}', 'label': label}
            for g in range(18) for label in scaling.LABELS]
    df = pd.DataFrame(rows)
    sets = [part for _, part in scaling.nested_subsets(df, 17)]
    assert set(sets[0].id) < set(sets[1].id) < set(sets[2].id) == set(df.id)
    for part in sets:
        assert (part.groupby('group').size() == 3).all()
    assert [part.id.tolist() for part in sets] == [part.id.tolist() for _, part in scaling.nested_subsets(df, 17)]


def test_changed_training_split_blocks_expansion_verification(tmp_path, monkeypatch):
    (tmp_path / 'train.csv').write_text('modified')
    (tmp_path / 'split_manifest.json').write_text(json.dumps({'splits': {'train': {'sha256': 'frozen'}}}))
    monkeypatch.setattr(scaling, 'LOCAL', tmp_path); monkeypatch.setattr(scaling, 'OUT', tmp_path)
    with pytest.raises(RuntimeError, match='Split changed'):
        scaling.verified()


def test_completed_fresh_test_cannot_be_opened_for_another_probe(tmp_path, monkeypatch):
    (tmp_path / 'protocol.json').write_text('{}')
    (tmp_path / 'trained.json').write_text(json.dumps({'models': [], 'protocol_sha256': digest(b'{}')}))
    (tmp_path / 'evaluation.lock').write_text(json.dumps({'status': 'complete'}))
    monkeypatch.setattr(scaling, 'OUT', tmp_path)
    monkeypatch.setattr(scaling, 'gpu', lambda: None)
    monkeypatch.setattr(scaling, 'verified', lambda: {})
    def forbidden(*args):
        raise AssertionError('Finished Test must stop before loading inputs or a model')
    monkeypatch.setattr(scaling.pd, 'read_csv', forbidden)
    monkeypatch.setattr(scaling, 'predict_rows', forbidden)
    with pytest.raises(RuntimeError, match='already started/completed'):
        scaling.evaluate(resume=True)


def test_metrics_expose_missed_positive_class_and_language_difference():
    result = scaling.metrics(np.array([0, 1, 2, 0, 1, 2]),
                             np.array([0, 1, 1, 1, 1, 0]), np.array(['de'] * 3 + ['en'] * 3))
    assert result['f1_positive'] == result['recall_positive'] == 0
    assert result['recall_negative'] == .5
    assert result['macro_f1_de'] > result['macro_f1_en']
    assert result['balanced_language_macro_f1'] == pytest.approx((result['macro_f1_de'] + result['macro_f1_en']) / 2)
