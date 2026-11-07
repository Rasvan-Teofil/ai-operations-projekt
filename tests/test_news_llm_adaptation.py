from contextlib import contextmanager
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src import local_llm
from src import news_llm_experiment as experiment
from src.news_data import digest
from src.news_llm_profiles import examples, demonstrations, PROMPT


def test_demonstrations_are_deterministic_balanced_and_fit_only():
    fit = pd.DataFrame([{'id': f'fit-{lang}-{label}-{i}', 'language': lang, 'label': label,
                         'text': f'Synthetic {lang} {label} example{i}', 'text_sha256': str(i)}
                        for lang in ['de', 'en'] for label in experiment.LABELS for i in range(4)])
    first = examples(fit, 77); second = examples(fit, 77)
    assert first == second and len(first) == 6
    assert {r['id'] for r in first} <= set(fit.id)
    assert {(r['language'], r['label']) for r in first} == {(l, y) for l in ['de', 'en'] for y in experiment.LABELS}
    turns = demonstrations(first)
    assert len(turns) == 12
    assert [r['role'] for r in turns] == ['user', 'assistant'] * 6
    assert {turns[i]['content'] for i in range(1, 12, 2)} == {'A', 'B', 'C'}
    assert 'across all topics' in PROMPT


def test_prompt_profiles_share_one_base_allocation(tmp_path, monkeypatch):
    calls = []
    class Engine:
        def __init__(self, weights): calls.append(weights)
    monkeypatch.setattr(local_llm, 'LocalEngine', Engine)
    a = local_llm.shared_engine(tmp_path / 'weights')
    b = local_llm.shared_engine(tmp_path / 'weights')
    assert a is b and len(calls) == 1
    assert local_llm.shared_engine(tmp_path / 'other') is not a


def test_unadapted_calls_disable_adapter_and_restore_its_state():
    class Net:
        active = 'trained'
        def set_adapter(self, name): self.active = name
        @contextmanager
        def disable_adapter(self):
            old = self.active; self.active = None
            try: yield
            finally: self.active = old
        def __call__(self, **kwargs):
            assert kwargs['use_cache'] is False and kwargs['logits_to_keep'] == 1
            value = 9. if self.active else 1.
            return SimpleNamespace(logits=torch.tensor([[[value, 0., 0.]]]))
    import threading
    engine = local_llm.LocalEngine.__new__(local_llm.LocalEngine)
    engine.lock = threading.RLock(); engine.net = Net(); engine.has_adapters = True
    assert engine.logits({}, 'trained')[0] == 9
    assert engine.logits({}, None)[0] == 1
    assert engine.net.active == 'trained'
    assert engine.logits({}, 'trained')[0] == 9


def test_base_dependency_cannot_escape_local_model_directory():
    with pytest.raises(ValueError, match='local model artifacts'):
        local_llm.local_dependency('/etc/passwd')


def test_finished_v5_test_blocks_even_resume_before_model_load(tmp_path, monkeypatch):
    (tmp_path / 'protocol.json').write_text('{}')
    (tmp_path / 'trained.json').write_text(json.dumps({'protocol_sha256': digest(b'{}'), 'models': []}))
    (tmp_path / 'evaluation.lock').write_text(json.dumps({'status': 'complete'}))
    monkeypatch.setattr(experiment, 'OUT', tmp_path)
    monkeypatch.setattr(experiment, 'gpu', lambda: None)
    monkeypatch.setattr(experiment, 'verified', lambda: {})
    def forbidden(*args): raise AssertionError('Do not reopen completed Test')
    monkeypatch.setattr(experiment.pd, 'read_csv', forbidden)
    monkeypatch.setattr(experiment, 'artifact_hash', forbidden)
    with pytest.raises(RuntimeError, match='already started/completed'):
        experiment.evaluate(resume=True)


def test_profiles_preserve_example_provenance_and_unchanged_base(tmp_path, monkeypatch):
    base = tmp_path / 'base'; base.mkdir()
    original = {'kind': 'local_llm', 'system_prompt': 'old prompt', 'labels': experiment.LABELS,
                'max_article_tokens': 3072, 'score_method': 'restricted logits'}
    (base / 'artifact.json').write_text(json.dumps(original))
    monkeypatch.setattr(experiment, 'BASE', base)
    rows = pd.DataFrame([{'id': l + y, 'language': l, 'label': y, 'text': 'Synthetic example', 'text_sha256': 'sha'}
                        for l in ['de', 'en'] for y in experiment.LABELS])
    target = experiment.make_profile(tmp_path / 'profile', 'fewshot', rows)
    meta = json.loads((target / 'artifact.json').read_text())
    assert len(meta['demonstrations']) == 12 and len(meta['example_provenance']) == 6
    assert meta['system_prompt'] == PROMPT and meta['base_artifact'] == 'models/local_llm'
    assert json.loads((base / 'artifact.json').read_text()) == original
    with pytest.raises(RuntimeError, match='already exists'):
        experiment.make_profile(target, 'news')


def test_reproduction_keeps_frozen_evidence_and_rejects_path_escape(tmp_path, monkeypatch):
    from src import news_llm_reproduce as reproduce
    source = tmp_path / 'reports/llm_adaptation_v5'; source.mkdir(parents=True)
    for name in ['protocol.json', 'source_manifest.json', 'split_manifest.json', 'annotations.csv', 'overlap_audit.json']:
        (source / name).write_text('frozen-' + name)
    monkeypatch.setattr(reproduce, 'ROOT', tmp_path)
    env = reproduce.prepare('tower-01')
    assert env['SENTIMENT_LLM_ADAPT_MODELS'] == 'models/reproduction/llm-adaptation-tower-01'
    assert env['SENTIMENT_LLM_ADAPT_REPRODUCTION'] == 'tower-01'
    with pytest.raises(RuntimeError, match='already exists'): reproduce.prepare('tower-01')
    with pytest.raises(ValueError): reproduce.prepare('../overwrite')
    assert (source / 'protocol.json').read_text() == 'frozen-protocol.json'


@pytest.mark.parametrize('nested', [False, True])
def test_standalone_size_includes_shared_base_and_counts_adapter_once(tmp_path, monkeypatch, nested):
    base, profile = tmp_path / 'models/base', tmp_path / 'models/profile'
    adapter = profile / 'adapter' if nested else tmp_path / 'models/adapter'
    base.mkdir(parents=True); profile.mkdir(parents=True); adapter.mkdir(parents=True)
    (base / 'weights.bin').write_bytes(b'b' * 1000)
    (adapter / 'adapter.bin').write_bytes(b'a' * 100)
    (profile / 'artifact.json').write_text(json.dumps({'base_artifact': 'models/base',
        'adapter_path': str(adapter.relative_to(tmp_path))}))
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    complete, incremental = experiment.full_size(profile)
    assert complete == 1000 + incremental
    assert incremental == (profile / 'artifact.json').stat().st_size + 100


def test_pinned_profile_hash_also_detects_base_and_adapter_changes(tmp_path, monkeypatch):
    base, profile, adapter = [tmp_path / 'models' / n for n in ['base', 'profile', 'adapter']]
    for path in [base, profile, adapter]: path.mkdir(parents=True)
    (base / 'weights.bin').write_bytes(b'base')
    (adapter / 'adapter.bin').write_bytes(b'adapter')
    (profile / 'artifact.json').write_text(json.dumps({'base_artifact': 'models/base', 'adapter_path': 'models/adapter'}))
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    before = experiment.artifact_hash(profile)
    (base / 'weights.bin').write_bytes(b'changed base')
    middle = experiment.artifact_hash(profile)
    (adapter / 'adapter.bin').write_bytes(b'changed adapter')
    assert len({before, middle, experiment.artifact_hash(profile)}) == 3
