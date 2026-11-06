import json
import pytest

from src.news_data import digest, lead_text, source_key
from src import news_experiment as experiment


def test_collects_only_substantive_lead_and_drops_media_metadata():
    sentence = 'The company reported heavy losses and announced substantial job cuts across its factories in several countries.'
    html = f'<div class="mw-parser-output"><table><tr><td><p>Do not use this navigation paragraph with a very long irrelevant description of the news service.</p></td></tr></table><p><strong class="published">Tuesday</strong></p><p>{sentence}</p><p>Later body paragraph.</p></div>'
    assert lead_text(html, 'Company cuts jobs') == 'Company cuts jobs\n\n' + sentence


def test_quoted_lead_is_excluded_instead_of_selecting_a_different_paragraph():
    html = '<div class="mw-parser-output"><p>The minister said "success" while discussing a very long policy announcement about the future of the regional economy.</p><p>A later paragraph should not silently replace the rejected lead.</p></div>'
    with pytest.raises(ValueError, match='Quoted'):
        lead_text(html, 'Policy announcement')


def test_never_truncates_an_overlong_sentence_into_a_false_claim():
    html = '<div class="mw-parser-output"><p>' + ('long clause ' * 100) + '.</p></div>'
    with pytest.raises(ValueError, match='length'):
        lead_text(html, 'Long article')


def test_source_grouping_ignores_boilerplate_but_preserves_article_urls():
    assert source_key('https://creativecommons.org/licenses/by/4.0/') is None
    assert source_key('https://en.wikinews.org/wiki/Example') is None
    assert source_key('mailto:example@example.com') is None
    assert source_key('https://www.publisher.example/story/42#top') == '//publisher.example/story/42'


def test_linked_languages_and_shared_source_cannot_cross_splits():
    rows = [
        {'id': 'a', 'title_key': 'de:Artikel', 'text': 'Fabrik schließt nach schwerem Verlust.', 'linked_articles': ['en:Story'], 'source_keys': []},
        {'id': 'b', 'title_key': 'en:Story', 'text': 'A factory closes after reporting a major loss.', 'linked_articles': [], 'source_keys': ['//source.example/story/42']},
        {'id': 'c', 'title_key': 'en:Followup', 'text': 'Employees are seeking replacement jobs.', 'linked_articles': [], 'source_keys': ['//source.example/story/42']},
        {'id': 'd', 'title_key': 'de:Sport', 'text': 'Ein Tennisspieler gewinnt das internationale Turnier.', 'linked_articles': [], 'source_keys': []}]
    groups = experiment.groups(rows)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_unlabelled_items_block_split_preparation(tmp_path, monkeypatch):
    local, out = tmp_path / 'local', tmp_path / 'reports'
    local.mkdir(); out.mkdir()
    raw = (json.dumps({'id': 'one', 'text': 'Unlabelled news'}) + '\n').encode()
    (local / 'articles.jsonl').write_bytes(raw)
    (out / 'source_manifest.json').write_text(json.dumps({'sha256': digest(raw)}))
    (out / 'annotations.csv').write_text('id,label,reason\n')
    monkeypatch.setattr(experiment, 'LOCAL', local); monkeypatch.setattr(experiment, 'OUT', out)
    with pytest.raises(ValueError, match='exactly one'):
        experiment.prepare()
    assert not (local / 'train.csv').exists()


def test_changed_split_blocks_training_input_verification(tmp_path, monkeypatch):
    (tmp_path / 'train.csv').write_text('changed')
    (tmp_path / 'split_manifest.json').write_text(json.dumps({'splits': {'train': {'sha256': 'frozen-digest'}}}))
    (tmp_path / 'overlap_audit.json').write_text(json.dumps({'matched_ids': []}))
    monkeypatch.setattr(experiment, 'LOCAL', tmp_path); monkeypatch.setattr(experiment, 'OUT', tmp_path)
    with pytest.raises(RuntimeError, match='Split changed'):
        experiment.verified()


def test_metrics_keep_all_three_classes_when_a_model_misses_one():
    result = experiment.scores([0, 1, 2], [1, 1, 2])
    assert result['f1_negative'] == 0
    assert result['accuracy'] == pytest.approx(2 / 3)
    assert result['macro_f1'] == pytest.approx((0 + 2 / 3 + 1) / 3)
