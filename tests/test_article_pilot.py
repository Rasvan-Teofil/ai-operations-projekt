"""Synthetic layout and intake checks; no external articles, network or GPU."""
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.article import ARTICLE_POLICY, ArticleError, extract_full_article, substantive
from src.highlighting import sentence_spans
from src.next_stage import ArticleRecord, AnnotationRecord, audit, digest, historical_records, profile
from src.next_stage_protocol import RunIdentity, primary_score, select_candidate

BODY = 'The council restored access to the public library after completing the repair of its damaged entrance.'


def article(identifier='one', **updates):
    values = dict(schema_version='article-pilot-v1', id=identifier, document_id='example:en:' + identifier,
        language='en', title='Library reopens', text=BODY, text_sha256=digest(BODY),
        source_url='https://news.example/library', revision_url='https://news.example/library?revision=1',
        history_url='https://news.example/history', revision='1', attribution='Synthetic test fixture',
        published_at='2026-11-10', retrieved_at='2026-11-10', source_license='synthetic-test-only',
        source_license_url='https://news.example/license', rights_evidence_url='https://news.example/license',
        rights_status='reviewed_compatible', rights_review_note='Synthetic test text, not collected news',
        third_party_passages_reviewed=True, extraction_version=ARTICLE_POLICY, transformation='Fixture',
        event_ids=['library-reopening'], role='pilot')
    values.update(updates)
    return ArticleRecord(**values)


def annotation(row, **updates):
    review = dict(label='positive', target='Library users', rationale='Access has been restored', ambiguous=False,
                  ambiguity_note='', reviewed_at='2026-11-10', candidate_predictions_seen=False)
    values = dict(schema_version='article-annotation-v1', id=row.id, text_sha256=row.text_sha256,
        annotation_version='test-1', rubric_version='outcome-rubric-v1', provenance='AI-labelled · Codex-reviewed',
        annotator='Codex', annotator_revision='unavailable', annotation_license='CC0-1.0 to the extent project rights apply',
        first_pass=deepcopy(review), second_pass=deepcopy(review), resolution_note='Consistent self-review')
    values.update(updates)
    return AnnotationRecord(**values)


@pytest.mark.parametrize('layout', ['article', 'main', 'microdata', 'wikinews'])
def test_full_article_retains_middle_and_order_across_layouts(layout):
    middle = 'The repairs were delayed after inspectors found additional damage to the roof.'
    content = f'<h1>Library reopens</h1><p>{BODY}</p><p>{middle}</p><p>Opening hours are unchanged.</p>'
    wrappers = {'article': ('<article>', '</article>'), 'main': ('<main>', '</main>'),
                'microdata': ('<div itemprop="articleBody">', '</div>'),
                'wikinews': ('<div class="mw-parser-output">', '</div>')}
    left, right = wrappers[layout]
    html = f'<body><nav><p>Unrelated links</p></nav>{left}{content}<p hidden>Advertisement</p>{right}<footer>Footer</footer></body>'
    text = extract_full_article(html)
    assert text == '\n\n'.join(['Library reopens', BODY, middle, 'Opening hours are unchanged.'])


@pytest.mark.parametrize('html', [
    '<title>Access denied</title><body><p>' + BODY * 3 + '</p></body>',
    '<title>Just a moment...</title><main><p>' + BODY + '</p></main>',
    '<body><form id="challenge-form"><p>' + BODY + '</p></form></body>',
    '<main><h1>Sign in</h1><form><input type="password"></form><p>' + BODY + '</p></main>',
])
def test_success_status_access_pages_are_not_articles(html):
    with pytest.raises(ArticleError) as error:
        extract_full_article(html)
    assert error.value.code == 'access_page'


def test_reporting_about_access_denial_is_not_a_gate_page():
    html = '<article><h1>Residents report access denied at the border</h1><p>' + BODY + '</p></article>'
    assert BODY in extract_full_article(html)


def test_extractor_rejects_overlength_instead_of_truncating():
    with pytest.raises(ArticleError, match='no text was truncated'):
        extract_full_article('<article><p>' + BODY * 220 + '</p></article>')


def test_japanese_requires_no_spaces_and_preserves_sentence_offsets():
    text = '📰「図書館が再開しました。」利用者は入口を使えるようになりました！工事は完了しましたか？はい。'
    assert substantive(text * 2, 'ja')
    assert not substantive('これは短い。', 'ja')
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ['📰「図書館が再開しました。」', '利用者は入口を使えるようになりました！', '工事は完了しましたか？', 'はい。']
    assert ''.join(text[a:b] for a, b in spans) == text


def test_hash_and_extra_fields_cannot_silently_change_pilot_input():
    with pytest.raises(ValidationError, match='exact annotated text'):
        article(text=BODY + ' Changed.')
    with pytest.raises(ValidationError):
        article(candidate_prediction='positive')


def test_ready_intake_needs_two_passes_rights_events_and_historical_evidence():
    row = article()
    assert audit([row], [annotation(row)])['ready_for_development_intake']
    blocked = article(rights_status='unresolved', event_ids=[])
    report = audit([blocked], missing_protection=['data/processed/test.csv'])
    assert {b['code'] for b in report['blockers']} == {
        'rights_review_pending', 'event_review_pending', 'annotations_pending', 'historical_evidence_missing'}
    assert report['final_test_frozen'] is False


def test_annotation_change_needs_resolution_and_cannot_use_candidate_predictions():
    row = article()
    old = annotation(row).model_dump()
    old['second_pass']['label'] = 'neutral'
    old['resolution_note'] = ''
    with pytest.raises(ValidationError, match='resolution'):
        AnnotationRecord(**old)
    old['resolution_note'] = 'Reviewed the dominant subject again'
    old['second_pass']['candidate_predictions_seen'] = True
    with pytest.raises(ValidationError):
        AnnotationRecord(**old)
    wrong = annotation(row).model_copy(update={'text_sha256': '0' * 64})
    with pytest.raises(ValueError, match='another text'):
        audit([row], [wrong])


def test_unresolved_cases_remain_in_audit_without_becoming_neutral():
    row = article()
    values = annotation(row).model_dump()
    values['second_pass'].update(label='unresolved', ambiguous=True, ambiguity_note='No dominant target can be established')
    reviewed = AnnotationRecord(**values)
    result = audit([row], [reviewed])
    assert result['article_rows'] == 1
    assert result['language_label_counts'] == {'en:unresolved': 1}
    assert not result['ready_for_development_intake']
    assert {'id': row.id, 'code': 'unresolved_annotation'} in result['blockers']


def test_cross_language_event_and_transitive_bridge_protect_old_tests():
    first = article('one', text=BODY + ' First.', text_sha256=digest(BODY + ' First.'))
    second = article('two', title='Different title', text=BODY + ' Second.', text_sha256=digest(BODY + ' Second.'),
                     source_keys=['//publisher.example/old-test'])
    protected = [{'id': 'old', 'text': 'Completely different historical lead.', 'source_keys': "['//publisher.example/old-test']"}]
    report = audit([first, second], protected=protected)
    assert {b['id'] for b in report['blockers'] if b['code'] == 'historical_overlap'} == {'one', 'two'}
    assert any(b['code'] == 'cross_role_group' for b in audit([first, second.model_copy(update={'role': 'test'})])['blockers'])


def test_revision_change_cannot_reintroduce_a_protected_page():
    row = article(document_id='wikinews:ja:42')
    protected = [{'id': 'old-revision', 'language': 'ja', 'page_id': '42'}]
    assert any(b['code'] == 'historical_overlap' for b in audit([row], protected=protected)['blockers'])


def test_missing_historical_files_are_explicit(tmp_path):
    rows, missing, hashes = historical_records(tmp_path)
    assert rows == [] and hashes == {} and 'data/processed/test.csv' in missing


def test_changed_historical_file_cannot_pass_protection(tmp_path):
    data = tmp_path / 'data/processed'; data.mkdir(parents=True)
    (data / 'test.csv').write_text('id,text\none,changed\n')
    reports = tmp_path / 'reports'; reports.mkdir()
    (reports / 'data_manifest.json').write_text('{"splits":{"test":{"sha256":"old"}}}')
    _, issues, _ = historical_records(tmp_path)
    assert 'data/processed/test.csv (hash differs from frozen manifest)' in issues


def test_token_profile_reports_middle_loss_without_loading_a_model():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(4000))
        def __call__(self, text, **kwargs):
            return {'input_ids': [[1] * 128] * 32}
    report = profile([article()], Tokenizer())
    assert report['items'][0]['retained_fraction_at_article_limit'] == pytest.approx(.768)
    assert report['languages']['en']['over_article_limit'] == 1


def test_language_metric_and_close_rule_do_not_reward_a_japanese_only_gain():
    assert primary_score({'de': .7, 'en': .9, 'ja': 1.}) == pytest.approx(.8)
    rows = [dict(model='quality', comparable_group='same', primary_score=.8, inference_ms_per_text=100., full_model_size_bytes=8000),
            dict(model='fast', comparable_group='same', primary_score=.795, inference_ms_per_text=3., full_model_size_bytes=250)]
    assert select_candidate(rows) == 'fast'
    rows[1]['primary_score'] = .7949
    assert select_candidate(rows) == 'quality'
    rows[1]['comparable_group'] = 'other-test'
    with pytest.raises(ValueError, match='comparison group'):
        select_candidate(rows)


def test_run_identity_groups_inputs_and_labels_not_candidate_artifacts():
    values = dict(research_question='core', stage='development', data_version='full-v1', annotation_version='ai-v1',
        split_version='event-v1', model_family='MiniLM', adaptation='fine-tune', fold='0', seed=1,
        precision='bf16', context_policy=ARTICLE_POLICY, git_commit='a' * 40,
        **{key: 'b' * 64 for key in ['protocol_sha256', 'input_sha256', 'annotation_sha256', 'split_sha256', 'artifact_sha256', 'base_model_sha256']})
    first = RunIdentity(**values)
    second = RunIdentity(**dict(values, model_family='Qwen', artifact_sha256='c' * 64))
    assert first.tags()['comparable_group'] == second.tags()['comparable_group']
    assert first.tags()['comparable_group'] != first.model_copy(update={'annotation_sha256': 'd' * 64}).tags()['comparable_group']


@pytest.mark.parametrize('name', ['development', 'common_data_v3', 'expanded_news_v4', 'llm_adaptation_v5'])
def test_report_destinations_preserve_original_evidence(name, tmp_path):
    from src.reporting import ROOT, report_destination
    with pytest.raises(RuntimeError, match='Historical reports'):
        report_destination(ROOT / 'reports' / name)
    assert report_destination(tmp_path / name) == tmp_path / name


def test_definition_list_interview_answers_and_nested_paragraphs_are_retained_once():
    answer = 'The repaired entrance now allows every library user to enter safely.'
    html = f'<article><p>{BODY}</p><dl><dt>What changed?</dt><dd><p>{answer}</p><ul><li>Access restored.</li></ul></dd></dl></article>'
    text = extract_full_article(html)
    assert 'What changed?' in text
    assert text.count(answer) == 1
    assert text.count('Access restored.') == 1


def test_table_caption_and_declared_image_alternative_preserve_outcome_context():
    html = f'<article><p>{BODY}</p><table><caption>Seat changes</caption><tr><td>Party A</td><td><img alt="Loss" src="green-up.svg">20</td></tr></table></article>'
    text = extract_full_article(html)
    assert 'Seat changes' in text and 'Party A [Loss] 20' in text
    assert 'Gain' not in text  # Never silently correct a contradictory source alt.


def test_deep_wiki_footer_heading_stops_external_material():
    html = f'<div class="mw-parser-output"><p>{BODY}</p><h4>Weblinks</h4><p>Unrelated external link commentary.</p></div>'
    assert extract_full_article(html) == BODY


def test_pilot_finalize_rejects_changed_text_and_preserves_frozen_outputs(tmp_path, monkeypatch):
    import json
    from src import next_stage_pilot_report as report
    row = article()
    source = tmp_path / 'articles.jsonl'; source.write_text(row.model_dump_json() + '\n')
    review = annotation(row).first_pass.model_dump()
    item = dict(id=row.id, text_sha256=row.text_sha256, review=review, event_ids=row.event_ids,
                rights_status=row.rights_status, rights_review_note=row.rights_review_note,
                third_party_passages_reviewed=True)
    first = tmp_path / 'first.jsonl'; second = tmp_path / 'second.jsonl'
    first.write_text(json.dumps(item) + '\n'); second.write_text(json.dumps(item) + '\n')
    monkeypatch.setattr(report, 'historical_records', lambda: ([], [], {}))
    result, _ = report.finalize(source, first, second, tmp_path / 'out', input_version='v4', annotation_version='fixture', resolutions={})
    frozen = (tmp_path / 'out/annotations-v1-input-v4.jsonl').read_bytes()
    assert result['same_annotator_label_agreement']['independent_validation'] is False
    item['text_sha256'] = '0' * 64; second.write_text(json.dumps(item) + '\n')
    with pytest.raises(ValueError, match='Review input changed'):
        report.finalize(source, first, second, tmp_path / 'out', input_version='v4', resolutions={})
    assert (tmp_path / 'out/annotations-v1-input-v4.jsonl').read_bytes() == frozen


@pytest.mark.parametrize('language,text,expected', [
    ('de', 'Titel\n\nMainz (Deutschland), 21.04.2025 – ' + BODY, '2025-04-21'),
    ('en', 'Title\n\nWednesday, October 23, 2019\n\n' + BODY, '2019-10-23'),
    ('ja', '見出し\n\n【2010年3月1日】\n\n本文', '2010-03-01'),
    ('de', 'Titel\n\n31.02.2025 – ' + BODY, None),
])
def test_publication_dateline_is_not_page_creation(language, text, expected):
    from src.article import visible_publication_date
    assert visible_publication_date(text, language) == expected


def test_language_title_link_protects_an_uncollected_foreign_revision():
    row = article(related_document_ids=['wikinews:de:title:verwandter bericht'])
    protected = [{'id': 'old', 'language': 'de', 'title_key': 'de:Verwandter Bericht', 'text': 'An old lead.'}]
    assert any(b['code'] == 'historical_overlap' for b in audit([row], protected=protected)['blockers'])


def test_reviewed_components_preserve_transitive_protected_bridge():
    from src.next_stage_groups import components
    rows = [dict(id='a', event_ids=['one']), dict(id='b', event_ids=['one','two']), dict(id='protected',event_ids=['two'])]
    assert len(set(components(rows))) == 1


def test_full_article_encoder_pools_features_before_the_head():
    import numpy as np
    import torch
    from src.inference import Predictor
    class Tokenizer:
        model_input_names = ['input_ids','attention_mask']
        def __call__(self, text, **kwargs):
            return {'input_ids':[[1],[3]],'attention_mask':[[1],[1]]}
        def pad(self, rows, **kwargs):
            class Tokens(dict):
                def to(self, device): return self
            return Tokens(input_ids=torch.tensor([r['input_ids'] for r in rows]), attention_mask=torch.ones(len(rows),1))
    class Net:
        def features(self, tokens): return tokens['input_ids'].float()
        def head(self, features): return torch.cat([features, -features, features*0],dim=-1)
    predictor=Predictor.__new__(Predictor);predictor.kind='article_encoder';predictor.device='cpu'
    predictor.meta={'max_length':128,'article_feature_pooling':'mean'};predictor.net=Net();predictor.tokenizer=Tokenizer()
    expected=torch.tensor([[2.,-2.,0.]]).softmax(-1).numpy()
    np.testing.assert_allclose(predictor.predict_proba(['A complete synthetic article.']),expected)


def test_qwen_full_text_overflow_fails_instead_of_inheriting_legacy_middle_omission():
    from src.local_llm import LocalLLMPredictor
    predictor=LocalLLMPredictor.__new__(LocalLLMPredictor)
    class Tokenizer:
        def encode(self,text,**kwargs):return list(range(5))
    predictor.tokenizer=Tokenizer();predictor.meta={'max_article_tokens':4,'article_overflow':'reject'}
    with pytest.raises(ArticleError,match='no text was truncated'):
        predictor.messages(BODY)


def test_fresh_test_quota_keeps_whole_groups_and_does_not_use_class_labels():
    from src.next_stage_test_prepare import select_groups
    rows=[article('a'),article('b'),article('c')]
    selected,excluded=select_groups(rows,['pair','pair','single'],{'a','b','c'},1)
    assert selected=={'c'}
    assert {r['id'] for r in excluded}=={'a','b'}
    selected,_=select_groups(rows,['pair','pair','single'],{'a','c'},3)
    assert selected=={'c'}  # One blocked member reserves the entire component.


def test_source_quality_review_is_complete_pre_prediction_and_hash_bound():
    from copy import deepcopy
    from src.next_stage_test_prepare import quality_exclusions,select_groups
    rows=[article('a'),article('b'),article('c')]
    review={'status':'reviewed-before-candidate-test-inference','candidate_predictions_seen':False,
            'decisions':[{'id':'a','text_sha256':rows[0].text_sha256,
                          'decision':'exclude_from_clean_full_article_test','note':'Unrelated link list in extraction.'}]}
    assert quality_exclusions(rows,review)=={'a':'Unrelated link list in extraction.'}
    selected,_=select_groups(rows,['pair','pair','single'],{'b','c'},300)
    assert selected=={'c'}
    for field,value in [('status','in-progress-before-test-inference'),('candidate_predictions_seen',True)]:
        invalid=deepcopy(review);invalid[field]=value
        with pytest.raises(ValueError):quality_exclusions(rows,invalid)
    for field,value in [('id','unknown'),('text_sha256','0'*64),('decision','remove_candidate_error'),('note','')]:
        invalid=deepcopy(review);invalid['decisions'][0][field]=value
        with pytest.raises(ValueError):quality_exclusions(rows,invalid)
    invalid=deepcopy(review);invalid['decisions']*=2
    with pytest.raises(ValueError,match='duplicate'):quality_exclusions(rows,invalid)


def test_resolved_class_disagreement_is_not_presented_as_clear_reference():
    from src.next_stage_test_report import reference_ambiguous
    def item(first_label,second_label,first_flag=False,second_flag=False):
        return {'first_pass':{'label':first_label,'ambiguous':first_flag},
                'second_pass':{'label':second_label,'ambiguous':second_flag}}
    assert reference_ambiguous(item('positive','negative'))
    assert reference_ambiguous(item('neutral','neutral',first_flag=True))
    assert reference_ambiguous(item('neutral','neutral',second_flag=True))
    assert not reference_ambiguous(item('negative','negative'))


def test_upstream_aliases_remove_tracking_without_losing_content_queries():
    from src.next_stage_final_collect import source_aliases
    original='//publisher.example/article?id=42&utm_source=mail&fbclid=abc'
    assert source_aliases([original])==['//publisher.example/article?id=42',original]
    assert '//publisher.example/article?id=43' not in source_aliases([original])


@pytest.mark.parametrize('name',['../original','/tmp/overwrite','a/b','.', ''])
def test_reproduction_namespace_cannot_target_original_or_external_paths(name,tmp_path):
    from src.next_stage_reproduce import paths
    with pytest.raises(ValueError,match='namespace'):
        paths(name,tmp_path)


def test_reproduction_paths_preserve_original_measurement_directories(tmp_path):
    from src.next_stage_reproduce import paths
    reports,models=paths('audit_1',tmp_path)
    assert reports==tmp_path/'reports/next_stage/reproduction/audit_1'
    assert models==tmp_path/'models/next_stage/reproduction/audit_1'
