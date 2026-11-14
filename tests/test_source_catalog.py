"""Readable public sources must preserve membership, references and provenance."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_source_catalog as catalog

ROOT = Path(__file__).resolve().parents[1]


def test_selected_and_excluded_sources_match_frozen_membership_and_labels():
    groups = catalog.catalog_rows()
    assert Counter(r['language'] for r in groups['train']) == {'de': 78, 'en': 71, 'ja': 91}
    assert Counter(r['language'] for r in groups['test']) == {'de': 273, 'en': 118, 'ja': 300}
    manifest = json.loads((ROOT / catalog.INPUTS[4]).read_text())
    assert {r['id'] for r in groups['excluded']} == {r['id'] for r in manifest['exclusions']}
    assert not {r['id'] for r in groups['test']} & {r['id'] for r in groups['excluded']}
    annotations = catalog.records(ROOT / catalog.INPUTS[3])
    different = {r['id']: r for r in annotations if r['first_pass']['label'] != r['second_pass']['label']}
    assert different
    for row in groups['test']:
        if row['id'] in different:
            assert row['label'] == different[row['id']]['second_pass']['label']
    # Quota rows use the original singular `code` key, unlike most exclusions.
    quota = [r for r in groups['excluded'] if 'whole_group_source_order_quota' in r['exclusion_codes']]
    assert len(quota) == 2


def test_source_annotation_hash_mismatch_is_rejected(monkeypatch):
    original = catalog.records

    def changed(path):
        rows = deepcopy(original(path))
        if path.name == 'train-annotations-v1.jsonl':
            rows[0]['text_sha256'] = '0' * 64
        return rows

    monkeypatch.setattr(catalog, 'records', changed)
    with pytest.raises(ValueError, match='hash mismatch'):
        catalog.catalog_rows()


@pytest.mark.parametrize('value', [
    'javascript:alert(1)', 'http://de.wikinews.org/wiki/Example',
    'https://de.wikinews.org@evil.example/wiki/Example',
])
def test_non_source_or_unsafe_links_are_rejected(value):
    with pytest.raises(ValueError, match='source URL'):
        catalog.source_url(value)


def test_markdown_title_cannot_inject_an_extra_link_or_html():
    value = catalog.cell('<script>bad</script> [x](javascript:bad)|next')
    assert '<script>' not in value
    assert '\\[x\\]' in value and '\\|' in value
    assert '%28' in catalog.source_url('https://en.wikinews.org/wiki/Example_(test)')


def test_derived_presentation_is_bound_to_original_results_and_review():
    path = ROOT / 'reports/data-review/comparison-presentation-v1.json'
    manifest = json.loads(path.read_text())
    assert manifest['model_calls'] == 0
    assert hashlib.sha256((ROOT / manifest['source_results']).read_bytes()).hexdigest() == manifest['source_sha256']
    assert hashlib.sha256((ROOT / manifest['review_record']).read_bytes()).hexdigest() == manifest['review_record_sha256']
    for name, sha in manifest['output_files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha
