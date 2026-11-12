"""Explicit v4-to-v5 pilot migration and conservative development intake.

Original checkpoint files remain immutable. Only exact-hash reviews transfer;
changed texts need new explicit reviews. No candidate inference or MLflow writes.
"""
import json
from collections import Counter
from pathlib import Path

from src.article import visible_publication_date
from src.next_stage import (ROOT, ArticleRecord, AnnotationRecord, audit, digest,
                            historical_records, normalized, overlap_keys, read_records)
from src.next_stage_collect import LOCAL, OUT, write_json


def frozen_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != content:
        raise RuntimeError('Refusing to overwrite frozen output: ' + str(path))
    path.write_text(content)


def jsonl(path, rows):
    frozen_write(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def related_titles(row):
    return sorted({'wikinews:' + link['lang'] + ':title:' + normalized(link['title'])
                   for link in row.get('langlinks', [])})


def prepare():
    old = {a.id: a for a in read_records(LOCAL / 'reviewed-articles-v1-input-v4.jsonl', ArticleRecord)}
    annotations = {a.id: a for a in read_records(LOCAL / 'annotations-v1-input-v4.jsonl', AnnotationRecord)}
    rows = read_records(LOCAL / 'articles-full-article-v5.jsonl', ArticleRecord)
    changes = []
    for number in (1, 2):
        changes.append({r['id']: r for r in map(json.loads, (LOCAL / f'review-v5-changed-pass{number}.jsonl').read_text().splitlines())})
    if set(changes[0]) != set(changes[1]):
        raise ValueError('Changed texts need both new reading passes')
    materials = {r['id']: r for r in map(json.loads, (LOCAL / 'review_material-full-article-v4.jsonl').read_text().splitlines())}
    protected, missing, hashes = historical_records()
    # Protect all old news collection pages and their source/translation links,
    # a conservative superset of the held-out-only rule.
    protection_hashes = dict(hashes)
    for name in ('common_data_v3', 'expanded_news_v4', 'llm_adaptation_v5'):
        path = ROOT / 'data' / name / 'articles.jsonl'
        protection_hashes[str(path.relative_to(ROOT))] = digest(path.read_text())
        protected.extend(json.loads(x) for x in path.read_text().splitlines())
    migrated, accepted, accepted_annotations, exclusions, dates = [], [], [], [], []
    for a in rows:
        prior = annotations[a.id]
        if a.text_sha256 == prior.text_sha256:
            values = prior.model_dump()
            values['annotation_version'] = 'pilot-300-v2-input-v5-exact-hash-transfer'
            annotation = AnnotationRecord.model_validate(values)
            transfer = 'Exact text hash unchanged; original two saved review passes transferred explicitly.'
        elif a.id in changes[0]:
            first, second = [c[a.id] for c in changes]
            if any(c['text_sha256'] != a.text_sha256 for c in (first, second)):
                raise ValueError('Changed review refers to another input: ' + a.id)
            values = prior.model_dump()
            values.update(text_sha256=a.text_sha256, annotation_version='pilot-300-v2-input-v5-re-read',
                          first_pass=first['review'], second_pass=second['review'],
                          resolution_note='New exact v5 text was read twice after extraction changed. The second-pass target is authoritative; original v4 reviews remain preserved.')
            annotation = AnnotationRecord.model_validate(values)
            transfer = 'Changed text: two explicit new reviews on the v5 hash.'
        else:
            exclusions.append({'id': a.id, 'codes': ['changed_text_review_pending', 'rights_review_pending']})
            continue
        migrated.append(annotation.model_dump())
        value = a.model_dump()
        value['related_document_ids'] = sorted(set(value['related_document_ids']) | set(related_titles(materials[a.id])))
        displayed = visible_publication_date(a.text, a.language)
        dates.append({'id': a.id, 'page_created_at': old[a.id].published_at,
                      'displayed_publication_date': displayed,
                      'basis': 'Pinned visible dateline; not independently verified event/publication chronology. Original v4 published_at was incorrectly page creation time.'})
        if displayed:
            value['published_at'] = displayed
        value['role'] = 'train'
        article = ArticleRecord.model_validate(value)
        report = audit([article], [annotation], protected, missing)
        codes = sorted({b['code'] for b in report['blockers']})
        if displayed is None:
            codes.append('publication_dateline_unresolved')
        if codes:
            exclusions.append({'id': a.id, 'codes': codes})
        else:
            accepted.append(article)
            accepted_annotations.append(annotation)
    # Keep selection separate from the unmodified complete checkpoint.
    target = LOCAL / 'development-v1'
    jsonl(target / 'articles.jsonl', [r.model_dump() for r in accepted])
    jsonl(target / 'annotations.jsonl', [r.model_dump() for r in accepted_annotations])
    jsonl(OUT / 'annotations-v2-input-v5.jsonl', migrated)
    jsonl(OUT / 'publication-provenance-v1.jsonl', dates)
    jsonl(OUT / 'development-provenance-v1.jsonl', [{k: v for k, v in r.model_dump().items() if k != 'text'} for r in accepted])
    report = audit(accepted, accepted_annotations, protected, missing)
    report.update(input_version='full-article-v5', train_only=True, exclusions=exclusions,
                  original_rows=len(rows), migrated_annotation_rows=len(migrated),
                  original_input_sha256=digest((LOCAL / 'articles-full-article-v4.jsonl').read_text()),
                  new_input_sha256=digest((LOCAL / 'articles-full-article-v5.jsonl').read_text()),
                  protection_hashes=protection_hashes,
                  migration='Exact-hash transfer only; five changed compatible texts re-read twice. Four changed rights-blocked inputs remain unreviewed in v5.',
                  protected_scope='Original held-outs plus all 1319 old news collection pages and their observed source/translation components.',
                  population='Second seeded quote-restricted pilot; all development-exposed; no final test.',
                  publication_metadata='Visible dateline replaces mistaken page-creation value in development records; old raw bytes preserved. Do not infer verified chronology.')
    write_json(OUT / 'development-intake-v1.json', report)
    print(json.dumps({k: report[k] for k in ('article_rows', 'language_label_counts', 'observed_groups', 'ready_for_development_intake', 'exclusions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    prepare()
