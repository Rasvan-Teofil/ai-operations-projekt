"""Join two explicit assistant reviews to exact inputs and report pilot gates.

No model predictions, automatic labels, training, split writing or MLflow writes.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.next_stage import AnnotationRecord, ArticleRecord, audit, historical_records, read_records
from src.next_stage_collect import write_json


def finalize(articles_path, first_path, second_path, output, *, input_version="v3",
             annotation_version="pilot-90-v1-input-v3", resolutions=None):
    if input_version not in ("v3", "v4", "v5"):
        raise ValueError("Unsupported pilot input version")
    articles = read_records(articles_path, ArticleRecord)
    passes = []
    for path in (first_path, second_path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        indexed = {row['id']: row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != {a.id for a in articles}:
            raise ValueError('Each pass must contain exactly one review per exact input')
        passes.append(indexed)
    annotations, reviewed = [], []
    changes = {
        'wikinews:de:85997': 'Second review resolves the first-pass uncertainty for the officially declared winner Maduro. The positive label concerns the declared electoral result, not the fairness of the election or an uncontested mandate.',
        'wikinews:de:80559': 'Second review resolves the first-pass uncertainty for the arrested accused. Arrest is an actual adverse legal consequence for that central actor; no finding of guilt is implied. The investigative perspective would differ.',
    }
    if resolutions is not None:
        changes = resolutions
    unknown = set(changes) - {a.id for a in articles}
    if unknown:
        raise ValueError("Resolution references unknown input: " + str(sorted(unknown)))
    for article in articles:
        first, second = (p[article.id] for p in passes)
        if first['text_sha256'] != article.text_sha256 or second['text_sha256'] != article.text_sha256:
            raise ValueError('Review input changed: ' + article.id)
        changed = first['review']['label'] != second['review']['label']
        if changed and article.id not in changes:
            raise ValueError('Explicit label-resolution note missing: ' + article.id)
        note = changes.get(article.id, '')
        if not note and first['review']['target'] != second['review']['target']:
            note = 'Second reading specifies the central actor/development more precisely; the class is unchanged. The second-pass target and rationale are authoritative for this annotation version.'
        annotation = AnnotationRecord(schema_version='article-annotation-v1', id=article.id,
            text_sha256=article.text_sha256, annotation_version=annotation_version,
            rubric_version='outcome-rubric-v1', provenance='AI-labelled · Codex-reviewed',
            annotator='Codex', annotator_revision='unavailable',
            annotation_license='CC0-1.0 to the extent project rights apply',
            first_pass=first['review'], second_pass=second['review'], resolution_note=note)
        annotations.append(annotation)
        updated = article.model_dump()
        for key in ('event_ids', 'rights_status', 'rights_review_note', 'third_party_passages_reviewed'):
            updated[key] = second[key]
        reviewed.append(ArticleRecord.model_validate(updated))
    output.mkdir(parents=True, exist_ok=True)
    annotated_path = output / f'annotations-v1-input-{input_version}.jsonl'
    reviewed_path = output / f'reviewed-articles-v1-input-{input_version}.jsonl'
    for path, rows in ((annotated_path, annotations), (reviewed_path, reviewed)):
        content = ''.join(row.model_dump_json() + '\n' for row in rows)
        if path.exists() and path.read_text() != content:
            raise ValueError('Refusing to replace a different finalized pilot: ' + str(path))
        path.write_text(content)
    protected, missing, hashes = historical_records()
    result = audit(reviewed, annotations, protected, missing)
    result['protected_file_hashes'] = hashes
    result['input_file_sha256'] = hashlib.sha256(articles_path.read_bytes()).hexdigest()
    result['annotation_file_sha256'] = hashlib.sha256(annotated_path.read_bytes()).hexdigest()
    result['reviewed_article_file_sha256'] = hashlib.sha256(reviewed_path.read_bytes()).hexdigest()
    result['same_annotator_label_agreement'] = {
        'matches': sum(a.first_pass.label == a.second_pass.label for a in annotations),
        'total': len(annotations), 'independent_validation': False,
        'interpretation': 'Self-consistency only; neither annotation accuracy nor human agreement.'}
    result['rights_by_language'] = {lang: dict(Counter(a.rights_status for a in reviewed if a.language == lang)) for lang in ('de', 'en', 'ja')}
    eligible = {a.id for a in reviewed if a.rights_status == 'reviewed_compatible'}
    result['rights_eligible_class_counts'] = dict(Counter(f'{article.language}:{annotation.second_pass.label}' for article, annotation in zip(reviewed, annotations) if article.id in eligible))
    times = [datetime.fromisoformat(a.first_pass.reviewed_at) for a in annotations] + [datetime.fromisoformat(a.second_pass.reviewed_at) for a in annotations]
    result['annotation_timing'] = {'first_saved_review': min(times).isoformat(), 'last_saved_review': max(times).isoformat(),
        'elapsed_saved_review_window_seconds': (max(times) - min(times)).total_seconds(),
        'limits': 'Elapsed wall-clock window between saved reviews includes other work; not measured active annotation labor or a scalable per-item estimate.'}
    result['scope'] = '90 first-checkpoint pilot inputs, 30/language; all are development-exposed. No fresh final test, development score or new training run.'
    result['extraction_findings'] = ['Some Japanese topic-navigation tables precede body text.', 'Some English related/sister-link sections and hidden original-language quotation tooltips remain.', 'Some German comment links remain.']
    result['next_decision'] = 'Repair extraction in a new version and preserve this exact reviewed checkpoint. Expand the rights-reviewed source frame before freezing any train/test sample; current eligible English inputs have no positive class.'
    if input_version != 'v3':
        result['scope'] = f'{len(articles)} second-checkpoint pilot inputs; all are development-exposed. No fresh final test, development score or new training run.'
        result['extraction_findings'] = ['Some interview answers in definition lists are missing.', 'Some Weblinks/original-reporting footers remain.', 'Image-rendered table signs need verification.']
        result['next_decision'] = 'Preserve this exact reviewed checkpoint. Repair extraction in a new version, review changed texts, and check source/event groups before freezing training or a fresh test.'
    result['input_version'] = input_version
    result['annotation_version'] = annotation_version
    return result, annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--articles', type=Path, required=True)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--local-output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--annotation-metadata', type=Path, required=True)
    parser.add_argument('--input-version', default='v3')
    parser.add_argument('--annotation-version', default='pilot-90-v1-input-v3')
    parser.add_argument('--resolutions', type=Path)
    args = parser.parse_args()
    result, annotations = finalize(args.articles, args.first, args.second, args.local_output,
        input_version=args.input_version, annotation_version=args.annotation_version,
        resolutions=json.loads(args.resolutions.read_text()) if args.resolutions else None)
    write_json(args.report, result)
    args.annotation_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.annotation_metadata.write_text(''.join(a.model_dump_json()+'\n' for a in annotations))
    print(json.dumps({k: result[k] for k in ('article_rows', 'language_label_counts', 'rights_by_language', 'rights_eligible_class_counts', 'same_annotator_label_agreement')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
