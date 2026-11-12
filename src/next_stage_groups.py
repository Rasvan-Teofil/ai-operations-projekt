"""Reviewed transitive event components and a frozen grouped development split."""
import json
from collections import Counter

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.next_stage import ROOT, ArticleRecord, AnnotationRecord, audit, digest, historical_records, overlap_keys, read_records
from src.next_stage_collect import LOCAL, OUT, write_json
from src.next_stage_prepare import frozen_write, jsonl


def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        for key in overlap_keys(row):
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    names = {}
    for i, row in enumerate(rows):
        root = find(i)
        names[root] = min(names.get(root, row['id']), row['id'])
    return [names[find(i)] for i in range(len(rows))]


def prepare():
    articles = read_records(LOCAL / 'development-v1/articles.jsonl', ArticleRecord)
    annotations = {a.id: a for a in read_records(LOCAL / 'development-v1/annotations.jsonl', AnnotationRecord)}
    protected, missing, hashes = historical_records()
    for name in ('common_data_v3', 'expanded_news_v4', 'llm_adaptation_v5'):
        protected.extend(json.loads(x) for x in (ROOT / 'data' / name / 'articles.jsonl').read_text().splitlines())
    rows = [a.model_dump() for a in articles] + protected
    indexed = {}
    for row in rows:
        indexed.setdefault(row['id'], []).append(row)
    review = json.loads((OUT / 'group-review-v1.json').read_text())
    for pair in review['pairs']:
        if pair['decision'] == 'join_conservative_component':
            key = 'reviewed-edge:' + digest(':'.join(sorted([pair['id'], pair['other']])))
            for identifier in (pair['id'], pair['other']):
                for row in indexed[identifier]:
                    row['event_ids'] = sorted(set(row.get('event_ids', [])) | {key})
    all_groups = components(rows)
    protected_groups = set(all_groups[len(articles):])
    accepted, excluded = [], []
    for row, group in zip(rows[:len(articles)], all_groups):
        if group in protected_groups:
            excluded.append({'id': row['id'], 'code': 'reviewed_related_historical_component', 'group': group})
        else:
            accepted.append((ArticleRecord.model_validate(row), group))
    accepted_annotations = [annotations[a.id] for a, _ in accepted]
    result = audit([a for a, _ in accepted], accepted_annotations, rows[len(articles):], missing)
    if not result['ready_for_development_intake']:
        raise ValueError('Intake remains blocked: ' + repr(result['blockers']))
    target = LOCAL / 'development-v2'
    jsonl(target / 'articles.jsonl', [a.model_dump() for a, _ in accepted])
    jsonl(target / 'annotations.jsonl', [a.model_dump() for a in accepted_annotations])
    table = pd.DataFrame([{**a.model_dump(), 'label': annotations[a.id].second_pass.label, 'group': group}
                          for a, group in accepted])
    frozen_write(target / 'train.csv', table.to_csv(index=False))
    membership = []
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261110)
    strata = table.language + ':' + table.label
    for fold, (fit, dev) in enumerate(splitter.split(table, strata, table['group'])):
        train, validation = table.iloc[fit], table.iloc[dev]
        assert not set(train['group']) & set(validation['group'])
        for part in (train, validation):
            counts = Counter(part.language + ':' + part.label)
            if any(counts.get(lang + ':' + label, 0) == 0 for lang in ('de', 'en', 'ja') for label in ('negative', 'neutral', 'positive')):
                raise ValueError('Fixed split lacks language/class; do not search seeds after scoring')
        membership.append({'fold': fold, 'fit_ids': train.id.tolist(), 'dev_ids': validation.id.tolist(),
                           'fit_language_label_counts': dict(Counter(train.language + ':' + train.label)),
                           'dev_language_label_counts': dict(Counter(validation.language + ':' + validation.label))})
    frozen_write(OUT / 'cv-membership-v1.json', json.dumps(membership, ensure_ascii=False, indent=2) + '\n')
    result.update(exclusions=excluded, train_csv_sha256=digest((target / 'train.csv').read_text()),
                  article_sha256=digest((target / 'articles.jsonl').read_text()),
                  annotation_sha256=digest((target / 'annotations.jsonl').read_text()),
                  group_review_sha256=digest((OUT / 'group-review-v1.json').read_text()),
                  cv_membership_sha256=digest((OUT / 'cv-membership-v1.json').read_text()),
                  seed=20261110, folded_inside_train_only=True,
                  quantity_decision='Use all eligible second-pilot inputs for the bounded first full-article experiment, not an assumed 3000-row optimum. Scarce positive/neutral cells and source-frame restriction limit precision. Nested learning curves determine whether a later independently frozen expansion is warranted.',
                  final_test='No final sample collected, annotated or predicted yet; all current articles are development-exposed.')
    write_json(OUT / 'development-intake-v2.json', result)
    print(json.dumps({k: result[k] for k in ('article_rows','language_label_counts','observed_groups','exclusions')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    prepare()
