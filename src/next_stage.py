"""Offline pilot intake, historical-overlap audit and tokenizer diagnostics.

No collection, annotation, split writing, training, model inference or MLflow writes.
Raw pilot JSONL belongs under ignored data/next_stage/, never in Git.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.article import ARTICLE_POLICY, substantive

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('negative', 'neutral', 'positive')
SHA = r'^[0-9a-f]{64}$'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def normalized(text):
    return ' '.join(re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=False)


class ArticleRecord(Record):
    schema_version: Literal['article-pilot-v1']
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1, description='Stable across revisions, e.g. wikinews:ja:123')
    language: Literal['de', 'en', 'ja']
    title: str = Field(min_length=1)
    text: str = Field(min_length=40, max_length=20000)
    text_sha256: str = Field(pattern=SHA)
    source_url: str
    revision_url: str
    revision: str = Field(min_length=1)
    history_url: str
    attribution: str = Field(min_length=1)
    published_at: str = Field(min_length=10)
    retrieved_at: str = Field(min_length=10)
    source_license: str = Field(min_length=1)
    source_license_url: str
    rights_evidence_url: str
    rights_status: Literal['reviewed_compatible', 'unresolved', 'restricted']
    rights_review_note: str = Field(min_length=1)
    third_party_passages_reviewed: bool
    extraction_version: Literal['full-article-v1', 'full-article-v2', 'full-article-v3', 'full-article-v4', 'full-article-v5']
    transformation: str = Field(min_length=1)
    related_document_ids: list[str] = Field(default_factory=list)
    source_keys: list[str] = Field(default_factory=list, description='Canonical upstream article URLs')
    event_ids: list[str] = Field(default_factory=list, description='Reviewed cross-language event groups')
    role: Literal['pilot', 'train', 'dev', 'test'] = 'pilot'

    @field_validator('source_url', 'revision_url', 'history_url', 'source_license_url', 'rights_evidence_url')
    @classmethod
    def http_url(cls, value):
        u = urlsplit(value)
        if u.scheme not in ('http', 'https') or not u.hostname or u.username or u.password or any(c.isspace() for c in value):
            raise ValueError('An explicit HTTP(S) provenance URL is required')
        return value

    @model_validator(mode='after')
    def exact_input(self):
        if digest(self.text) != self.text_sha256:
            raise ValueError('text_sha256 must identify the exact annotated text')
        if not substantive(self.text, self.language):
            raise ValueError('No substantive text for this language')
        return self


class ReviewPass(Record):
    label: Literal['negative', 'neutral', 'positive', 'unresolved']
    target: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    ambiguous: bool
    ambiguity_note: str
    reviewed_at: str = Field(min_length=10)
    candidate_predictions_seen: Literal[False]

    @model_validator(mode='after')
    def ambiguity_documented(self):
        if not self.target.strip() or not self.rationale.strip():
            raise ValueError('Target and rationale cannot be blank')
        if self.ambiguous and not self.ambiguity_note.strip():
            raise ValueError('Ambiguity needs an explicit note, not a neutral fallback')
        if self.label == 'unresolved' and not self.ambiguous:
            raise ValueError('An unresolved label must be marked ambiguous')
        return self


class AnnotationRecord(Record):
    schema_version: Literal['article-annotation-v1']
    id: str = Field(min_length=1)
    text_sha256: str = Field(pattern=SHA)
    annotation_version: str = Field(min_length=1)
    rubric_version: Literal['outcome-rubric-v1']
    provenance: Literal['AI-labelled · Codex-reviewed']
    annotator: Literal['Codex']
    annotator_revision: str = Field(min_length=1, description='Use unavailable when exact revision is unknown')
    annotation_license: Literal['CC0-1.0 to the extent project rights apply']
    first_pass: ReviewPass
    second_pass: ReviewPass
    resolution_note: str

    @model_validator(mode='after')
    def changes_explained(self):
        if (self.first_pass.label != self.second_pass.label or self.first_pass.target != self.second_pass.target) and not self.resolution_note.strip():
            raise ValueError('Changed label or target requires a resolution note')
        return self


def read_records(path, schema):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(schema.model_validate_json(line))
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
    if not rows or len({r.id for r in rows}) != len(rows):
        raise ValueError(f'{path}: nonempty records with unique IDs required')
    return rows


PROTECTED_FILES = (
    'data/processed/val.csv', 'data/processed/test.csv',
    'data/benchmark/shared.csv', 'data/local_llm/shared.csv',
    'data/common_data_v3/val.csv', 'data/common_data_v3/test.csv',
    'data/expanded_news_v4/test.csv', 'data/llm_adaptation_v5/test.csv',
)


def historical_records(root=ROOT):
    """Read known held-out rows only; fail closed if local evidence is missing."""
    rows, missing, files = [], [], {}
    for name in PROTECTED_FILES:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name.startswith('data/processed/'):
            manifest = root / 'reports/data_manifest.json'
        elif name == 'data/benchmark/shared.csv':
            manifest = root / 'reports/development/benchmark_manifest.json'
        elif name == 'data/local_llm/shared.csv':
            manifest = root / 'reports/local_llm/benchmark_manifest.json'
        else:
            manifest = root / 'reports' / path.parent.name / 'split_manifest.json'
        try:
            expected = json.loads(manifest.read_text())
            expected_hash = expected['splits'][path.stem]['sha256'] if 'splits' in expected else expected['sha256']
            if files[name] != expected_hash:
                missing.append(name + ' (hash differs from frozen manifest)')
        except (OSError, ValueError, KeyError):
            missing.append(str(manifest.relative_to(root)) + ' (frozen hash unavailable)')
        with path.open(encoding='utf-8') as stream:
            rows.extend(dict(row, protected_file=name) for row in csv.DictReader(stream))
    return rows, missing, files


def _list(value):
    # Historical CSVs encode list columns as JSON or Python list literals.
    if isinstance(value, list):
        return value
    if not value:
        return []
    import ast
    try:
        result = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    return result if isinstance(result, list) else [value]


def overlap_keys(row):
    """Stable pages, exact text/title, upstream sources and reviewed event links.

    This deliberately does not claim semantic near-duplicate/event detection.
    """
    keys = set()
    for field in ('id', 'document_id', 'text_sha256'):
        if row.get(field):
            keys.add(f'{field}:{row[field]}')
    if row.get('page_id') and row.get('language'):
        keys.add(f'document_id:wikinews:{row["language"]}:{row["page_id"]}')
    for value in [row.get('text'), row.get('headline'), row.get('sentence')]:
        if value:
            keys.add('text:' + normalized(value))
            keys.add('title:' + normalized(value.split('\n')[0]))
    if row.get('title'):
        keys.add('title:' + normalized(row['title']))
    if row.get('title_key'):
        keys.add('title:' + normalized(row['title_key'].split(':', 1)[-1]))
    for field, prefix in [('related_document_ids', 'document_id'), ('source_keys', 'source'), ('event_ids', 'event')]:
        keys.update(f'{prefix}:{value}' for value in _list(row.get(field)))
    # Language-qualified title aliases connect links even when their remote
    # page IDs were never collected. Historical lists use 'language:title'.
    language = row.get('language')
    title = row.get('title') or (row.get('title_key', '').split(':', 1)[-1])
    if language and title:
        keys.add('document_id:wikinews:' + language + ':title:' + normalized(title))
    for link in _list(row.get('linked_articles')):
        if isinstance(link, str) and ':' in link:
            lang, linked_title = link.split(':', 1)
            keys.add('document_id:wikinews:' + lang + ':title:' + normalized(linked_title))
    if row.get('group'):
        keys.add('legacy_group:' + row['group'])
    return keys


def audit(articles, annotations=None, protected=(), missing_protection=()):
    """Transitive grouping includes held-out rows: a bridge cannot hide overlap."""
    raw = [r.model_dump() for r in articles]
    all_rows = raw + list(protected)
    parent = list(range(len(all_rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, row in enumerate(all_rows):
        for key in overlap_keys(row):
            if key in owners:
                parent[find(i)] = find(owners[key])
            owners[key] = i
    protected_groups = {find(i) for i in range(len(raw), len(all_rows))}
    groups = defaultdict(list)
    blockers = []
    for i, row in enumerate(articles):
        group = find(i)
        groups[group].append(row)
        if group in protected_groups:
            blockers.append({'id': row.id, 'code': 'historical_overlap'})
        if row.rights_status != 'reviewed_compatible' or not row.third_party_passages_reviewed:
            blockers.append({'id': row.id, 'code': 'rights_review_pending'})
        if not row.event_ids:
            blockers.append({'id': row.id, 'code': 'event_review_pending'})
    for members in groups.values():
        if len({r.role for r in members}) > 1:
            blockers.append({'ids': [r.id for r in members], 'code': 'cross_role_group'})
    if missing_protection:
        blockers.append({'code': 'historical_evidence_missing', 'files': list(missing_protection)})
    counts = Counter()
    if annotations is None:
        blockers.append({'code': 'annotations_pending'})
    else:
        indexed = {r.id: r for r in annotations}
        if len(indexed) != len(annotations) or set(indexed) != {r.id for r in articles}:
            raise ValueError('Exactly one two-pass annotation per article is required')
        for row in articles:
            annotation = indexed[row.id]
            if annotation.text_sha256 != row.text_sha256:
                raise ValueError(f'{row.id}: annotation refers to another text')
            counts[f'{row.language}:{annotation.second_pass.label}'] += 1
            if annotation.second_pass.label == 'unresolved':
                blockers.append({'id': row.id, 'code': 'unresolved_annotation'})
    return {'schema_version': 'pilot-audit-v1', 'article_rows': len(articles),
            'language_counts': dict(Counter(r.language for r in articles)),
            'language_label_counts': dict(counts), 'observed_groups': len(groups),
            'blockers': blockers, 'ready_for_development_intake': not blockers,
            'final_test_frozen': False, 'label_status': 'AI-labelled · Codex-reviewed' if annotations else 'unlabelled',
            'limits': 'Exact/provenance/event-key audit only. Semantic near-duplicates, translation history and residual events require review before splitting. Two passes are one AI annotator, not independent verification.'}


def profile(articles, tokenizer, chunk_length=128, article_limit=3072):
    """CPU tokenization only; lengths include every article token, no truncation."""
    by_language = defaultdict(list)
    items = []
    for row in articles:
        n = len(tokenizer.encode(row.text, add_special_tokens=False, verbose=False))
        encoded = tokenizer(row.text, truncation=True, max_length=chunk_length, return_overflowing_tokens=True)
        chunks = len(encoded['input_ids'])
        by_language[row.language].append(n)
        items.append({'id': row.id, 'text_sha256': row.text_sha256, 'tokens': n,
                      'chunks': chunks, 'retained_fraction_at_article_limit': min(1., article_limit / max(1, n))})
    return {'extraction_versions': sorted({r.extraction_version for r in articles}), 'chunk_length': chunk_length,
            'article_limit': article_limit, 'items': items,
            'languages': {language: {'rows': len(values), 'min': min(values), 'median': statistics.median(values),
                                     'max': max(values), 'over_article_limit': sum(n > article_limit for n in values)}
                          for language, values in by_language.items()},
            'limits': 'Article tokens only; prompt, demonstrations and special-token overhead need a separate capacity check. No model inference or GPU allocation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('audit')
    check.add_argument('--articles', type=Path, required=True)
    check.add_argument('--annotations', type=Path)
    tokens = sub.add_parser('profile')
    tokens.add_argument('--articles', type=Path, required=True)
    tokens.add_argument('--tokenizer', type=Path, required=True, help='Existing local tokenizer directory; no downloads')
    tokens.add_argument('--chunk-length', type=int, default=128)
    tokens.add_argument('--article-limit', type=int, default=3072)
    sub.add_parser('schema')
    args = parser.parse_args()
    if args.action == 'schema':
        result = {'articles': ArticleRecord.model_json_schema(), 'annotations': AnnotationRecord.model_json_schema()}
    else:
        articles = read_records(args.articles, ArticleRecord)
        if args.action == 'audit':
            annotations = read_records(args.annotations, AnnotationRecord) if args.annotations else None
            protected, missing, files = historical_records()
            result = audit(articles, annotations, protected, missing)
            result['protected_file_hashes'] = files
        else:
            if not args.tokenizer.is_dir() or args.chunk_length < 3 or args.article_limit < 1:
                parser.error('Existing local tokenizer, chunk length >=3 and positive article limit required')
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), local_files_only=True)
            result = profile(articles, tokenizer, args.chunk_length, args.article_limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.action == 'audit' and not result['ready_for_development_intake']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
