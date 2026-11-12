"""Seeded, revision-pinned full-article pilot; no sentiment predictions or labels.

Network access is confined to the three Wikinews APIs. Raw responses and texts
remain in ignored data/next_stage/. Native-translation and rights flags are
recorded, not silently converted to class filters.
"""
import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from src.article import ARTICLE_POLICY, extract_full_article, visible_publication_date
from src.next_stage import ArticleRecord, ROOT, digest, historical_records, overlap_keys
from src.news_data import source_key

LOCAL = ROOT / 'data/next_stage/pilot'
OUT = ROOT / 'reports/next_stage/pilot'
SEED = 20261110
CATEGORIES = {'de': 'Kategorie:Veröffentlicht', 'en': 'Category:Published', 'ja': 'カテゴリ:公開中'}
POLICIES = {'de': ('Wikinews:Lizenzbestimmungen', 'CC-BY-2.5'),
            'en': ('Wikinews:Copyright', 'CC-BY-2.5/CC-BY-4.0 by contribution'),
            'ja': ('MediaWiki:Copyrightwarning', 'CC-BY-2.5/CC-BY-4.0 by contribution')}
AGENT = 'ArticleSentimentResearch/1.0 (https://github.com/Rasvan-Teofil/ai-operations-projekt)'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class API:
    def __init__(self, client):
        self.client = client

    def get(self, language, **params):
        params = {'format': 'json', 'formatversion': 2, 'maxlag': 5, **params}
        key = digest(json.dumps([language, params], sort_keys=True, ensure_ascii=False))
        path = LOCAL / 'api_cache' / (key + '.json')
        if path.exists():
            return json.loads(path.read_text())
        for attempt in range(8):
            response = self.client.get(f'https://{language}.wikinews.org/w/api.php', params=params)
            if response.status_code in (429, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))
                continue
            response.raise_for_status()
            value = response.json()
            if value.get('error', {}).get('code') == 'maxlag':
                time.sleep(min(2 ** attempt, 30))
                continue
            if 'error' in value:
                raise ValueError(value['error'])
            write_json(path, value)
            return value
        raise RuntimeError('Wikinews retry limit reached; preserve partial collection')


def pool(api, language):
    path = LOCAL / ('pool-' + language + '.json')
    if path.exists():
        return json.loads(path.read_text())
    members, continuation = [], {}
    while len(members) < 1500:
        value = api.get(language, action='query', list='categorymembers', cmtitle=CATEGORIES[language],
                        cmnamespace=0, cmtype='page', cmsort='timestamp', cmdir='desc', cmlimit=500, **continuation)
        members.extend(value['query']['categorymembers'])
        if 'continue' not in value:
            break
        continuation = value['continue']
    members = members[:1500]
    random.Random(SEED + list(CATEGORIES).index(language)).shuffle(members)
    write_json(path, members)
    return members


def historical_pages():
    pages = set()
    for name in ('common_data_v3', 'expanded_news_v4', 'llm_adaptation_v5'):
        path = ROOT / 'data' / name / 'articles.jsonl'
        if not path.exists():
            raise RuntimeError('Historical collection evidence missing: ' + str(path))
        for line in path.read_text().splitlines():
            row = json.loads(line)
            pages.add((row['language'], int(row['page_id'])))
    return pages


def flags(text):
    translation = bool(re.search(r'(?:この記事は.{0,80}翻訳|英語版.{0,30}から.{0,15}翻訳|translated (?:from|by)|übersetzt (?:aus|von)|übersetzung (?:aus|von))', text, re.I))
    quotations = bool(re.search(r'[„“”«»「」『』]|\"[^\"]{12,}\"', text))
    return translation, quotations


def long_quotation(text, language):
    """Class-blind source-frame rule, not a legal clearance decision."""
    threshold = 40 if language == 'ja' else 80
    patterns = (r'「([^「」]*)」', r'『([^『』]*)』', r'„([^„“]*)“',
                r'“([^“”]*)”', r'«([^«»]*)»', r'"([^"\n]*)"')
    return any(len(value) >= threshold for pattern in patterns for value in re.findall(pattern, text))


def collect(target):
    LOCAL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = LOCAL / f'articles-{ARTICLE_POLICY}.jsonl'
    records = [ArticleRecord.model_validate_json(line).model_dump() for line in path.read_text().splitlines()] if path.exists() else []
    if any(row['extraction_version'] != ARTICLE_POLICY for row in records):
        raise RuntimeError('Extraction changed after collection; do not mix policies')
    protected, missing, hashes = historical_records()
    if missing:
        raise RuntimeError('Historical protection incomplete: ' + repr(missing))
    protected_keys = set().union(*(overlap_keys(row) for row in protected))
    prior = historical_pages()
    checkpoint = LOCAL / 'articles-full-article-v3.jsonl'
    exposed = {json.loads(line)['id'] for line in checkpoint.read_text().splitlines()} if checkpoint.exists() and ARTICLE_POLICY != 'full-article-v3' else set()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest() if exposed else None
    manifest_path = OUT / f'collection-{ARTICLE_POLICY}.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'seed': SEED, 'extraction_version': ARTICLE_POLICY, 'status': 'unlabelled pilot; rights/event review pending',
        'sampling': 'Seeded first eligible pages from the frozen recent-1500 published-category pool/language. New v4 frame excludes all 90 development-exposed v3 pages and long direct quotations (80 characters DE/EN, 40 JA) before new annotation. No sentiment hints or predictions. Native archive articles only.',
        'prior_checkpoint_sha256': checkpoint_hash,
        'scope_limitation': 'Quotation-length eligibility changes the source population and may change topic/class distribution. It is a feasibility rule for a publication-oriented pool, not blanket legal clearance. Do not generalize scores to all news articles.',
        'exclusions': [], 'sites': {}, 'pool_hashes': {}, 'protected_file_hashes': hashes,
        'limits': 'Category membership time is not publication time. Single archive, source declarations and native/translation flags require manual review. Pretraining overlap unknown.'}
    if manifest['seed'] != SEED or manifest['protected_file_hashes'] != hashes or manifest.get('prior_checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError('Frozen seed or historical inputs changed')
    seen = {row['id'] for row in records} | {row['id'] for row in manifest['exclusions']}
    with httpx.Client(timeout=30, headers={'User-Agent': AGENT}, follow_redirects=False) as client:
        api = API(client)
        for language in CATEGORIES:
            rights = api.get(language, action='query', meta='siteinfo', siprop='rightsinfo|general')['query']
            policy_title, licence = POLICIES[language]
            policy = api.get(language, action='query', titles=policy_title, prop='revisions', rvprop='ids|timestamp', rvlimit=1)['query']['pages'][0]
            evidence = f'https://{language}.wikinews.org/w/index.php?oldid={policy["revisions"][0]["revid"]}'
            manifest['sites'][language] = {'rightsinfo': rights['rightsinfo'], 'readonly_observed': rights['general']['readonly'], 'policy_revision_url': evidence}
            members = pool(api, language)
            manifest['pool_hashes'][language] = hashlib.sha256((LOCAL / ('pool-' + language + '.json')).read_bytes()).hexdigest()
            accepted = sum(row['language'] == language for row in records)
            for page in members:
                identifier = f'wikinews:{language}:{page["pageid"]}'
                if accepted >= target:
                    break
                if identifier in seen:
                    continue
                seen.add(identifier)
                if identifier in exposed:
                    manifest['exclusions'].append({'id': identifier, 'reason': 'Previously reviewed v3 checkpoint; separate development exposure'})
                    continue
                if (language, page['pageid']) in prior:
                    manifest['exclusions'].append({'id': identifier, 'reason': 'previously collected page; not a new pilot input'})
                    continue
                try:
                    meta = api.get(language, action='query', pageids=page['pageid'], prop='revisions|info', inprop='url',
                                   rvdir='newer', rvlimit=1, rvprop='ids|timestamp')['query']['pages'][0]
                    created = meta['revisions'][0]['timestamp']
                    if created[:10] <= '2005-09-25':
                        raise ValueError('Outside chosen attribution-licence era')
                    revision = meta['lastrevid']
                    parsed = api.get(language, action='parse', oldid=revision, prop='text|revid|externallinks|langlinks|categories')['parse']
                    if parsed['revid'] != revision or parsed['pageid'] != page['pageid']:
                        raise ValueError('Pinned revision mismatch')
                    text = extract_full_article(parsed['text'], meta['title'])
                    publication = visible_publication_date(text, language)
                    if publication is None:
                        raise ValueError('Displayed publication dateline unresolved; page creation is not publication')
                    translated, _ = flags(BeautifulSoup(parsed['text'], 'html.parser').get_text(' ', strip=True))
                    _, quoted = flags(text)
                    if translated:
                        raise ValueError('Explicitly translated article; native pilot excludes translations')
                    if ARTICLE_POLICY in ('full-article-v4', 'full-article-v5') and long_quotation(text, language):
                        raise ValueError('Outside predeclared v4 source frame: long direct quotation')
                    row = ArticleRecord(schema_version='article-pilot-v1', id=identifier, document_id=identifier,
                        language=language, title=meta['title'], text=text, text_sha256=digest(text),
                        source_url=meta['canonicalurl'], revision_url=f'https://{language}.wikinews.org/w/index.php?oldid={revision}',
                        revision=str(revision), history_url=meta['canonicalurl']+'?action=history',
                        attribution=f'{language}.wikinews.org contributors; per-page history link retained',
                        published_at=publication, retrieved_at=datetime.now(timezone.utc).isoformat(),
                        source_license=licence, source_license_url=evidence, rights_evidence_url=evidence,
                        rights_status='unresolved', rights_review_note='Per-article review pending; site licence is not a warranty of third-party passages',
                        third_party_passages_reviewed=False, extraction_version=ARTICLE_POLICY,
                        transformation='Visible full article body, headings/lists/table rows; no source/footer section, navigation, images or hidden content; no truncation',
                        source_keys=sorted({key for link in parsed['externallinks'] if (key := source_key(link))}),
                        related_document_ids=[], event_ids=[], role='pilot').model_dump()
                    if overlap_keys(row) & protected_keys:
                        raise ValueError('Observed historical held-out text/page/source overlap')
                    records.append(row)
                    accepted += 1
                    review = {'id': identifier, 'text_sha256': row['text_sha256'], 'quotation_flag': quoted,
                              'categories': parsed['categories'], 'langlinks': parsed['langlinks'],
                              'html_cache_policy': 'Pinned parse response retained in ignored API cache', 'revision_url': row['revision_url']}
                    with (LOCAL / f'review_material-{ARTICLE_POLICY}.jsonl').open('a') as stream:
                        stream.write(json.dumps(review, ensure_ascii=False)+'\n')
                    print(f'{language}: {accepted}/{target} inputs, revision {revision}', flush=True)
                except (ValueError, KeyError) as error:
                    manifest['exclusions'].append({'id': identifier, 'reason': str(error)})
                path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records))
                manifest['language_counts'] = dict(Counter(row['language'] for row in records))
                manifest['raw_file_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                write_json(manifest_path, manifest)
            if accepted < target:
                raise RuntimeError(f'Insufficient eligible {language} inputs: {accepted}/{target}; report feasibility, do not change seed after labels')
    print('Collected', len(records), 'unlabelled full articles. Rights, event and annotation review remain pending.', flush=True)


def reextract(source):
    """Explicit pre-annotation migration; preserve original bytes and sample IDs."""
    source = source.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Source must be inside the project checkout')
    if any(LOCAL.glob('*annotation*.jsonl')) or (LOCAL / 'review_pass1.jsonl').exists():
        raise RuntimeError('Annotation has begun; refusing extraction migration')
    output = LOCAL / f'articles-{ARTICLE_POLICY}.jsonl'
    records = [json.loads(line) for line in source.read_text().splitlines()]
    with httpx.Client(timeout=30, headers={'User-Agent': AGENT}, follow_redirects=False) as client:
        api = API(client)
        for row in records:
            parsed = api.get(row['language'], action='parse', oldid=int(row['revision']), prop='text|revid|externallinks|langlinks|categories')['parse']
            if parsed['revid'] != int(row['revision']):
                raise RuntimeError('Pinned revision mismatch')
            row['text'] = extract_full_article(parsed['text'], row['title'])
            row['text_sha256'] = digest(row['text'])
            row['extraction_version'] = ARTICLE_POLICY
            ArticleRecord.model_validate(row)
    content = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in records)
    if output.exists() and output.read_text() != content:
        raise RuntimeError('Different target extraction already exists')
    output.write_text(content)
    old_manifest = json.loads((OUT / 'collection.json').read_text())
    old_manifest.update(extraction_version=ARTICLE_POLICY, raw_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                        migrated_from={'file': str(source.relative_to(ROOT)), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                       'reason': 'Before annotation: remove observed Wikinews publication/comment/gallery boilerplate; same pages and revisions, no class filtering'})
    write_json(OUT / f'collection-{ARTICLE_POLICY}.json', old_manifest)
    print('Explicitly re-extracted', len(records), 'same revisions; original file preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-language', type=int, choices=(30, 100), default=30)
    parser.add_argument('--reextract-from', type=Path)
    args = parser.parse_args()
    reextract(args.reextract_from) if args.reextract_from else collect(args.per_language)
