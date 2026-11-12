"""Collect revision-pinned Wikinews leads. No automatic sentiment labels."""
import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from src.config import ROOT, REPORTS

LOCAL = ROOT / 'data/common_data_v3'
OUT = Path(os.getenv('SENTIMENT_NEWS_REPORTS', str(REPORTS / 'common_data_v3'))).resolve()
SEED = 20261107
USER_AGENT = 'AIOpsSecondPathDataReview/1.0 (https://github.com/Rasvan-Teofil/ai-operations-projekt)'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return re.sub(r'\W+', ' ', text.casefold()).strip()


def lead_text(html, title):
    """Use the first substantive paragraph, excluding metadata and quoted leads."""
    soup = BeautifulSoup(html, 'html.parser')
    root = soup.select_one('.mw-parser-output')
    if root is None:
        raise ValueError('No article content')
    for node in root.select('table, blockquote, aside, script, style, .thumb, .noprint, .infobox, .metadata, .published, .Z3988, sup'):
        node.decompose()
    for p in root.find_all('p'):
        text = ' '.join(p.get_text(' ', strip=True).split())
        if len(text) < 100 or len(text.split()) < 15:
            continue
        if any(mark in title + text for mark in ['"', '„', '“', '”', '«', '»', '‹', '›']):
            raise ValueError('Quoted lead/title excluded')
        if len(text) > 650:
            # Keep complete sentences only; never label a silently cut sentence.
            sentences = re.split(r'(?<=[.!?])\s+', text)
            kept = []
            for sentence in sentences:
                if len(' '.join(kept + [sentence])) > 650:
                    break
                kept.append(sentence)
            text = ' '.join(kept)
        if len(text) < 100:
            raise ValueError('Lead length outside policy')
        return title + '\n\n' + text
    raise ValueError('No substantive lead')


def source_key(url):
    p = urlsplit(url)
    if p.scheme not in ('http', 'https'):
        return None
    host = (p.hostname or '').lower().removeprefix('www.')
    if any(host == h or host.endswith('.' + h) for h in
           ['wikinews.org', 'wikimedia.org', 'wikipedia.org', 'creativecommons.org',
            'facebook.com', 'twitter.com', 'x.com', 'telegram.me', 'whatsapp.com', 'linkedin.com']):
        return None
    if not p.path or p.path == '/':
        return None
    return urlunsplit(('', host, p.path.rstrip('/'), p.query, ''))


def get_api(client, lang, params):
    full = {'format': 'json', 'formatversion': 2, 'maxlag': 5, **params}
    key = digest(json.dumps([lang, full], sort_keys=True).encode())
    cache = LOCAL / 'api_cache' / f'{key}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(4):
        r = client.get(f'https://{lang}.wikinews.org/w/api.php', params=full)
        if r.status_code in (429, 502, 503, 504):
            time.sleep(min(2 ** attempt, 8))
            continue
        r.raise_for_status()
        data = r.json()
        if data.get('error', {}).get('code') == 'maxlag':
            time.sleep(min(2 ** attempt, 8))
            continue
        if 'error' in data:
            raise ValueError(data['error'])
        cache.write_text(json.dumps(data, ensure_ascii=False))
        return data
    raise RuntimeError('Wikimedia API retry limit reached')


def collect(per_language=180):
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'api_cache').mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if (LOCAL / 'articles.jsonl').exists():
        raise RuntimeError('Collection already frozen; restore from source_manifest instead of overwriting')
    records, excluded, sites = [], Counter(), {}
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}, follow_redirects=False) as client:
        for lang, category in [('de', 'Kategorie:Veröffentlicht'), ('en', 'Category:Published')]:
            sites[lang] = get_api(client, lang, {'action': 'query', 'meta': 'siteinfo', 'siprop': 'rightsinfo|general'})['query']
            pool, continuation = [], {}
            while len(pool) < 1500:
                data = get_api(client, lang, {'action': 'query', 'list': 'categorymembers',
                    'cmtitle': category, 'cmnamespace': 0, 'cmtype': 'page', 'cmsort': 'timestamp',
                    'cmdir': 'desc', 'cmlimit': 500, **continuation})
                pool.extend(data['query']['categorymembers'])
                if 'continue' not in data:
                    break
                continuation = data['continue']
            random.Random(SEED + (0 if lang == 'de' else 1)).shuffle(pool)

            def fetch(page):
                try:
                    meta = get_api(client, lang, {'action': 'query', 'pageids': page['pageid'],
                        'prop': 'revisions|info', 'inprop': 'url', 'rvdir': 'newer',
                        'rvlimit': 1, 'rvprop': 'ids|timestamp'})['query']['pages'][0]
                    creation = meta['revisions'][0]['timestamp']
                    revid = meta['lastrevid']
                    parsed = get_api(client, lang, {'action': 'parse', 'oldid': revid,
                        'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
                    if parsed['revid'] != revid or parsed['pageid'] != page['pageid']:
                        raise ValueError('Revision mismatch')
                    text = lead_text(parsed['text'], meta['title'])
                    # EN migration applies by original publication era; use oldest revision
                    # conservatively for older articles. DE has its own CC BY 2.5 policy.
                    version = '4.0' if lang == 'en' and creation[:10] > '2024-12-16' else '2.5'
                    item = {'id': f'wikinews-{lang}-{page["pageid"]}-{revid}', 'language': lang,
                        'text': text, 'text_sha256': digest(text.encode()), 'page_id': page['pageid'],
                        'revision': revid, 'created_at': creation, 'source_url': meta['canonicalurl'],
                        'revision_url': f'https://{lang}.wikinews.org/w/index.php?oldid={revid}',
                        'history_url': meta['canonicalurl'] + '?action=history',
                        'license': f'CC-BY-{version}', 'license_url': f'https://creativecommons.org/licenses/by/{version}/',
                        'attribution': f'{lang}.wikinews.org / Wikinews contributors; history link retained',
                        'transformation': 'Title plus first substantive paragraph; complete-sentence cap 650 lead characters; metadata/images/quoted leads excluded',
                        'source_keys': sorted({k for u in parsed['externallinks'] if (k := source_key(u))}),
                        'linked_articles': [f'{x["lang"]}:{x["title"]}' for x in parsed['langlinks'] if x['lang'] in ('de', 'en')],
                        'title_key': lang + ':' + meta['title']}
                    return item, None
                except (ValueError, KeyError, RuntimeError, httpx.HTTPError) as error:
                    return None, str(error)[:140]

            selected, seen = [], set()
            # Bounded parallel reads; selection follows the seeded order, not completion time.
            with ThreadPoolExecutor(max_workers=2) as executor:
                for start in range(0, len(pool), 40):
                    for item, error in executor.map(fetch, pool[start:start + 40]):
                        if item is None:
                            excluded[error] += 1
                        elif normalized(item['text']) in seen:
                            excluded['normalized duplicate'] += 1
                        elif len(selected) < per_language:
                            selected.append(item)
                            seen.add(normalized(item['text']))
                    print(f'{lang}: selected {len(selected)}/{per_language}; examined {start + 40}', flush=True)
                    if len(selected) >= per_language:
                        break
            if len(selected) != per_language:
                raise RuntimeError(f'Insufficient accepted {lang} leads: {len(selected)}')
            records.extend(selected)
    raw = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records).encode()
    (LOCAL / 'articles.jsonl').write_bytes(raw)
    manifest = {'data_version': 'wikinews-ai-v3-pilot', 'seed': SEED, 'rows': len(records),
        'per_language': per_language, 'sha256': digest(raw), 'label_status': 'unlabelled',
        'source': 'Published-category seeded sample from most recent 1500 memberships per language',
        'licenses': dict(Counter(x['license'] for x in records)), 'excluded': dict(excluded),
        'site_rights': {lang: x['rightsinfo'] for lang, x in sites.items()},
        'site_readonly_flag': {lang: x['general']['readonly'] for lang, x in sites.items()},
        'limitations': ['Single-source archive; short leads, not full articles',
            'Quotations/images excluded; site declarations do not warrant every contribution',
            'Public LLM pretraining overlap unknown; no independent human labels'],
        'articles': [{k: v for k, v in x.items() if k != 'text'} for x in records]}
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Collected', len(records), 'revision-pinned leads; no sentiment labels generated.', flush=True)


def restore():
    """Rebuild ignored inputs from the frozen public revisions and verify every byte."""
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'api_cache').mkdir(exist_ok=True)
    rows = []
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}) as client:
        for i, item in enumerate(manifest['articles']):
            parsed = get_api(client, item['language'], {'action': 'parse', 'oldid': item['revision'],
                'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
            if parsed['revid'] != item['revision'] or parsed['pageid'] != item['page_id']:
                raise RuntimeError('Restored revision mismatch')
            text = lead_text(parsed['text'], item['title_key'].split(':', 1)[1])
            if digest(text.encode()) != item['text_sha256']:
                raise RuntimeError('Rendered lead changed; preserve evidence and inspect, do not silently substitute')
            rows.append({'id': item['id'], 'language': item['language'], 'text': text, **item})
            if i % 40 == 0:
                print(f'Restored {i + 1}/{manifest["rows"]}', flush=True)
    raw = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows).encode()
    if digest(raw) != manifest['sha256']:
        raise RuntimeError('Collection bytes differ from frozen manifest')
    path = LOCAL / 'articles.jsonl'
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError('Existing collection differs; refusing overwrite')
    path.write_bytes(raw)
    print('Original collection restored and SHA-256 verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-language', type=int, default=180)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore: restore()
    else: collect(args.per_language)
