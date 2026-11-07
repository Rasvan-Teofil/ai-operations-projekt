"""Random fresh v5 Test; every historical Wikinews input is protected."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import random

import httpx

from src.config import ROOT
from src.news_data import get_api, digest, lead_text, source_key, USER_AGENT
from src.news_expand import item_keys
from src.news_experiment import groups, read_json, write_json

LOCAL = ROOT / 'data/llm_adaptation_v5'
OUT = ROOT / 'reports/llm_adaptation_v5'
SEED = 20261109


def historical():
    return [json.loads(line) for directory in ['common_data_v3', 'expanded_news_v4']
            for line in (ROOT / 'data' / directory / 'articles.jsonl').read_text().splitlines()]


def collect():
    LOCAL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (LOCAL / 'articles.jsonl').exists(): raise RuntimeError('Fresh inputs already frozen')
    old = historical(); old_ids = {(r['language'], r['page_id']) for r in old}
    old_keys = set().union(*(item_keys(r) for r in old))
    records, rejected, hashes = [], Counter(), {}
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}, follow_redirects=False) as client:
        for lang, category in [('de', 'Kategorie:Veröffentlicht'), ('en', 'Category:Published')]:
            pool, continuation = [], {}
            while len(pool) < 1500:
                result = get_api(client, lang, {'action': 'query', 'list': 'categorymembers', 'cmtitle': category,
                    'cmnamespace': 0, 'cmtype': 'page', 'cmsort': 'timestamp', 'cmdir': 'desc', 'cmlimit': 500, **continuation})
                pool.extend(result['query']['categorymembers'])
                if 'continue' not in result: break
                continuation = result['continue']
            hashes[lang] = digest(json.dumps(pool, ensure_ascii=False).encode())
            pool = [p for p in pool if (lang, p['pageid']) not in old_ids]
            random.Random(SEED + (lang == 'en')).shuffle(pool)

            def fetch(page):
                try:
                    meta = get_api(client, lang, {'action': 'query', 'pageids': page['pageid'], 'prop': 'revisions|info',
                        'inprop': 'url', 'rvdir': 'newer', 'rvlimit': 1, 'rvprop': 'ids|timestamp'})['query']['pages'][0]
                    created, revision = meta['revisions'][0]['timestamp'], meta['lastrevid']
                    if created[:10] <= '2005-09-25': raise ValueError('Outside chosen attribution-licence era')
                    parsed = get_api(client, lang, {'action': 'parse', 'oldid': revision,
                        'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
                    assert parsed['revid'] == revision and parsed['pageid'] == page['pageid']
                    text = lead_text(parsed['text'], meta['title'])
                    licence = '4.0' if lang == 'en' and created[:10] > '2024-12-16' else '2.5'
                    item = {'id': f'wikinews-{lang}-{page["pageid"]}-{revision}', 'language': lang, 'text': text,
                        'text_sha256': digest(text.encode()), 'page_id': page['pageid'], 'revision': revision,
                        'created_at': created, 'source_url': meta['canonicalurl'],
                        'revision_url': f'https://{lang}.wikinews.org/w/index.php?oldid={revision}',
                        'history_url': meta['canonicalurl'] + '?action=history', 'license': 'CC-BY-' + licence,
                        'license_url': f'https://creativecommons.org/licenses/by/{licence}/',
                        'attribution': f'{lang}.wikinews.org / Wikinews contributors; author history retained',
                        'transformation': 'Same v3/v4 title+lead extraction; metadata/images/detected quoted leads omitted; complete-sentence lead cap650characters',
                        'source_keys': sorted({k for url in parsed['externallinks'] if (k := source_key(url))}),
                        'linked_articles': [f'{x["lang"]}:{x["title"]}' for x in parsed['langlinks'] if x['lang'] in ('de', 'en')],
                        'title_key': lang + ':' + meta['title']}
                    if item_keys(item) & old_keys: raise ValueError('Connected to historical page/reference')
                    return item, None
                except (ValueError, RuntimeError, KeyError, AssertionError, httpx.HTTPError) as error:
                    return None, str(error)[:150]
            selected = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                for start in range(0, len(pool), 30):
                    for item, error in executor.map(fetch, pool[start:start + 30]):
                        if item is None: rejected[error] += 1
                        elif len(selected) < 120: selected.append(item)
                    print(lang, len(selected), '/120 fresh random inputs', flush=True)
                    if len(selected) == 120: break
            if len(selected) != 120: raise RuntimeError('Insufficient eligible fresh inputs; do not change seed after labels')
            records.extend(selected)
    all_rows = old + records; component = groups(all_rows); protected = set(component[:len(old)])
    accepted, exclusions = [], []
    for row, group in zip(records, component[len(old):]):
        if group in protected: exclusions.append({'id': row['id'], 'reason': 'Historical transitive source/near-duplicate overlap'})
        else: accepted.append({**row, 'group': group})
    random.Random(SEED).shuffle(accepted)
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in accepted).encode()
    (LOCAL / 'articles.jsonl').write_bytes(raw)
    write_json(OUT / 'source_manifest.json', {'data_version': 'wikinews-llm-adaptation-v5', 'seed': SEED,
        'rows': len(accepted), 'sha256': digest(raw), 'pool_sha256': hashes,
        'sampling': 'Random120/language before labels, no class hints/balancing; all1080 historical Wikinews inputs/groups excluded',
        'language_counts': dict(Counter(r['language'] for r in accepted)),
        'licenses': dict(Counter(r['license'] for r in accepted)), 'fetch_exclusions': dict(rejected),
        'group_exclusions': exclusions, 'historical_protected_inputs': len(old),
        'label_status_at_collection': 'unlabelled; annotation ledger created separately before fitting',
        'articles': [{k: v for k, v in r.items() if k != 'text'} for r in accepted]})
    print('Fresh random inputs frozen:', len(accepted), flush=True)


def restore():
    manifest = read_json(OUT / 'source_manifest.json'); rows = []
    LOCAL.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/common_data_v3/api_cache').mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}) as client:
        for item in manifest['articles']:
            parsed = get_api(client, item['language'], {'action': 'parse', 'oldid': item['revision'],
                'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
            if parsed['revid'] != item['revision'] or parsed['pageid'] != item['page_id']:
                raise RuntimeError('Revision mismatch')
            text = lead_text(parsed['text'], item['title_key'].split(':', 1)[1])
            if digest(text.encode()) != item['text_sha256']: raise RuntimeError('Pinned rendering changed; do not substitute')
            rows.append({'id': item['id'], 'language': item['language'], 'text': text, **item})
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode()
    target = LOCAL / 'articles.jsonl'
    if digest(raw) != manifest['sha256'] or (target.exists() and target.read_bytes() != raw):
        raise RuntimeError('Frozen input bytes differ')
    target.write_bytes(raw); print('Fresh inputs restored with exact SHA-256 verification', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore: restore()
    else: collect()
