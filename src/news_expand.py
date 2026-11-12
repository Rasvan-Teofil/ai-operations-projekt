"""New source examples for a data-scaling study; title hints never create labels."""
import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

from src.config import ROOT, REPORTS
from src.news_data import get_api, lead_text, source_key, digest, USER_AGENT
from src.news_experiment import groups

LOCAL = ROOT / 'data/expanded_news_v4'
OUT = REPORTS / 'expanded_news_v4'
SEED = 20261108
POSITIVE = {
    'de': r'gewinn|siegt|erfolg|rekord|auszeichn|preis|gerettet|befreit|entdeck|eröffn|erholt|wachstum|verbessert|wiederauf|gesenkt|aufstieg|steigt auf|medaille|meister|durchbruch|einigung|rettung|hilfe|förder|vereinfacht',
    'en': r'\bwin\b|\bwins\b|victory|success|record|award|prize|rescue|freed|discover|reopen|recover|growth|improv|restore|breakthrough|champion|medal|survive|avert|agreement|approv|protect|aid|funding|qualif|defeats|beats|launch'}
NEUTRAL = {
    'de': r'besuch|treffen|tagung|konferenz|interview|jubiläum|erinner|gedenk|statistik|veröffentlicht|kündigt|plan|gesetz|vorstell|programm|termine|findet statt|versammlung',
    'en': r'interview|meeting|visits|conference|announces|schedule|anniversary|commemorat|statistics|publishes|plans|presents|opens|debate|election|budget'}


def item_keys(item):
    return set(['page:' + item['title_key']] + ['page:' + k for k in item['linked_articles']]
               + ['source:' + k for k in item['source_keys']])


def collect():
    LOCAL.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (LOCAL / 'articles.jsonl').exists():
        raise RuntimeError('Expansion already collected; preserve the frozen inputs')
    historical = [json.loads(x) for x in (ROOT / 'data/common_data_v3/articles.jsonl').read_text().splitlines()]
    old_pages = {(x['language'], x['page_id']) for x in historical}
    old_keys = set().union(*(item_keys(x) for x in historical))
    records, rejected, attempted, pool_hashes = [], Counter(), set(), {}
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}, follow_redirects=False) as client:
        pools = {}
        for lang, category in [('de', 'Kategorie:Veröffentlicht'), ('en', 'Category:Published')]:
            pool, continuation = [], {}
            while len(pool) < 1500:
                data = get_api(client, lang, {'action': 'query', 'list': 'categorymembers',
                    'cmtitle': category, 'cmnamespace': 0, 'cmtype': 'page', 'cmsort': 'timestamp',
                    'cmdir': 'desc', 'cmlimit': 500, **continuation})
                pool.extend(data['query']['categorymembers'])
                if 'continue' not in data: break
                continuation = data['continue']
            pool_hashes[lang] = digest(json.dumps(pool, ensure_ascii=False).encode())
            pool = [p for p in pool if (lang, p['pageid']) not in old_pages]
            random.Random(SEED + (lang == 'en')).shuffle(pool); pools[lang] = pool

        def fetch(job):
            lang, page, role, hint = job
            try:
                meta = get_api(client, lang, {'action': 'query', 'pageids': page['pageid'],
                    'prop': 'revisions|info', 'inprop': 'url', 'rvdir': 'newer',
                    'rvlimit': 1, 'rvprop': 'ids|timestamp'})['query']['pages'][0]
                created, revision = meta['revisions'][0]['timestamp'], meta['lastrevid']
                if created[:10] <= '2005-09-25':
                    raise ValueError('Outside chosen attribution-licence era')
                parsed = get_api(client, lang, {'action': 'parse', 'oldid': revision,
                    'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
                assert parsed['revid'] == revision and parsed['pageid'] == page['pageid']
                text = lead_text(parsed['text'], meta['title'])
                licence = '4.0' if lang == 'en' and created[:10] > '2024-12-16' else '2.5'
                item = {'id': f'wikinews-{lang}-{page["pageid"]}-{revision}', 'language': lang,
                    'text': text, 'text_sha256': digest(text.encode()), 'page_id': page['pageid'],
                    'revision': revision, 'created_at': created, 'source_url': meta['canonicalurl'],
                    'revision_url': f'https://{lang}.wikinews.org/w/index.php?oldid={revision}',
                    'history_url': meta['canonicalurl'] + '?action=history', 'license': 'CC-BY-' + licence,
                    'license_url': f'https://creativecommons.org/licenses/by/{licence}/',
                    'attribution': f'{lang}.wikinews.org / Wikinews contributors; author history retained',
                    'transformation': 'Same v3 title+lead extraction; metadata/images/quoted leads excluded, lead capped at complete sentences within650characters',
                    'source_keys': sorted({k for url in parsed['externallinks'] if (k := source_key(url))}),
                    'linked_articles': [f'{x["lang"]}:{x["title"]}' for x in parsed['langlinks'] if x['lang'] in ('de', 'en')],
                    'title_key': lang + ':' + meta['title'], 'assigned_role': role, 'sampling_hint': hint}
                if item_keys(item) & old_keys:
                    raise ValueError('Connected to a historical source page/event reference')
                if role == 'train' and item_keys(item) & fresh_test_keys:
                    raise ValueError('Connected to a reserved fresh test page/reference')
                return item, None
            except (ValueError, RuntimeError, KeyError, AssertionError, httpx.HTTPError) as error:
                return None, str(error)[:160]

        fresh_test_keys = set()

        def sample(lang, role, count, hint, predicate=None):
            candidates = [p for p in pools[lang] if (lang, p['pageid']) not in attempted and
                          (predicate is None or predicate(p['title']))]
            picked = 0
            with ThreadPoolExecutor(max_workers=2) as executor:
                for start in range(0, len(candidates), 30):
                    chunk = candidates[start:start + 30]
                    attempted.update((lang, p['pageid']) for p in chunk)
                    for item, error in executor.map(fetch, [(lang, p, role, hint) for p in chunk]):
                        if item is None: rejected[error] += 1
                        elif picked < count:
                            records.append(item); picked += 1
                            if role == 'test': fresh_test_keys.update(item_keys(item))
                    print(f'{lang} {role}/{hint}: {picked}/{count}; total {len(records)}', flush=True)
                    if picked >= count: break
            return picked

        # Test is randomly sampled first, without sentiment hints or class balancing.
        for lang in ['de', 'en']:
            if sample(lang, 'test', 120, 'random') != 120:
                raise RuntimeError('Unable to collect the planned random test pool')
        # Training enrichment changes sampling, never the labels. Shortfalls use random fills.
        for lang in ['de', 'en']:
            count = sample(lang, 'train', 120, 'positive-title-hint', lambda t: bool(re.search(POSITIVE[lang], t, re.I)))
            count += sample(lang, 'train', 60, 'neutral-title-hint', lambda t: bool(re.search(NEUTRAL[lang], t, re.I)))
            count += sample(lang, 'train', 240 - count, 'random-fill')
            if count != 240: raise RuntimeError('Unable to collect planned training enrichment')
    # A second, transitive audit includes historical rows and all new candidates.
    combined = historical + records
    components = groups(combined)
    historical_groups = set(components[:len(historical)])
    test_groups = {g for r, g in zip(combined, components) if r.get('assigned_role') == 'test'}
    accepted, group_exclusions = [], []
    for item, group in zip(records, components[len(historical):]):
        if group in historical_groups or (item['assigned_role'] == 'train' and group in test_groups):
            group_exclusions.append({'id': item['id'], 'reason': 'Transitive group overlaps historical input or reserved Test'})
        else: accepted.append({**item, 'group': group})
    # Hide role/hint in annotation batches by interleaving all new examples.
    random.Random(SEED).shuffle(accepted)
    raw = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in accepted).encode()
    (LOCAL / 'articles.jsonl').write_bytes(raw)
    manifest = {'data_version': 'wikinews-ai-v4-expanded', 'seed': SEED, 'rows': len(accepted),
        'sha256': digest(raw), 'pool_sha256': pool_hashes, 'label_status': 'unlabelled',
        'roles_before_annotation': dict(Counter(x['assigned_role'] for x in accepted)),
        'language_roles': dict(Counter(x['language'] + ':' + x['assigned_role'] for x in accepted)),
        'sampling_hints': dict(Counter(x['sampling_hint'] for x in accepted)),
        'policy': 'Fresh Test random without class hints; training title-based enrichment only. Hints do not assign labels.',
        'group_exclusions': group_exclusions, 'fetch_exclusions': dict(rejected),
        'licenses': dict(Counter(x['license'] for x in accepted)),
        'historical_rows_excluded_from_new_collection': len(historical),
        'articles': [{k: v for k, v in x.items() if k != 'text'} for x in accepted]}
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print('Expansion collected:', len(accepted), 'new inputs; roles fixed, no labels created.', flush=True)


def restore():
    """Rebuild pinned texts, never silently substitute updated page renderings."""
    manifest = json.loads((OUT / 'source_manifest.json').read_text())
    LOCAL.mkdir(parents=True, exist_ok=True)
    (ROOT / 'data/common_data_v3/api_cache').mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=35, headers={'User-Agent': USER_AGENT}) as client:
        for i, item in enumerate(manifest['articles']):
            parsed = get_api(client, item['language'], {'action': 'parse', 'oldid': item['revision'],
                'prop': 'text|externallinks|langlinks', 'disableeditsection': 1})['parse']
            if parsed['revid'] != item['revision'] or parsed['pageid'] != item['page_id']:
                raise RuntimeError('Restored revision mismatch')
            text = lead_text(parsed['text'], item['title_key'].split(':', 1)[1])
            if digest(text.encode()) != item['text_sha256']:
                raise RuntimeError('Source rendering changed; preserve evidence, do not substitute')
            rows.append({'id': item['id'], 'language': item['language'], 'text': text, **item})
            if i % 100 == 0: print(f'Restored {i + 1}/{len(manifest["articles"])}', flush=True)
    raw = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows).encode()
    if digest(raw) != manifest['sha256']:
        raise RuntimeError('Restored collection differs from frozen bytes')
    target = LOCAL / 'articles.jsonl'
    if target.exists() and target.read_bytes() != raw:
        raise RuntimeError('Existing collection differs; refusing overwrite')
    target.write_bytes(raw)
    print('Expansion restored; exact source and text hashes verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore: restore()
    else: collect()
