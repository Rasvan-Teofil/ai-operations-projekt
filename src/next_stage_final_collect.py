"""Class-blind fresh Test source intake after candidate selection is frozen.

No annotation or candidate prediction. Source quotas never depend on class counts.
"""
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from src.article import ARTICLE_POLICY, extract_full_article, visible_publication_date
from src.next_stage import ROOT, ArticleRecord, digest, historical_records, overlap_keys, _list
from src.next_stage_collect import API, AGENT, CATEGORIES, LOCAL, POLICIES, flags, historical_pages, long_quotation, pool, write_json
from src.next_stage_prepare import related_titles
from src.news_data import source_key

DESTINATION = ROOT / 'data/next_stage/final-test-v1'
REPORTS = ROOT / 'reports/next_stage/final-test-v1'


def source_aliases(values):
    result = set(_list(values))
    for value in list(result):
        parsed = urlsplit(value)
        query = [(key, val) for key, val in parse_qsl(parsed.query, keep_blank_values=True)
                 if not key.casefold().startswith('utm_') and key.casefold() not in ('fbclid', 'gclid', 'dclid', 'msclkid')]
        result.add(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), '')))
    return sorted(result)


def protection():
    rows, missing, hashes = historical_records()
    if missing:
        raise RuntimeError('Historical protection incomplete: ' + repr(missing))
    paths = [ROOT / 'data' / name / 'articles.jsonl' for name in ('common_data_v3','expanded_news_v4','llm_adaptation_v5')]
    paths += [LOCAL / name for name in ('reviewed-articles-v1-input-v3.jsonl','reviewed-articles-v1-input-v4.jsonl','articles-full-article-v5.jsonl','development-v2/articles.jsonl')]
    for path in paths:
        content = path.read_text(); hashes[str(path.relative_to(ROOT))] = digest(content)
        rows.extend(json.loads(line) for line in content.splitlines())
    # Include observed foreign-language title aliases from both exposed pilots.
    for name in ('review_material.jsonl','review_material-full-article-v4.jsonl'):
        for item in map(json.loads,(LOCAL/name).read_text().splitlines()):
            rows.append({'id':item['id'],'related_document_ids':related_titles(item)})
    addendum = ROOT / 'reports/next_stage/pilot/group-addendum-v1.json'
    if addendum.exists():
        hashes[str(addendum.relative_to(ROOT))] = digest(addendum.read_text())
        for component in json.loads(addendum.read_text())['components']:
            for row in rows:
                if row.get('id') in component['ids']:
                    row['event_ids'] = sorted(set(row.get('event_ids', [])) | {component['event_id']})
    for row in rows:
        if row.get('source_keys'):
            row['source_keys'] = source_aliases(row['source_keys'])
    return rows, hashes


def collect():
    declaration_path = ROOT / 'reports/next_stage/final-candidates-v1.json'
    declaration = json.loads(declaration_path.read_text())
    if digest(Path(__file__).read_text()) != declaration['source_collector_sha256']:
        raise RuntimeError('Source collector changed after candidate declaration')
    if declaration['status'] != 'frozen-before-fresh-test' or declaration['final_test_predicted']:
        raise RuntimeError('Candidates must be fixed before any fresh source intake')
    if ARTICLE_POLICY != 'full-article-v5':
        raise RuntimeError('Extractor changed after declaration')
    source_target = declaration['test_source_frame']['source_target_per_language']
    protected, hashes = protection()
    protected_keys = set().union(*(overlap_keys(row) for row in protected))
    prior = historical_pages()
    exposed = {r.get('document_id') or r['id'] for r in protected}
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/local_llm/weights',local_files_only=True)
    DESTINATION.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    raw = DESTINATION/'articles.jsonl'; manifest_path=REPORTS/'source-manifest.json'
    records=[ArticleRecord.model_validate_json(line).model_dump() for line in raw.read_text().splitlines()] if raw.exists() else []
    declaration_sha=digest(declaration_path.read_text())
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'version':'fresh-full-article-test-source-v1','status':'unlabelled; rights and event review pending',
        'source_target_per_language':source_target,'candidate_declaration_sha256':declaration_sha,
        'source_collector_sha256':declaration['source_collector_sha256'],
        'protected_file_hashes':hashes,'exclusions':[],'sites':{},'pool_hashes':{},
        'sampling':'Next previously unexposed eligible pages in the same frozen shuffled recent-1500 category-member frame/language. No class hints, candidate scores or class balancing. Explicit translations, long quotes, over-4096-token articles and observed past source/page/translation links excluded before annotation.',
        'source_aliases':'Preserve raw upstream keys and additionally remove known utm_*/fbclid/gclid/dclid/msclkid tracking parameters; content query parameters remain.',
        'rights':'Contributor licences recorded per fixed revision; third-party rights require manual review. Quotation filter is not clearance.',
        'selection_after_annotation':'First up to 300 rights-compatible, resolvable, group-disjoint articles/language in this frozen source order; never choose by class or candidate score. Report shortages and all exclusions.',
        'population_limits':'One quote-restricted archive, max 20000 characters and 4096 Qwen article tokens; not representative of all live publishers. Historical/public-pretraining overlap cannot be proved absent.'}
    if manifest['candidate_declaration_sha256']!=declaration_sha or manifest['protected_file_hashes']!=hashes:
        raise RuntimeError('Frozen source declaration/protection changed')
    seen={r['id'] for r in records}|{r['id'] for r in manifest['exclusions']}
    with httpx.Client(timeout=30,headers={'User-Agent':AGENT},follow_redirects=False) as client:
        api=API(client)
        for language in CATEGORIES:
            policy_title,licence=POLICIES[language]
            policy=api.get(language,action='query',titles=policy_title,prop='revisions',rvprop='ids|timestamp',rvlimit=1)['query']['pages'][0]
            evidence=f'https://{language}.wikinews.org/w/index.php?oldid={policy["revisions"][0]["revid"]}'
            manifest['sites'][language]={'policy_revision_url':evidence,'source_license':licence}
            members=pool(api,language);manifest['pool_hashes'][language]=digest((LOCAL/('pool-'+language+'.json')).read_text())
            count=sum(r['language']==language for r in records)
            for rank,page in enumerate(members):
                if count>=source_target:break
                identifier=f'wikinews:{language}:{page["pageid"]}'
                if identifier in seen:continue
                seen.add(identifier)
                try:
                    if identifier in exposed or (language,page['pageid']) in prior:
                        raise ValueError('Previously exposed/collected page')
                    meta=api.get(language,action='query',pageids=page['pageid'],prop='revisions|info',inprop='url',rvdir='newer',rvlimit=1,rvprop='ids|timestamp')['query']['pages'][0]
                    created=meta['revisions'][0]['timestamp']
                    if created[:10]<='2005-09-25':raise ValueError('Outside chosen attribution-licence era')
                    revision=meta['lastrevid'];parsed=api.get(language,action='parse',oldid=revision,prop='text|revid|externallinks|langlinks|categories')['parse']
                    if parsed['revid']!=revision or parsed['pageid']!=page['pageid']:raise ValueError('Pinned revision mismatch')
                    text=extract_full_article(parsed['text'],meta['title']);published=visible_publication_date(text,language)
                    if published is None:raise ValueError('Visible publication dateline unresolved')
                    if flags(BeautifulSoup(parsed['text'],'html.parser').get_text(' ',strip=True))[0]:raise ValueError('Explicit translation')
                    if long_quotation(text,language):raise ValueError('Long direct quotation outside frozen source frame')
                    token_count=len(tokenizer.encode(text,add_special_tokens=False,verbose=False))
                    if token_count>4096:raise ValueError('Outside frozen full-text 4096-token population')
                    row=ArticleRecord(schema_version='article-pilot-v1',id=identifier,document_id=identifier,language=language,title=meta['title'],text=text,text_sha256=digest(text),
                        source_url=meta['canonicalurl'],revision_url=f'https://{language}.wikinews.org/w/index.php?oldid={revision}',revision=str(revision),history_url=meta['canonicalurl']+'?action=history',
                        attribution=f'{language}.wikinews.org contributors; per-page history link retained',published_at=published,retrieved_at=datetime.now(timezone.utc).isoformat(),
                        source_license=licence,source_license_url=evidence,rights_evidence_url=evidence,rights_status='unresolved',rights_review_note='Exact-text manual rights review pending',third_party_passages_reviewed=False,
                        extraction_version=ARTICLE_POLICY,transformation='Visible full-article v5 body, definition lists, captions and table alt text; no truncation; displayed source dateline, page creation separately retained',
                        related_document_ids=related_titles(parsed),source_keys=source_aliases([key for link in parsed['externallinks'] if (key:=source_key(link))]),event_ids=[],role='test').model_dump()
                    if overlap_keys(row)&protected_keys:raise ValueError('Observed protected source/page/translation relation')
                    records.append(row);count+=1
                    review={'id':identifier,'text_sha256':row['text_sha256'],'frame_rank':rank,'page_created_at':created,'displayed_publication_date':published,'qwen_article_tokens':token_count,
                            'categories':parsed['categories'],'langlinks':parsed['langlinks'],'revision_url':row['revision_url']}
                    with (DESTINATION/'review-material.jsonl').open('a') as stream:stream.write(json.dumps(review,ensure_ascii=False)+'\n')
                    print(language,count,'/',source_target,'fresh unlabelled inputs',flush=True)
                except (ValueError,KeyError) as error:
                    manifest['exclusions'].append({'id':identifier,'frame_rank':rank,'reason':str(error)})
                raw.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records));manifest['language_counts']=dict(Counter(r['language'] for r in records));manifest['raw_sha256']=digest(raw.read_text());write_json(manifest_path,manifest)
            manifest.setdefault('supply',{})[language]={'collected':count,'target':source_target,'frame_exhausted':count<source_target}
            write_json(manifest_path,manifest)
    manifest['status']='Frozen unlabelled source collection; exact-text two-pass annotation, rights and residual event review pending'
    write_json(manifest_path,manifest)
    print(json.dumps(manifest['supply'],indent=2),flush=True)


if __name__=='__main__':
    collect()
