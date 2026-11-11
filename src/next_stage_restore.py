"""Restore exact frozen Train inputs from public pinned revisions and metadata.

No automatic labels, downloads of model weights, training or historical rewrites.
Hash mismatch fails closed (MediaWiki template rendering can change over time).
"""
import hashlib
import json

import httpx
import pandas as pd

from src.article import ARTICLE_POLICY, extract_full_article
from src.next_stage import ROOT, ArticleRecord, AnnotationRecord, digest
from src.next_stage_collect import API, AGENT, LOCAL
from src.next_stage_prepare import frozen_write, jsonl


def restore_train():
    protocol=json.loads((ROOT/'reports/next_stage/development-protocol-v1.json').read_text())
    if ARTICLE_POLICY!=protocol['extraction_policy']:raise RuntimeError('Extractor differs from frozen data')
    provenance=[json.loads(x) for x in (ROOT/'reports/next_stage/train-provenance-v1.jsonl').read_text().splitlines()]
    annotations=[AnnotationRecord.model_validate_json(x) for x in (ROOT/'reports/next_stage/train-annotations-v1.jsonl').read_text().splitlines()]
    if {r['id'] for r in provenance}!={a.id for a in annotations}:raise RuntimeError('Frozen annotation/provenance IDs differ')
    articles=[];table=[]
    with httpx.Client(timeout=30,headers={'User-Agent':AGENT},follow_redirects=False) as client:
        api=API(client)
        for row in provenance:
            parsed=api.get(row['language'],action='parse',oldid=int(row['revision']),prop='text|revid|externallinks|langlinks|categories')['parse']
            if str(parsed['revid'])!=row['revision']:raise RuntimeError('Revision mismatch')
            text=extract_full_article(parsed['text'],row['title'])
            if digest(text)!=row['text_sha256']:raise RuntimeError('Pinned revision currently renders differently; use the original raw/API-cache backup instead of changing frozen data: '+row['id'])
            article=ArticleRecord.model_validate({**{k:v for k,v in row.items() if k not in ('label','group')},'text':text})
            articles.append(article.model_dump());table.append({**article.model_dump(),'label':row['label'],'group':row['group']})
    target=LOCAL/'development-v2'
    # Validate all final bytes before writing any target.
    article_bytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in articles)
    annotation_bytes=''.join(json.dumps(a.model_dump(),ensure_ascii=False)+'\n' for a in annotations)
    train_bytes=pd.DataFrame(table).to_csv(index=False)
    for content,key in [(article_bytes,'input_sha256'),(annotation_bytes,'annotation_sha256'),(train_bytes,'train_sha256')]:
        if digest(content)!=protocol[key]:raise RuntimeError('Restoration bytes differ from frozen '+key)
    frozen_write(target/'articles.jsonl',article_bytes);frozen_write(target/'annotations.jsonl',annotation_bytes);frozen_write(target/'train.csv',train_bytes)
    print('Verified exact restoration of',len(articles),'frozen Train articles; existing identical files retained.',flush=True)


if __name__=='__main__':restore_train()
