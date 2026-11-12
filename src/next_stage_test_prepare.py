"""Join exact two-pass fresh-Test annotations; freeze class-blind source quotas.

No candidate predictions are read. Unknown rights/labels and related historical
components remain explicit exclusions. All source inputs remain Test-reserved.
"""
import json
from collections import Counter

import pandas as pd

from src.next_stage import ArticleRecord, AnnotationRecord, ROOT, audit, digest, read_records
from src.next_stage_collect import write_json
from src.next_stage_final_collect import DESTINATION, REPORTS, protection
from src.next_stage_groups import components
from src.next_stage_prepare import frozen_write, jsonl


def assemble():
    articles=read_records(DESTINATION/'articles.jsonl',ArticleRecord)
    reviews=[]
    for number in (1,2):
        rows=[json.loads(x) for x in (DESTINATION/f'review-pass{number}.jsonl').read_text().splitlines()]
        indexed={r['id']:r for r in rows}
        if len(indexed)!=len(rows) or set(indexed)!={a.id for a in articles}:raise ValueError('Exactly one explicit review per source input and pass required')
        reviews.append(indexed)
    resolution_path=DESTINATION/'resolutions.json';resolutions=json.loads(resolution_path.read_text()) if resolution_path.exists() else {}
    annotations=[];reviewed=[]
    for article in articles:
        first,second=[r[article.id] for r in reviews]
        if any(r['text_sha256']!=article.text_sha256 for r in (first,second)):raise ValueError('Review input hash changed: '+article.id)
        note=resolutions.get(article.id,'')
        if first['review']['label']!=second['review']['label'] and not note:raise ValueError('Explicit label-resolution note missing: '+article.id)
        if not note and first['review']['target']!=second['review']['target']:
            note='Second reading clarifies the dominant target without changing the class. The second-pass rationale is authoritative for this version.'
        a=AnnotationRecord(schema_version='article-annotation-v1',id=article.id,text_sha256=article.text_sha256,annotation_version='fresh-test-ai-two-pass-v1',rubric_version='outcome-rubric-v1',
            provenance='AI-labelled · Codex-reviewed',annotator='Codex',annotator_revision='unavailable',annotation_license='CC0-1.0 to the extent project rights apply',
            first_pass=first['review'],second_pass=second['review'],resolution_note=note)
        value=article.model_dump()
        for key in ('event_ids','rights_status','rights_review_note','third_party_passages_reviewed'):value[key]=second[key]
        value['related_document_ids']=sorted(set(value['related_document_ids'])|set(second.get('related_document_ids',[])))
        reviewed.append(ArticleRecord.model_validate(value));annotations.append(a)
    return reviewed,annotations


def select_groups(rows,groups,eligible,target):
    """First whole components in fixed source order; labels are never consulted."""
    members={}
    for row,group in zip(rows,groups):members.setdefault(group,[]).append(row)
    selected=set();counts=Counter();quota_exclusions=[]
    for group,part in members.items():
        if not all(r.id in eligible for r in part):
            # Related rights-blocked or unresolvable group members keep the entire
            # component out, rather than retaining only its easy/compatible half.
            continue
        added=Counter(r.language for r in part)
        if any(counts[language]+n>target for language,n in added.items()):
            quota_exclusions.extend({'id':r.id,'code':'whole_group_source_order_quota','group':group} for r in part)
            continue
        selected.update(r.id for r in part);counts.update(added)
    return selected,quota_exclusions


def quality_exclusions(rows, review):
    """Validate the pre-inference source review against the exact pinned inputs.

    Only source-form decisions are accepted here; this function has no labels or
    predictions. Whole-component exclusion remains the responsibility of selection.
    """
    if review.get('status') != 'reviewed-before-candidate-test-inference':
        raise ValueError('Source-quality review unfinished')
    if review.get('candidate_predictions_seen') is not False:
        raise ValueError('Source-quality review must precede candidate predictions')
    by_id={row.id:row for row in rows};excluded={}
    for decision in review['decisions']:
        identifier=decision['id']
        if identifier not in by_id or identifier in excluded:
            raise ValueError('Unknown or duplicate source-quality input: '+identifier)
        if decision['text_sha256'] != by_id[identifier].text_sha256:
            raise ValueError('Source-quality input hash changed: '+identifier)
        if decision['decision'] != 'exclude_from_clean_full_article_test' or not decision.get('note','').strip():
            raise ValueError('Invalid source-quality exclusion: '+identifier)
        excluded[identifier]=decision['note']
    return excluded


def prepare():
    declaration_path=ROOT/'reports/next_stage/final-candidates-v1.json';declaration=json.loads(declaration_path.read_text())
    if (ROOT/'reports/next_stage/final-test-evaluation.lock').exists():raise RuntimeError('Final inference already started; no Test changes allowed')
    source=json.loads((REPORTS/'source-manifest.json').read_text())
    if digest((DESTINATION/'articles.jsonl').read_text())!=source['raw_sha256']:raise RuntimeError('Frozen source text changed')
    if digest(declaration_path.read_text())!=source['candidate_declaration_sha256']:raise RuntimeError('Candidate declaration changed during annotation')
    rows,annotations=assemble();protected,protection_hashes=protection()
    quality_review_path=REPORTS/'source-quality-review.json'
    quality_review=json.loads(quality_review_path.read_text())
    quality_blocked=quality_exclusions(rows,quality_review)
    # Mandatory manual residual-event review is separate from sentiment decisions.
    group_review_path=REPORTS/'group-review.json';group_review=json.loads(group_review_path.read_text())
    if group_review['status']!='reviewed-before-candidate-test-inference':raise RuntimeError('Residual group review unfinished')
    raw=[r.model_dump() for r in rows]+protected;by_id={}
    for row in raw:by_id.setdefault(row['id'],[]).append(row)
    for pair in group_review['pairs']:
        if pair['decision']=='join_conservative_component':
            key='final-reviewed-edge:'+digest(':'.join(sorted([pair['id'],pair['other']])))
            for identifier in (pair['id'],pair['other']):
                for row in by_id[identifier]:row['event_ids']=sorted(set(row.get('event_ids',[]))|{key})
    groups=components(raw);protected_groups=set(groups[len(rows):]);changed=[ArticleRecord.model_validate(r) for r in raw[:len(rows)]]
    indexed={a.id:a for a in annotations};eligible=set();exclusions=[]
    for article,group in zip(changed,groups):
        codes=[]
        if group in protected_groups:codes.append('related_protected_component')
        if article.rights_status!='reviewed_compatible' or not article.third_party_passages_reviewed:codes.append('rights_review_pending')
        if not article.event_ids:codes.append('event_review_pending')
        if indexed[article.id].second_pass.label=='unresolved':codes.append('unresolved_annotation')
        if article.id in quality_blocked:codes.append('source_quality_exclusion')
        if codes:exclusions.append({'id':article.id,'group':group,'codes':codes})
        else:eligible.add(article.id)
    selected,quota=select_groups(changed,groups[:len(rows)],eligible,declaration['test_source_frame']['final_target_per_language'])
    exclusions.extend(quota);selected_rows=[r for r in changed if r.id in selected];selected_annotations=[indexed[r.id] for r in selected_rows]
    dropped_eligible=eligible-selected-{r['id'] for r in quota}
    exclusions.extend({'id':identifier,'code':'related_blocked_source_group'} for identifier in sorted(dropped_eligible))
    if not selected_rows:raise RuntimeError('No eligible fresh Test remains')
    report=audit(selected_rows,selected_annotations,raw[len(rows):])
    if report['blockers']:raise RuntimeError('Test audit blocked: '+repr(report['blockers']))
    table=pd.DataFrame([{**r.model_dump(),'label':indexed[r.id].second_pass.label,'group':groups[i]} for i,r in enumerate(changed) if r.id in selected])
    counts=dict(Counter(table.language+':'+table.label))
    if any(counts.get(language+':'+label,0)<1 for language in ('de','en','ja') for label in ('negative','neutral','positive')):
        raise RuntimeError('Frozen sample lacks a language/class; report supply failure instead of score/class-based resampling')
    jsonl(DESTINATION/'annotations.jsonl',[a.model_dump() for a in selected_annotations]);jsonl(DESTINATION/'reviewed-all-source-annotations.jsonl',[a.model_dump() for a in annotations])
    frozen_write(DESTINATION/'test.csv',table.to_csv(index=False))
    jsonl(REPORTS/'annotations.jsonl',[a.model_dump() for a in annotations])
    jsonl(REPORTS/'provenance.jsonl',[{k:v for k,v in r.model_dump().items() if k!='text'} for r in changed])
    report.update(version='fresh-full-article-test-v1',candidate_declaration_sha256=digest(declaration_path.read_text()),source_sha256=source['raw_sha256'],
        test_sha256=digest((DESTINATION/'test.csv').read_text()),annotation_sha256=digest((DESTINATION/'annotations.jsonl').read_text()),
        all_source_annotation_sha256=digest((DESTINATION/'reviewed-all-source-annotations.jsonl').read_text()),group_review_sha256=digest(group_review_path.read_text()),
        source_quality_review_sha256=digest(quality_review_path.read_text()),source_quality_exclusions=quality_blocked,
        protected_file_hashes=protection_hashes,source_rows=len(rows),eligible_before_quotas=len(eligible),exclusions=exclusions,
        language_label_counts=counts,final_test_frozen=True,final_test_predicted=False,
        same_annotator_label_agreement={'matches':sum(a.first_pass.label==a.second_pass.label for a in annotations),'total':len(annotations),'independent_validation':False},
        selection='First whole rights-compatible, resolvable, protected-disjoint groups in frozen source order; no class balancing or candidate scores.',
        annotation_limits='Same AI annotator with prior development exposure; no predictions on these inputs seen during annotation. Self-review is not independent human validation.',
        all_source_inputs_permanently_test_reserved=True)
    frozen_write(REPORTS/'test-manifest.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'frozen_test_rows':len(table),'language_label_counts':counts,'exclusions':len(exclusions)},indent=2),flush=True)


if __name__=='__main__':
    prepare()
