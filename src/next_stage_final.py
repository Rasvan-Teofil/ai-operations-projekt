"""Four frozen candidates, complete-Train fitting and one immutable fresh Test.

Candidate selection uses development only. Test predictions are append-only;
loading/warmups use Train and never repeat Test inference for timing.
"""
import argparse
import hashlib
import json
import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from pathlib import Path

from src.config import ROOT, LABELS, ENCODER, ENCODER_REVISION
from src.next_stage import digest
from src.next_stage_experiment import (ART, OUT, TRAIN, BASE, SEED, artifact_info, clean,
    fit_encoder, fit_lora, gpu, metrics, qwen_profile, verified)
from src.next_stage_prepare import frozen_write
from src.next_stage_protocol import RunIdentity, select_candidate
from src.next_stage_collect import write_json
from src.serving import load_artifact
from src.tracking import fingerprint, tracking

FINAL_DATA = ROOT / 'data/next_stage/final-test-v1'


def declaration():
    p=verified(); result=json.loads((OUT/'development-v1.json').read_text())
    capacity=json.loads((OUT/'inference-capacity-v1.json').read_text())
    if capacity['article_tokens']<4090 or capacity['full_sequence_tokens']>4608:
        raise RuntimeError('Full-window inference capacity has not been established')
    if result['scored_evaluations']!=24 or result['final_test_predicted']:
        raise RuntimeError('Complete the frozen 24-condition Train-only comparison first')
    if (FINAL_DATA/'articles.jsonl').exists():
        raise RuntimeError('Fresh source intake already began; candidate declaration cannot change')
    rows=result['out_of_fold']; shared=digest((OUT/'development-v1.json').read_text())
    def choose(candidates):
        return select_candidate([{**r,'comparable_group':shared} for r in candidates])
    mini=choose([r for r in rows if r['model'] in ('minilm128','minilm256','minilm512')])
    names=['frozen128',mini,'qwen-zero','qwen-lora']
    selected=choose([r for r in rows if r['model'] in names])
    shortlist=sorted([r for r in rows if r['model'] in names],key=lambda r:r['primary_score'],reverse=True)
    # Optional seed study needs an explicit decision before fresh Test intake.
    if (OUT/'stability-decision-v1.json').exists():
        stability=json.loads((OUT/'stability-decision-v1.json').read_text())
        if stability.get('status') not in ('complete','not_triggered'):
            raise RuntimeError('Optional stability decision remains unfinished')
    else:
        if shortlist[0]['primary_score']-shortlist[1]['primary_score']<=.02:
            raise RuntimeError('Frozen closeness rule requires the optional seed-stability decision before final declaration')
        stability={'status':'not_triggered','performed':False,'reason':'Top-two complete-Train candidate conditions differ by more than .02 primary OOF F1; optional seed study not triggered.',
                   'primary_gap':shortlist[0]['primary_score']-shortlist[1]['primary_score']}
        frozen_write(OUT/'stability-decision-v1.json',json.dumps(stability,ensure_ascii=False,indent=2)+'\n')
    value={'version':'full-article-final-candidates-v1','status':'frozen-before-fresh-test','final_test_predicted':False,
        'selected_on_development':selected,'minilm_window_selected_on_development':int(mini.removeprefix('minilm')),
        'models':names,'selection_rule':'OOF primary DE/EN F1; within .005 use warm latency then full standalone bytes. Final Test will not change the selected model or recipes.',
        'development_sha256':shared,'development_protocol_sha256':hashlib.sha256((OUT/'development-protocol-v1.json').read_bytes()).hexdigest(),
        'train_sha256':p['train_sha256'],'annotation_sha256':p['annotation_sha256'],'working_tree_code_sha256':p['working_tree_code_sha256'],
        'engineering_amendment_sha256':hashlib.sha256((OUT/'engineering-amendment-v1.json').read_bytes()).hexdigest(),
        'final_runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_collector_sha256':hashlib.sha256((ROOT/'src/next_stage_final_collect.py').read_bytes()).hexdigest(),
        'inference_capacity_sha256':hashlib.sha256((OUT/'inference-capacity-v1.json').read_bytes()).hexdigest(),'seed':SEED,'full_train_rows':p['train_rows'],
        'stability':stability,'test_source_frame':{'source_target_per_language':350,'final_target_per_language':300,
            'source_order':'Same frozen shuffled recent-1500 published-category frame, skip all previously exposed pages and observed related source/translation groups',
            'source_eligibility':'Native DE/EN/JA; quote-restricted v5, <=20000 characters and <=4096 Qwen article tokens; compatible rights and resolvable article-level target after two explicit AI reviews.',
            'selection':'First up to 300 eligible articles/language in frozen source order, whole event groups; no class balancing, score-based removal or seed changes. Disclose supply/rights/grouping shortages.',
            'final_test_access':'No candidate predictions or candidate-error examples in annotation. Same AI annotator already exposed to development findings; this is not independent blind human evaluation.'},
        'no_post_test_tuning':True,'human_gold':False,'label_status':'AI-labelled · Codex-reviewed'}
    frozen_write(OUT/'final-candidates-v1.json',json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'frozen_candidates':names,'selected_on_development':selected,'stability':stability},indent=2),flush=True)


def frozen_declaration():
    value=json.loads((OUT/'final-candidates-v1.json').read_text())
    verified()
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=value['final_runner_sha256']:
        raise RuntimeError('Final runner changed after candidate declaration')
    if hashlib.sha256(TRAIN.read_bytes()).hexdigest()!=value['train_sha256']:
        raise RuntimeError('Complete Train changed')
    return value


def train():
    d=frozen_declaration();gpu();tracking();fit=pd.read_csv(TRAIN);base_hash=fingerprint(BASE)
    records_path=OUT/'final-fit-progress-v1.json';records=json.loads(records_path.read_text()) if records_path.exists() else []
    for name in d['models']:
        if any(r['model']==name for r in records):continue
        path=ART/'final'/name
        if path.exists():raise RuntimeError('Unfinished full-Train artifact requires recovery audit: '+str(path))
        with mlflow.start_run(run_name='Full article · complete Train · '+name) as run:
            mlflow.set_tags({'phase':'article-next-stage','stage':'engineering','purpose':'complete-train-final-fit','model':name,'human_gold':'false','label_status':'AI-labelled · Codex-reviewed'})
            mlflow.log_params({'train_sha256':d['train_sha256'],'annotation_sha256':d['annotation_sha256'],'seed':SEED,'fit_rows':len(fit),
                'candidate_declaration_sha256':digest((OUT/'final-candidates-v1.json').read_text()),'development_protocol_sha256':d['development_protocol_sha256'],'final_runner_sha256':d['final_runner_sha256']})
            if name.startswith('qwen'):
                adapter=None
                if name=='qwen-lora':
                    adapter=ART/'final/qwen-adapter';fit_lora(fit,adapter)
                qwen_profile(path,adapter)
            else:
                fit_encoder(fit,path,128 if name=='frozen128' else int(name.removeprefix('minilm')),frozen=name=='frozen128')
            sha,full_size=artifact_info(path,base_hash)
            mlflow.log_param('artifact_sha256',sha);mlflow.log_metric('full_model_size_bytes',full_size)
            record={'model':name,'path':str(path.relative_to(ROOT)),'run_id':run.info.run_id,'artifact_sha256':sha,'full_model_size_bytes':full_size,'full_train_rows':len(fit)}
            records.append(record);write_json(records_path,records)
            print('Complete-Train artifact fixed:',name,flush=True)
    frozen_write(OUT/'final-artifacts-v1.json',json.dumps({'candidate_declaration_sha256':digest((OUT/'final-candidates-v1.json').read_text()),'models':records,'final_test_predicted':False},ensure_ascii=False,indent=2)+'\n')


def one_test_pass(path,fit,test,checkpoint):
    predictor=load_artifact(path)
    for i in range(20):predictor.predict_proba([fit.iloc[i%len(fit)].text]);torch.cuda.synchronize()
    records=[json.loads(x) for x in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
    if len({r['id'] for r in records})!=len(records):raise RuntimeError('Duplicate checkpoint predictions')
    expected=test.set_index('id')
    for record in records:
        if record['id'] not in expected.index or record['text_sha256']!=expected.loc[record['id'],'text_sha256'] or record['reference']!=expected.loc[record['id'],'label']:
            raise RuntimeError('Checkpoint refers to a changed Test input/reference')
    done={r['id'] for r in records};checkpoint.parent.mkdir(parents=True,exist_ok=True);torch.cuda.reset_peak_memory_stats()
    with checkpoint.open('a') as stream:
        for row in test.itertuples():
            if row.id in done:continue
            torch.cuda.synchronize();started=time.perf_counter();scores=predictor.predict_proba([row.text])[0];torch.cuda.synchronize()
            elapsed=(time.perf_counter()-started)*1000
            record={'id':row.id,'text_sha256':row.text_sha256,'language':row.language,'reference':row.label,'prediction':LABELS[int(np.argmax(scores))],
                    'scores':[float(v) for v in scores],'milliseconds':elapsed}
            stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno());records.append(record)
    if len(records)!=len(test):raise RuntimeError('Incomplete Test checkpoint')
    ordered=test.set_index('id').loc[[r['id'] for r in records]].reset_index();result=metrics(ordered,[LABELS.index(r['prediction']) for r in records])
    times=[r['milliseconds'] for r in records];result.update(inference_ms_per_text=float(np.mean(times)),latency_p50_ms=float(np.percentile(times,50)),latency_p95_ms=float(np.percentile(times,95)),
        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(),test_inference_calls=len(records),warmup_fit_calls=20,model_version=predictor.version)
    del predictor;clean();return result


def evaluate(resume=False):
    d=frozen_declaration();artifacts=json.loads((OUT/'final-artifacts-v1.json').read_text());manifest=json.loads((OUT/'final-test-v1/test-manifest.json').read_text())
    declaration_sha=digest((OUT/'final-candidates-v1.json').read_text())
    if artifacts['candidate_declaration_sha256']!=declaration_sha or manifest['candidate_declaration_sha256']!=declaration_sha:
        raise RuntimeError('Candidates changed after final preparation')
    if hashlib.sha256((FINAL_DATA/'articles.jsonl').read_bytes()).hexdigest()!=manifest['source_sha256']:raise RuntimeError('Frozen Test source changed')
    test_path=FINAL_DATA/'test.csv';test_bytes=test_path.read_bytes()
    if hashlib.sha256(test_bytes).hexdigest()!=manifest['test_sha256']:raise RuntimeError('Final Test changed')
    if hashlib.sha256((FINAL_DATA/'annotations.jsonl').read_bytes()).hexdigest()!=manifest['annotation_sha256']:raise RuntimeError('Final Test annotations changed')
    lock_path=OUT/'final-test-evaluation.lock';lock=json.loads(lock_path.read_text()) if lock_path.exists() else None
    identity={'candidate_declaration_sha256':declaration_sha,'test_sha256':manifest['test_sha256'],'test_manifest_sha256':digest((OUT/'final-test-v1/test-manifest.json').read_text())}
    if lock and (lock['status']=='complete' or not resume or any(lock[k]!=v for k,v in identity.items())):
        raise RuntimeError('Final Test already started/completed or identity changed')
    gpu();tracking();base_hash=fingerprint(BASE)
    for record in artifacts['models']:
        if artifact_info(ROOT/record['path'],base_hash)[0]!=record['artifact_sha256']:raise RuntimeError('Frozen final artifact changed')
    if not lock:write_json(lock_path,{'status':'started',**identity})
    table=pd.read_csv(TRAIN);test=pd.read_csv(test_path);results_path=OUT/'final-test-progress-v1.json';results=json.loads(results_path.read_text()) if results_path.exists() else []
    encoder_hash=fingerprint(Path(snapshot_download(ENCODER,revision=ENCODER_REVISION,local_files_only=True)))
    for record in artifacts['models']:
        if any(r['model']==record['model'] for r in results):continue
        with mlflow.start_run(run_name='Full article · one fresh Test · '+record['model']) as run:
            run_identity=RunIdentity(research_question='frozen-final-comparison',stage='final-test',data_version='fresh-full-article-test-v1',annotation_version='fresh-test-ai-two-pass-v1',
                split_version='protected-event-groups-final-v1',model_family='Qwen' if record['model'].startswith('qwen') else 'MiniLM',adaptation=record['model'],fold='full-train',seed=SEED,precision='bf16',context_policy='full-article-v5',
                git_commit=verified()['git_commit'],protocol_sha256=d['development_protocol_sha256'],input_sha256=manifest['test_sha256'],annotation_sha256=manifest['annotation_sha256'],
                split_sha256=identity['test_manifest_sha256'],artifact_sha256=record['artifact_sha256'],base_model_sha256=base_hash if record['model'].startswith('qwen') else encoder_hash)
            mlflow.set_tags(run_identity.tags());mlflow.log_params({'candidate_declaration_sha256':declaration_sha,'full_train_rows':len(table),'test_rows':len(test),'selected_on_development':d['selected_on_development']})
            result=one_test_pass(ROOT/record['path'],table,test,ART/'final-predictions'/(record['model']+'.jsonl'))
            mlflow.log_metrics({k:v for k,v in result.items() if isinstance(v,(float,int))});mlflow.log_dict(result,'final_test_metrics.json')
            mlflow.log_metric('full_model_size_bytes',record['full_model_size_bytes'])
            results.append({**record,'test_run_id':run.info.run_id,**result});write_json(results_path,results)
    write_json(OUT/'final-test-v1.json',{'selected_on_development':d['selected_on_development'],'no_post_test_tuning':True,'test_manifest_sha256':identity['test_manifest_sha256'],
        'label_status':'AI-labelled · Codex-reviewed','human_gold':False,'models':results})
    write_json(lock_path,{'status':'complete',**identity});print('One fresh Test pass per frozen candidate complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['declare','train','evaluate']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='evaluate':evaluate(args.resume)
    else:{'declare':declaration,'train':train}[args.action]()
