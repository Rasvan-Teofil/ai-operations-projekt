"""One predeclared additional seed for close development finalists only.

A deterministic zero-shot finalist reuses its already fixed OOF predictions;
changing a training seed cannot create a new zero-shot model.
"""
import hashlib
import json

import mlflow
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from pathlib import Path

from src.config import ROOT, LABELS, ENCODER, ENCODER_REVISION
from src.next_stage import digest
from src.next_stage_collect import write_json
from src.next_stage_experiment import (ART, OUT, TRAIN, BASE, SEED, artifact_info,
    fit_encoder, fit_lora, gpu, measure, metrics, qwen_profile, verified)
from src.next_stage_prepare import frozen_write
from src.next_stage_protocol import RunIdentity, select_candidate
from src.tracking import fingerprint, tracking


def run():
    p=verified();development=json.loads((OUT/'development-v1.json').read_text())
    if development['scored_evaluations']!=24 or (ROOT/'data/next_stage/final-test-v1/articles.jsonl').exists():
        raise RuntimeError('Complete initial 24 conditions; fresh Test must remain absent')
    rows=development['out_of_fold'];shared=digest((OUT/'development-v1.json').read_text())
    mini=select_candidate([{**r,'comparable_group':shared} for r in rows if r['model'] in ('minilm128','minilm256','minilm512')])
    names=['frozen128',mini,'qwen-zero','qwen-lora']
    ranked=sorted([r for r in rows if r['model'] in names],key=lambda r:r['primary_score'],reverse=True);gap=ranked[0]['primary_score']-ranked[1]['primary_score']
    path=OUT/'stability-decision-v1.json'
    if path.exists():raise RuntimeError('Stability decision already finalized')
    if gap>.02:
        frozen_write(path,json.dumps({'status':'not_triggered','performed':False,'primary_gap':gap,'reason':'Top-two candidate OOF primary scores differ by more than the frozen .02 trigger.'},indent=2)+'\n');return
    selected=[r['model'] for r in ranked[:2]];seed=SEED+1
    frozen_write(OUT/'stability-protocol-v1.json',json.dumps({'seed':seed,'finalists':selected,'same_three_folds':True,'recipe_change':'Only the training/random-initialization seed changes; no input, label, prompt, epoch, optimiser or context search.',
        'trigger_gap':gap,'development_sha256':shared,'zero_shot':'Reuse deterministic unadapted model predictions; no meaningless re-inference for a changed training seed.',
        'interpretation':'Diagnostic seed sensitivity. Original-seed OOF primary metric and tie rule remain the deployment selection rule; no final Test available.'},ensure_ascii=False,indent=2)+'\n')
    gpu();tracking();table=pd.read_csv(TRAIN);members=json.loads((OUT/'experiment-membership-v1.json').read_text());base_hash=fingerprint(BASE)
    encoder_hash=fingerprint(Path(snapshot_download(ENCODER,revision=ENCODER_REVISION,local_files_only=True)))
    progress=OUT/'stability-progress-v1.json';records=json.loads(progress.read_text()) if progress.exists() else []
    for name in selected:
        if name=='qwen-zero':continue
        for member in members:
            fold=member['fold']
            if any(r['model']==name and r['fold']==fold for r in records):continue
            fit=table[table.id.isin(member['fit_ids'])];dev=table[table.id.isin(member['dev_ids'])]
            folder=ART/'stability'/f'seed{seed}'/f'fold{fold}';artifact=folder/name
            if artifact.exists():raise RuntimeError('Unfinished stability artifact requires recovery audit')
            with mlflow.start_run(run_name=f'Full article · seed stability · fold{fold} · {name}') as run:
                mlflow.set_tags({'phase':'article-next-stage','stage':'development','research_question':'seed-stability','human_gold':'false','label_status':'AI-labelled · Codex-reviewed'})
                mlflow.log_params({'seed':seed,'fit_rows':len(fit),'dev_rows':len(dev),'fold':fold,'stability_protocol_sha256':digest((OUT/'stability-protocol-v1.json').read_text())})
                if name=='qwen-lora':
                    adapter=folder/'adapter';fit_lora(fit,adapter,seed);qwen_profile(artifact,adapter)
                else:fit_encoder(fit,artifact,128 if name=='frozen128' else int(name.removeprefix('minilm')),frozen=name=='frozen128',seed=seed)
                sha,size=artifact_info(artifact,base_hash)
                identity=RunIdentity(research_question='seed-stability',stage='development',data_version='article-train-240-v1',annotation_version='pilot-300-v2-input-v5',split_version='grouped-three-fold-v1',
                    model_family='Qwen' if name.startswith('qwen') else 'MiniLM',adaptation=name,fold=str(fold),seed=seed,precision='bf16',context_policy='full-article-v5',git_commit=p['git_commit'],
                    protocol_sha256=hashlib.sha256((OUT/'development-protocol-v1.json').read_bytes()).hexdigest(),input_sha256=p['input_sha256'],annotation_sha256=p['annotation_sha256'],split_sha256=p['split_sha256'],
                    artifact_sha256=sha,base_model_sha256=base_hash if name.startswith('qwen') else encoder_hash)
                mlflow.set_tags(identity.tags());measured=measure(artifact,fit,dev,folder/(name+'-dev-predictions.jsonl'))
                mlflow.log_metrics({k:v for k,v in measured.items() if isinstance(v,(float,int))});mlflow.log_dict(measured,'stability_metrics.json')
                records.append({'model':name,'fold':fold,'seed':seed,'run_id':run.info.run_id,'artifact_sha256':sha,'full_model_size_bytes':size,**measured});write_json(progress,records)
                print('Additional seed completed',fold,name,measured['primary_score'],flush=True)
    summaries=[]
    for name in selected:
        original=next(r for r in rows if r['model']==name)
        if name=='qwen-zero':summaries.append({'model':name,'original_primary_score':original['primary_score'],'additional_seed_primary_score':original['primary_score'],'reused_deterministic_predictions':True});continue
        predictions=[]
        for member in members:
            checkpoint=ART/'stability'/f'seed{seed}'/f'fold{member["fold"]}'/(name+'-dev-predictions.jsonl')
            predictions.extend(json.loads(x) for x in checkpoint.read_text().splitlines())
        ordered=table.set_index('id').loc[[r['id'] for r in predictions]].reset_index();score=metrics(ordered,[LABELS.index(r['prediction']) for r in predictions])
        summaries.append({'model':name,'original_primary_score':original['primary_score'],'additional_seed_primary_score':score['primary_score'],'additional_seed_metrics':score,'reused_deterministic_predictions':False})
    frozen_write(path,json.dumps({'status':'complete','performed':True,'seed':seed,'finalists':selected,'primary_gap':gap,'additional_scored_evaluations':len(records),
        'total_scored_development_evaluations':24+len(records),'summaries':summaries,'selection_rule_unchanged':True,
        'limits':'Two seeds and three folds describe limited sensitivity, not an independent uncertainty estimate or a hyperparameter search.'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    run()
