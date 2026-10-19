"""Bounded v2 ablations. Reads only original train.csv; leaves v1 artifacts/reports intact."""
import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from second_path.config import ROOT, DATA, MODELS, REPORTS, LABELS
from second_path.inference import Predictor, SentimentNet, save_model
from second_path.tracking import tracking, fingerprint, iteration_run
from second_path.fetch import extract_article, FetchError

DEV = Path(os.getenv('SENTIMENT_DEV_REPORTS',str(REPORTS/'development'))).resolve()
LOCAL = DATA.parent/'development'
ART = MODELS/'development'
STAGES = ['B0_frozen','R0_reference','R1_quality','R2_article','R3_deploy','R4_combined']
SEED = 20261019


def setup():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA GPU with BF16 required')
    torch.set_num_threads(4)
    tracking()
    ART.mkdir(parents=True,exist_ok=True)
    LOCAL.mkdir(parents=True,exist_ok=True)
    DEV.mkdir(parents=True,exist_ok=True)
    if not (DEV/'protocol.json').exists():
        shutil.copyfile(REPORTS/'development'/'protocol.json',DEV/'protocol.json')
    protocol = json.loads((DEV/'protocol.json').read_text())
    return protocol


def train_data():
    expected=json.loads((REPORTS/'data_manifest.json').read_text())['splits']['train']['sha256']
    assert hashlib.sha256((DATA/'train.csv').read_bytes()).hexdigest()==expected
    return pd.read_csv(DATA/'train.csv')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_frozen(df, seed):
    set_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(MODELS/'frozen'/'tokenizer',local_files_only=True)
    net=SentimentNet(AutoModel.from_pretrained(MODELS/'frozen'/'encoder',local_files_only=True)).cuda().eval()
    y=np.array([LABELS.index(v) for v in df.label])
    tokens=tokenizer(df.text.tolist(),truncation=True,max_length=128,padding=False)
    vectors=[]
    with torch.inference_mode():
        for start in range(0,len(df),64):
            rows=[{k:v[i] for k,v in tokens.items()} for i in range(start,min(start+64,len(df)))]
            batch=tokenizer.pad(rows,padding=True,return_tensors='pt').to('cuda')
            vectors.append(net.features(batch).cpu().numpy())
    head=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=seed)
    head.fit(np.concatenate(vectors),y)
    with torch.no_grad():
        net.head.weight.copy_(torch.as_tensor(head.coef_,dtype=torch.float32,device='cuda'))
        net.head.bias.copy_(torch.as_tensor(head.intercept_,dtype=torch.float32,device='cuda'))
    return net,tokenizer


def tune(net,tokenizer,df,quality,seed):
    set_seed(seed)
    net.train()
    tokens=tokenizer(df.text.tolist(),truncation=True,max_length=128,padding=False)
    y=np.array([LABELS.index(v) for v in df.label])
    strata=(df.language+':'+df.label) if quality else df.label
    counts=strata.value_counts()
    weights=np.array([len(df)/(len(counts)*counts[v]) for v in strata],dtype=np.float32)
    opt=torch.optim.AdamW([{'params':net.encoder.parameters(),'lr':2e-5},
                          {'params':net.head.parameters(),'lr':2e-4}],weight_decay=.01)
    steps=3*int(np.ceil(len(df)/32))
    schedule=get_linear_schedule_with_warmup(opt,int(.1*steps),steps)
    for epoch in range(3):
        order=np.random.default_rng(seed+epoch).permutation(len(df))
        total,seen=0.,0
        for start in range(0,len(df),32):
            ix=order[start:start+32]
            batch=tokenizer.pad([{k:v[int(i)] for k,v in tokens.items()} for i in ix],padding=True,return_tensors='pt').to('cuda')
            target=torch.as_tensor(y[ix],device='cuda')
            w=torch.as_tensor(weights[ix],device='cuda')
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=net(batch).float()
                losses=torch.nn.functional.cross_entropy(logits,target,reduction='none')
                loss=(losses*w).sum()/w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
            schedule.step()
            total+=loss.item()*len(ix)
            seen+=len(ix)
        mlflow.log_metric('train_loss',total/seen,step=epoch+1)
        mlflow.log_metric('train_rows_seen',seen,step=epoch+1)
        print(f'epoch {epoch+1}/3 {seen} rows loss={total/seen:.4f}',flush=True)
    net.eval()
    return net


def meta(stage,fold,run_id):
    return {'kind':'finetuned' if stage!='B0_frozen' else 'frozen',
            'model_version':f'dev2-{stage}-fold{fold}-{SEED}', 'run_id':run_id,
            'precision':'bf16' if stage in ['R3_deploy','R4_combined'] else 'fp32',
            'aggregation':'token_weighted' if stage in ['R2_article','R4_combined'] else 'uniform',
            'improved_article':stage in ['R2_article','R4_combined'], 'development_stage':stage}


def derive(source,dest,metadata):
    if metadata['precision']=='bf16':
        p=Predictor(source,device='cuda')
        p.net.to(dtype=torch.bfloat16)
        save_model(dest,p.net,p.tokenizer,metadata)
        del p
    else:
        shutil.copytree(source,dest,copy_function=os.link,ignore=shutil.ignore_patterns('artifact.json'))
        original=json.loads((source/'artifact.json').read_text())
        (dest/'artifact.json').write_text(json.dumps({**original,**metadata},indent=2))


def metrics(y,p):
    pred=np.asarray(p).argmax(-1)
    out={'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0),
         'negative_precision':precision_score(y,pred,labels=[0],average=None,zero_division=0)[0],
         'negative_recall':recall_score(y,pred,labels=[0],average=None,zero_division=0)[0]}
    for i,label in enumerate(LABELS):
        out['f1_'+label]=f1_score(y,pred,labels=[i],average=None,zero_division=0)[0]
    return {k:float(v) for k,v in out.items()}


def controlled_cases():
    # Authored fixtures, deliberately synthetic; never presented as real-article gold.
    anchors=[('de','negative','Das Unternehmen meldet schwere Verluste und streicht zahlreiche Arbeitsplätze.'),
             ('de','neutral','Das Unternehmen veröffentlicht seinen Geschäftsbericht am Dienstag in Berlin.'),
             ('de','positive','Das Unternehmen meldet einen Rekordgewinn und erhöht seine Umsatzprognose.'),
             ('en','negative','The company reported heavy losses and announced substantial job cuts.'),
             ('en','neutral','The company will publish its annual report on Tuesday in London.'),
             ('en','positive','The company reported record profits and raised its revenue forecast.')]
    texts,labels=[],[]
    for lang,label,anchor in anchors:
        other=next(a for l,v,a in anchors if l==lang and v!=label)
        for text in [anchor,(anchor+' ')*20,(anchor+' ')*20+other]:
            texts.append(text)
            labels.append(LABELS.index(label))
    return texts,np.array(labels)


def extraction_cases():
    a='The company reported a quarterly loss and announced job cuts across its factories.'
    b='The board will publish its annual report on Tuesday and hold a meeting in London.'
    return [
      (f'<article><p>{a}</p></article>',a),
      (f'<article><h1>Profit warning</h1><p>{a}</p></article>','Profit warning '+a),
      (f'<article><p>{a}</p><p hidden>{b}</p></article>',a),
      (f'<article><p>{a}</p><p>{a}</p></article>',a),
      (f'<article><p>Related story</p></article><article><p>{a}</p></article>',a),
      (f'<main><p>{b}</p><aside><p>{a}</p></aside></main>',b)]


def evaluate(path,df):
    p=Predictor(path,device='cuda')
    y=np.array([LABELS.index(v) for v in df.label])
    probs=p.predict_proba(df.text.tolist())
    out=metrics(y,probs)
    for lang in ['de','en']:
        ix=df.language.to_numpy()==lang
        out['macro_f1_'+lang]=metrics(y[ix],probs[ix])['macro_f1']
    for _ in range(20): p.predict_proba(['Die Firma meldet ihre Quartalszahlen.'])
    duration=[]
    for text in df.text.iloc[:180]:
        torch.cuda.synchronize()
        start=time.perf_counter()
        p.predict_proba([text])
        torch.cuda.synchronize()
        duration.append(1000*(time.perf_counter()-start))
    out.update(inference_ms_per_text=float(np.mean(duration)),latency_p50_ms=float(np.median(duration)),
               latency_p95_ms=float(np.percentile(duration,95)),
               model_size_bytes=sum(f.stat().st_size for f in path.rglob('*') if f.is_file()))
    texts,labels=controlled_cases()
    out['synthetic_article_macro_f1']=metrics(labels,p.predict_proba(texts))['macro_f1']
    matches=[]
    for html,expected in extraction_cases():
        try:
            matches.append(extract_article(html,improved=p.meta['improved_article'])==expected)
        except FetchError:
            matches.append(False)
    out['html_fixture_exact_match']=float(np.mean(matches))
    cm=confusion_matrix(y,probs.argmax(-1),labels=[0,1,2])
    del p
    gc.collect()
    torch.cuda.empty_cache()
    return out,probs,cm


def run(resume=False):
    protocol=setup()
    df=train_data()
    if (DEV/'runs.json').exists():
        raise SystemExit('v2 execution complete; preserve completed evidence')
    if (LOCAL/'started.json').exists() and not resume:
        raise SystemExit('v2 execution started; explicit --resume required after examining failure')
    digest=hashlib.sha256((DEV/'protocol.json').read_bytes()).hexdigest()
    if (LOCAL/'started.json').exists():
        assert json.loads((LOCAL/'started.json').read_text())['protocol_sha256']==digest, 'Protocol changed after starting'
    else:
        (LOCAL/'started.json').write_text(json.dumps({'protocol_sha256':digest}))
    cv=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
    folds=list(cv.split(df,df.language+':'+df.label,df.source))
    manifest=[]
    records=json.loads((DEV/'partial_runs.json').read_text()) if (DEV/'partial_runs.json').exists() else []
    for fold,(fit_ix,dev_ix) in enumerate(folds):
        fit,dev=df.iloc[fit_ix].reset_index(drop=True),df.iloc[dev_ix].reset_index(drop=True)
        assert not (set(fit.source)&set(dev.source))
        manifest.append({'fold':fold,'fit_rows':len(fit),'dev_rows':len(dev),'fit_sources':sorted(fit.source.unique()),
                         'dev_sources':sorted(dev.source.unique()),
                         'fit_id_sha256':hashlib.sha256('\n'.join(fit.id).encode()).hexdigest(),
                         'dev_id_sha256':hashlib.sha256('\n'.join(dev.id).encode()).hexdigest()})
        (DEV/'folds.json').write_text(json.dumps(manifest,indent=2))
        for stage in STAGES:
            if any(r['stage']==stage and r['fold']==fold for r in records):
                continue
            path=ART/f'fold{fold}'/stage
            if path.exists():
                old=json.loads((path/'artifact.json').read_text())
                path.rename(path.with_name(stage+'_aborted_'+old['run_id'][:8]))
            with iteration_run(run_name=f'dev2-{stage}-fold{fold}',tags={'phase':'development-v2','stage':stage,'fold':str(fold)}) as run:
                metadata=meta(stage,fold,run.info.run_id)
                mlflow.log_params({'stage':stage,'fold':fold,'seed':SEED+fold,'train_rows':len(fit),'dev_rows':len(dev),
                    'group':'source','stratum':'language:label','original_test_used':False,'original_val_used':False,
                    'epochs':0 if stage=='B0_frozen' else 3,'batch_size':32,'max_length':128,
                    'encoder_lr':2e-5,'head_lr':2e-4,'weight_decay':.01,'warmup':.1,
                    'loss_balance':'language:label' if stage in ['R1_quality','R4_combined'] else 'label',
                    'precision':metadata['precision'],'aggregation':metadata['aggregation'],
                    'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
                start=time.perf_counter()
                if stage=='B0_frozen':
                    net,tokenizer=fit_frozen(fit,SEED+fold)
                    save_model(path,net,tokenizer,metadata)
                    del net,tokenizer
                elif stage in ['R0_reference','R1_quality']:
                    p=Predictor(ART/f'fold{fold}'/'B0_frozen',device='cuda')
                    tune(p.net,p.tokenizer,fit,stage=='R1_quality',SEED+fold)
                    save_model(path,p.net,p.tokenizer,metadata)
                    del p
                else:
                    source=ART/f'fold{fold}'/('R1_quality' if stage=='R4_combined' else 'R0_reference')
                    derive(source,path,metadata)
                torch.cuda.synchronize()
                training_seconds=time.perf_counter()-start
                gc.collect()
                torch.cuda.empty_cache()
                scores,probs,cm=evaluate(path,dev)
                scores['training_or_packaging_seconds']=training_seconds
                mlflow.log_metrics(scores)
                mlflow.log_param('artifact_sha256',fingerprint(path))
                mlflow.log_artifact(str(DEV/'protocol.json'))
                cm_path=DEV/f'{stage}_fold{fold}_confusion_matrix.csv'
                pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(cm_path)
                mlflow.log_artifact(str(cm_path))
                np.savez(LOCAL/f'{stage}_fold{fold}.npz',indices=dev_ix,probs=probs)
                record={'stage':stage,'fold':fold,'run_id':run.info.run_id,**scores}
                records.append(record)
                (DEV/'partial_runs.json').write_text(json.dumps(records,indent=2))
                print(json.dumps(record),flush=True)
    (DEV/'runs.json').write_text(json.dumps(records,indent=2))
    print('All v2 development runs complete')


def summarize():
    if (DEV/'decision.json').exists():
        raise SystemExit('Selection already frozen; do not select again after final audit')
    setup()
    runs=json.loads((DEV/'runs.json').read_text())
    frame=pd.DataFrame(runs)
    summary=[]
    for stage in STAGES:
        part=frame[frame.stage==stage]
        row={'stage':stage,'folds':len(part),'mean_macro_f1':float(part.macro_f1.mean()),
             'sd_macro_f1':float(part.macro_f1.std(ddof=1)),
             'median_latency_ms':float(part.inference_ms_per_text.median()),
             'model_size_bytes':int(part.model_size_bytes.median()),
             'mean_macro_f1_de':float(part.macro_f1_de.mean()),'mean_macro_f1_en':float(part.macro_f1_en.mean()),
             'mean_negative_recall':float(part.negative_recall.mean()),'mean_negative_precision':float(part.negative_precision.mean()),
             'synthetic_article_macro_f1':float(part.synthetic_article_macro_f1.mean()),
             'html_fixture_exact_match':float(part.html_fixture_exact_match.mean())}
        summary.append(row)
        with iteration_run(run_name=f'dev2-summary-{stage}',tags={'phase':'development-v2-summary','stage':stage}) as run:
            mlflow.log_params({'folds':3,'group':'source','selection_metric':'mean_macro_f1','original_test_used':False,'original_val_used':False})
            mlflow.log_metrics({k:v for k,v in row.items() if isinstance(v,(float,int))})
            row['summary_run_id']=run.info.run_id
    best=max(r['mean_macro_f1'] for r in summary)
    close=[r for r in summary if best-r['mean_macro_f1']<=.005]
    selected=min(close,key=lambda r:(r['median_latency_ms'],r['model_size_bytes']))
    decision={'selected':selected['stage'],'selection_locked_before_external_comparison':True,
              'rule':'mean source-group CV macro-F1; within .005 prefer latency then bytes', 'results':summary}
    (DEV/'decision.json').write_text(json.dumps(decision,indent=2))
    pd.DataFrame(summary).to_csv(DEV/'curve.csv',index=False)
    print(json.dumps(decision,indent=2))


def release():
    setup()
    selected=json.loads((DEV/'decision.json').read_text())['selected']
    dest=ART/'release'
    if dest.exists(): raise SystemExit('Release already exists')
    df=train_data()
    with iteration_run(run_name=f'dev2-release-{selected}',tags={'phase':'development-v2-release','stage':selected}) as run:
        net,tokenizer=fit_frozen(df,SEED)
        if selected!='B0_frozen': tune(net,tokenizer,df,selected in ['R1_quality','R4_combined'],SEED)
        metadata=meta(selected,'release',run.info.run_id)
        if metadata['precision']=='bf16': net.to(dtype=torch.bfloat16)
        save_model(dest,net,tokenizer,metadata)
        mlflow.log_params({'stage':selected,'seed':SEED,'train_rows':len(df),'epochs':0 if selected=='B0_frozen' else 3,
                           'test_evaluated':False,'original_val_used':False,'selection_source':'source-group training CV'})
        mlflow.log_param('artifact_sha256',fingerprint(dest))
        mlflow.log_artifact(str(DEV/'decision.json'))
        mlflow.log_artifact(str(dest/'artifact.json'))
        # Keep original deployment selection alongside the new artifact for rollback.
        if (MODELS/'selected.json').exists():
            shutil.copyfile(MODELS/'selected.json',ART/'previous_selected.json')
        (DEV/'release.json').write_text(json.dumps({'stage':selected,'run_id':run.info.run_id,'artifact_sha256':fingerprint(dest),
                                                  'model_version':metadata['model_version'],'original_test_evaluated':False},indent=2))
    print('Frozen full-train release',selected)

def activate():
    release=json.loads((DEV/'release.json').read_text())
    if fingerprint(ART/'release')!=release['artifact_sha256']:
        raise RuntimeError('Release differs from frozen selected artifact')
    (MODELS/'selected.json').write_text(json.dumps({'model':'development/release',
        'model_version':release['model_version']},indent=2))
    print('Activated frozen release; restart uvicorn to load it')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['run','summarize','release','activate'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='run': run(resume=args.resume)
    else: {'summarize':summarize,'release':release,'activate':activate}[args.action]()
