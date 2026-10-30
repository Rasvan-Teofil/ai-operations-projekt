"""Final shared external audit, after v2 selection. Never selects or retrains a model."""
import argparse
import ast
import gc
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from second_path.config import ROOT,DATA,MODELS,REPORTS,LABELS
from second_path.development import metrics, DEV
from second_path.prepare import normalized
from second_path.inference import Predictor
from second_path.tracking import tracking,fingerprint,iteration_run

OUT=DEV
LOCAL=DATA.parent/'benchmark'
REV='eba43610fe3cb57a3e5773cd6c037da1f8992ad5'
URL=f'https://raw.githubusercontent.com/pyRis/SEntFiN/{REV}/SEntFiN.csv'
TRANSLATOR='Helsinki-NLP/opus-mt-en-de'
TRANS_REV='6183067f769a302e3861815543b9f312c71b0ca4'
SEED=20261019


def prepare(reference):
    if (OUT/'benchmark_manifest.json').exists(): raise SystemExit('External benchmark already frozen')
    assert (OUT/'release.json').exists(), 'Freeze our release before constructing final benchmark'
    LOCAL.mkdir(parents=True,exist_ok=True)
    raw=urllib.request.urlopen(URL,timeout=30).read()
    source=pd.read_csv(io.BytesIO(raw))
    rows=[]
    for _,r in source.iterrows():
        try: values=ast.literal_eval(r['Decisions'])
        except (ValueError,SyntaxError): continue
        if not isinstance(values,dict) or not values: continue
        labels={str(v).strip().lower() for v in values.values()}
        if len(labels)!=1 or not labels.issubset(LABELS): continue
        text=str(r['Title']).strip()
        if not 20<=len(text)<=20000: continue
        rows.append({'paired_id':'sentfin-'+str(r['S No.']),'text':text,'label':next(iter(labels))})
    frame=pd.DataFrame(rows)
    unanimous=len(frame)
    frame['normalized']=frame.text.map(normalized)
    frame=frame.drop_duplicates('normalized')
    training=pd.read_csv(DATA/'train.csv')
    peer_training=pd.read_csv(reference/'data'/'processed'/'train.csv')
    seen=set(training.text.map(normalized))|set(peer_training.text.map(normalized))
    before=len(frame)
    frame=frame[~frame.normalized.isin(seen)].copy()
    # Class-balanced by a fixed seed, independent of all model predictions.
    selected=frame.groupby('label',group_keys=False).sample(n=200,random_state=SEED).sort_values('paired_id').reset_index(drop=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(TRANSLATOR,revision=TRANS_REV)
    translator=AutoModelForSeq2SeqLM.from_pretrained(TRANSLATOR,revision=TRANS_REV).cuda().eval()
    translations=[]
    source_truncated=0
    with torch.inference_mode():
        for start in range(0,len(selected),24):
            texts=selected.text.iloc[start:start+24].tolist()
            source_truncated+=sum(len(tokenizer.encode(t))>256 for t in texts)
            tokens=tokenizer(texts,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
            generated=translator.generate(**tokens,num_beams=4,max_new_tokens=128,do_sample=False)
            translations.extend(tokenizer.batch_decode(generated,skip_special_tokens=True))
            print('translated',len(translations),'/',len(selected),flush=True)
    en=selected[['paired_id','text','label']].copy()
    en['language']='en'
    en['origin']='human-labelled native English unanimous-entity headline'
    de=en.copy()
    de['text']=translations
    de['language']='de'
    de['origin']='machine translation; labels inherited from English, not native German gold'
    assert all(text.strip() for text in translations)
    benchmark=pd.concat([en,de],ignore_index=True)
    benchmark['id']=benchmark.paired_id+':'+benchmark.language
    target=LOCAL/'shared.csv'
    benchmark.to_csv(target,index=False)
    manifest={'dataset':'pyRis/SEntFiN','revision':REV,'source_url':URL,'source_sha256':hashlib.sha256(raw).hexdigest(),
              'license':'MIT, original repository; copyright Ankur Sinha, Satishwar Kedas, Rishu Kumar, Pekka Malo (2021)',
              'raw_rows':len(source),'unanimous_valid_length_rows':unanimous,'normalized_training_overlap_removed':before-len(frame),
              'eligible_after_dedup_overlap':len(frame),'selection':'200 per class, random_state=20261019, no classifier predictions used',
              'rows':len(benchmark),'native_english_rows':len(en),'translated_german_rows':len(de),
              'counts':{f'{lang}:{label}':int(n) for (lang,label),n in benchmark.groupby(['language','label']).size().items()},
              'translator':TRANSLATOR,'translator_revision':TRANS_REV,'translator_license':'CC-BY-4.0',
              'translation':'CUDA fp32, batch24, beam4, do_sample=False, encoder256, max_new_tokens128',
              'translation_source_truncations':source_truncated,
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'our_selection_sha256':hashlib.sha256((OUT/'decision.json').read_bytes()).hexdigest(),
              'our_release_sha256':json.loads((OUT/'release.json').read_text())['artifact_sha256'],
              'caveats':['Entity-level labels mapped only when all entities agree.','Exactly normalized training overlap removed; semantic/event overlap and pretraining contamination unknown.',
                         'German translations are a transfer check, not native German annotation.','600 paired headlines, not 1200 independent articles.','Balanced headline benchmark, not natural production prevalence or full-article gold.'],
              'reference_training_sha256':hashlib.sha256((reference/'data'/'processed'/'train.csv').read_bytes()).hexdigest()}
    (OUT/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2))
    del translator
    gc.collect()
    torch.cuda.empty_cache()
    print('Shared external benchmark frozen')


def audit(reference):
    tracking()
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'benchmark_manifest.json').read_text())
    csv=LOCAL/'shared.csv'
    assert hashlib.sha256(csv.read_bytes()).hexdigest()==manifest['sha256']
    assert hashlib.sha256((OUT/'decision.json').read_bytes()).hexdigest()==manifest['our_selection_sha256']
    assert fingerprint(MODELS/'development'/'release')==manifest['our_release_sha256']
    with (LOCAL/'audit.lock').open('x') as f: json.dump({'status':'started','benchmark_sha256':manifest['sha256']},f)
    # Reference source executes only now, with its own pinned lightweight environment.
    subprocess.run([str(reference/'.venv'/'bin'/'python'),str(ROOT/'src'/'peer_reference.py'),'predict',
                    '--checkout',str(reference),'--benchmark',str(csv),'--report',str(LOCAL/'peer_predictions.json')],check=True,cwd=reference)
    df=pd.read_csv(csv)
    truth=np.array([LABELS.index(v) for v in df.label])
    candidates=[('second_path_v1',MODELS/'finetuned'),('second_path_v2',MODELS/'development'/'release'),('colleague_main_default',None)]
    records=[]
    all_prob={}
    for name,path in candidates:
        if path:
            predictor=Predictor(path,device='cuda')
            for _ in range(20): predictor.predict_proba(['The firm reports quarterly results.'])
            prob=[]
            duration=[]
            for text in df.text:
                torch.cuda.synchronize()
                start=time.perf_counter()
                prob.append(predictor.predict_proba([text])[0])
                torch.cuda.synchronize()
                duration.append(1000*(time.perf_counter()-start))
            probabilities=np.stack(prob)
            size=sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            version=predictor.version
            digest=fingerprint(path)
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
        else:
            cached=np.load(LOCAL/'peer_predictions.npz')
            probabilities=cached['probs']
            duration=cached['durations']
            reference_info=json.loads((OUT/'peer_provenance.json').read_text())
            size=reference_info['model_size_bytes']
            version='tfidf-linearsvc-v1, reconstructed at pinned colleague commit'
            digest=reference_info['artifact_sha256']
        all_prob[name]=probabilities
        values=metrics(truth,probabilities)
        for language in ['en','de']:
            ix=df.language.to_numpy()==language
            values[f'macro_f1_{language}']=metrics(truth[ix],probabilities[ix])['macro_f1']
        values.update(inference_ms_per_text=float(np.mean(duration)),latency_p50_ms=float(np.median(duration)),
                      latency_p95_ms=float(np.percentile(duration,95)),model_size_bytes=size)
        cm=confusion_matrix(truth,probabilities.argmax(-1),labels=[0,1,2])
        cm_path=OUT/f'external_{name}_confusion_matrix.csv'
        pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(cm_path)
        with iteration_run(run_name=f'final-shared-audit-{name}',tags={'phase':'final-branch-comparison','candidate':name}) as run:
            mlflow.log_params({'benchmark_sha256':manifest['sha256'],'rows':len(df),'native_en_rows':600,'translated_de_rows':600,
                               'model_version':version,'artifact_sha256':digest,'used_for_model_selection':False,
                               'backend':'CPU' if path is None else 'CUDA','original_tests_reused':False})
            mlflow.log_metrics(values)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(OUT/'benchmark_manifest.json'))
            mlflow.log_artifact(str(OUT/'decision.json'))
            if path is None: mlflow.log_artifact(str(OUT/'peer_provenance.json'))
            record={'model':name,'run_id':run.info.run_id,**values}
            records.append(record)
            print(record,flush=True)
        np.savez(LOCAL/f'{name}_probabilities.npz',probs=probabilities)
    # Resample existing paired predictions only; no additional model evaluation or selection.
    ix=np.flatnonzero(df.language.to_numpy()=='en')
    rng=np.random.default_rng(SEED)
    diffs=[]
    for _ in range(1000):
        sample=rng.choice(ix,size=len(ix),replace=True)
        diffs.append(metrics(truth[sample],all_prob['second_path_v2'][sample])['macro_f1']-
                     metrics(truth[sample],all_prob['colleague_main_default'][sample])['macro_f1'])
    result={'benchmark_sha256':manifest['sha256'],'our_selection_unchanged':True,
            'primary_comparison':'native English macro-F1; German MT reported separately',
            'native_en_delta_v2_minus_peer':next(r['macro_f1_en'] for r in records if r['model']=='second_path_v2')-
                                           next(r['macro_f1_en'] for r in records if r['model']=='colleague_main_default'),
            'native_en_paired_bootstrap_95_percentile_interval':list(map(float,np.percentile(diffs,[2.5,97.5]))),
            'results':records}
    (OUT/'branch_comparison.json').write_text(json.dumps(result,indent=2))
    pd.DataFrame(records).to_csv(OUT/'branch_comparison.csv',index=False)
    (LOCAL/'audit.lock').write_text(json.dumps({'status':'complete','benchmark_sha256':manifest['sha256']},indent=2))
    # Complete the release run's external audit metrics with explicit evaluation provenance.
    release=json.loads((OUT/'release.json').read_text())
    record=next(r for r in records if r['model']=='second_path_v2')
    client=mlflow.MlflowClient()
    client.set_tag(release['run_id'],'final_audit_run_id',record['run_id'])
    for key,value in record.items():
        if isinstance(value,(int,float)): client.log_metric(release['run_id'],'external_'+key,float(value))
    print('Final shared audit complete; deployment selection unchanged')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','audit'])
    p.add_argument('--reference-checkout',type=Path,default=ROOT.parent/'iteration1-reference')
    args=p.parse_args()
    {'prepare':prepare,'audit':audit}[args.action](args.reference_checkout.resolve())
