"""Independently audit saved grouped-validation evidence; never infer on Test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(partial=False,bootstrap=False):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
    from mlflow import MlflowClient
    from src.studies.optimizer_validation.runner import verify
    from src.tracking import fingerprint
    protocol=verify(check_base=True);out=ROOT/'reports/optimizer_validation'
    from src.next_stage_experiment import TRAIN
    train=pd.read_csv(TRAIN).set_index('id')
    labels=['negative','neutral','positive'];results=[];predictions={}
    client=MlflowClient(tracking_uri=f'sqlite:///{ROOT/"mlflow.db"}')
    for config in protocol['runs']:
        path=out/'runs'/(config['id']+'.json')
        if not path.exists():
            if partial:continue
            raise AssertionError('Missing result '+config['id'])
        r=json.loads(path.read_text());assert r['config']==config and r['protocol_sha256']==digest(out/'protocol-v1.json')
        fold=next(m for m in protocol['folds'] if m['fold']==config['fold'])
        fit=set(fold['fit_ids']);dev=set(fold['dev_ids'])
        assert not fit&dev and fit|dev==set(train.index)
        assert not set(train.loc[list(fit),'group'])&set(train.loc[list(dev),'group'])
        paths={kind:out/kind/(config['id']+'.json') for kind in ('predictions','telemetry')}
        for kind,path in paths.items():assert digest(path)==r[kind+'_sha256']
        records=json.loads(paths['predictions'].read_text());events=json.loads(paths['telemetry'].read_text())
        assert len(records)==len(dev) and {x['id'] for x in records}==dev
        for x in records:
            t=train.loc[x['id']];assert (x['reference'],x['language'],x['text_sha256'])==(t.label,t.language,t.text_sha256)
            assert len(x['scores'])==3 and all(math.isfinite(s) and 0<=s<=1 for s in x['scores'])
            assert abs(sum(x['scores'])-1)<1e-5 and labels[int(np.argmax(x['scores']))]==x['prediction']
        y=[x['reference'] for x in records];pred=[x['prediction'] for x in records]
        values={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0))}
        assert confusion_matrix(y,pred,labels=labels).tolist()==r['metrics']['confusion_matrix']
        for lang in ('de','en','ja'):
            subset=[x for x in records if x['language']==lang]
            values['macro_f1_'+lang]=float(f1_score([x['reference'] for x in subset],[x['prediction'] for x in subset],labels=labels,average='macro',zero_division=0))
        values['primary_score']=(values['macro_f1_de']+values['macro_f1_en'])/2
        for key,value in values.items():assert abs(value-r['metrics'][key])<1e-12
        steps=3*math.ceil(len(fit)/16);warmup=int(config['warmup_fraction']*steps)
        assert len(events)==steps and [e['step'] for e in events]==list(range(1,steps+1))
        for e in events:
            assert all(math.isfinite(e[k]) for k in ('train_loss','gradient_norm','adapter_update_l2','learning_rate'))
            assert e['clipped']==int(config['gradient_clip'] is not None and e['gradient_norm']>config['gradient_clip'])
            s=e['step']-1;rate=config['learning_rate']*(s/max(1,warmup) if s<warmup else (steps-s)/max(1,steps-warmup))
            assert math.isclose(e['learning_rate'],rate,rel_tol=1e-10,abs_tol=1e-12)
        adapter=ROOT/'models'/('optimizer_validation/'+config['id'] if r['evidence_origin']=='new-fit' else 'optimizer_ablation/'+r['original_run_id'])
        assert fingerprint(adapter)==r['adapter_sha256']
        run=client.get_run(r['mlflow_run_id']);assert run.info.status=='FINISHED'
        history=client.get_metric_history(r['mlflow_run_id'],'gradient_norm')
        assert len(history)==steps
        assert {m.step:m.value for m in history}=={e['step']:e['gradient_norm'] for e in events}
        for key,value in r['metrics'].items():
            if isinstance(value,(int,float)):
                assert math.isclose(run.data.metrics[key],value,rel_tol=1e-10,abs_tol=1e-12)
        for key in ('learning_rate','weight_decay','warmup_fraction','seed'):
            assert float(run.data.params[key])==config[key]
        assert run.data.params['gradient_clip']==('disabled' if config['gradient_clip'] is None else str(config['gradient_clip']))
        results.append(r);predictions[config['id']]=records
    report=dict(status='partial' if len(results)!=27 else 'complete',checked_records=len(results),expected_records=27,
        distinct_mlflow_run_ids=len({r['mlflow_run_id'] for r in results}),new_fits=sum(r['evidence_origin']=='new-fit' for r in results),
        reused_fits=sum(r['evidence_origin']!='new-fit' for r in results),hashes_membership_predictions_metrics_telemetry_artifacts_and_mlflow_verified=True,
        final_test_inference_performed=False,source_protocol_sha256=digest(out/'protocol-v1.json'))
    if len(results)==27:
        summary=json.loads((out/'summary-v1.json').read_text());analysis=summary['analysis'];pooled={};scores={}
        for seed in protocol['seeds']:
            pooled[seed]={};scores[seed]={}
            for recipe in protocol['recipes']:
                rows=[x for fold in range(3) for x in predictions[f'f{fold}-s{seed}-{recipe}']]
                assert len(rows)==240 and {x['id'] for x in rows}==set(train.index)
                pooled[seed][recipe]=sorted(rows,key=lambda x:x['id']);scores[seed][recipe]={}
                for lang in ('de','en','ja'):
                    sub=[x for x in rows if x['language']==lang]
                    scores[seed][recipe]['macro_f1_'+lang]=float(f1_score([x['reference'] for x in sub],[x['prediction'] for x in sub],labels=labels,average='macro',zero_division=0))
                scores[seed][recipe]['primary_score']=(scores[seed][recipe]['macro_f1_de']+scores[seed][recipe]['macro_f1_en'])/2
                for key,value in scores[seed][recipe].items():assert abs(value-analysis['per_seed'][str(seed)]['oof'][recipe][key])<1e-12
        comparisons={'recipe':('candidate','baseline'),'clipping':('candidate','candidate_no_clip')}
        for name,(a,b) in comparisons.items():
            for key in scores[protocol['seeds'][0]][a]:
                delta=float(np.mean([scores[s][a][key]-scores[s][b][key] for s in protocol['seeds']]))
                assert abs(delta-analysis['mean_contrasts'][name][key])<1e-12
        report['pooled_oof_scores_and_paired_contrasts_independently_reproduced']=True
        if bootstrap:
            # Independent sklearn sample-weight implementation of the paired group draws.
            groups=sorted(train.group.unique());gi={g:i for i,g in enumerate(groups)}
            draws=np.random.default_rng(20261126).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
            values={name:{key:[] for key in analysis['mean_contrasts'][name]} for name in comparisons}
            prepared={}
            for seed in protocol['seeds']:
                for recipe,rows in pooled[seed].items():
                    for lang in ('de','en','ja'):
                        sub=[x for x in rows if x['language']==lang]
                        prepared[seed,recipe,lang]=([x['reference'] for x in sub],[x['prediction'] for x in sub],[gi[train.loc[x['id'],'group']] for x in sub])
            for draw in draws:
                sampled={}
                for recipe in protocol['recipes']:
                    sampled[recipe]={}
                    for lang in ('de','en','ja'):
                        parts=[]
                        for seed in protocol['seeds']:
                            y,p,g=prepared[seed,recipe,lang]
                            parts.append(f1_score(y,p,labels=labels,average='macro',zero_division=0,sample_weight=draw[g]))
                        sampled[recipe]['macro_f1_'+lang]=float(np.mean(parts))
                    sampled[recipe]['primary_score']=(sampled[recipe]['macro_f1_de']+sampled[recipe]['macro_f1_en'])/2
                for name,(a,b) in comparisons.items():
                    for key in values[name]:values[name][key].append(sampled[a][key]-sampled[b][key])
            for name,by_metric in values.items():
                for key,vals in by_metric.items():assert np.allclose(np.quantile(vals,[.025,.975]),analysis['conditional_group_bootstrap_95_intervals'][name][key],atol=1e-12)
            report['bootstrap_2000_paired_group_draws_independently_reproduced_with_sklearn_weights']=True
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--partial',action='store_true');parser.add_argument('--bootstrap',action='store_true')
    parser.add_argument('--write',action='store_true');args=parser.parse_args();report=audit(args.partial,args.bootstrap)
    if args.write:
        if report['status']!='complete':raise RuntimeError('Refusing final audit for incomplete study')
        path=ROOT/'reports/optimizer_validation/verification-v1.json'
        with path.open('x') as f:f.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
