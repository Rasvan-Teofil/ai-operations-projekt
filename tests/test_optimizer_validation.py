"""Paired recipes, group isolation and honest OOF aggregation without GPU work."""
import copy

import pytest

from src.studies.optimizer_validation.design import (
    RECIPES, REUSE, SEEDS, configurations, paired_analysis, validate_folds,
)


def fixture():
    rows=[];members=[];predictions={};results=[]
    for fold in range(3):
        for lang in ('de','en','ja'):
            for label in ('negative','neutral','positive'):
                rows.append(dict(id=f'{fold}-{lang}-{label}',group=f'g{fold}-{lang}-{label}',language=lang,label=label,fold=fold))
    for fold in range(3):
        members.append(dict(fold=fold,fit_ids=[r['id'] for r in rows if r['fold']!=fold],dev_ids=[r['id'] for r in rows if r['fold']==fold]))
    for c in configurations():
        records=[]
        for r in rows:
            if r['fold']!=c['fold']:continue
            pred=r['label']
            if c['recipe']!='candidate' and r['label']=='positive':pred='neutral'
            records.append(dict(id=r['id'],language=r['language'],reference=r['label'],prediction=pred,text_sha256='same-'+r['id']))
        predictions[c['id']]=records;results.append(dict(config=c))
    return rows,members,results,predictions


def test_three_arms_share_every_parameter_except_the_declared_contrast():
    assert len(configurations())==27
    assert len({c['id'] for c in configurations()})==27
    assert len(REUSE)==6
    assert RECIPES['baseline']['gradient_clip']==RECIPES['candidate']['gradient_clip']==1
    diff={k for k in RECIPES['candidate'] if RECIPES['candidate'][k]!=RECIPES['candidate_no_clip'][k]}
    assert diff=={'gradient_clip'}
    assert {c['seed'] for c in configurations()}==set(SEEDS)


def test_grouped_folds_partition_train_and_reject_cross_fold_group_leakage():
    rows,members,_,_=fixture();validate_folds(rows,members)
    rows[-1]['group']=rows[0]['group']
    with pytest.raises(ValueError,match='groups overlap'):validate_folds(rows,members)


def test_pooled_oof_analysis_uses_paired_metadata_and_does_not_claim_test_evidence():
    rows,_,results,predictions=fixture();ids=[r['id'] for r in rows];groups={r['id']:r['group'] for r in rows}
    analysis=paired_analysis(results,predictions,ids,groups,bootstrap_samples=100)
    assert analysis['status']=='complete'
    assert analysis['mean_contrasts']['clipping']['primary_score']>0
    assert analysis['conditional_group_bootstrap_95_intervals']['clipping']['primary_score'][0]>0
    assert analysis['clipping_supported_in_this_exploratory_cv']
    assert not analysis['independent_test_evidence']
    assert not analysis['serving_selection_changed']
    altered=copy.deepcopy(predictions)
    altered[next(iter(altered))][0]['reference']='different'
    with pytest.raises(ValueError,match='metadata differs'):paired_analysis(results,altered,ids,groups,bootstrap_samples=100)
    assert paired_analysis(results[:-1],predictions,ids,groups)['status']=='pending'


def test_frozen_protocol_supports_run_setup_before_loading_any_model(monkeypatch,tmp_path):
    """Exercise the real pre-fit logging path with no GPU or model load."""
    import hashlib
    import json
    from pathlib import Path
    import mlflow
    import pandas as pd
    import torch
    from src.studies.optimizer_validation import runner
    from src import next_stage_experiment as experiment
    repo=Path(__file__).resolve().parents[1]
    protocol=json.loads((repo/'reports/optimizer_validation/protocol-v1.json').read_text())
    config=next(c for c in protocol['runs'] if c['fold']==1)
    member=next(m for m in protocol['folds'] if m['fold']==1)
    path=tmp_path/'train.csv'
    pd.DataFrame([dict(id=i,group=i,language=('de','en','ja')[n%3],label=('negative','neutral','positive')[(n//3)%3],text='synthetic',text_sha256='synthetic') for n,i in enumerate(member['fit_ids']+member['dev_ids'])]).to_csv(path,index=False)
    protocol['files_sha256']['train.csv']=hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'OUT',tmp_path/'reports')
    monkeypatch.setattr(runner,'ART',tmp_path/'models');monkeypatch.setattr(runner,'PROTOCOL',repo/'reports/optimizer_validation/protocol-v1.json')
    monkeypatch.setattr(runner,'verify',lambda **kw:protocol)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *args,**kw:'synthetic-code-commit')
    monkeypatch.setattr(experiment,'TRAIN',path);monkeypatch.setattr(experiment,'gpu',lambda:'synthetic')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(16*1024**3,16*1024**3))
    # Real grouping is separately checked above; this regression targets protocol/schema setup.
    monkeypatch.setattr(runner,'validate_membership',lambda *args:None)
    for name in ('set_tracking_uri','set_experiment','log_params','log_artifact'):
        monkeypatch.setattr(mlflow,name,lambda *args,**kw:None)
    tags={}
    monkeypatch.setattr(mlflow,'set_tags',lambda value:tags.update(value))
    class Context:
        def __enter__(self):
            from types import SimpleNamespace
            return SimpleNamespace(info=SimpleNamespace(run_id='synthetic'))
        def __exit__(self,*args):return False
    monkeypatch.setattr(mlflow,'start_run',lambda **kw:Context())
    def no_model():raise RuntimeError('intentional stop before model load')
    monkeypatch.setattr(experiment,'lora',no_model)
    with pytest.raises(RuntimeError,match='intentional stop before model load'):runner.one(config['id'])
    assert tags['train_sha256']==protocol['files_sha256']['train.csv']
    assert tags['fold']=='1'
