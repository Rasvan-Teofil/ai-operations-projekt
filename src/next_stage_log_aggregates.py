"""Log saved OOF summaries; no additional model call or scored evaluation."""
import hashlib
import json
from pathlib import Path

import mlflow
from huggingface_hub import snapshot_download

from src.config import ENCODER, ENCODER_REVISION
from src.next_stage import digest
from src.next_stage_collect import write_json
from src.next_stage_experiment import OUT, BASE, verified
from src.next_stage_protocol import RunIdentity
from src.tracking import fingerprint, tracking


def log():
    p=verified();tracking();source=json.loads((OUT/'development-v1.json').read_text());protocol_sha=hashlib.sha256((OUT/'development-protocol-v1.json').read_bytes()).hexdigest()
    path=OUT/'oof-aggregation-runs-v1.json';records=json.loads(path.read_text()) if path.exists() else []
    base_hash=fingerprint(BASE);encoder_hash=fingerprint(Path(snapshot_download(ENCODER,revision=ENCODER_REVISION,local_files_only=True)))
    for row in source['out_of_fold']:
        name=row['model']
        if any(r['model']==name for r in records):continue
        runs=[r for r in source['runs'] if r['model']==name]
        with mlflow.start_run(run_name='Full article · saved OOF aggregation · '+name) as run:
            identity=RunIdentity(research_question=row['research_question'],stage='development',data_version='article-train-240-v1',annotation_version='pilot-300-v2-input-v5',
                split_version='grouped-three-fold-v1',model_family='Qwen' if name.startswith('qwen') else 'MiniLM',adaptation=name,fold='out-of-fold',seed=20261110,precision='bf16',context_policy='full-article-v5',git_commit=p['git_commit'],
                protocol_sha256=protocol_sha,input_sha256=p['input_sha256'],annotation_sha256=p['annotation_sha256'],split_sha256=p['split_sha256'],
                artifact_sha256=digest('\n'.join(r['artifact_sha256'] for r in runs)),base_model_sha256=base_hash if name.startswith('qwen') else encoder_hash)
            mlflow.set_tags({**identity.tags(),'purpose':'saved-prediction-aggregation','metric_scope':'out-of-fold','scored_evaluation':'false','artifact_scope':'three-fold collection hash',
                'cost_size_scope':'One serving artifact, maximum across folds; not storage of all CV checkpoints'})
            mlflow.log_params({'source_development_sha256':digest((OUT/'development-v1.json').read_text()),'source_fold_runs':','.join(r['run_id'] for r in runs),'oof_rows':240,
                'method':'Concatenate exact saved OOF predictions, then calculate per-language macro-F1; no inference or new labels'})
            mlflow.log_metrics({k:v for k,v in row.items() if isinstance(v,(float,int))});mlflow.log_dict(row,'out_of_fold_metrics.json')
            records.append({'model':name,'run_id':run.info.run_id,'scored_evaluation':False});write_json(path,records)
    print('Logged',len(records),'OOF summaries from saved predictions; no new inference.',flush=True)


if __name__=='__main__':log()
