"""Execute an unchanged reference checkout in its own Python environment.

This adapter imports the first iteration's implementation (commit a840bb7); it contains no copied model code.
Never call the reference src.train: that would evaluate its old test and log to its experiment.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','predict'])
    parser.add_argument('--checkout',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path)
    args=parser.parse_args()
    checkout=args.checkout.resolve()
    sys.path.insert(0,str(checkout))
    import pandas as pd
    import sklearn
    import numpy as np
    from src.sklearn_models import new_linearsvc,load_sklearn_model
    if args.action=='prepare':
        from src.prepare_data import prepare
        from src.config import PREPARED_TRAIN_PATH,TEXT_COLUMN,LABEL_COLUMN
        from unittest.mock import patch
        from huggingface_hub import hf_hub_download
        revisions={'takala/financial_phrasebank': '8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964', 'Kenpache/multilingual-financial-sentiment': '9d43950d151602f623f28bba62f30554b5d40dd2'}
        def pinned_download(repo_id,filename,**kwargs):
            return hf_hub_download(repo_id,filename,revision=revisions[repo_id],**kwargs)
        with patch('huggingface_hub.hf_hub_download',side_effect=pinned_download):
            info=prepare()  # original filters/split, no evaluation or MLflow calls
        train=pd.read_csv(PREPARED_TRAIN_PATH)
        model=new_linearsvc().fit(train[TEXT_COLUMN].tolist(),train[LABEL_COLUMN].tolist())
        artifact=model.save()
        source_files={str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted((checkout/'src').glob('*.py'))}
        result={'reference_commit':'a840bb7bc8f1656f1700ccabaa89837ce347ca2b',
                'method':'Unmodified src.sklearn_models.new_linearsvc().fit on unmodified prepare() training split',
                'original_weights_available':False,'published_0_720_not_reproduced_or_used':True,
                'training_rows':len(train),'train_sha256':hashlib.sha256(PREPARED_TRAIN_PATH.read_bytes()).hexdigest(),
                'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
                'model_size_bytes':artifact.stat().st_size,'tracking_params':model.tracking_params(),
                'pandas':pd.__version__,'sklearn':sklearn.__version__,'python':sys.version.split()[0],
                'source_files':source_files,'data_metadata':info}
    else:
        frame=pd.read_csv(args.benchmark)
        model=load_sklearn_model('tfidf_linearsvc')
        for _ in range(20): model.predict_proba(['The firm reports quarterly results.'])
        probabilities=[]
        durations=[]
        for text in frame.text:
            begin=time.perf_counter()
            row=model.predict_proba([text])[0]
            durations.append(1000*(time.perf_counter()-begin))
            probabilities.append([row[label] for label in ['negative','neutral','positive']])
        np.savez(args.report.with_suffix('.npz'),probs=np.asarray(probabilities),durations=np.asarray(durations))
        result={'rows':len(frame),'inference_ms_per_text':float(np.mean(durations)),
                'latency_p50_ms':float(np.median(durations)),'latency_p95_ms':float(np.percentile(durations,95)),
                'model_version':model.model_version,'model_size_bytes':model.artifact_path.stat().st_size,
                'backend':'CPU (native reference implementation)'}
    args.report.write_text(json.dumps(result,indent=2))
    print(args.action,'complete',flush=True)

if __name__=='__main__': main()
