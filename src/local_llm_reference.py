"""Run the pinned colleague recipe in an isolated checkout, without its test evaluator."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkout, output = args.checkout.resolve(), args.output.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
    if commit != '8668255ed7b94c24e16d52bec1d992f9ad507e87':
        raise RuntimeError('Reference checkout is not the frozen v2 commit')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=checkout):
        raise RuntimeError('Reference source has local modifications')
    sys.path.insert(0, str(checkout))
    import pandas as pd
    import sklearn
    from unittest.mock import patch
    from huggingface_hub import hf_hub_download
    from src.prepare_data import prepare
    from src.data import load_work_splits
    from src.sklearn_models import new_linearsvc
    revisions = {'takala/financial_phrasebank': '8d3fe0c36d5feec6b3cc5e455b0fcb4820fb9964',
                 'Kenpache/multilingual-financial-sentiment': '9d43950d151602f623f28bba62f30554b5d40dd2'}

    def pinned_download(repo_id, filename, **kwargs):
        return hf_hub_download(repo_id, filename, revision=revisions[repo_id], **kwargs)

    with patch('huggingface_hub.hf_hub_download', side_effect=pinned_download):
        data_info = prepare()
    train, val, test = load_work_splits()
    start = time.perf_counter()
    model = new_linearsvc().fit(train.text.tolist(), train.label.tolist())
    output.mkdir(parents=True, exist_ok=True)
    artifact = model.save(output)
    info = {'reference_commit': commit, 'original_weights_available': False,
            'method': 'Unmodified pinned prepare(), load_work_splits(), new_linearsvc().fit(train); no reference test evaluation',
            'train_rows_including_supplement': len(train), 'val_rows_not_evaluated': len(val),
            'test_rows_not_evaluated': len(test), 'training_seconds': time.perf_counter() - start,
            'model_version': model.model_version, 'tracking_params': model.tracking_params(),
            'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'python': sys.version.split()[0],
            'dataset_revisions': revisions, 'data_metadata': data_info,
            'supplement_sha256': hashlib.sha256((checkout / 'data/sample/opinion_supplement.csv').read_bytes()).hexdigest(),
            'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'source_files': {str(p.relative_to(checkout)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((checkout / 'src').glob('*.py'))}}
    (output / 'artifact.json').write_text(json.dumps({'kind': 'reference_sklearn',
        'labels': ['negative', 'neutral', 'positive'], 'model_version': model.model_version,
        'sklearn': sklearn.__version__, 'improved_article': False}, indent=2))
    (output / 'provenance.json').write_text(json.dumps(info, indent=2))
    print('Reconstructed reference v2:', len(train), 'training rows; original val/test not scored', flush=True)


if __name__ == '__main__':
    main()
