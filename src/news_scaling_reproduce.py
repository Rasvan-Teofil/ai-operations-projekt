"""New reproduction runs in separate ignored paths; measured evidence is immutable."""
import argparse
import os
import re
import shutil
import subprocess
import sys

from src.config import ROOT


def prepare(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('Use a unique short alphanumeric/hyphen/underscore name')
    out = ROOT / 'reports/reproduction' / ('expanded-' + name)
    art = ROOT / 'models/reproduction' / ('expanded-' + name)
    if out.exists() or art.exists():
        raise RuntimeError('Reproduction name already exists; preserve it')
    out.mkdir(parents=True)
    for filename in ['protocol.json', 'source_manifest.json', 'split_manifest.json', 'annotations.csv', 'overlap_audit.json']:
        shutil.copyfile(ROOT / 'reports/expanded_news_v4' / filename, out / filename)
    env = {**os.environ, 'SENTIMENT_SCALING_REPORTS': str(out),
        'SENTIMENT_SCALING_MODELS': str(art.relative_to(ROOT)), 'SENTIMENT_SCALING_REPRODUCTION': name,
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    print('Separate reproduction reports:', out, flush=True)
    print('Separate reproduction artifacts:', art, flush=True)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True)
    parser.add_argument('--run', action='store_true', help='Actually fit CV/final and evaluate; stop GPU API first')
    args = parser.parse_args(); environment = prepare(args.name)
    if args.run:
        for argv in [['src.news_scaling', 'cv'], ['src.news_scaling', 'train'],
                     ['src.news_scaling', 'evaluate'], ['src.news_scaling_report']]:
            subprocess.run([sys.executable, '-m', *argv], cwd=ROOT, env=environment, check=True)
    else:
        print('Prepared only; --run with a fresh name executes new runs. Existing evidence stays unchanged.', flush=True)
