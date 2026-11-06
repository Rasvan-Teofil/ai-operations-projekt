"""Reproduce a frozen pilot into new local paths, preserving measured evidence."""
import argparse
import os
import re
import shutil
import subprocess
import sys

from src.config import ROOT, REPORTS, MODELS


def prepare(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('Use a unique short alphanumeric/hyphen/underscore name')
    source = REPORTS / 'common_data_v3'
    out = REPORTS / 'reproduction' / ('news-' + name)
    art = MODELS / 'reproduction' / ('news-' + name)
    if out.exists() or art.exists():
        raise RuntimeError('Reproduction name already exists; preserve it and choose a new name')
    out.mkdir(parents=True)
    for filename in ['protocol.json', 'source_manifest.json', 'split_manifest.json', 'annotations.csv', 'overlap_audit.json']:
        shutil.copyfile(source / filename, out / filename)
    environment = {**os.environ, 'SENTIMENT_NEWS_REPORTS': str(out),
        'SENTIMENT_NEWS_MODELS': str(art.relative_to(ROOT)),
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    print('Separate reproduction reports:', out, flush=True)
    print('Separate reproduction artifacts:', art, flush=True)
    return environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--run', action='store_true', help='Actually train and evaluate; stop the GPU API first')
    args = parser.parse_args(); environment = prepare(args.name)
    if args.run:
        for module_args in [['src.news_experiment', 'train'], ['src.news_experiment', 'evaluate'], ['src.news_report']]:
            subprocess.run([sys.executable, '-m', *module_args], cwd=ROOT, env=environment, check=True)
    else:
        print('Prepared only; no training or inference. Use --run with a fresh name to execute.', flush=True)
