"""Repeat the frozen matched-task protocol in new ignored paths, retaining history."""
import argparse
import os
import re
import shutil
import subprocess
import sys

from src.config import ROOT


def prepare(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name): raise ValueError('Use a unique simple reproduction name')
    out = ROOT / 'reports/reproduction' / ('llm-adaptation-' + name)
    art = ROOT / 'models/reproduction' / ('llm-adaptation-' + name)
    if out.exists() or art.exists(): raise RuntimeError('Reproduction name already exists; preserve evidence')
    out.mkdir(parents=True)
    for filename in ['protocol.json', 'source_manifest.json', 'split_manifest.json', 'annotations.csv', 'overlap_audit.json']:
        shutil.copyfile(ROOT / 'reports/llm_adaptation_v5' / filename, out / filename)
    return {**os.environ, 'SENTIMENT_LLM_ADAPT_REPORTS': str(out),
        'SENTIMENT_LLM_ADAPT_MODELS': str(art.relative_to(ROOT)), 'SENTIMENT_LLM_ADAPT_REPRODUCTION': name,
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--name', required=True)
    parser.add_argument('--run', action='store_true', help='Actually probe, fit CV/full Train and audit; stop GPU API first')
    args = parser.parse_args(); environment = prepare(args.name)
    print('New reproduction reports:', environment['SENTIMENT_LLM_ADAPT_REPORTS'], flush=True)
    print('New reproduction artifacts:', environment['SENTIMENT_LLM_ADAPT_MODELS'], flush=True)
    if args.run:
        for argv in [['src.news_llm_experiment', 'probe'], ['src.news_llm_experiment', 'cv'],
                     ['src.news_llm_experiment', 'train'], ['src.news_llm_experiment', 'evaluate'], ['src.news_llm_report']]:
            subprocess.run([sys.executable, '-m', *argv], cwd=ROOT, env=environment, check=True)
    else: print('Prepared only. Use --run with a new name for real new runs; historical evidence is not overwritten.', flush=True)
