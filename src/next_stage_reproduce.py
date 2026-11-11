"""Explicit isolated rerun of the frozen Train-only 24-condition experiment.

Original reports, models, historical runs and final Test are never overwritten.
The default validates inputs only; --execute starts a separate measured rerun.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import mlflow

from src.config import ROOT
from src import next_stage_experiment as experiment
from src.next_stage_prepare import frozen_write


def paths(name,root=ROOT):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,47}',name):
        raise ValueError('Use a short namespace starting with a letter; no paths or traversal')
    return root/'reports/next_stage/reproduction'/name, root/'models/next_stage/reproduction'/name


def reproduce(name,execute=False,resume=False):
    original=experiment.OUT;original_models=experiment.ART;experiment.verified();reports,models=paths(name)
    if not execute:
        print(json.dumps({'inputs_verified':True,'execute':False,'reports':str(reports),'models':str(models),
            'scope':'24 Train-only development evaluations in an isolated namespace; no final Test read or inference.'},indent=2));return
    if (reports.exists() or models.exists()) and not resume:
        raise RuntimeError('Reproduction namespace exists; choose another or explicitly --resume')
    reports.mkdir(parents=True,exist_ok=True)
    for name in ['development-protocol-v1.json','engineering-amendment-v1.json','experiment-membership-v1.json','capacity-v1.json']:
        frozen_write(reports/name,(original/name).read_text())
    original_start=mlflow.start_run
    driver_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def tagged_start(*args,**kwargs):
        tags=kwargs.pop('tags',{}) or {}
        return original_start(*args,tags={**tags,'reproduction':str(reports.relative_to(ROOT)),
            'measurement_origin':'explicit-isolated-reproduction','reproduction_driver_sha256':driver_sha},**kwargs)
    experiment.OUT=reports;experiment.ART=models;mlflow.start_run=tagged_start
    try:
        experiment.cv()
    finally:
        experiment.OUT=original;experiment.ART=original_models;mlflow.start_run=original_start
    print('Isolated reproduction complete; original results and final Test unchanged.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',required=True);parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args();reproduce(args.name,args.execute,args.resume)
