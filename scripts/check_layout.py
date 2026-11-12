"""Dependency-free layout and frozen-runtime check; reads Git-visible paths only."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {'README.md', 'CONTRIBUTING.md', 'requirements.txt', 'pytest.ini',
              '.gitignore', '.gitattributes', '.dockerignore', 'Dockerfile', 'docker-compose.yml'}
ROOT_DIRS = {'.github', 'app', 'data', 'docs', 'licenses', 'models', 'notebooks',
             'reports', 'requirements', 'scripts', 'src', 'tests'}


def violations(paths, baseline):
    errors=[]
    for name in paths:
        path=PurePosixPath(name);parts=path.parts
        if not parts:continue
        if len(parts)==1 and name not in ROOT_FILES:
            errors.append(f'Loose root file: {name}')
        elif len(parts)>1 and parts[0] not in ROOT_DIRS:
            errors.append(f'Unexpected root directory: {name}')
        if len(parts)==2 and parts[0]=='src' and parts[1] not in baseline['flat_src_files']:
            errors.append(f'New study code belongs in src/studies/<name>/: {name}')
        if len(parts)==2 and parts[0]=='reports' and parts[1] not in baseline['root_report_files']:
            errors.append(f'New measurements belong in reports/<study>/: {name}')
        if path.suffix in {'.safetensors','.joblib','.pkl','.pt','.pth','.db','.log'}:
            errors.append(f'Machine-local artifact/log must not be committed: {name}')
        if parts[0]=='models' and name!='models/README.md':
            errors.append(f'Local model path: {name}')
        if parts[0]=='data' and path.name!='README.md' and 'sample' not in parts:
            errors.append(f'Raw/processed data belongs outside Git: {name}')
    return errors


def frozen_runtime_errors(root=ROOT):
    folder=root/'reports/next_stage';declaration=json.loads((folder/'final-candidates-v1.json').read_text())
    files=('next_stage_experiment.py','article.py','inference.py','local_llm.py','serving.py',
           'tokenization.py','aggregation.py','config.py','next_stage_protocol.py','next_stage.py')
    value='\n'.join('src/'+name+':'+hashlib.sha256((root/'src'/name).read_bytes()).hexdigest() for name in sorted(files))
    expected=json.loads((folder/'engineering-amendment-v1.json').read_text())['corrected_code_sha256']
    errors=[]
    if hashlib.sha256(value.encode()).hexdigest()!=expected:errors.append('Frozen study runtime changed; preserve historical identity or create an explicitly versioned new study.')
    for name,key in [('next_stage_final.py','final_runner_sha256'),('next_stage_final_collect.py','source_collector_sha256')]:
        if hashlib.sha256((root/'src'/name).read_bytes()).hexdigest()!=declaration[key]:errors.append('Frozen candidate file changed: src/'+name)
    return errors


def main():
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    # git ls-files also shows staged removals until the move is staged.
    paths=sorted({p for p in paths if p and (ROOT/p).is_file()})
    baseline=json.loads((ROOT/'docs/layout-baseline.json').read_text())
    errors=violations(paths,baseline)+frozen_runtime_errors()
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Layout and frozen runtime verified: {len(paths)} Git-visible files; no raw data, weights or loose new study modules.')


if __name__=='__main__':main()
