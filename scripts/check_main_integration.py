"""Verify the complete main archive, frozen evidence and presentation entry links."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = (
    'README.md', 'docs/README.md', 'docs/project-overview.md',
    'docs/languages/japanese.md', 'docs/presentation-guide.md',
    'docs/reproduce-full-article.md', 'docs/repository-structure.md',
    'docs/public-hosting.md', 'reports/README.md',
    'docs/archive/main-before-full-article/README.md',
)


def checked_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Evidence path leaves repository: ' + name)
    return path


def verify_records(records, path_key):
    names = [r[path_key] for r in records]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate preserved path')
    for row in records:
        data = checked_path(row[path_key]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Preserved bytes changed: ' + row[path_key])


def main():
    archive = json.loads((ROOT / 'docs/archive/main-before-full-article/manifest.json').read_text())
    integration = json.loads((ROOT / 'reports/main-integration/integration-v1.json').read_text())
    verify_records(archive['files'], 'archived_path')
    verify_records(integration['protected_source_and_report_files'], 'path')
    if len(archive['files']) != integration['historical_main_files_preserved']:
        raise ValueError('Incomplete main archive')
    for name in ENTRY_DOCS:
        path = ROOT / name
        for target in re.findall(r'\]\(([^)\s]+)\)', path.read_text()):
            if re.match(r'[a-zA-Z]+://|mailto:|#', target):
                if '/blob/second-path/' in target or '/tree/second-path/' in target:
                    raise ValueError('Presentation entry leaves main: ' + name)
                continue
            relative = target.strip('<>').split('#')[0]
            if not (path.parent / relative).exists():
                raise ValueError(f'Broken entry link in {name}: {target}')
    readme = (ROOT / 'README.md').read_text()
    if 'Japanisch / Japanese / 日本語' not in readme:
        raise ValueError('Japanese missing from project entry')
    for name in ('app/main.py', 'app/schemas.py', 'src/next_stage_final.py',
                 'tests/test_api.py', 'reports/next_stage/comparison-v1.json'):
        if not (ROOT / name).is_file():
            raise ValueError('Current implementation missing on main: ' + name)
    print(f'Main integration verified: {len(archive["files"])} original files preserved, '
          f'{len(integration["protected_source_and_report_files"])} frozen source/report files unchanged; '
          'Japanese and presentation links available in this checkout.')


if __name__ == '__main__':
    main()
