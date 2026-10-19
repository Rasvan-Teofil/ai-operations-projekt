"""Download one pinned dataset; split unique normalized sentences, never raw row order."""
import hashlib
import json
import unicodedata
import re
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from second_path.config import DATA, DATASET, DATA_REVISION, REPORTS, SEED, LABELS


def normalized(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()


def main():
    if (DATA / 'manifest.json').exists():
        raise SystemExit('Splits already exist. Reuse them; do not overwrite an evaluated split.')
    path = hf_hub_download(DATASET, 'all_languages_clean.csv', repo_type='dataset', revision=DATA_REVISION)
    raw = pd.read_csv(path)
    df = raw.loc[raw.language.isin(['de', 'en']) & raw.label.isin(LABELS)].copy()
    before = len(df)
    df = df.dropna(subset=['sentence'])
    df['text'] = df.sentence.astype(str).str.strip()
    df = df[df.text.str.len().between(20, 20000)].copy()
    df['normalized'] = df.text.map(normalized)
    conflicts = df.groupby('normalized').label.nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    conflict_rows = int(df.normalized.isin(conflicts).sum())
    df = df[~df.normalized.isin(conflicts)].copy()
    pre_dedup = len(df)
    df = df.sort_values(['normalized', 'language', 'source']).drop_duplicates('normalized')
    df['id'] = df.normalized.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    df = df.sort_values('id').reset_index(drop=True)
    strata = df.language + ':' + df.label
    trainval, test = train_test_split(df, test_size=.15, stratify=strata, random_state=SEED)
    train, val = train_test_split(trainval, test_size=.15/.85,
                                stratify=trainval.language + ':' + trainval.label, random_state=SEED)
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': DATASET, 'revision': DATA_REVISION, 'seed': SEED,
                'source_sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
                'raw_rows': len(raw), 'de_en_before_filter': before,
                'conflicting_rows_removed': conflict_rows, 'duplicates_removed': pre_dedup-len(df),
                'filtered_rows': len(df), 'splits': {}}
    for name, split in [('train', train), ('val', val), ('test', test)]:
        dest = DATA / f'{name}.csv'
        split[['id', 'text', 'label', 'language', 'source']].sort_values('id').to_csv(dest, index=False)
        manifest['splits'][name] = {
            'rows': len(split), 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'counts': split.groupby(['language', 'label']).size().to_dict()}
        manifest['splits'][name]['counts'] = {':'.join(k): int(v) for k,v in manifest['splits'][name]['counts'].items()}
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
