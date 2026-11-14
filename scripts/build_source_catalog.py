"""Build readable source links from public frozen metadata, never raw articles."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {'de': 'Deutsch', 'en': 'Englisch', 'ja': 'Japanisch / 日本語'}
INPUTS = (
    'reports/next_stage/train-provenance-v1.jsonl',
    'reports/next_stage/train-annotations-v1.jsonl',
    'reports/next_stage/final-test-v1/provenance.jsonl',
    'reports/next_stage/final-test-v1/annotations.jsonl',
    'reports/next_stage/final-test-v1/test-manifest.json',
    'reports/data-review/project-review-v1.json',
)
LEGACY = {
    'v3': ('Nachrichtenpilot v3', 'reports/common_data_v3'),
    'v4': ('Erweiterung v4', 'reports/expanded_news_v4'),
    'v5': ('LLM-Vergleich v5', 'reports/llm_adaptation_v5'),
}


def records(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source/annotation ID: ' + str(path))
    return rows


def source_url(value):
    parsed = urlsplit(value)
    allowed = {lang + '.wikinews.org' for lang in LANGUAGES} | {'creativecommons.org'}
    if (parsed.scheme != 'https' or parsed.hostname not in allowed or
            parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Unexpected public source URL: ' + value)
    return quote(value, safe=':/?=&%#@+;,_~.-')


def cell(value):
    value = html.escape(str(value), quote=False).replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'([\\\[\]`*_|])', r'\\\1', value)


def catalog_rows(root=ROOT):
    train = records(root / INPUTS[0]); train_notes = records(root / INPUTS[1])
    sources = records(root / INPUTS[2]); notes = records(root / INPUTS[3])
    manifest = json.loads((root / INPUTS[4]).read_text())
    annotation_by_id = {r['id']: r for r in train_notes + notes}
    if len(annotation_by_id) != len(train_notes) + len(notes):
        raise ValueError('Train and Test source IDs overlap')
    excluded = {r['id']: r['codes'] if 'codes' in r else [r['code']]
                for r in manifest['exclusions']}
    if len(excluded) != len(manifest['exclusions']) or not set(excluded) <= {r['id'] for r in sources}:
        raise ValueError('Invalid exclusion membership')
    groups = {'train': [], 'test': [], 'excluded': []}
    for split, rows in [('train', train), ('test', sources)]:
        for row in rows:
            note = annotation_by_id[row['id']]
            if row['text_sha256'] != note['text_sha256']:
                raise ValueError('Annotation input hash mismatch: ' + row['id'])
            label = note['second_pass']['label']
            if split == 'train' and row['label'] != label:
                raise ValueError('Train reference differs from frozen provenance')
            target = 'excluded' if split == 'test' and row['id'] in excluded else split
            if target != 'excluded' and label not in ('negative', 'neutral', 'positive'):
                raise ValueError('Unresolved reference entered selected catalog')
            for key in ('source_url', 'revision_url', 'history_url', 'source_license_url'):
                source_url(row[key])
            groups[target].append({**row, 'label': label, 'exclusion_codes': excluded.get(row['id'], [])})
    if len(groups['train']) != 240 or len(groups['test']) != manifest['article_rows']:
        raise ValueError('Selected catalog sizes differ from frozen study')
    counts = Counter(r['language'] + ':' + r['label'] for r in groups['test'])
    if dict(counts) != manifest['language_label_counts']:
        raise ValueError('Test references differ from frozen class counts')
    return groups


def table(rows, excluded=False):
    header = '| ID | Originalartikel | Label | Datum | Verwendete Revision | Autoren / Lizenz | Text-SHA-256 (Anfang) |'
    if excluded:
        header += ' Ausschlussgründe |'
    lines = [header, '|' + '---|' * (8 if excluded else 7)]
    for row in rows:
        parts = [f'`{row["id"]}`', f'[{cell(row["title"])}]({source_url(row["source_url"])})',
                 row['label'], cell(row['published_at']),
                 f'[{cell(row["revision"])}]({source_url(row["revision_url"])})',
                 f'[Historie]({source_url(row["history_url"])}) · [Lizenz]({source_url(row["source_license_url"])})',
                 f'`{row["text_sha256"][:16]}`']
        if excluded:
            parts.append(', '.join(cell(v) for v in row['exclusion_codes']))
        lines.append('| ' + ' | '.join(parts) + ' |')
    return '\n'.join(lines) + '\n'


def outputs(root=ROOT):
    groups = catalog_rows(root)
    review = json.loads((root / INPUTS[-1]).read_text())
    for artifact in review['annotation_artifacts']:
        if hashlib.sha256((root / artifact['path']).read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError('Project review refers to changed annotation artifact: ' + artifact['path'])
    result = {}
    legacy_sources = []
    legacy_index = []
    for version, (title, folder) in LEGACY.items():
        provenance = folder + '/source_manifest.json'; annotation = folder + '/annotations.csv'
        legacy_sources.extend((provenance, annotation))
        sources = json.loads((root / provenance).read_text())['articles']
        with (root / annotation).open(newline='') as stream:
            original_annotations = list(csv.DictReader(stream))
        notes = {r['id']: r for r in original_annotations}
        if len(notes) != len(original_annotations) or set(notes) != {r['id'] for r in sources}:
            raise ValueError('Historical annotation membership differs: ' + version)
        counts = {}
        for language in ('de', 'en'):
            rows = []
            for source in sources:
                if source['language'] != language:
                    continue
                note = notes[source['id']]
                if source['text_sha256'] != note['text_sha256']:
                    raise ValueError('Historical annotation hash differs: ' + source['id'])
                row = {**source, 'label': note['label'],
                       'title': unquote(urlsplit(source['source_url']).path.removeprefix('/wiki/')).replace('_', ' '),
                       'published_at': source['created_at'][:10], 'source_license_url': source['license_url']}
                for key in ('source_url', 'revision_url', 'history_url', 'source_license_url'):
                    source_url(row[key])
                rows.append(row)
            counts[language] = len(rows)
            result[f'docs/data/{version}-{language}.md'] = (
                f'# {title} · {LANGUAGES[language]}\n\n[Zur Datenübersicht](README.md).\n\n'
                f'**{len(rows)} erhobene Quellen**, einschließlich späterer Ausschlüsse. Dies ist keine\n'
                'neue Train-/Test-Zuordnung. `exclude` bleibt ein Ausschluss, kein Sentimentlabel.\n'
                'Titel werden aus der Original-URL lesbar dargestellt; Datum aus dem gespeicherten\n'
                'Seitenerstellungsdatum. Labels und Hashes stammen unverändert aus den ursprünglichen\n'
                f'[Annotationen](../../{annotation}) und [Quellenmetadaten](../../{provenance}).\n'
                'Die ausgewerteten Splitgrößen stehen im jeweiligen historischen Ergebnisbericht.\n'
                '[Spätere projektseitige Prüfbestätigung](project-review.md).\n\n' + table(rows))
        legacy_index.append(f'| {title} | [{counts["de"]} DE-Quellen]({version}-de.md) · [{counts["en"]} EN-Quellen]({version}-en.md) | [Quellenmetadaten](../../{provenance}) · [Annotationen](../../{annotation}) |')
    index = ['# Daten und Originalquellen', '',
        'Die folgenden Listen machen die aktuelle Vollartikelstudie direkt auf `main` nachvollziehbar.',
        'Sie werden ausschließlich aus bereits veröffentlichten Metadaten erzeugt; keine Rohtexte,',
        'neuen Labels, Stichproben oder Modellvorhersagen werden erzeugt.', '',
        f'**{review["display_label_de"]} · Bestätigung vom {review["reviewed_on"]}.**',
        '[Prüfstatus und ursprüngliche Annotation](project-review.md).', '',
        '| Sprache | Train | Abschlusstest |', '|---|---:|---:|']
    for language, title in LANGUAGES.items():
        counts = {}
        for split in ('train', 'test'):
            rows = [r for r in groups[split] if r['language'] == language]
            counts[split] = len(rows)
            name = f'{split}-{language}.md'
            origin = INPUTS[0] if split == 'train' else INPUTS[2]
            text = (f'# {"Train" if split == "train" else "Abschlusstest"} · {title}\n\n'
                    f'**{len(rows)} ausgewählte native Vollartikel.** [Zur Datenübersicht](README.md).\n\n'
                    'Die Originalartikel können sich ändern. „Verwendete Revision“ öffnet den beim\n'
                    'Datensatz vermerkten Seitenstand. Historie und Lizenz verlinken die Herkunftsnachweise.\n'
                    f'Der vollständige Text-Hash, Extraktionsversion und Rechteprüfung stehen im\n'
                    f'[Metadatensatz](../../{origin}). Der Hash identifiziert den extrahierten Eingabetext,\n'
                    'nicht die HTML-Seite. [Projektseitige Prüfbestätigung](project-review.md).\n\n')
            result['docs/data/' + name] = text + table(rows)
        index.append(f'| {title} | [{counts["train"]} Artikel](train-{language}.md) | [{counts["test"]} Artikel](test-{language}.md) |')
    index += ['', '**Gesamt: 240 Train + 691 Test = 931 verwendete Artikel.**', '',
        'Die Testquellenliste enthält ursprünglich 856 Artikel. Davon wurden 165 vor der',
        'Modellbewertung ausgeschlossen; sie zählen nicht zum 691er Test.',
        '[Alle ausgeschlossenen Quellen mit Gründen](excluded-test.md).', '',
        '## Herkunft und Rekonstruktion', '',
        '- [Train-Herkunft mit vollständigen Hashes](../../' + INPUTS[0] + ')',
        '- [Train-Annotationen und ursprüngliche Begründungen](../../' + INPUTS[1] + ')',
        '- [Alle 856 Testquellen](../../' + INPUTS[2] + ')',
        '- [Test-Annotationen und ursprüngliche Begründungen](../../' + INPUTS[3] + ')',
        '- [Testmanifest: Auswahl, Ausschlüsse, Gruppen und Hashbindungen](../../' + INPUTS[4] + ')',
        '- [Modell-/Datennutzung und Lizenzen](../../licenses/README.md)',
        '- [Exakte Extraktion und Wiederherstellung](../reproduce-full-article.md)', '',
        'Der Quellenkatalog verlinkt Originale und feste Revisionen. Er verteilt keine',
        'Volltexte neu. Eine spätere Änderung des Wikinews-Renderings kann die erneute',
        'Extraktion beeinflussen; die ursprünglichen Eingabe-Hashes bleiben maßgeblich.', '',
        '## Frühere Studien', '',
        '| Studie | Lesbare Quellenlisten | Ursprüngliche Nachweise |', '|---|---|---|',
        *legacy_index,
        '| Frühere Finanzdatensätze / Modelle | [Inventar mit Original-Datasetlinks](../../licenses/inventory.json) | [Historischer Hauptzweig](../archive/main-before-full-article/README.md) |', '',
        'Erhobene Quellen enthalten auch Ausschlüsse und entsprechen nicht automatisch',
        'den ausgewerteten Splitgrößen. Versionen können dieselben Seiten in anderen',
        'Revisionen oder Extraktionsformen enthalten; ihre Zahlen nicht zusammenaddieren.', '',
        '## Pflege', '', '`python scripts/build_source_catalog.py` erzeugt diese Listen neu.',
        '`python scripts/build_source_catalog.py --check` prüft in CI, dass sie weiterhin',
        'exakt zu den unveränderten Herkunfts- und Annotationsdateien passen.', '']
    result['docs/data/README.md'] = '\n'.join(index)
    result['docs/data/excluded-test.md'] = (
        '# Ausgeschlossene Testquellen\n\n[Zur Datenübersicht](README.md).\n\n'
        '**165 Quellen, nicht Bestandteil des 691er Abschlusstests.** Die Ausschlussgründe\n'
        'stammen unverändert aus dem eingefrorenen Testmanifest. Eine nachträgliche\n'
        'Projektprüfung hebt keine Rechte-, Gruppen- oder ursprünglichen Auswahlentscheidungen auf.\n'
        'Alle Rohfälle bleiben Test-reserviert. `unresolved` wird nicht als neutral umgedeutet.\n\n'
        'Vollständige Definition und Gruppen: [Testmanifest](../../' + INPUTS[4] + ').\n\n'
        + table(groups['excluded'], excluded=True))
    manifest = {
        'version': 'readable-source-catalog-v1', 'contains_raw_text': False,
        'selected_rows': {k: len(v) for k, v in groups.items()},
        'source_files': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (*INPUTS, *legacy_sources)},
        'generated_files': {name: hashlib.sha256(value.encode()).hexdigest() for name, value in result.items()},
    }
    result['docs/data/catalog-manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    generated = outputs()
    for name, value in generated.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text() != value:
                raise SystemExit('Source catalog is stale or missing: ' + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(value)
    print(f'Source catalog {"verified" if args.check else "built"}: 931 selected articles, '
          f'165 exclusions, {len(generated)} metadata-only files.')


if __name__ == '__main__':
    main()
