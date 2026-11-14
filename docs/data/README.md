# Daten und Originalquellen

Die folgenden Listen machen die aktuelle Vollartikelstudie direkt auf `main` nachvollziehbar.
Sie werden ausschließlich aus bereits veröffentlichten Metadaten erzeugt; keine Rohtexte,
neuen Labels, Stichproben oder Modellvorhersagen werden erzeugt.

**Projektseitig geprüft · Bestätigung vom 2026-11-14.**
[Prüfstatus und ursprüngliche Annotation](project-review.md).

| Sprache | Train | Abschlusstest |
|---|---:|---:|
| Deutsch | [78 Artikel](train-de.md) | [273 Artikel](test-de.md) |
| Englisch | [71 Artikel](train-en.md) | [118 Artikel](test-en.md) |
| Japanisch / 日本語 | [91 Artikel](train-ja.md) | [300 Artikel](test-ja.md) |

**Gesamt: 240 Train + 691 Test = 931 verwendete Artikel.**

Die Testquellenliste enthält ursprünglich 856 Artikel. Davon wurden 165 vor der
Modellbewertung ausgeschlossen; sie zählen nicht zum 691er Test.
[Alle ausgeschlossenen Quellen mit Gründen](excluded-test.md).

## Herkunft und Rekonstruktion

- [Train-Herkunft mit vollständigen Hashes](../../reports/next_stage/train-provenance-v1.jsonl)
- [Train-Annotationen und ursprüngliche Begründungen](../../reports/next_stage/train-annotations-v1.jsonl)
- [Alle 856 Testquellen](../../reports/next_stage/final-test-v1/provenance.jsonl)
- [Test-Annotationen und ursprüngliche Begründungen](../../reports/next_stage/final-test-v1/annotations.jsonl)
- [Testmanifest: Auswahl, Ausschlüsse, Gruppen und Hashbindungen](../../reports/next_stage/final-test-v1/test-manifest.json)
- [Modell-/Datennutzung und Lizenzen](../../licenses/README.md)
- [Exakte Extraktion und Wiederherstellung](../reproduce-full-article.md)

Der Quellenkatalog verlinkt Originale und feste Revisionen. Er verteilt keine
Volltexte neu. Eine spätere Änderung des Wikinews-Renderings kann die erneute
Extraktion beeinflussen; die ursprünglichen Eingabe-Hashes bleiben maßgeblich.

## Frühere Studien

| Studie | Lesbare Quellenlisten | Ursprüngliche Nachweise |
|---|---|---|
| Nachrichtenpilot v3 | [180 DE-Quellen](v3-de.md) · [180 EN-Quellen](v3-en.md) | [Quellenmetadaten](../../reports/common_data_v3/source_manifest.json) · [Annotationen](../../reports/common_data_v3/annotations.csv) |
| Erweiterung v4 | [360 DE-Quellen](v4-de.md) · [360 EN-Quellen](v4-en.md) | [Quellenmetadaten](../../reports/expanded_news_v4/source_manifest.json) · [Annotationen](../../reports/expanded_news_v4/annotations.csv) |
| LLM-Vergleich v5 | [120 DE-Quellen](v5-de.md) · [119 EN-Quellen](v5-en.md) | [Quellenmetadaten](../../reports/llm_adaptation_v5/source_manifest.json) · [Annotationen](../../reports/llm_adaptation_v5/annotations.csv) |
| Frühere Finanzdatensätze / Modelle | [Inventar mit Original-Datasetlinks](../../licenses/inventory.json) | [Historischer Hauptzweig](../archive/main-before-full-article/README.md) |

Erhobene Quellen enthalten auch Ausschlüsse und entsprechen nicht automatisch
den ausgewerteten Splitgrößen. Versionen können dieselben Seiten in anderen
Revisionen oder Extraktionsformen enthalten; ihre Zahlen nicht zusammenaddieren.

## Pflege

`python scripts/build_source_catalog.py` erzeugt diese Listen neu.
`python scripts/build_source_catalog.py --check` prüft in CI, dass sie weiterhin
exakt zu den unveränderten Herkunfts- und Annotationsdateien passen.
