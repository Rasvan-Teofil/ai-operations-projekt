# Daten, Quellen und Referenzlabels

**[Alle Daten und Originalquellen nachvollziehen](../docs/data/README.md)**

| Sprache | Train: Artikel, Revisionen und Labels | Test: Artikel, Revisionen und Labels |
|---|---:|---:|
| Deutsch | [78 Artikel](../docs/data/train-de.md) | [273 Artikel](../docs/data/test-de.md) |
| Englisch | [71 Artikel](../docs/data/train-en.md) | [118 Artikel](../docs/data/test-en.md) |
| Japanisch / 日本語 | [91 Artikel](../docs/data/train-ja.md) | [300 Artikel](../docs/data/test-ja.md) |

Die Listen verlinken für jeden verwendeten Artikel das Original, die festgehaltene
Revision, Autorenhistorie und Lizenz. Referenzlabel und Hash-Anfang sind direkt
lesbar; die vollständigen Hashes und Begründungen stehen in den verlinkten Metadaten.

**Projektseitig geprüft:** Der Projektverantwortliche hat am 14. November 2026
bestätigt, dass er die erstellten beziehungsweise klassifizierten Daten geprüft
hat und sie passen. [Prüfstatus und ursprüngliche Erstellung](../docs/data/project-review.md).

## Umfang und historische Daten

240 Train- und 691 Testartikel bilden den aktuellen verwendeten Bestand.
[165 ausgeschlossene Testquellen mit Gründen](../docs/data/excluded-test.md)
bleiben nachvollziehbar, zählen aber nicht zum ausgewerteten Test.
[Frühere Studien und Originaldatensätze](../docs/data/README.md#frühere-studien)
sind separat verlinkt.

## Lokale Dateien

Rohtexte, API-Cache und Train-/Test-Splits liegen in ignorierten Unterordnern.
Der Katalog enthält Quellenlinks und Metadaten, keine neu veröffentlichten
Volltexte. Alle 856 erhobenen Test-Rohfälle bleiben dauerhaft Test-reserviert;
keine ausgeschlossenen Testfälle nachträglich in Train übernehmen.

[Extraktion und Wiederherstellung](../docs/reproduce-full-article.md) ·
[Quellen und Lizenzen](../licenses/README.md) ·
[Japanisch im Projekt](../docs/languages/japanese.md)
