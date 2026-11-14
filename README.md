# Article Sentiment

**AI Operations · Hochschule Heilbronn · M.Sc. Wirtschaftsinformatik · Wintersemester 2026/27**

Medienbeobachtung für native deutsche, englische und japanische Nachrichten: vollständigen Text oder genau eine HTTP(S)-URL als `negative`, `neutral` oder `positive` klassifizieren. **Der aktuelle Projektstand liegt vollständig auf `main`.** Frühere Studien bleiben im Archiv erhalten.

## Stand · 14. November 2026

Die Vollartikelstudie ist abgeschlossen: 27 Train-interne Entwicklungsauswertungen, vier Fits auf denselben 240 Train-Artikeln und ein gemeinsamer frischer Test mit **691 Artikeln: DE273, EN118, JA300**. Qwen LoRA wurde vor Erhebung des Tests gewählt und ist jetzt der lokale API-Standard.

**Projektseitig geprüfte Referenzen**: Der Projektverantwortliche hat die erstellten beziehungsweise klassifizierten Daten geprüft und am 14. November 2026 bestätigt. Die ursprüngliche KI-unterstützte Annotation und der spätere Review sind im [Prüfnachweis](docs/data/project-review.md) getrennt dokumentiert. Der Hauptwert gewichtet DE- und EN-macro-F1 je zur Hälfte; JA bleibt separat. Keine künstliche Auffüllung des englischen Tests.

| Methode | DE/EN-macro-F1 | DE | EN | JA | warme ms/Artikel | vollständige MiB |
|---|---:|---:|---:|---:|---:|---:|
| frozen128 | 0.5150 | 0.5281 | 0.5019 | 0.6257 | 3.72 | 240.73 |
| minilm128 | 0.5928 | 0.4734 | 0.7122 | 0.5655 | 3.66 | 240.73 |
| qwen-zero | 0.7796 | 0.7508 | 0.8085 | 0.8143 | 172.33 | 7687.49 |
| qwen-lora | 0.8218 | 0.7612 | 0.8825 | 0.8132 | 184.26 | 7698.77 |

- **[Daten und Originalquellen öffnen](docs/data/README.md)** · [Prüfstatus](docs/data/project-review.md)
- [Ergebnisse mit Unsicherheit, Kosten und Laufkennungen](reports/next_stage/comparison-v1.md) · [vollständiges JSON](reports/next_stage/comparison-v1.json)
- [Modellkarten und Grenzen](reports/next_stage/model-cards-v1.md) · [unveränderte Referenzen und Fehlerfälle](reports/next_stage/error-analysis-v1.json)
- [Reproduzieren](docs/reproduce-full-article.md) · [Projektstruktur und Dateiplatzierung](docs/repository-structure.md)
- [Quellen und Lizenzen](licenses/README.md) · [Inventar](licenses/inventory.json)
- [Historische Studien und Befehle](docs/archive/project-history-through-v5.md) · [Phasenprotokoll](docs/next-stage.md)

## Sprachen: Deutsch, Englisch und Japanisch

| Sprache | Native Train-Artikel | Native Testartikel | Qwen-LoRA-Macro-F1 |
|---|---:|---:|---:|
| Deutsch / German | 78 | 273 | 0.7612 |
| Englisch / English | 71 | 118 | 0.8825 |
| **Japanisch / Japanese / 日本語** | **91** | **300** | **0.8132** |

Japanisch ist implementiert und wurde separat ausgewertet. Es handelt sich um
native japanische Vollartikel, keine übersetzten DE/EN-Testtexte.
[Japanisch: Daten, Ergebnisse, Bedienung und Grenzen](docs/languages/japanese.md).
DE/EN bleiben die vorab festgelegte Hauptkennzahl; JA wird nicht nachträglich
zur Hauptkennzahl hinzugenommen.

## Projekt direkt von main erklären

1. [Problem, Zielgruppe und Aufbau](docs/project-overview.md).
2. [Aktuelle Ergebnisse einschließlich Japanisch](reports/next_stage/comparison-v1.md).
3. [Projektstruktur nach Vorlesungsvorlage](docs/repository-structure.md).
4. [Vorführungsleitfaden](docs/presentation-guide.md) und [Reproduktion](docs/reproduce-full-article.md).
5. [Quellen und Lizenzen](licenses/README.md), [Prüfnachweise](reports/next_stage/verification-final-v1.json)
   und [ursprünglicher main-Stand](docs/archive/main-before-full-article/README.md).

Alle verlinkten Dateien, die aktuelle Anwendung und ihre Tests liegen auf diesem
Hauptzweig. `second-path` bleibt nur als historische Entwicklungskennung in
Messprotokollen und MLflow erhalten. Für die Präsentation ist kein Branchwechsel nötig.

**Öffentliche Demo:** noch nicht bereitgestellt. Die laufende Anwendung und MLflow
sind bisher lokal. Kostenloses Hosting ohne Tower ist eine separate, noch offene
Bereitstellungsaufgabe; eine GitHub-Seite allein führt das Modell nicht aus.
[Hostingstand und kostenfreie Optionen](docs/public-hosting.md).

## Was daraus folgt

Qwen LoRA erzielt die höchste Übereinstimmung im gleichgewichteten DE/EN-Hauptwert. Die gepaarte 95%-Bootstrapspanne der Differenz zu Qwen Zero liegt bei [0,0070; 0,0826], bedingt auf diesen Quellenbestand und die festen Referenzlabels. Das ist kein Nachweis von 95% Genauigkeit. Auf Japanisch liegt LoRA geringfügig unter Zero-Shot; MiniLMs Verbesserung gilt ebenfalls nicht für jede Sprache.

Das angepasste MiniLM benötigt hier etwa 3,66 statt 184,26 warme ms/Artikel und rund 241 statt 7699 MiB. Diese Größen enthalten eigenständig benötigte Basisgewichte und Adapter. Zeiten enthalten Tokenisierung, Transfer, gesamten Kontext und Aggregation; Modellladen und URL-Abruf sind ausgenommen. GPU: RTX 5060 Ti, Batch 1.

Die Trainingsmenge 240 ist kein belegtes Optimum: die kleine Mengenkurve war nicht gesättigt. 3000 war eine Obergrenze, 6000 bleibt spätere Forschung. Ein einzelnes Archiv, begrenzte direkte Zitate, maximal 20000 Zeichen/4096 Qwen-Artikeltokens, unvollständige Erkennung semantischer Verbindungen und mögliche Vortrainingsüberschneidungen begrenzen die Übertragbarkeit. Die projektseitige Prüfung ist dokumentiert; eine zusätzliche verblindete Doppelannotation mit gemessener Annotatorenübereinstimmung wurde nicht protokolliert.

## Daten und Testsperre

Train: DE78/EN71/JA91 nach zwei Lesedurchgängen, Rechte- und Ereignisprüfung. Im frischen Test-Quellenrahmen wurden 856 Artikel erhoben und jeweils zweimal vollständig gelesen. 41 Klassenabweichungen wurden vor Modellvorhersagen begründet; beide Originalprüfungen bleiben erhalten. Vier weiterhin unauflösbare Annotationen sowie offene Rechte, Quellformprobleme und geschützte Ereignisverbindungen sind dokumentiert ausgeschlossen. Ausschlussgründe können sich überlappen. Ganze Gruppen werden gemeinsam behandelt.

691 Fälle in 658 beobachteten Gruppen bleiben nach vorab festgelegter Quellenreihenfolge und Sprachobergrenze. Die englische 1500-Seiten-Stichprobenquelle war erschöpft; dies bedeutet nicht, dass es weltweit zu wenige englische Nachrichten gibt. Quellenverluste und Sprach-/Klassenhäufigkeiten stehen im [Test-Manifest](reports/next_stage/final-test-v1/test-manifest.json). Keine Klassenbalance, Textumstellungen, Duplikate oder synthetischen Testfälle. Alle 856 Rohfälle bleiben dauerhaft Test-reserviert.

Vier Kandidaten × 691 erfolgreiche gespeicherte Testvorhersagen, kein erneuter Testdurchlauf zur Zeitmessung und keine nachträgliche Optimierung. Warmups nutzen nur Train. Unsicherheit und Fehleranalyse verwenden dieselben gespeicherten Vorhersagen. Die historischen acht Schutzdateien, alten Berichte und MLflow-Läufe bleiben erhalten; v1–v5 und Vollartikelwerte werden nicht zu einer Fortschrittskurve verbunden.

## Lokal starten und verwenden

Python 3.12 mit bestehender CUDA-PyTorch-Umgebung; Rohdaten, Gewichte und MLflow-Datenbank liegen ignoriert außerhalb Git. Ein frischer Clone enthält zunächst Nachweise und Code. API-Start trainiert nichts und lädt keine Gewichte aus dem Netz.

```bash
source .venv/bin/activate
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Bestehenden Listener identifizieren und keinen zweiten Dienst starten. [Lokale Anwendung](http://127.0.0.1:8000/) · [API-Dokumentation](http://127.0.0.1:8000/docs) · [MLflow](http://127.0.0.1:5000/). `127.0.0.1` ist lokal erreichbar; kein öffentliches Deployment behauptet.

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"Das Unternehmen meldet einen Gewinneinbruch."}'
```

Genau eines von `text` oder `url`. Optionale Query-Aliasse: `article_frozen_v1`, `article_minilm_v1`, `article_qwen_zero_v1`, `article_qwen_lora_v1`. Historische Modelle bleiben auswählbar. `/predict` liefert `label`, drei normalisierte `scores`, tatsächliche `model_version`, `input_policy` und bei URL einen Auszug. Scores sind nicht kalibriert. `/analyze` ergänzt Satzlabels mit Originaltext-Offsets, maximal 80 Sätze; Farben sind zusätzliche Vorhersagen, keine kausalen Worterklärungen.

Vollartikel-Qwen weist über 4096 Artikeltokens zurück und kürzt nichts aus der Mitte. MiniLM verarbeitet alle 128-Token-Chunks als gleich gemittelte Features vor dem Artikelkopf. Historisches Qwen behält seine 3072-Token-Anfang/Ende-Regel, historische MiniLM-Profile ihre jeweilige Aggregation.

Modellunabhängiger URL-Abruf mit `full-article-v5`, ohne stille Zeichenkürzung. Private Adressen, Redirects, robots-Verbote und erkannte Login-/Sperrseiten bleiben blockiert. 422/403/502/503 und optionale `X-Article-Error`-Diagnosen bleiben erhalten. Fehlerhafte/veränderte Verlagslayouts können weiterhin scheitern; Textpaste bleibt möglich.

## Verifikation und Reproduktion

**189 Offline-Tests bestanden**; synthetische GPU-/HTTP-Prüfungen aller vier Modelle in DE/EN/JA, japanische Satzoffsets, Kontextgrenzen und Browserprüfung sind separat dokumentiert. Keine Testtexte wurden dafür erneut klassifiziert. Siehe [Prüfnachweis für Quellenkatalog und Reviewdarstellung](reports/data-review/verification-v1.json) und den [ursprünglichen Studien-Prüfnachweis](reports/next_stage/verification-final-v1.json).

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
python -m src.next_stage_reproduce --name review-only
```

Der zweite Befehl prüft standardmäßig nur. Ein ausdrücklich gestarteter Reproduktionslauf verwendet einen eigenen Namensraum und neue MLflow-Kennungen. [Anleitung](docs/reproduce-full-article.md) beschreibt Gewichte, exakte Datenwiederherstellung, Fehler bei abweichendem Rendering und die Testsperre. Die Quellenattribution und alten Lizenzbeschränkungen bleiben erhalten.

## Team

Teofil Wetzel und Nick Fronius · AI Operations, Hochschule Heilbronn.
