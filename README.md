# Artikel-Sentiment

**Lehrveranstaltung:** AI Operations  
**Hochschule:** HHN Hochschule Heilbronn  
**Studiengang:** Master Wirtschaftsinformatik  
**Semester:** WiSe 2026/27

Arbeitsstand. Der Titel kann sich noch ändern, die Aufgabe nicht: Stimmung von Nachrichtentexten.

## Kurzbeschreibung

Das Werkzeug bekommt entweder den Text eines Artikels oder die URL genau einer Seite, liest den Haupttext und schätzt die Stimmung: negativ, neutral oder positiv. Gedacht für Medienbeobachtung, PR- und Kommunikations-Teams und für Leute, die einen Text schnell einordnen wollen, bevor sie ihn weitergeben.

Später, nicht in dieser Hülle: ein Score für wertende oder aufgeladene Sprache. Dafür gibt es noch kein Label und kein Modell.

## Problem & Stakeholder

| | |
| --- | --- |
| Geschäftsproblem | Eingehende Artikel schnell nach Stimmung sortieren, ohne jeden Text von Hand zu lesen |
| Entscheidung | Was ist dringend, was ist neutrale Meldung, was ist positiv belegbar |
| Stakeholder | Medienbeobachtung, PR/Kommunikation, kritische Leserinnen und Leser |
| Sprachen | Deutsch und Englisch |
| Fehler | Ein Fehlalarm (neutral als Skandal) und ein übersehener negativer Text kosten unterschiedlich viel. Die Kosten sind noch nicht beziffert |

Die Idee ist mit dem Professor abgestimmt. Er erwartet einen Vergleich mehrerer Modelle und ein eigenes Fine-Tuning, nicht nur ein vortrainiertes Modell von der Stange.

## Vorhersageziel

| | |
| --- | --- |
| Ziel | Stimmung eines Artikels |
| Klassen im Code | `negative`, `neutral`, `positive` |
| Fünf Stufen | Very Negative und Negative → `negative`, Neutral → `neutral`, Positive und Very Positive → `positive` |
| Auswahlmaß | macro-F1 auf dem gemeinsamen Testsplit, danach Inferenzzeit und Modellgröße |

Die Abbildung passiert über die Label-Namen, nicht über geratene Indizes (`src/labels.py`).

## Datensatz & Split-Logik

| | |
| --- | --- |
| Im Repo | `data/sample/sentiment_sample.csv`: 18 handgeschriebene Sätze, DE und EN, Spalte `origin=dummy` |
| Echter Datensatz | **offen** (Meilenstein 1). Gelabelte Nachrichten, Deutsch und Englisch |
| Eine Zeile | Platzhalter: ein Satz. Später: ein Artikel |
| Split | nur in `src/data.py`: stratifiziert 80/20, `random_state=42` |

Der Zufallssplit gilt, solange die Zeilen unabhängig sind. Bei echten Artikeln muss geprüft werden, ob Medium, Ereignis oder Datum zusammenbleiben müssen. Dieselbe Regel für alle fünf Modelle, sonst ist der Vergleich nichts wert.

## Modellvergleich

Alle laufen auf **demselben** Testsplit und im **selben** Drei-Klassen-Raum. Ein MLflow-Experiment: `article-sentiment`. Dieselben Metriken: Accuracy, macro-F1, F1 je Klasse, Inferenzzeit (ms pro Text), Konfusionsmatrix als CSV-Artefakt.

| # | Rolle | Modell | Anmerkung |
| --- | --- | --- | --- |
| 1 | Baseline | TF-IDF + LogisticRegression | Klein, CPU, Wortgewichte. `tfidf_logreg` |
| 2 | Klassische Alternative | TF-IDF + LinearSVC | Ähnlich leicht. Scores sind ein Softmax der Entscheidungsfunktion, kein kalibriertes `predict_proba` |
| 3 | Zero-Shot | `tabularisai/multilingual-sentiment-analysis` | DistilBERT, fünf Klassen, geprüft am `id2label`. Chunks ≤ 512 Tokens, Mittel der Wahrscheinlichkeiten, dann auf drei Klassen falten |
| 4 | Zero-Shot | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Existiert. `id2label`: 0 negative, 1 neutral, 2 positive. Auf Tweets trainiert, nicht auf Nachrichten. `max_position_embeddings` ist 514, die Inferenz bleibt bei ≤ 512 |
| 5 | Eigenes Fine-Tuning | `distilbert-base-multilingual-cased`, Kopf mit 3 Labels | Alternativ `--base-model tabularisai/multilingual-sentiment-analysis` (Kopf wird neu aufgesetzt) |

Jedes Modell hat `predict_proba(texts)` und liefert die drei Klassen (`src/interface.py`). Die API lädt darüber, welches Modell `SENTIMENT_MODEL` nennt. Standard und CI: `tfidf_logreg`.

Lange Artikel: TF-IDF sieht den ganzen Text. Transformer zerlegen ihn in Stücke à höchstens 512 Tokens (abzüglich Sonderzeichen) und mitteln die Verteilungen. Beim Fine-Tuning wird der Trainingskontext auf `--max-length` gekürzt; die Inferenz chunked danach mit dieser Länge. Das ist ein Trainings-/Inferenz-Unterschied und gehört in die Trade-off-Diskussion.

### Auswahl (Meilenstein 2)

1. Höchstes macro-F1 auf dem gemeinsamen Testsplit. Jede Klasse zählt gleich, solange die Fehlerkosten offen sind.
2. Bei ähnlichem macro-F1 die niedrigere Inferenzzeit und die kleinere Modellgröße.
3. Danach die fachliche Passung: Twitter-Modell auf Nachrichten, fünf Klassen gemappt, Dummy-Daten.

Die Tabelle sortiert nur zur Ansicht (macro-F1 absteigend, dann Latenz). Sie schaltet die API nicht um. Solange der Dummy-Datensatz liegt, sind die Zahlen keine Modellwahl.

## Fine-Tuning

Leichter Check, ein Optimierungsschritt, vier Trainingssätze, CPU, Kontext 32. Schreibt nach `models/finetuned-smoke` und ist kein abgabefähiges Modell. Der erste Lauf lädt trotzdem das Basismodell (hundert bis mehrere hundert MB) und gehört deshalb nicht in die CI:

```bash
python -m src.finetune --smoke
```

Echtes Training auf einer GPU. In Google Colab: Laufzeit → GPU ändern, Repository öffnen, im Projektroot:

```bash
pip install -r requirements.txt -r requirements-transformer.txt
python -m src.finetune --epochs 3 --lr 2e-5 --batch-size 16 --max-length 256
```

Auf Colab das CPU-Wheel aus `requirements-transformer.txt` nicht erzwingen, sondern das CUDA-Wheel von PyTorch installieren und danach `transformers` und `accelerate`. Gewichte liegen in `models/finetuned` (gitignoriert). Ordner von Colab herunterladen und lokal an dieselbe Stelle legen. Die API liest nur diesen Ordner, nicht den Smoke-Ordner:

```bash
SENTIMENT_MODEL=finetuned uvicorn app.main:app --reload
```

Parameter: `--epochs`, `--lr`, `--batch-size`, `--max-length`, `--base-model`. `--smoke` setzt das Budget fest (1 Epoche, 1 Schritt, Batch 2, Länge 32, 4 Sätze) und ignoriert die anderen Budget-Flags.

## Setup

Python 3.12, Projektroot. Das reicht für Baseline, API, Tests und CI:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell): `.venv\Scripts\Activate.ps1`, danach derselbe pip-Aufruf.

Transformer nur wenn Zero-Shot oder Fine-Tuning wirklich laufen sollen:

```bash
python -m pip install -r requirements-transformer.txt
```

## Training, Vergleich, MLflow

Nur die beiden TF-IDF-Modelle, kein Download:

```bash
python -m src.train
```

Alle fünf, sobald die Transformer-Pakete da sind. Fehlt `models/finetuned`, läuft der Vergleich ohne das Fine-Tuning und sagt das in der Tabelle:

```bash
python -m src.train --all
```

Dasselbe ist `python -m src.compare`.

MLflow-UI:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Im Browser <http://127.0.0.1:5000>, Experiment `article-sentiment`.

- Ein Lauf je Modell: Parameter, Accuracy, macro-F1, F1 je Klasse, `inference_ms_per_text`, Artefakt `confusion_matrix.csv`.
- Lauf `comparison`: `comparison.csv` und `comparison.md` (dieselbe Tabelle lokal unter `reports/`, nicht im Git).

Tests:

```bash
python -m src.train
pytest
```

Die CI macht genau das und setzt `SENTIMENT_MODEL=tfidf_logreg`. Sie lädt kein Hugging-Face-Modell und ruft keine URL auf.

## API

```bash
uvicorn app.main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- Doku: <http://127.0.0.1:8000/docs>

Text:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The clinic reported fewer infections after the new treatment."}'
```

Eine einzelne Artikel-URL (nicht beides, nicht keins):

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/news/library"}'
```

Antwort: `label`, `scores` (negative, neutral, positive), `model_version`.

| Situation | HTTP |
| --- | --- |
| Text oder URL fehlt, beides gesetzt, leerer Text, falscher Typ | 422 |
| Seite ohne lesbaren Artikeltext | 422 |
| `robots.txt` verbietet den Abruf | 403 |
| Timeout, HTTP-Fehler, Weiterleitung, Host nicht auflösbar | 502 |
| Modelldatei fehlt | 503 |
| Anderes Modell | `SENTIMENT_MODEL=tfidf_linearsvc` oder `tabularisai`, `cardiffnlp`, `finetuned` |

### Abruf einzelner Artikel

`src/scraping.py` holt genau eine http(s)-URL. Keine Links weiterverfolgen, keine Redirects. Timeout 15 Sekunden. User-Agent `AIOpsSentimentBot/0.1`. `robots.txt` gilt für diesen Agenten; fehlt die Datei (404), ist der einzelne Abruf erlaubt, bei einem Lesefehler bricht er ab. Lokale und nicht-öffentliche Adressen werden nicht geholt.

Trotzdem die Nutzungsbedingungen der Seite lesen. Das ist ein einzelner Artikel für die Übung, kein Crawler.

## Repository-Struktur

```text
.
├── README.md
├── requirements.txt                 # Baseline, API, Tests
├── requirements-transformer.txt     # PyTorch CPU, Transformers, Accelerate
├── data/sample/                     # Dummy-Sätze, echter Datensatz fehlt
├── notebooks/01_exploration.ipynb
├── src/                             # Split, Modelle, Vergleich, Fine-Tuning, Abruf
├── models/                          # joblib und Fine-Tuning, lokal
├── app/                             # FastAPI, Vertrag in schemas.py
├── tests/
└── .github/workflows/ci.yml
```

Exploration im Notebook. Was deployed wird, liegt in `src/` und `app/` und trifft sich nur über `predict_proba` und das gespeicherte Artefakt.

## Meilensteine

### M1 – Problem, Daten, Baseline

- [x] Geschäftsidee abgestimmt (Artikel-Stimmung, DE/EN)
- [x] Vorhersageziel: negative / neutral / positive
- [ ] Echter gelabelter Nachrichten-Datensatz gewählt, Herkunft und Lizenz notiert
- [ ] Split-Regel am echten Datensatz geprüft (Zeit, Medium, Ereignis)
- [x] Repository und TF-IDF-Baseline laufen auf den Dummy-Sätzen

### M2 – Vergleich und Auswahl

- [x] Fünf Modellplätze, gleiche Metriken, ein MLflow-Experiment
- [ ] Vergleich auf dem echten Testsplit gelaufen, inklusive eigenem Fine-Tuning
- [x] Auswahlkriterien festgehalten (macro-F1, dann Latenz und Größe)
- [ ] Trade-off am echten Ergebnis aufgeschrieben, nicht nur die höhere Kennzahl

### M3 – Artefakt, Vertrag, lokales Deployment

- [x] Training und Inferenz getrennt, gemeinsames `predict_proba`
- [x] Vertrag: genau Text oder URL, Antwort mit Label, Scores, Modellversion
- [x] FastAPI lokal, gültige und ungültige Fälle, Abruf- und Leertextfehler
- [ ] Dokumentation und Präsentation auf dem echten Stand

## Offene Punkte

- Echter gelabelter Nachrichten-Datensatz Deutsch/Englisch ist noch nicht gewählt.
- Fehlerkosten (Fehlalarm gegen übersehenen negativen Text) sind noch nicht beziffert.
- Wertende oder aufgeladene Sprache bleibt ein späteres Ziel, ohne Label und ohne Modell.
- Ob das Fine-Tuning die Zero-Shot-Modelle schlägt, sagt der Dummy-Datensatz nicht.

## Team

| Name | Kontakt | Verantwortung |
| --- | --- | --- |
| _offen_ | _offen_ | _offen_ |
