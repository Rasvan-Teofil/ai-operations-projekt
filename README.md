# Artikel-Sentiment

**Lehrveranstaltung:** AI Operations  
**Hochschule:** HHN Hochschule Heilbronn  
**Studiengang:** Master Wirtschaftsinformatik  
**Semester:** WiSe 2026/27

Das Werkzeug liest einen Nachrichtentext oder genau eine Artikel-URL und schätzt die Stimmung: negativ, neutral oder positiv. Deutsch und Englisch.

## Kurzbeschreibung

Gedacht für Medienbeobachtung, PR- und Kommunikations-Teams und für Leute, die einen Text einordnen wollen, bevor sie ihn weitergeben. Später, hier noch ohne Label und ohne Modell: ein Score für wertende oder aufgeladene Sprache.

## Problem & Stakeholder

| | |
| --- | --- |
| Geschäftsproblem | Eingehende Artikel nach Stimmung sortieren, ohne jeden Text von Hand zu lesen |
| Entscheidung | Was ist dringend negativ, was ist eine neutrale Meldung, was ist positiv belegbar |
| Stakeholder | Medienbeobachtung, PR/Kommunikation, kritische Leserinnen und Leser |
| Sprachen | Deutsch und Englisch |
| Fehler | Ein Fehlalarm und ein übersehener negativer Text kosten unterschiedlich viel. Die Kosten sind noch offen. Auswahlmaß ist deshalb macro-F1 |

Die Idee ist mit dem Professor abgestimmt. Er erwartet einen Vergleich mehrerer Modelle und ein eigenes Fine-Tuning.

## Vorhersageziel

| | |
| --- | --- |
| Ziel | Stimmung eines Textes |
| Klassen | `negative`, `neutral`, `positive` |
| Fünf Stufen bei Zero-Shot | Very Negative und Negative → `negative`, Neutral → `neutral`, Positive und Very Positive → `positive` |
| Auswahl | höchstes macro-F1 auf dem gemeinsamen Testsplit, bei knappem Abstand Inferenzzeit und Modellgröße |

Die Abbildung läuft über die Label-Namen (`src/labels.py`).

## Datensatz

Eine Zeile ist ein Satz aus einer Finanznachricht. Die Sammlung ist akademisch nutzbar und bleibt lokal: `python -m src.prepare_data` lädt sie, `.gitignore` hält die Texte aus dem Git. Die Dummy-Datei `data/sample/sentiment_sample.csv` (18 handgeschriebene Sätze, `origin=dummy`) gilt nur für Tests und CI (`SENTIMENT_DATA=dummy`).

| Sprache | Quelle | Ausschnitt | Lizenz | Zeilen nach Filter |
| --- | --- | --- | --- | --- |
| Englisch | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | `Sentences_75Agree.txt` (≥ 75 % Annotator-Einigkeit) | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | 3 440 |
| Deutsch | [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) | nur `language=de` | akademisch / nicht-kommerziell; die Karte nennt zusätzlich Apache-2.0, hier gilt die engere Lesart | 5 023 |

Phrasebank-Labels sind die Sicht einer Investorin oder eines Investors: würde der Satz den Kurs eher heben, senken oder unverändert lassen. Kenpache sind kürzere Sätze aus deutschsprachigen Finanzmedien; die Labels sind brauchbar und gröber (Glossarzeilen, gelegentlich zusammengeklebte Meldungen). Beide Quellen liefern die drei Klassennamen bereits. Zusammen 8 463 Sätze.

**Split** (nur in `src/prepare_data.py`, Seed 42, Schicht `label|language`): zuerst 15 % Test, aus dem Rest Val als 15 % der Gesamtheit (`0.15/0.85`), Train der Rest.

| Split | Zeilen |
| --- | --- |
| Train | 5 923 |
| Val | 1 270 |
| Test | 1 270 |

Val bleibt unangetastet. Alle fünf Modelle sehen denselben Testsplit. Es gibt keine Artikel-ID, deshalb können Sätze derselben Meldung in Train und Test liegen. Herkunft, Labelabbildung und Grenzen stehen in `data/README.md`.

## Schnellstart

Python 3.12, Projektroot.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell): `.venv\Scripts\Activate.ps1`, danach derselbe pip-Aufruf.

Zero-Shot und Fine-Tuning brauchen zusätzlich PyTorch (CPU-Wheel) und Transformers. Die CI installiert diese Datei nicht:

```bash
python -m pip install -r requirements-transformer.txt
```

Daten holen und aufteilen (lädt die beiden Hugging-Face-Dateien, schreibt `data/processed/`, committet sie nicht):

```bash
python -m src.prepare_data
```

Die beiden TF-IDF-Modelle trainieren und in MLflow schreiben:

```bash
python -m src.train
```

Eigenes Fine-Tuning. Ohne GPU ist der belegte Lauf dieses Repos das CPU-Subset: eine Epoche, 2 000 stratifizierte Trainingssätze, Kontext 128. Die Bewertung nutzt den vollen Testsplit. Das dauert auf einer kleinen CPU wenige Minuten, sobald `distilbert-base-multilingual-cased` im Cache liegt.

```bash
python -m src.finetune --epochs 1 --lr 2e-5 --batch-size 8 --max-length 128 --max-samples 2000
```

Voller Lauf auf einer GPU (Google Colab, Laufzeit → GPU): `notebooks/02_finetune_colab.ipynb`. Dort drei Epochen, der ganze Train-Split, Kontext 256, CUDA-PyTorch statt des CPU-Wheels. Den Ordner `models/finetuned` danach lokal an dieselbe Stelle legen.

Alle fünf Modelle auf demselben Testsplit, eine Tabelle unter `reports/`:

```bash
python -m src.train --all
```

Dasselbe ist `python -m src.compare`. Fehlt `models/finetuned`, läuft der Vergleich ohne das Fine-Tuning und schreibt das in die Tabelle.

MLflow-UI:

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns --port 5000
```

Im Browser <http://127.0.0.1:5000>, Experiment `article-sentiment`. Ein Lauf je Modell (Accuracy, macro-F1, F1 je Klasse, `inference_ms_per_text`, Artefakt `confusion_matrix.csv`) und ein Lauf `comparison`.

API, Standardmodell `tfidf_linearsvc`:

```bash
uvicorn app.main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- Doku: <http://127.0.0.1:8000/docs>

Anderes Modell: `SENTIMENT_MODEL=tfidf_logreg`, `tabularisai`, `cardiffnlp` oder `finetuned`. Transformer und das Fine-Tuning brauchen `requirements-transformer.txt` und die Gewichte auf der Platte.

### Beispielaufrufe

Health nach `python -m src.train`:

```json
{"status":"ok","model_loaded":true,"model_name":"tfidf_linearsvc","model_version":"tfidf-linearsvc-v1"}
```

Deutsch, Rohtext:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Die Aktie verlor nach der Gewinnwarnung deutlich an Wert."}'
```

```json
{"label":"negative","scores":{"negative":0.7077840993250278,"neutral":0.14219203378102802,"positive":0.15002386689394404},"model_version":"tfidf-linearsvc-v1"}
```

Englisch, Rohtext:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Samsung expects a nine-fold surge in quarterly profits, driven by demand for AI memory chips."}'
```

```json
{"label":"positive","scores":{"negative":0.2651965190241689,"neutral":0.2732371307859462,"positive":0.4615663501898849},"model_version":"tfidf-linearsvc-v1"}
```

Artikel-URL (genau eine, am 8. Oktober 2026 abgerufen). BBC: „AI chip boom pushes Samsung profits to record $80bn“, <https://www.bbc.com/news/articles/c687z8127302o>.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.bbc.com/news/articles/c687z8127302o"}'
```

```json
{"label":"positive","scores":{"negative":0.20056848069748298,"neutral":0.20885450235096512,"positive":0.590577016951552},"model_version":"tfidf-linearsvc-v1"}
```

Die LinearSVC-Scores sind ein Softmax der Entscheidungsfunktion, keine kalibrierte Wahrscheinlichkeit. Text und URL zusammen, oder keins von beiden, ergibt HTTP 422.

| Situation | HTTP |
| --- | --- |
| Text oder URL fehlt, beides gesetzt, leerer Text, falscher Typ | 422 |
| Seite ohne lesbaren Artikeltext | 422 |
| `robots.txt` verbietet den Abruf | 403 |
| Timeout, HTTP-Fehler, Weiterleitung, Host nicht auflösbar | 502 |
| Modelldatei fehlt | 503 |

`src/scraping.py` holt genau eine http(s)-URL. Keine Links weiterverfolgen, keine Redirects. Timeout 15 Sekunden. User-Agent `AIOpsSentimentBot/0.1`. Fehlt `robots.txt` (404), ist der einzelne Abruf erlaubt; ein Lesefehler bricht ab. Lokale und nicht-öffentliche Adressen werden nicht geholt. Trotzdem die Nutzungsbedingungen der Seite lesen.

## Modellvergleich

Die gemessene Tabelle, die Trade-off-Diskussion und die Empfehlung stehen in [`reports/model_comparison.md`](reports/model_comparison.md). Kurz, CPU, gemeinsamer Test von 1 270 Sätzen:

| Modell | macro-F1 | ms pro Text | Größe |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 0.720 | 0.02 | 4 MB |
| TF-IDF + LogisticRegression | 0.656 | 0.03 | 4 MB |
| Fine-Tuning, CPU-Subset (2 000 Sätze, 1 Epoche, Kontext 128) | 0.614 | 8.3 | 516 MB |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 0.486 | 19.0 | 1 061 MB |
| tabularisai/multilingual-sentiment-analysis (5→3) | 0.364 | 8.8 | 516 MB |

**Empfehlung und API-Standard: `tfidf_linearsvc`.** Höchstes macro-F1, klein, schnell, läuft mit `requirements.txt`. Der volle GPU-Lauf (3 Epochen, alle 5 923 Trainingssätze, Kontext 256) ist vorbereitet und hier nicht gemessen. Die CI setzt `SENTIMENT_MODEL=tfidf_logreg` und `SENTIMENT_DATA=dummy`, installiert nur `requirements.txt` und ruft kein Netz auf.

Jedes Modell hat `predict_proba(texts)` (`src/interface.py`). TF-IDF sieht den ganzen Text. Transformer zerlegen ihn in Stücke à höchstens 512 Tokens und mitteln die Verteilungen. Das Fine-Tuning kürzt beim Training auf `--max-length` und chunkt bei der Inferenz mit dieser Länge.

## Tests

```bash
SENTIMENT_DATA=dummy SENTIMENT_MODEL=tfidf_logreg python -m src.train
pytest
```

Die CI macht genau das.

Exploration der echten Splits, mit gespeicherter Ausgabe: `notebooks/01_exploration.ipynb`.

## Repository-Struktur

```text
.
├── README.md
├── requirements.txt
├── requirements-transformer.txt
├── data/README.md                 # Lizenzen, Split, Grenzen
├── data/sample/                   # Dummy, nur Tests und CI
├── notebooks/01_exploration.ipynb
├── notebooks/02_finetune_colab.ipynb
├── src/                           # Prepare, Training, Vergleich, Fine-Tuning, Abruf
├── models/                        # joblib und Fine-Tuning, lokal
├── reports/model_comparison.md
├── app/
├── tests/
└── .github/workflows/ci.yml
```

## Meilensteine

### M1 – Problem, Daten, Baseline

- [x] Geschäftsidee abgestimmt (Artikel-Stimmung, DE/EN)
- [x] Vorhersageziel: negative / neutral / positive
- [x] Gelabelte Nachrichten-Sätze gewählt, Herkunft und Lizenz notiert
- [x] Split-Regel festgehalten (70/15/15, Label und Sprache, Seed 42; keine Artikel-ID)
- [x] TF-IDF-Baseline auf dem echten Train-Split

### M2 – Vergleich und Auswahl

- [x] Fünf Modelle, gleiche Metriken, ein MLflow-Experiment
- [x] Vergleich auf dem echten Testsplit, inklusive CPU-Fine-Tuning
- [x] Auswahl: `tfidf_linearsvc` (macro-F1, Latenz, Größe)
- [x] Trade-off in `reports/model_comparison.md`

### M3 – Artefakt, Vertrag, lokales Deployment

- [x] Training und Inferenz getrennt, gemeinsames `predict_proba`
- [x] Vertrag: genau Text oder URL, Antwort mit Label, Scores, Modellversion
- [x] FastAPI lokal, gültige und ungültige Fälle
- [x] README auf dem gemessenen Stand
- [ ] Präsentation

## Offene Punkte

- Fehlerkosten (Fehlalarm gegen übersehenen negativen Text) sind noch offen.
- Wertende oder aufgeladene Sprache bleibt ein späteres Ziel, ohne Label und ohne Modell.
- Der GPU-Lauf aus dem Colab-Notebook ist beschrieben und auf diesem Stand nicht gemessen.
- Die Trainingsdomäne ist der Finanzsatz. Ein allgemeiner Nachrichtenartikel kann danebenliegen.
- Team und Präsentation sind noch offen.

## Team

| Name | Kontakt | Verantwortung |
| --- | --- | --- |
| _offen_ | _offen_ | _offen_ |
