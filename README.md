# [Projekttitel]

**Lehrveranstaltung:** AI Operations  
**Hochschule:** HHN Hochschule Heilbronn  
**Studiengang:** Master Wirtschaftsinformatik  
**Semester:** WiSe 2026/27

## Kurzbeschreibung

> Platzhalter: Hier steht in ein bis zwei Sätzen das Geschäftsproblem, sobald es gewählt ist.  
> Wer muss welche Entscheidung treffen, und welche Vorhersage hilft dabei?

Geschäftsidee und Datensatz sind noch offen. Dieses Repository ist die lauffähige Hülle dafür: ein Platzhalter-Datensatz, eine Baseline, Experiment-Tracking und ein lokaler Inferenzdienst. Der Platzhalter ist überall mit `PLACEHOLDER` markiert (Iris aus scikit-learn) und wird ausgetauscht, nicht umbaut.

## Problem & Stakeholder

| | |
| --- | --- |
| Geschäftsproblem | _offen_ |
| Entscheidung, die besser werden soll | _offen_ |
| Stakeholder (wer nutzt die Vorhersage?) | _offen_ |
| Was passiert bei einem Fehler? | _offen_ |

Offene Fragen für M1: Wer trägt die Konsequenz einer falschen Vorhersage, und was ist teurer – ein Fehlalarm oder ein übersehener Fall?

## Vorhersageziel

| | |
| --- | --- |
| Zielvariable | _offen_ (Platzhalter: Iris-Art `species`) |
| Aufgabe | _offen_ (Platzhalter: Klassifikation in drei Klassen) |
| Erfolgsmaß fachlich | _offen_ |
| Technisches Auswahlmaß der Hülle | macro-F1 auf dem Testdatensatz, bei Gleichstand Accuracy, danach das einfachere Modell |

Der Platzhalter sagt eine von drei Arten vorher: `setosa`, `versicolor`, `virginica`. Das ist kein Geschäftsproblem, nur damit Training, Tracking und API schon vor der Themenwahl durchlaufen.

## Datensatz & Split-Logik

| | |
| --- | --- |
| Quelle | **PLACEHOLDER:** `sklearn.datasets.load_iris` |
| Echte Quelle, Stand, Lizenz | _offen_ |
| Eine Zeile bedeutet | _offen_ (Platzhalter: eine Blüte) |
| Merkmale | Platzhalter: `sepal_length_cm`, `sepal_width_cm`, `petal_length_cm`, `petal_width_cm` |
| Split | stratifiziert 80/20, `random_state=42`, implementiert nur in `src/data.py` |

Der Zufallssplit gilt nur, solange die Zeilen unabhängig sind. Zeitreihen, mehrere Zeilen pro Kunde oder Informationen aus der Zukunft brauchen eine andere Regel. Die wird an derselben Stelle ersetzt.

Rohdaten liegen später unter `data/` und werden nicht ins Git gelegt, wenn sie groß oder sensibel sind. Siehe `data/README.md`.

## Setup

Python 3.12, im Projektroot:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows (PowerShell), nach dem Anlegen der Umgebung:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Training

Ebenfalls im Projektroot, mit aktivierter Umgebung:

```bash
python -m src.train
```

Das Skript trainiert zwei Modelle auf demselben Split:

- `dummy_most_frequent` – Baseline ohne Signal (häufigste Klasse)
- `logistic_regression` – einfaches lineares Modell

Parameter, Metriken und das Artefakt des ausgewählten Modells gehen nach MLflow (`mlruns/`, nicht im Git). Das ausgewählte Modell liegt zusätzlich als `models/baseline.joblib`. Diese Datei ist die einzige Brücke zur API; die API importiert keinen Trainingscode.

Auswahlregel, bis die fachlichen Kosten feststehen: höchstes macro-F1 auf dem Testsplit, bei Gleichstand höhere Accuracy, bei weiterem Gleichstand das einfachere Modell. Der Trade-off gehört in M2 ausformuliert, sobald klar ist, welche Fehler teuer sind.

## MLflow UI

Nach dem Training, im Projektroot:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Im Browser: <http://127.0.0.1:5000>

Im Experiment `ai-operations-baseline` liegen die Läufe mit Parametern, `accuracy`, `f1_macro` und dem Tag `selected`.

## API starten und `/predict` aufrufen

Training muss vorher gelaufen sein, sonst antwortet `/predict` mit HTTP 503 und einem Hinweis. Der Prozess selbst startet auch ohne Artefakt.

```bash
uvicorn app.main:app --reload
```

- Health: <http://127.0.0.1:8000/health>
- interaktive Doku: <http://127.0.0.1:8000/docs>
- Vorhersage:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}'
```

Beispielantwort:

```json
{"prediction": "setosa", "model_name": "logistic_regression"}
```

Der Vertrag steht in `app/schemas.py`. Eine Anfrage mit fehlendem Feld, falschem Typ oder einem unbekannten Feld liefert HTTP 422, ohne das Modell aufzurufen. Ein Fehler im Modell selbst liefert HTTP 500 mit einer kurzen Meldung, ohne Traceback an die aufrufende Stelle.

Tests, lokal wie in GitHub Actions:

```bash
python -m src.train
pytest
```

Die Action trainiert zuerst, damit das Artefakt für die API-Tests existiert. `pytest` allein trainiert nach, falls die Datei noch fehlt.

## Repository-Struktur

```text
.
├── README.md                 # Kontext, Setup, Meilensteine
├── requirements.txt          # Abhängigkeiten
├── data/                     # Rohdaten (großes bleibt lokal)
├── notebooks/                # Exploration, bevor der Code fest wird
├── src/                      # Laden, Split, Training, Metriken
├── models/                   # gespeicherte Artefakte (joblib bleibt lokal)
├── app/                      # FastAPI: Vertrag und /predict
├── tests/                    # Split und gültige/ungültige Anfragen
└── .github/workflows/ci.yml  # pytest auf Push und Pull Request
```

Warum diese Trennung: Die Erkundung darf im Notebook unordentlich sein. Was deployed wird, liegt in `src/` und `app/`, mit einem Vertrag dazwischen. Review und Wiederholung werden dadurch einfacher, weil Training und Dienst sich nur über das Artefakt treffen.

Tauschstellen, sobald das echte Thema feststeht:

- `src/config.py` – Spalten, Ziel, Seed, Pfade
- `src/data.py` – Laden aus `data/` und die Split-Regel
- `notebooks/01_exploration.ipynb` – EDA am echten Datensatz
- `app/schemas.py` – Input/Output-Vertrag passend zu den Merkmalen
- diese README – die Tabellen oben, die jetzt noch `_offen_` sind

## Meilensteine

Die Hülle (Platzhalter-Datensatz, Baseline, MLflow, API, Tests) steht. Die Haken setzt das Team, wenn der Inhalt zum gewählten Problem passt.

### M1 – Problem, Daten, Baseline

- [ ] Geschäftsproblem gewählt
- [ ] Vorhersageziel festgelegt
- [ ] Datensatz gewählt und Split-Logik beschrieben (und in `src/data.py` umgesetzt)
- [ ] Repository steht, README ist auf das echte Thema umgeschrieben
- [ ] Baseline-Modell ist gebaut und läuft (`python -m src.train`)

### M2 – Verbesserung und Auswahl

- [ ] Verbessertes Modell gegen die Baseline verglichen
- [ ] Läufe in MLflow nachvollziehbar (Parameter, Metriken, Artefakte)
- [ ] Bestes Modell ausgewählt, Trade-offs dokumentiert (nicht nur die höhere Kennzahl)

### M3 – Artefakt, Vertrag, lokales Deployment

- [ ] Modellartefakt gespeichert, Training und Inferenz getrennt
- [ ] Input-/Output-Vertrag steht (Schema für gültige Anfragen und Antworten)
- [ ] Lokales Deployment mit FastAPI (`/health`, `/predict`)
- [ ] Gültige und ungültige Anfragen getestet, Fehlerfälle behandelt
- [ ] Dokumentation und Präsentation

## Team

| Name | Kontakt | Verantwortung |
| --- | --- | --- |
| _offen_ | _offen_ | _offen_ |
