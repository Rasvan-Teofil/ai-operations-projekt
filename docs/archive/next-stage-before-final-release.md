> Ablagehinweis 11.11.2026: Die historischen `requirements-*.txt`-Dateien liegen jetzt unverändert unter `requirements/`. In alten Installationsbefehlen entsprechend `-r requirements/requirements-…txt` verwenden. Lauf-/Messnachweise behalten ihre alten Pfade und Hashes.

# Article Sentiment

Gemeinsames Gruppenprojekt für Medienbeobachtung und PR: deutsche und englische
Nachrichtentexte oder genau eine HTTP(S)-URL als `negative`, `neutral` oder
`positive` klassifizieren. Japanisch ist die bevorzugte, noch zu prüfende
Erweiterung. `second-path` ist der Arbeitsbranch; `main` bleibt unverändert.

## Aktueller Stand · 11. November 2026

Der neue Quellenpilot umfasst 300 native Volltexte (je 100 DE/EN/JA), die zweimal
von Codex gelesen wurden. Nach Rechte-, Übersetzungs- und konservativen
Ereignisprüfungen sind 240 Artikel für Train geeignet: 78 DE, 71 EN, 91 JA.
`full-article-v5` ergänzt fehlende Interview-Antworten und Tabellenkontext; alte
Textfassungen und Prüfungen bleiben erhalten. **Die 24 vorab festgelegten
Entwicklungsauswertungen laufen. Ein frischer Abschlusstest liegt noch nicht vor.**

- [Nächste Phase: Rubrik, Quellenpilot, Umfang und Grenzen](docs/next-stage.md)
- [Maschinenlesbares Protokoll](reports/next_stage/protocol.json)
- [Daten- und Modelllizenzen](licenses/README.md) · [Inventar](licenses/inventory.json)
- [Projektarchiv mit bisherigen Reproduktionsbefehlen](docs/archive/project-history-through-v5.md)

Die vorgeschlagene Entwicklung ist auf 24–30 Auswertungen innerhalb Train
begrenzt. Primärmaß bleibt das gleichgewichtete Mittel von DE- und EN-macro-F1;
innerhalb von 0,005 absolut zum besten Ergebnis entscheiden warme Latenz, dann
vollständige Artefaktgröße einschließlich LLM-Basis. Japanisch wird separat und
im ergänzenden Dreisprachenmittel ausgewiesen. Daten, Rezepte und finale
Kandidaten werden vor dem Abschlusstest festgeschrieben. Eingaben, Falten,
Mengenabstufungen und Entwicklungsrezepte sind bereits eingefroren;
[Fortschritt](reports/next_stage/progress.json) und
[Entwicklungsprotokoll](reports/next_stage/development-protocol-v1.json) halten den Stand fest.

## Belegte Ergebnisse und ihre Reichweite

Auf **einem gemeinsamen v5-Test mit 227 Titel-/Lead-Texten** und 692 Train-Texten:

| Verfahren | Mittlerer DE/EN-macro-F1 | Warme ms/Text | Vollständige MiB |
|---|---:|---:|---:|
| Frozen MiniLM mit gelerntem Kopf | 0,6467 | 3,340 | 465,14 |
| Eigenes MiniLM-Fine-Tuning, BF16 | 0,7012 | 2,927 | 240,73 |
| Qwen3 4B mit Nachrichtenrubrik | 0,7235 | 106,665 | 7.687,49 |
| Qwen3 4B mit sechs Train-Beispielen | 0,7914 | 246,481 | 7.687,50 |
| Eigenes Qwen3-4B-LoRA | 0,8049 | 114,160 | 7.698,77 |

Das sind gespeicherte Messungen aus v5, keine neuen Messungen. Referenzlabels:
**AI-labelled · Codex-reviewed**, ein AI-Annotator mit Eigenprüfung. Keine
unabhängig verifizierte Ground Truth und kein Nachweis vollständiger Artikel-
oder japanischer Qualität. Alle sieben Bedingungen, Klassenwerte, Unsicherheit,
Trainingskosten und Laufkennungen stehen im [v5-Bericht](reports/llm_adaptation_v5/comparison.md).
Die historischen Tests v1–v5 unterscheiden sich; ihre Zahlen bilden keine
kontinuierliche Lernkurve. Die historische main-Zahl 0,720 wird nicht dagegen gerankt.

Die UI gruppiert die v5-Forschungsauswahl und schnelle Alternative vor weiteren
Modellen. Der **API-Standard bleibt das historische Qwen-Zero-Shot-Profil**;
`?model=qwen_lora_v5` wählt LoRA ausdrücklich für eine Anfrage. Während des GPU-Trainings läuft die lokale API vorübergehend mit dem bisherigen
MiniLM auf der CPU (`SENTIMENT_ARTIFACT`); diese Sitzung überschreibt die
gespeicherte Modellauswahl nicht. Ein neuer Vollartikelstandard wird erst nach dem
neuen Vergleich gewählt. Modell-, Daten-
und Softwareversionen sind verschiedene Kennungen desselben Gruppenprojekts.

## Lokal verwenden

Vorhandene Tower-Umgebung: Python 3.12, CUDA-PyTorch, lokale Artefakte unter dem
ignorierten `models/`. Die API trainiert und lädt nichts aus dem Netz herunter.

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

[Analyse](http://127.0.0.1:8000/) · [API-Dokumentation](http://127.0.0.1:8000/docs)
· [Health](http://127.0.0.1:8000/health) · [MLflow](http://127.0.0.1:5000/)

Bestehende Dienste nicht doppelt starten. Ein Listener auf `127.0.0.1` ist lokal;
öffentliche Erreichbarkeit ist ein eigener Betriebsschritt. Gewichte und Daten
liegen nicht in Git. Ein frischer Clone hat deshalb noch kein lauffähiges Modell.
Für die bestehende CUDA-Umgebung gelten `requirements-cuda.txt`, `requirements.txt`
und die jeweiligen Zusatzdateien im Archiv. `requirements-transformer.txt` nicht
verwenden; vor jeder GPU-Arbeit `nvidia-smi`, ohne sichtbare GPU kein Training.

```bash
# Genau eines der Felder text oder url; Modell optional als Queryparameter:
python - <<'REQUEST'
import json, urllib.request
request = urllib.request.Request(
    'http://127.0.0.1:8000/predict?model=qwen_lora_v5',
    data=json.dumps({'text': 'Das Unternehmen meldet einen Gewinneinbruch.'}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request) as response:
    print(json.load(response))
REQUEST
```

`/predict`: `label`, drei `scores`, tatsächliche `model_version`, zusätzlich
`input_policy` und bei URL ein Auszug. `/analyze` ergänzt Satzvorhersagen mit
Offsets im Originaltext, bis 80 Sätze. Satzfarben sind Vorhersagen, keine kausalen
Worterklärungen. Scores sind normalisiert, noch nicht kalibriert.

URL-Eingaben verwenden modellunabhängig `full-article-v5`. Private Adressen,
Redirects und robots-Verbote werden weiter blockiert. Bekannte Login-/Sperrseiten
werden zurückgewiesen; andere Layouts können weiterhin scheitern. 422/403/502/503
bleiben erhalten. Der optionale Header `X-Article-Error` unterscheidet etwa
Upstream-403, Timeout und Extraktionsfehler. Die UI erhält Eingabe und Modellauswahl
und bietet Textpaste an. Keine pauschale Zusage für jeden Verlag.

**Grenzen:** maximal 20.000 Zeichen; keine stille Kürzung in der Extraktion.
Qwen lässt bei mehr als 3.072 Artikeltokens jedoch den Mittelteil weg. MiniLM
aggregiert 128-Token-Chunks. Die neue Aufbereitung ist noch kein Nachweis einer
angemessenen Vollartikelklassifikation. Alte Benchmarkdateien und Artefaktgewichte
wurden durch diese Softwareänderung nicht neu ausgewertet.

## Pilot prüfen und reproduzieren

```bash
python -m src.next_stage schema
# Nach Sammlung/Annotation; Rohtexte ausschließlich im ignorierten Verzeichnis:
python -m src.next_stage audit --articles data/next_stage/pilot/articles.jsonl --annotations data/next_stage/pilot/annotations.jsonl
# CPU-Tokenisierung mit vorhandenem lokalen Tokenizer, keine Inferenz:
python -m src.next_stage profile --articles data/next_stage/pilot/articles.jsonl --tokenizer models/local_llm/weights
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
```

Der Prüfer schreibt keine Splits oder MLflow-Läufe. Ungeprüfte Rechte, fehlende
historische Schutzdateien und fehlende Annotationen werden als offene Punkte
sichtbar; Exitcode 2. Er prüft bekannte exakte/provenienzbasierte und manuell
zugeordnete Ereignisgruppen, ersetzt aber keinen semantischen Überschneidungsaudit.
Zwei Annotationseigenprüfungen sind keine unabhängigen Annotatoren.

Prüfung dieses Pakets: **145 Offline-Tests bestanden**, dazu synthetische CLI-, Tokenizer-, HTTP- und Browserprüfungen. Die acht historischen Schutzdateien stimmen mit ihren eingefrorenen Hashes überein. [Prüfnachweis und Grenzen](reports/next_stage/verification.json).

Offline-Tests sind getrennt von Live-URL-/GPU-Prüfungen. Historische
Reproduktionswege, eingefrorene Protokolle, Quellenattribution und Berichte bleiben
im [Archiv](docs/archive/project-history-through-v5.md) erreichbar. Kein neuer
personenbezogener Vergleich und keine Präsentation.
