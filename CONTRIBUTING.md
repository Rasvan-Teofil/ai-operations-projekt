# Weiterarbeiten

Die [Projektstruktur](docs/repository-structure.md) folgt der Vorlesungsvorlage.
Im Wurzelverzeichnis bleiben Einstieg und zentrale Konfiguration. Abhängigkeiten
liegen unter `requirements/`, Methodenbeschreibungen unter `docs/`, gemessene
Ergebnisse im eigenen Studienordner unter `reports/`.

Neue Studien implementieren wir als `src/studies/<name>/` mit einem klaren
CLI-Einstieg und eigenem README. Bestehende eingefrorene Module bleiben an ihrem
gehashten Ort; keine weitere Serie loser Präfixdateien unter `src/` beginnen.
Anwendungslogik gehört nach `app/`, der HTTP-Vertrag nach `app/schemas.py`.
Notebooks dürfen erkunden und darstellen; Training und Datenverarbeitung gehören
in importierbare Module. Wartungsbefehle liegen unter `scripts/`.

Rohtexte, Zugriffsdaten, Gewichte, lokale Datenbanken und Prozesslogs bleiben
ignoriert. Neue Logs direkt nach `runtime/logs/` schreiben. Vor jeder GPU-Arbeit
`nvidia-smi`; vorhandene CUDA-Umgebung erhalten. Gemeinsame Basispakete über
`requirements.txt`, Zusätze über `requirements/requirements-*.txt`. Keine
CPU-Torch-Installation auf dem Trainingstower durch unbeabsichtigtes Dependency-Update.

Neue Daten brauchen Quelle, Revision, Datum, Transformation, Rechteentscheidung,
Text-Hash und Annotationsherkunft. AI-Eigenprüfung als solche kennzeichnen.
Train-interne Entwicklung, endgültige Kandidaten und frischen Test zeitlich
trennen. Test-reservierte Fälle einschließlich ausgeschlossener Rohfälle nicht
später fürs Training verwenden. Synthetische Varianten teilen immer die Gruppe
ihres Originals und gehören nicht in den realen Haupttest. Historische
Vergleichsgruppen und Messdateien niemals überschreiben oder nachträglich umdeuten.

Vor dem Commit:

```bash
python scripts/check_layout.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
```

Bei neuen Messungen Protokoll, Auswahlregel, exakte Eingaben und Abhängigkeiten
vor Auswertung einfrieren, neue MLflow-Kennungen verwenden und Berichtsindizes
aktualisieren. Für Nachmessungen eigene Reproduktionsnamensräume verwenden.
Eine Ausnahme von der Layout-Baseline muss im PR begründet werden. Keine
automatische Neuannotation, kein Nachtraining und keine Testwiederholung durch CI.

Der vorstellbare Gesamtstand gehört auf `main`, einschließlich aktueller Anwendung,
Sprachübersicht, Messberichten und Startanleitung. Entwicklungsbranches sind
Arbeitsmittel. Historische Branch-Namen in eingefrorenen Nachweisen nicht umschreiben.
Vor Merge zusätzlich `python scripts/check_main_integration.py` ausführen; der
frühere Hauptzweig bleibt unter `docs/archive/main-before-full-article/source/`
bytegetreu erhalten und wird nicht als aktive zweite Anwendung eingebunden.

Lesbare Quellenlisten aus den öffentlichen Metadaten erzeugen:
`python scripts/build_source_catalog.py`; CI prüft sie mit `--check`.
Originale Referenzlabels, Quellrevisionen und Text-Hashes dabei nicht verändern.
Spätere menschliche Datenprüfungen mit Datum, Rolle, tatsächlichem Umfang und
Herkunftsnachweis als eigenen Schritt dokumentieren. Keine nachträgliche
Umdeutung der ursprünglichen Annotation oder des damaligen Messablaufs.
