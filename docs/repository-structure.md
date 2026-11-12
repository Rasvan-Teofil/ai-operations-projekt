# Projektstruktur und dauerhafte Ablage

Seit 12. November 2026 ist dies die aktive Struktur von `main`. Die aktuelle
Anwendung, DE/EN/JA-Studie, Tests und Nachweise sind hier integriert. Der gesamte
vorherige Hauptzweig bleibt [bytegetreu archiviert](archive/main-before-full-article/README.md).

Die Vorlesungsvorlage ist im Repository nachweisbar: Commit
[`6d1356a`](https://github.com/Rasvan-Teofil/ai-operations-projekt/commit/6d1356aa114e370eda3fb3971325608325ba9ead),
Branch `cursor/official-layout-4237`, README-Abschnitt „Repository structure“.
Auch `origin/main` enthält diese Zuordnung. Das folgende Layout folgt grob
dieser Vorlage; bestehende Studien bleiben eigenständige Vergleichsgruppen.

```text
ai-operations-projekt/
├── README.md                 # Einstieg, belegter aktueller Stand, Startbefehle
├── CONTRIBUTING.md           # Ablageregeln, Prüfung und unveränderliche Nachweise
├── requirements.txt          # Gemeinsame Python-Abhängigkeiten
├── requirements/             # CUDA, Evaluation, LLM/LoRA, Tower-Lock
├── data/                     # Herkunftshinweise; Rohtexte/Splits lokal ignoriert
├── notebooks/                # Lesende Exploration vorhandener Ergebnisdateien
├── src/                      # Daten, Features, Training, Auswertung, Serving
│   └── studies/              # Neue Studien künftig als eigene Pakete
├── models/                   # Lokale Gewichte/Adapter, nicht in Git
├── app/
│   ├── main.py               # FastAPI-Routen, Modellwahl und Serialisierung
│   ├── schemas.py            # Pydantic-Eingabe-/Ausgabevertrag
│   └── static/               # Browseransicht, Skripte und Styles
├── tests/                    # Offline-Vertrags-, Sicherheits- und Methodentests
├── reports/                  # Gemessene, versionierte Ergebnisse; Index im README
├── docs/                     # Methoden, Reproduktion, Vorführungsweg und Archiv
├── licenses/                 # Inventar, exakte Lizenztexte und Attribution
├── scripts/                  # Wartungswerkzeuge, keine Modellimplementierungen
└── .github/                  # CI und Reviewvorlage
```

Die Vorlagenrollen `src/data.py` und `src/features.py` entsprechen hier
`prepare.py`/`next_stage_collect.py`/`next_stage_prepare.py` beziehungsweise
`tokenization.py`/`aggregation.py`/`inference.py`. `src/train.py` und
`src/evaluate.py` bleiben vorhanden. Für die neue Vollartikelstudie sind
`next_stage_experiment.py`, `next_stage_final.py` und
`next_stage_test_report.py` die versionsgebundenen konkreten Treiber.
Es werden keine bedeutungslosen Platzhaltermodule angelegt, um Dateinamen zu imitieren.

## Umgesetzt

- Fünf ergänzende Dependency-Dateien unverändert nach `requirements/` verschoben;
  `requirements.txt` bleibt der Einstieg. Aktuelle Installationsverweise korrigiert.
- 49 lokale Prozesslogs nach `runtime/logs/archived-2026-11-11/` verschoben.
  Der gesamte lokale Runtime-Bereich ist ignoriert. Ein laufender Prozess kann
  einen bereits geöffneten Log-Deskriptor weiter beschreiben.
- HTTP-Schemas aus `app/main.py` nach `app/schemas.py` ausgegliedert.
- Themenindizes für Quellcode, Berichte, Dokumentation, Daten und Modelle ergänzt;
  historische Protokolle bleiben auffindbar. Die Notebookansicht liest nur Ergebnisse.
- Dateiplatzierung durch `CONTRIBUTING.md`, PR-Vorlage und einen CI-Layoutcheck
  abgesichert. Neue Studienmodule gehören in `src/studies/<studie>/`.

## Warum einige alte Pfade erhalten bleiben

Eingefrorene Quellmodule, Daten-, Modell- und Berichtspfade sind Teil von
Hashbindungen, Kommandoaufrufen, Zitaten und MLflow-Herkunft. Eine reine
Kosmetikverschiebung würde diese Nachweise beschädigen. Die gegenwärtigen
flachen `src/*.py`-Module und alten Baseline-Berichte im `reports/`-Wurzelverzeichnis
sind deshalb als Bestand explizit inventarisiert. Neue lose Dateien dort werden
vom Layoutcheck abgewiesen. Die Struktur ist navigierbar, ohne eine zweite
Modellimplementierung oder widersprüchliche Kopien einzuführen.

Die lokale MLflow-Datenbank bleibt an ihrem bestehenden Ort: Der laufende Dienst
und seine bisherigen Pfade sind daran gebunden. Gewichte, Rohtexte, Logs und die
Datenbank werden nicht in Git aufgenommen. Es wurde keine Datei inhaltlich gelöscht.
Alle Verschiebungen stehen mit Byte-Hashes im
[Ablagenachweis](../reports/next_stage/repository-reorganization-v1.json).

## Pflege

Vor einem Commit `python scripts/check_layout.py` und die passenden Tests
ausführen. Für eine neue Studie einen neuen Namensraum unter Code, Reports,
lokalen Daten und lokalen Modellen anlegen und den jeweiligen Index ergänzen.
Bestehende Ergebnisse nicht überschreiben. Eine absichtliche Änderung der
Vorlage muss mit Begründung und aktualisierter Baseline im Review sichtbar sein.
