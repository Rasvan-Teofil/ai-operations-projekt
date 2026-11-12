# Vollartikelstudie reproduzieren

Aktueller Arbeitsbranch `main`, Python 3.12 und bestehende CUDA-Umgebung. Die
Messläufe entstanden auf `second-path`; diese historische Kennung bleibt erhalten. Daten,
API-Cache, Gewichte und MLflow-Datenbank sind lokal und ignoriert. Ein Clone
enthält Nachweise und Code, aber noch keine Modellgewichte oder Rohartikel.
Die ursprünglichen Berichte und Testvorhersagen werden niemals überschrieben.

## Vorhandene Ergebnisse prüfen

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src.next_stage_reproduce --name review-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
```

Der erste Befehl validiert eingefrorene Trainingsdaten und Code, startet kein
Training und liest keinen Test. Versionsbindungen stehen in
`reports/next_stage/development-protocol-v1.json`, `engineering-amendment-v1.json`
und `final-candidates-v1.json`. Das ursprüngliche Git-Commit beschreibt den
Startzustand; der zusätzliche Code-Hash bindet die damals noch nicht committierten
Laufzeitdateien. Neue Dokumentation und Anwendung ändern diese Messhistorie nicht.

## Trainingsdaten wiederherstellen

```bash
python -m src.next_stage_restore
```

Dieser Netzwerkzugriff lädt ausschließlich die gespeicherten Wikinews-Revisionen,
rekonstruiert `full-article-v5` und prüft sämtliche Byte-Hashes vor dem Schreiben.
Er verwendet bereits veröffentlichte Annotationen, erzeugt keine Modelllabels und
keine neuen Splits. Verändertes MediaWiki-Template-Rendering kann trotz alter
Revision andere Texte ergeben: dann bricht er ab. Für exakt gleiche Daten wird
der ursprüngliche lokale Rohdaten-/API-Cache benötigt; Hashabweichungen werden
nicht passend gemacht.

## Entwicklung ausdrücklich neu ausführen

```bash
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m src.next_stage_reproduce --name replication-01 --execute
```

Das startet 24 Train-interne Auswertungen in eigenen Verzeichnissen
`reports/next_stage/reproduction/replication-01/` und
`models/next_stage/reproduction/replication-01/`. Neue MLflow-Läufe tragen den
Reproduktionsnamensraum. Bestehende Namen werden ohne `--resume` abgelehnt.
`--resume` setzt nur unvollständige Bedingungen dieses neuen Namensraums fort.
Die zusätzliche dreifache Seed-Studie des Originals ist separat dokumentiert;
der Treiber behauptet keine automatische Wiederholung aller 27 Läufe.

## Abschlusstest und Ergebnisbericht

Der finale Test wurde nach der Kandidatenfestlegung erhoben, zweimal gelesen,
auf Rechte und bekannte Ereignisverbindungen geprüft und dann eingefroren:
691 Artikel, DE273/EN118/JA300, 658 beobachtete Gruppen. Alle 856 Rohfälle bleiben
dauerhaft Test-reserviert. 41 Klassenabweichungen zwischen Eigenprüfungen und
vier weiterhin unauflösbare Fälle stehen im vollständigen Annotationsnachweis.
Ausgeschlossene Rohfälle sind nicht nachträglich Train-Daten.

`final-test-v1/test-manifest.json` bindet Texte, Referenzen, Quellenreihenfolge,
Gruppen-, Qualitäts- und Schutzdateien. `final-test-evaluation.lock` verhindert
eine zweite Testbewertung. Die vier ursprünglichen Vorhersagedateien liegen
lokal unter `models/next_stage/final-predictions/`; je Modell und Eingabe wurde
eine erfolgreiche Vorhersage gespeichert. Aufwärmen nutzte nur Train. Keine
erneute Testinferenz zur Latenzmessung und keine nachträgliche Modellwahl.

```bash
# Nur aus den bereits gespeicherten Vorhersagen; keine Modellaufrufe:
python -m src.next_stage_test_report
```

Dieser Bericht berechnet Klassenwerte, Sprachwerte, Fehlerlisten,
Referenz-Mehrdeutigkeit und 2000 gepaarte Bootstrapziehungen ganzer beobachteter
Gruppen. Er liest vorhandene MLflow-Trainingskosten; er trainiert oder optimiert
nichts. Originale Referenzen bleiben unverändert. Der Bericht ist reproduzierbar,
ein neuer Durchlauf der Testinferenz ist kein regulärer Reproduktionsschritt.

## Anwendung

```bash
nvidia-smi
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Bestehenden Listener zuerst identifizieren; keinen zweiten Dienst starten.
`models/serving_selected.json` bezeichnet das vor Test gewählte Qwen-LoRA-Profil.
Die API-Aliasse `article_frozen_v1`, `article_minilm_v1`,
`article_qwen_zero_v1` und `article_qwen_lora_v1` wählen die vier eingefrorenen
Artefakte. Die ursprüngliche Standardauswahl bleibt als Sicherung erhalten.
Model Cards und neue Nachweise liegen außerhalb der gehashten Modellordner.

Vollartikel-Qwen akzeptiert höchstens 4096 Artikeltokens und weist längere
Eingaben zurück. MiniLM aggregiert alle 128-Token-Chunks auf Feature-Ebene.
Historische Qwen-/MiniLM-Profile behalten ihre eigenen Kürzungs- und
Aggregationsregeln. Maximal 20000 Zeichen; URL-Abruf prüft weiterhin SSRF,
robots und Zugangssperren. Satzfarben sind zusätzliche Vorhersagen, keine
kausalen Erklärungen und kein Bestandteil der gemessenen Artikelgenauigkeit.

Die Kennzahlen messen Übereinstimmung mit **AI-labelled · Codex-reviewed**
Referenzen. Zwei Lesedurchgänge desselben AI-Annotators ersetzen keine
unabhängige menschliche Validierung. Das Archiv, die Längenbegrenzung und offene
Vortrainingsüberschneidungen begrenzen die Übertragbarkeit.

## Lokales MLflow anzeigen

```bash
MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

Nur starten, wenn Port 5000 frei ist. Die vorhandene Datenbank bleibt erhalten.
Die ungenutzte Hintergrund-Jobausführung ist abgeschaltet: Ihre Huey-Consumer
starteten in dieser Umgebung mit einem fehlenden Einstiegspfad. Tracking und
das Anzeigen der 88 historischen sowie 43 neuen Läufe funktionieren weiterhin;
die Studie nutzt diese Server-Jobfunktion nicht. Prozessausgaben nach
`runtime/logs/` umleiten. Das ist unabhängig von der Modellinferenz auf Port 8000.
