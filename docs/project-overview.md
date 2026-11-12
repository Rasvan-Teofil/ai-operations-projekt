# Projektübersicht für die Präsentation

**AI Operations · Hochschule Heilbronn · Wintersemester 2026/27**

Das Projekt unterstützt Medienbeobachtung: vollständigen Nachrichtentext oder
eine Artikel-URL eingeben und die berichtete Entwicklung als negativ, neutral
oder positiv einordnen. Zielgruppen sind Redaktionen, Kommunikation und Analyse.
Ein Modellscore ist keine kalibrierte Sicherheit und keine automatische
Entscheidung über Personen.

## Was der aktuelle Hauptzweig enthält

- Native deutsche, englische und **japanische** Vollartikel; drei gemeinsame Klassen.
- Vier auf demselben Test verglichene Verfahren: Frozen MiniLM mit Kopf,
  eigenes MiniLM-Fine-Tuning, Qwen-Nachrichtenrubrik, eigenes Qwen LoRA.
- 240 Train-Artikel, 27 Train-interne Entwicklungsauswertungen, vier feste
  vollständige Train-Fits und ein frischer Test mit 691 Artikeln.
- Auswahl von Qwen LoRA vor Erhebung des Tests; unveränderte Referenzen und
  Rezepte, keine nachträgliche Modellwahl.
- FastAPI, Browseroberfläche, Modellvergleich, Satzmarkierungen, Quellen- und
  Lizenzseiten, Reproduktion und Offline-Vertragstests.

## Vorstellungsweg ohne Branchwechsel

1. [README](../README.md): Problem, Stand und Sprachübersicht.
2. [Japanisch / 日本語](languages/japanese.md): 91 Train, 300 Test und separat
   gemessene Ergebnisse; keine übersetzten Testtexte.
3. [Ergebnisse](../reports/next_stage/comparison-v1.md): Hauptwert, Unsicherheit,
   Latenz, Speicher und Grenzen. Unterschiedliche historische Tests nicht mischen.
4. [Struktur](repository-structure.md): Daten/Modelle lokal, Code und HTTP-Vertrag
   getrennt, Messnachweise unter `reports/`, Methoden unter `docs/`.
5. [Reproduktion](reproduce-full-article.md) und [Lizenzen](../licenses/README.md).
6. [Vorführungsleitfaden](presentation-guide.md) für die lokal laufende Anwendung.

## Historie erhalten

Die erste Iteration und die frühe zweite Iteration des bisherigen Hauptzweigs
liegen vollständig im [main-Archiv](archive/main-before-full-article/README.md),
einschließlich Code, README, Pitch, Vorlagendateien und damaliger CI.
Ihre 168 versionierten Dateien werden über Hashes unverändert geprüft.
Die späteren v1–v5- und Vollartikel-Nachweise sind in den regulären Berichtsordnern
erhalten. Git-Historien beider Entwicklungswege bleiben Teil der Integration.

`second-path` in alten Protokollen und im MLflow-Experiment ist eine historische
Kennung. Die aktuelle Anwendung und ihre Nachweise stehen nun auf `main`.

## Was noch offen ist

Die öffentliche, kostenlose Bereitstellung ohne privaten Tower ist noch nicht
erfolgt. [Hostingstand](public-hosting.md). Die AI-Eigenprüfung ersetzt keine
unabhängige menschliche Validierung; 240 Train-Artikel sind kein belegtes Optimum.
Ein frei zugänglicher Dienst benötigt außerdem passende Betriebsgrenzen und
Quellen-/Lizenzhinweise. Die Datenumstellung ist keine pauschale rechtliche Garantie.
