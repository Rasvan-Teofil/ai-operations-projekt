# Optimizer-Ablation: 32 zusätzliche Entwicklungsfits

**Nachfolgende Validierung:** Die [27 gepaarten Auswertungen über drei Folds und
drei Seeds](../optimizer_validation/README.md) zeigen keinen belegten Vorteil
des ausgewählten neuen Rezepts oder des isolierten Clippings in diesem Rezept.
Der hier berichtete +1,75-Punkte-Haupteffekt gilt für den ursprünglichen Fold 0
und den gemittelten Parameterbereich; er ist kein allgemeiner Qualitätsnachweis.
Das bisherige API-Standardmodell bleibt ausgewählt.

[Versuchsplan und Bedienung](../../src/studies/optimizer_ablation/README.md).
Vollständiges 2⁴-Design mit zwei Seeds: Clipping, Lernrate, AdamW Weight Decay,
Warmup. Qwen-LoRA, 159 Fit-/81 Entwicklungsartikel aus bestehendem Train-Fold 0.
Keine neue Testauswertung und keine Änderung des Standardmodells.

**32/32 Fits abgeschlossen und geprüft**, 26. November 2026, 12:52 Uhr MEZ.
[Abschlussauswertung mit Einordnung der Parametereffekte](interpretation-v1.md).
Das vor dem ersten Fit eingefrorene Protokoll und der Trainingscode sind unverändert.
Alle Nachweise liegen auf `main`; das lokale MLflow-Experiment heißt
`article-optimizer-ablation-v1` (Experiment-ID 2).

Effekte gelten für diesen Split, die zwei Seeds und die geprüften Wertebereiche.
Sie belegen keine allgemeine Optimalität oder kausale Erklärung. AdamW ist in
allen Bedingungen derselbe Optimierer; variiert wird dessen Weight Decay.

## Nachweise und Auswertung

- [Eingefrorenes Protokoll](protocol-v1.json): alle 32 Bedingungen, Seeds,
  Datenaufteilung und Hashes vor dem ersten Fit festgelegt.
- [Ergebnistabelle](results-v1.md) und [maschinenlesbarer Stand](summary-v1.json):
  vollständiger Stand aller 32 abgeschlossenen Fits.
- [Prüfnachweis aller 32 Fits](validation-v1.json): Hashes, gespeicherte
  Vorhersagen, neu berechnete Metriken und MLflow-Status.
- `runs/`: unveränderliche Konfigurationen, Metriken und MLflow-Kennungen.
- `predictions/`: gespeicherte Entwicklungslabels und Modellvorhersagen ohne Rohtexte.
- `telemetry/`: Gradientennormen, Lernraten und tatsächliche Adapteränderungen.

Jede der 16 Kombinationen wurde mit beiden Seeds ausgeführt. Die gepaarten
Haupteffekte und alle sechs Zweifaktor-Wechselwirkungen liegen vollständig vor.
Die [Einordnung](interpretation-v1.md) beschreibt Sprach- und Seed-Unterschiede,
Telemetrie, Grenzen und praktische Folgerungen. MLflow enthält Konfiguration, Metriken
und Artefakte je Versuch; seine Datenbank und die Modellgewichte bleiben lokal.
