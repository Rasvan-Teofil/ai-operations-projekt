# Optimizer-Ablation: 32 zusätzliche Entwicklungsfits

[Versuchsplan und Bedienung](../../src/studies/optimizer_ablation/README.md).
Vollständiges 2⁴-Design mit zwei Seeds: Clipping, Lernrate, AdamW Weight Decay,
Warmup. Qwen-LoRA, 159 Fit-/81 Entwicklungsartikel aus bestehendem Train-Fold 0.
Keine neue Testauswertung und keine Änderung des Standardmodells.

`protocol-v1.json` wird vor dem ersten Fit eingefroren. Während der Ausführung
erzeugt der Worker `summary-v1.json`, `results-v1.md`, `runs/`, `predictions/`
und `telemetry/` automatisch. Gemessene Ergebnisse erscheinen nach Prüfung auf
`main`; laufender Fortschritt im lokalen MLflow-Experiment
`article-optimizer-ablation-v1` oder per CLI `status`.

Effekte gelten für diesen Split, die zwei Seeds und die geprüften Wertebereiche.
Sie belegen keine allgemeine Optimalität oder kausale Erklärung. AdamW ist in
allen Bedingungen derselbe Optimierer; variiert wird dessen Weight Decay.

## Nachweise und Auswertung

- [Eingefrorenes Protokoll](protocol-v1.json): alle 32 Bedingungen, Seeds,
  Datenaufteilung und Hashes vor dem ersten Fit festgelegt.
- [Ergebnistabelle](results-v1.md) und [maschinenlesbarer Stand](summary-v1.json):
  geprüfte Momentaufnahme der abgeschlossenen Fits.
- [Prüfnachweis der Momentaufnahme](validation-v1.json): Hashes, gespeicherte
  Vorhersagen, neu berechnete Metriken und MLflow-Status.
- `runs/`: unveränderliche Konfigurationen, Metriken und MLflow-Kennungen.
- `predictions/`: gespeicherte Entwicklungslabels und Modellvorhersagen ohne Rohtexte.
- `telemetry/`: Gradientennormen, Lernraten und tatsächliche Adapteränderungen.

Die laufende Warteschlange führt jede der 16 Kombinationen mit beiden Seeds aus.
Erst nach allen 32 vollständigen Fits werden die gepaarten Haupteffekte und
Wechselwirkungen berechnet. Einzelne Zwischenwerte erlauben keine belastbare
Aussage über den Einfluss eines Faktors. MLflow enthält Konfiguration, Metriken
und Artefakte je Versuch; seine Datenbank und die Modellgewichte bleiben lokal.
