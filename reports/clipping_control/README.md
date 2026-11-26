# Gradient Clipping: vollständige Gegenläufe

**0/7 neue Fits, 23/30 vollständige Paare.**

Jeder Lauf mit Clipping erhält einen ansonsten identischen Gegenlauf ohne Clipping. Lernrate, AdamW Weight Decay, Warmup, Seed, Datenaufteilung, LoRA und Trainingsdauer bleiben gleich. Vorhandene exakt passende Fits werden wiederverwendet, keine Messung wird erfunden oder überschrieben.

Die erste 32-Fit-Serie enthält bereits 16 Paare. Die spätere CV umfasst zwei Rezepte mit je neun Paaren; vier Paare überlappen die erste Serie. Insgesamt sind es 30 verschiedene Paare / 60 Fits: 53 vorhandene und sieben neue. Fehlend waren nur sieben Ohne-Clipping-Fits des bisherigen Rezepts.

**Vorzeichen überall: mit minus ohne Clipping. Positive Werte sprechen für Clipping.** F1-Unterschiede stehen in absoluten Prozentpunkten.

## Gleiche Rezepte, jeweils mit und ohne Clipping

| Rezept | Lernrate | Weight Decay | Warmup | Clipping-Stufen |
|---|---:|---:|---:|---|
| Bisheriges Rezept | 0.0001 | 0.01 | 0.1 | 1.0 / aus |
| Neues Rezept | 0.00005 | 0 | 0 | 1.0 / aus |

Drei gruppierte Train-Folds und drei feste Seeds (20261110/11/12). Pro Seed werden die drei zurückgehaltenen Folds auf allen 240 Artikeln gepoolt, dann Macro-F1 berechnet. Hauptwert ist das gleichgewichtete DE/EN-Mittel; Japanisch wird separat ausgewiesen. Kein Mittel über die beiden Rezepte als vermeintlich allgemeiner Clipping-Effekt.

**Die vollständige CV-Auswertung steht noch aus. Es gibt noch kein neues Gesamtergebnis.**

## Alle ursprünglichen Faktor-Paare

Diese 16 Paare waren vor dem neuen Auftrag bereits abgeschlossen. Nur Fold 0, zwei Seeds; nicht mit der CV zu unabhängigen Stichproben zusammenzählen.

| Seed | LR | Decay | Warmup | Δ DE/EN (pp) |
|---:|---:|---:|---:|---:|
| 20261110 | 0.0001 | 0.01 | 0.1 | +0.00 |
| 20261110 | 0.0001 | 0 | 0.1 | +0.00 |
| 20261110 | 0.0001 | 0 | 0 | +0.00 |
| 20261110 | 5e-05 | 0 | 0 | +0.00 |
| 20261110 | 0.0001 | 0.01 | 0 | +0.00 |
| 20261110 | 5e-05 | 0 | 0.1 | +0.00 |
| 20261110 | 5e-05 | 0.01 | 0.1 | +5.23 |
| 20261110 | 5e-05 | 0.01 | 0 | +0.00 |
| 20261111 | 0.0001 | 0.01 | 0.1 | +2.91 |
| 20261111 | 0.0001 | 0.01 | 0 | +1.29 |
| 20261111 | 5e-05 | 0.01 | 0 | +5.71 |
| 20261111 | 0.0001 | 0 | 0 | +1.29 |
| 20261111 | 0.0001 | 0 | 0.1 | -1.03 |
| 20261111 | 5e-05 | 0 | 0.1 | +5.23 |
| 20261111 | 5e-05 | 0 | 0 | +2.16 |
| 20261111 | 5e-05 | 0.01 | 0.1 | +5.23 |

## Nachweise, Betrieb und Grenzen

- [Vor Trainingsbeginn eingefrorenes Protokoll](protocol-v1.json), [aktuelle vollständige Zuordnung](summary-v1.json), [Paarliste als CSV](pairs-v1.csv).
- [Erste 32 Fits](../optimizer_ablation/README.md), [vorherige Drei-Rezept-Validierung](../optimizer_validation/README.md). Deren Ergebnisse bleiben unverändert.
- Neues MLflow-Experiment: `article-gradient-clipping-controls-v1`; jeder neue Run verlinkt seinen ursprünglichen Clipping-Partner. [Öffentliches Original-MLflow](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments).
- Nach bereits bekannten Ergebnissen geplante explorative Erweiterung. Keine unabhängige Bestätigung; kleine, wiederverwendete Train-Stichprobe. 2000 gepaarte Gruppen-Bootstrap-Ziehungen beschreiben bedingte Unsicherheit, keine Auswahlkorrektur und nicht alle Trainingsschwankungen.
- Volltrain-Adapter ohne zurückgehaltene Daten sind keine CV-Paare. Kein erneuter Zugriff auf den 691er Abschlusstest. Keine Änderung vorhandener Gewichte oder der Serving-Auswahl.
- Fits laufen einzeln mit Trainingssperre und frischem Prozess. Die Inferenz-API kann dafür vorübergehend pausieren und wird anschließend wieder gestartet. MLflow und dessen öffentlicher Lesezugang bleiben verfügbar. Rechner muss eingeschaltet bleiben.
