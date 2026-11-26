# Gradient Clipping: vollständige Gegenläufe

**7/7 neue Fits, 30/30 vollständige Paare.**

[Einordnung und Formulierung für den Prof](interpretation-v1.md) · [Unabhängige Verifikation](verification-v1.json).

Jeder Lauf mit Clipping erhält einen ansonsten identischen Gegenlauf ohne Clipping. Lernrate, AdamW Weight Decay, Warmup, Seed, Datenaufteilung, LoRA und Trainingsdauer bleiben gleich. Vorhandene exakt passende Fits werden wiederverwendet, keine Messung wird erfunden oder überschrieben.

Die erste 32-Fit-Serie enthält bereits 16 Paare. Die spätere CV umfasst zwei Rezepte mit je neun Paaren; vier Paare überlappen die erste Serie. Insgesamt sind es 30 verschiedene Paare / 60 Fits: 53 vorhandene und sieben neue. Fehlend waren nur sieben Ohne-Clipping-Fits des bisherigen Rezepts.

**Vorzeichen überall: mit minus ohne Clipping. Positive Werte sprechen für Clipping.** F1-Unterschiede stehen in absoluten Prozentpunkten.

## Gleiche Rezepte, jeweils mit und ohne Clipping

| Rezept | Lernrate | Weight Decay | Warmup | Clipping-Stufen |
|---|---:|---:|---:|---|
| Bisheriges Rezept | 0.0001 | 0.01 | 0.1 | 1.0 / aus |
| Neues Rezept | 0.00005 | 0 | 0 | 1.0 / aus |

Drei gruppierte Train-Folds und drei feste Seeds (20261110/11/12). Pro Seed werden die drei zurückgehaltenen Folds auf allen 240 Artikeln gepoolt, dann Macro-F1 berechnet. Hauptwert ist das gleichgewichtete DE/EN-Mittel; Japanisch wird separat ausgewiesen. Kein Mittel über die beiden Rezepte als vermeintlich allgemeiner Clipping-Effekt.

## Vollständige gepaarte CV-Ergebnisse

| Rezept | DE/EN mit | DE/EN ohne | Δ DE/EN [95%-Intervall] (pp) | Δ DE | Δ EN | Δ JA |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.7483 | 0.7451 | +0.32 [-2.13; +2.61] | -0.72 | +1.36 | +0.99 |
| candidate | 0.7310 | 0.7371 | -0.61 [-3.43; +1.75] | +0.00 | -1.22 | -1.86 |

| Rezept | Seed | Δ DE/EN (pp) |
|---|---:|---:|
| baseline | 20261110 | +0.00 |
| baseline | 20261111 | -0.23 |
| baseline | 20261112 | +1.19 |
| candidate | 20261110 | -1.49 |
| candidate | 20261111 | -0.09 |
| candidate | 20261112 | -0.25 |

### Ohne den früheren Auswahlfold 0

| Rezept | Δ DE/EN (pp) | Δ DE | Δ EN | Δ JA |
|---|---:|---:|---:|---:|
| baseline | +0.30 | -0.29 | +0.90 | -1.31 |
| candidate | -1.49 | +0.00 | -2.98 | -4.29 |

Die feste Regel für einen positiven Clipping-Nachweis verlangt einen positiven mittleren Hauptwert, positive Hauptwertunterschiede in allen drei Seeds und eine positive untere bedingte Bootstrap-Grenze.

- baseline: Regel nicht erfüllt.
- candidate: Regel nicht erfüllt.

Ein nicht erfüllter positiver Nachweis beweist keine Wirkungslosigkeit oder allgemeine Schädlichkeit. Der API-Standard wird durch diese explorative Studie nicht automatisch geändert.

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
