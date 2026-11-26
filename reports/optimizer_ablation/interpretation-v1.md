# Abschlussauswertung: 32 Optimizer-Fits

**Abgeschlossen am 26. November 2026, 12:52 Uhr MEZ: 32/32 Fits.**
16 vollständig gekreuzte Bedingungen × zwei Seeds; drei Epochen und 30 Optimierungsschritte je Fit.
159 Fit- und 81 Entwicklungsartikel aus dem bestehenden gruppengetrennten Train-Fold 0.
Kein erneuter endgültiger Test, keine Änderung am API-Standardmodell.

## Gemessener Einfluss der vier Faktoren

Differenz der hohen gegenüber der niedrigen Stufe, gemittelt über ansonsten identische
Bedingungen und beide Seeds. Einheit: **absolute Macro-F1-Prozentpunkte**, keine relative Verbesserung.
DE/EN ist das vorab definierte Mittel der beiden Sprachwerte; JA bleibt separat.

| Veränderung | DE/EN | Seed 1 DE/EN | Seed 2 DE/EN | DE | EN | JA |
|---|---:|---:|---:|---:|---:|---:|
| Clipping aus → globale L2-Grenze 1,0 | +1.75 | +0.65 | +2.85 | -0.10 | +3.60 | +3.76 |
| Lernrate 0,00005 → 0,0001 | +0.49 | +0.65 | +0.33 | -0.23 | +1.21 | -0.22 |
| AdamW Weight Decay 0 → 0,01 | -0.96 | -0.65 | -1.26 | -0.16 | -1.75 | -0.85 |
| Warmup 0 → 10 % (3 Schritte) | -1.15 | -0.65 | -1.64 | +0.42 | -2.71 | +0.57 |

## Was Clipping hier bewirkt und was die Messung erklärt

Clipping bei 1,0 verbessert den DE/EN-Wert im Mittel um 1,75 Punkte:
8 der 16 gepaarten Vergleiche verbessern sich, 7 sind gleich und einer verschlechtert sich.
Der deutsche Sprachwert ist im Mittel praktisch unverändert; der Vorteil im Hauptwert
kommt aus Englisch. Japanisch verbessert sich separat im Mittel um 3,76 Punkte.
Beide Seeds haben einen positiven mittleren DE/EN-Effekt, aber sehr unterschiedliche
Größen (+0,65 bzw. +2,85 Punkte). Im ersten Seed stammen die F1-Unterschiede
des Hauptwerts aus nur einer der acht Clipping-Paarungen.

Die Grenze greift in den Clipping-Läufen bei durchschnittlich 95,4 % der Schritte.
Sie ist hier also ein regelmäßig aktiver Eingriff in die Optimierung.
Die globale Gradientennorm wird vor dem AdamW-Schritt begrenzt. Daraus folgt keine
gleich große Begrenzung der tatsächlichen Parameteränderung: Über diese Bedingungen
liegt die mittlere Adapter-Update-L2-Norm mit Clipping bei 0,03269 und ohne bei 0,03225.
AdamWs Momentenschätzung und adaptive Skalierung wirken nach dem Clipping.
Die Telemetrie belegt daher keinen allgemeinen Mechanismus „kleinere Updates ergeben bessere F1“.

Auch der Entwicklungs-NLL stützt keine einheitliche Erklärung: Clipping senkt ihn
im ersten Seed um 0,1375, erhöht ihn im zweiten um 0,0090; gemittelt sinkt er um 0,0642.
F1 hängt von diskreten Klassenentscheidungen ab, NLL von den Zielklassenwahrscheinlichkeiten.
Dass die Messgrößen unterschiedlich reagieren, lässt sich hier direkt beobachten.
Warum genau einzelne Artikel ihre Klasse wechseln, ist damit noch nicht kausal erklärt.

## Lernrate, Weight Decay und Warmup

Die höhere Lernrate verbessert den Hauptwert im Mittel nur um 0,49 Punkte
(4 Paarungen besser, 8 gleich, 4 schlechter). Den Entwicklungs-NLL senkt sie dagegen
in allen 16 Paarungen, im Mittel um 0,4615. Bessere Zielklassenwahrscheinlichkeiten
führen somit nicht bei jeder Bedingung zu besseren Klassenlabels.
Die Wechselwirkung mit Clipping ist relevant: Bei Lernrate 0,00005 beträgt der
mittlere Clipping-Vorteil +2,95 Punkte, bei 0,0001 nur +0,56 Punkte.
Der Unterschied dieser Clipping-Effekte beträgt −2,39 Punkte.

Weight Decay 0,01 bringt gegenüber 0 im Hauptwert keinen Vorteil:
12 Paarungen sind gleich, 4 schlechter, im Mittel −0,96 Punkte.
Dies ist ein Vergleich zweier Regularisierungsstärken innerhalb von AdamW;
die Studie enthält keinen Vergleich von AdamW mit SGD, Adam oder anderen Optimierern.
Die Zahl der Schritte ist klein; eine pauschale Aussage gegen Weight Decay
bei längeren Trainings oder anderen Datensätzen wäre nicht gedeckt.

10 % Warmup entspricht hier nur drei von 30 Schritten. Im Hauptwert ist eine
Paarung besser, acht gleich und sieben schlechter (Mittel −1,15 Punkte).
Der Warmup verändert bei gleicher Gesamtschrittzahl auch die Lernratenfolge.
Die Vermutung, dass dieser kurze Fit vom unmittelbaren Lernen profitiert, ist
eine plausible Hypothese; ein gesonderter Test dieser Erklärung wurde nicht durchgeführt.
Japanisch reagiert im Mittel leicht positiv (+0,57 Punkte), Deutsch ebenfalls (+0,42).

## Beste Entwicklungsbedingungen und praktische Folgerung

Den höchsten DE/EN-Mittelwert über beide Seeds teilen sich zwei Bedingungen:
Clipping 1,0, Lernrate 0,00005, kein Warmup und Weight Decay entweder 0 oder 0,01.
Beide erreichen **0,7742** (Seed-Werte 0,7634 und 0,7850). Die vorab festgelegte
Referenzbedingung erreicht im Mittel 0,7570. Diese Rangfolge ist nachträglich
auf denselben Entwicklungsartikeln ermittelt und kein unabhängiger Wirksamkeitsnachweis.
Beim Gleichstand hat die Variante ohne Weight Decay den höheren japanischen Mittelwert
(0,7736 gegenüber 0,7442); JA war keine nachträgliche Auswahlregel.

Für die Präsentation ist die belastbare Formulierung:
**„In unserem explorativen 32-Fit-Versuch ist Clipping bei 1,0 im Mittel hilfreich.
Die höhere Lernrate verbessert vor allem den NLL. Warmup und Weight Decay liefern
bei diesem kurzen Training keinen mittleren DE/EN-F1-Vorteil.“**
Die Messung gibt keinen Anlass, Clipping allein aufgrund der Aufgabenart zu ersetzen.
Sie rechtfertigt ebenso keine automatische Umstellung des veröffentlichten Modells.

## Grenzen und nachvollziehbare Nachweise

Nur ein Train-interner Split und zwei Seeds; dieselben 81 Entwicklungsartikel werden
in allen Bedingungen verwendet (DE 26, EN 24, JA 31; darunter nur drei englische neutrale Fälle).
Einzelne geänderte Vorhersagen können den Macro-F1 deutlich bewegen. Im ersten Seed
haben 15 von 16 Bedingungen denselben DE/EN-Wert trotz unterschiedlicher Telemetrie
und Wahrscheinlichkeiten. Paarungen sind keine unabhängigen Stichproben;
keine Signifikanzbehauptung, kein allgemeines Parameteroptimum und kein bewiesener Lernmechanismus.
Die Studie wurde nach Kenntnis des früheren Abschlusstests geplant und ist ausdrücklich explorativ.
Die hier gemessenen Entwicklungswerte sind mit den früheren Testwerten nicht direkt vergleichbar.

- [Eingefrorenes Protokoll](protocol-v1.json) und [Versuchsplan/Bedienung](../../src/studies/optimizer_ablation/README.md).
- [Alle 32 Einzelwerte und Effekttabelle](results-v1.md).
- [Maschinenlesbare Effekte, Paarungen und alle sechs Wechselwirkungen je Metrik](summary-v1.json).
- [Prüfnachweis](validation-v1.json): 32 fertige MLflow-Kennungen, Hashes,
  neu berechnete Klassifikationsmetriken und unabhängig geprüfte Effekte.
- `runs/`, `predictions/` und `telemetry/`: gespeicherte Nachweise ohne Artikelrohtexte.
- Lokal: MLflow-Experiment `article-optimizer-ablation-v1`, Experiment-ID 2;
  alle 32 zugehörigen Ergebniskennungen sind `FINISHED`.
