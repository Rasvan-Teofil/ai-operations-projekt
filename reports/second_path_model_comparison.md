# Gemessener Vergleich (zweiter Durchlauf)

Die Tabelle ist der historische v1-Test aus Branch `second-path` und ersetzt nicht [`model_comparison.md`](model_comparison.md) des ersten Durchlaufs. Die Zahlen darunter sind unverändert übernommen.

# Gemessener Vergleich

Eigener gemeinsamer Test: 1.787 DE/EN-Sätze, Seed 20261018. Einmalige abschließende Bewertung am 18.10.2026 auf RTX 5060 Ti (CUDA 12.8). Keine Vergleichbarkeit mit dem Testsplit auf main.

| Modell | Accuracy | macro-F1 | F1 negativ | F1 neutral | F1 positiv | ms/Text | MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen | 0.6676 | 0.6656 | 0.6822 | 0.6813 | 0.6333 | 2.511 | 465.14 |
| finetuned | 0.7230 | 0.7187 | 0.7182 | 0.7416 | 0.6962 | 2.500 | 465.14 |
| centroid | 0.6402 | 0.6410 | 0.6758 | 0.6372 | 0.6100 | 2.563 | 465.14 |
| majority | 0.4841 | 0.2174 | 0.0000 | 0.6523 | 0.0000 | 0.005 | 0.00 |

**Ausgeliefert wird das eigene GPU-Fine-Tuning statt der eingefrorenen Baseline, weil es auf dem gemeinsamen Test macro-F1 von 0,6656 auf 0,7187 hebt (+0,0531) und dabei praktisch gleiche Artefaktgröße und Inferenzzeit hat.**

Der Abstand liegt über der vorab festgelegten Nähegrenze 0,005; Latenz entscheidet hier also nicht. Beide Encoderartefakte brauchen rund 465 MiB, die zusätzliche Fine-Tuning-Zeit beträgt 40,3 s nach Laden der Initialisierung (Baseline-Training 11,3 s ohne Download/Laden). Die Mehrheitskontrolle ist kleiner und schneller, übersieht aber sämtliche negativen und positiven Texte. Centroid ist ähnlich teuer in der Inferenz und ungenauer.

Fine-Tune pro Sprache: DE macro-F1 0,7541 (Baseline 0,7008), EN 0,6913 (Baseline 0,6383). Englisch bleibt die schwächere Sprache. Das sind Punktwerte eines eigenen Tests, kein Vergleich mit main und kein Signifikanznachweis.

## Run-IDs

- `frozen`: `6f942a83f7c949b2a3a70d9036b2d12d`
- `finetuned`: `5b1864e1e45e49cc99a74c33a39b5660`
- `centroid`: `d130c7f51fe749a5a17326f5503f4d78`
- `majority`: `8a7d115af9d54edc83a2d029e034df35`

Experiment: `article-sentiment-second-path`, lokale SQLite-Datenbank `mlflow.db`. (Stand Branch `second-path`. Auf `main` liegt der Store unter `./mlruns`; die Zahlen holt `python -m src.import_recorded`.) Konfusionsmatrizen: Zeilen wahr, Spalten vorhergesagt, Klassenfolge negative/neutral/positive. Latenz: 20 unabhängige Warmups, Batch 1, Tokenisierung und Transfer enthalten, CUDA-Synchronisierung, kein Laden oder Netzwerk. Alle vollständigen Modell- und Tokenizerdateien zählen zur Größe.

Die Val-Datei wurde nur zur Integritätsprüfung gehasht. Es gab keine Val-Metriken oder Checkpoint-Auswahl. Fine-Tune-Endpunkt: Epoche 3, je 8.335 Zeilen; erste und letzte Encoderschicht zeigen gegenüber der eingefrorenen Baseline geänderte Gewichte (siehe training_evidence.json).

Ein Warmup-Abbruch ohne Testvorhersagen ist separat in preflight_incident.json dokumentiert; daraus entstanden keine Scores und keine Modellanpassung. Der reparierte Inferenzpfad wurde vor dem Test nur an selbst geschriebenen Beispieltexten geprüft. Nach Bekanntwerden der Testzahlen wurden Modelle und Inferenz nicht mehr verändert.
