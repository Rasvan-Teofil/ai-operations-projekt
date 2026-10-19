# Entwicklung v2: getrennte Ablationen und Kombination

Alle Werte stammen aus denselben drei Quellengruppen-Folds **innerhalb des ursprünglichen Train-Splits**. Die alten Testzahlen werden nicht in diese Kurve gemischt. Val und ursprünglicher Test blieben unberührt.

| Stufe | macro-F1 Mittel ± Fold-SD | DE | EN | Negativ-Recall | ms/Text | MiB | HTML exakt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0_frozen | 0.6306 ± 0.0214 | 0.6566 | 0.6059 | 0.7029 | 2.497 | 465.1 | 33% |
| R0_reference | 0.6724 ± 0.0159 | 0.7053 | 0.6425 | 0.7183 | 2.519 | 465.1 | 33% |
| R1_quality | 0.6739 ± 0.0153 | 0.7076 | 0.6430 | 0.7230 | 2.489 | 465.1 | 33% |
| R2_article | 0.6727 ± 0.0157 | 0.7053 | 0.6430 | 0.7183 | 2.508 | 465.1 | 100% |
| R3_deploy | 0.6729 ± 0.0158 | 0.7062 | 0.6427 | 0.7188 | 2.381 | 240.7 | 33% |
| R4_combined | 0.6740 ± 0.0144 | 0.7065 | 0.6441 | 0.7220 | 2.419 | 240.7 | 100% |

![Gemessene Entwicklung](development_curve.png)

**Vor dem Kollegenvergleich ausgewählt: `R3_deploy`.** Regel: höchstes mittleres Quellen-CV-macro-F1; innerhalb 0,005 gewinnen Latenz, dann Größe.

B0 ist je Fold neu gefittet; R0 ist der ursprüngliche Fine-Tune-Ansatz, ebenfalls ausschließlich auf dem jeweiligen Fit-Fold neu trainiert. Das ursprüngliche Volltrain-Modell wurde nicht auf Trainings-Folds bewertet. R1 ändert nur die Gewichtung von Klassen zu Sprache×Klasse; R2 nutzt R0-Gewichte mit tokengewichteter Chunk-Aggregation und verbesserter HTML-Extraktion; R3 speichert und rechnet R0 in BF16. R4 kombiniert R1, R2 und R3.

- `R0_reference` gegenüber `B0_frozen`: Δmacro-F1 +0.0418, Δms/Text +0.022. Kein automatischer Vorteil durch Kombination unterstellt.
- `R1_quality` gegenüber `R0_reference`: Δmacro-F1 +0.0014, Δms/Text -0.030. Kein automatischer Vorteil durch Kombination unterstellt.
- `R2_article` gegenüber `R0_reference`: Δmacro-F1 +0.0003, Δms/Text -0.011. Kein automatischer Vorteil durch Kombination unterstellt.
- `R3_deploy` gegenüber `R0_reference`: Δmacro-F1 +0.0004, Δms/Text -0.138. Kein automatischer Vorteil durch Kombination unterstellt.
- `R4_combined` gegenüber `R0_reference`: Δmacro-F1 +0.0016, Δms/Text -0.101. Kein automatischer Vorteil durch Kombination unterstellt.

Die 18 synthetischen Artikeltexte und sechs HTML-Fixtures sind kontrollierte technische Prüfungen, keine unabhängig von Menschen gelabelten Nachrichtenartikel. Deshalb trägt deren hohe Trefferquote keinen Anspruch auf reale Artikelqualität. Quellen-CV verhindert dieselbe Quelle in Fit/Dev, aber nicht dasselbe Ereignis über verschiedene Verlage. Drei Folds mit einem festen Seed ergeben Fold-Streuung, keinen Signifikanznachweis. Ein unabhängiger Benchmark folgt erst nach dieser Auswahl.

Der ursprüngliche v1-Test und seine einmalige Auswertung bleiben als historischer Meilenstein erhalten. Zusätzliche Fine-Tuning-Ablationen sind die ausdrücklich angeforderte Erweiterung des ursprünglichen Ein-Modell-Protokolls. Der Harness-Abbruch ist in `harness_incident.json` dokumentiert und als fehlgeschlagener MLflow-Lauf erhalten.

Reproduktion: `python -m src.development run`, danach `summarize`, `release`, `activate`, `python -m src.service_benchmark` und `python -m src.development_report`. Auf diesem Tower sind abgeschlossene Läufe gesperrt; ein frischer Clone hat keine lokalen Daten/Modelle. Vorher Daten und die ursprüngliche eingefrorene Baseline mit `src.prepare` / `src.train frozen` herstellen. Für frische Reproduktionen `SENTIMENT_DEV_REPORTS=reports/reproduction` exportieren; so bleiben eingecheckte historische Messergebnisse unberührt. Die ursprünglichen v1-Test- und Fine-Tuning-Befehle sind für v2 nicht nötig.

MLflow: Experiment `article-sentiment-second-path`, Tags `phase=development-v2`, `stage`, `fold`. Jede Fold-Auswertung enthält Accuracy, Klassen-F1, macro-F1, Sprachwerte, Negativ-Precision/Recall, Latenz, Bytes und Konfusionsmatrix. `decision.json`, `runs.json`, `curve.csv`, `folds.json` und vollständige lokale Modellartefakte machen jede Entscheidung nachvollziehbar.

Methodische Referenzen: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).

## Lokaler Last- und Erholungstest

Deliberately low 10ms bounded queue exposes backpressure; production default is 2s. Synthetic local workload; no URL/network latency or real traffic claim.

| Profil | Steady p50/p95 ms | Burst 200 / 503 | Burst p95 ms | Erholt |
| --- | ---: | ---: | ---: | --- |
| reference | 3.06 / 4.19 | 40 / 0 | 221.59 | True |
| bounded_bf16 | 3.08 / 3.38 | 5 / 35 | 20.70 | True |
