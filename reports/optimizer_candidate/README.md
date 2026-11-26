# Neue Modellversion aus der Optimizer-Ablation

**Fit abgeschlossen und technisch geprüft:** 45 Schritte, 240 Train-Artikel,
199 Offline-Tests bestanden. Direkte DE/EN/JA-Inferenz, echte FastAPI-Anfragen
und Browserauswahl geprüft.

**[Diesen Modelllauf in MLflow öffnen](http://127.0.0.1:5000/#/experiments/3/runs/5230309ca351479980869ef61fc239f0)**
· [Experiment 3](http://127.0.0.1:5000/#/experiments/3)
· [MLflow-Startseite](http://127.0.0.1:5000/).
Diese Links funktionieren bei laufender lokaler Instanz auf dem Tower.
GitHub hostet die MLflow-Datenbank nicht.

Eigenständiger, explorativer Qwen-LoRA-Kandidat mit der Kennung
**`article-qwen-lora-optimizer-v1`**. Er verwendet das aus allen 32 Fits abgeleitete
Rezept und alle 240 bestehenden Train-Artikel. Das bisherige API-Standardmodell
und die ursprüngliche Teststudie bleiben erhalten.

- [Modellkarte und Grenzen](model-card-v1.md).
- [Vor dem Fit eingefrorenes Protokoll](protocol-v1.json).
- [Trainingsresultat und MLflow-Kennung](result-v1.json).
- [Telemetrie aller 45 Optimierungsschritte](telemetry-v1.json).
- [Technische Inferenzprüfung mit synthetischen Beispielen](smoke-v1.json).
- [Abschlussprüfung](verification-v1.json).
- [Quellcode, Bedienung und Reproduktion](../../src/studies/optimizer_candidate/README.md).
- [Zugrunde liegende 32-Fit-Ablation](../optimizer_ablation/interpretation-v1.md).

Der bisherige Entwicklungswert von 0,7742 bewertet die Rezeptauswahl im alten
159/81-Split. Er ist **kein Qualitätswert dieses neuen Artefakts**. Alle bisherigen
Entwicklungsartikel sind jetzt Trainingsartikel. Eine neue unabhängige Evaluation
steht aus; der alte endgültige Test wird dafür nicht wiederverwendet.
