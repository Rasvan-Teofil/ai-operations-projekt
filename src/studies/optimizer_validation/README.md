# Gepaarte Optimizer-Validierung

Drei feste Rezepte, drei bestehende gruppierte Train-Folds und drei feste Seeds:
27 Fold-/Seed-Auswertungen, davon sechs exakt passende vorhandene Fits und 21
neue Fits. Jeder neue Fit beginnt mit frischem ursprünglichem Basismodell und
frischem LoRA, nutzt drei Epochen und wird nur auf seinem zurückgehaltenen Fold
ausgewertet. Keine neue Vorhersage auf dem bisherigen Abschlusstest.

- `baseline`: Clipping 1, LR 1e-4, AdamW-Decay 0.01, Warmup 0.1.
- `candidate`: Clipping 1, LR 5e-5, Decay 0, Warmup 0.
- `candidate_no_clip`: gleiches neues Rezept ohne Clipping.

Der Vergleich candidate/baseline bewertet das gesamte geänderte Rezept. Der
Vergleich candidate/candidate_no_clip isoliert den Clipping-Schalter. Baseline
und neuer Kandidat verwenden beide bereits Clipping 1.

```bash
.venv/bin/python -m src.studies.optimizer_validation initialize
# Quellcode und Protokoll vor neuen Fits committen; bestehende GPU-Nutzer prüfen.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_validation launch
.venv/bin/python -m src.studies.optimizer_validation collect
```

`initialize` ist einmalig und erzeugt ein unveränderliches Protokoll. Der Worker
arbeitet sequenziell mit exklusiver Trainingssperre und frischem Kindprozess
pro Fit. Vorhandene Resultate werden geprüft, nicht überschrieben. Ergebnisse,
Vorhersagen und Telemetrie liegen unter `reports/optimizer_validation/`, lokale
Adapter und Prozesslogs bleiben außerhalb Git. Wenn die Inferenz-API für diese
Studie kontrolliert pausiert wurde, startet der Worker sie abschließend neu.
MLflow und sein öffentlicher lesender Zugang benötigen die GPU nicht.

Pro Seed werden die drei Fold-Vorhersagen auf allen 240 Artikeln gepoolt, bevor
F1 berechnet wird. Zusätzlich werden die beiden Folds außerhalb des zur
Rezeptwahl verwendeten Folds 0 separat gepoolt. Gepaarte Gruppen-Bootstrap-
Intervalle mit 2000 festen Ziehungen beschreiben die Unsicherheit bedingt auf
diesen Artikeln, Fits, Seeds und bereits ausgewählten Rezepten. Sie korrigieren
keine Auswahlverzerrung und ersetzen keinen unabhängigen Test. Keine weitere
Hyperparametersuche und keine automatische Änderung des API-Standards.
