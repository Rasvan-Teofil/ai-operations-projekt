# Gradient-Clipping-Gegenläufe

Ergänzt ausschließlich die sieben fehlenden Ohne-Clipping-Partner der bisherigen
Optimizer-Studien. Die 32er Faktorserie enthält bereits alle 16 Paare; beim neuen
Rezept liegen alle neun CV-Paare vor. Beim bisherigen Rezept existieren zwei von
neun Ohne-Clipping-Fits. Vier Paare gehören zu beiden Ansichten. Deshalb insgesamt
30 verschiedene Paare / 60 Fits, davon 53 vorhanden und sieben neu.

Keine neue Suche, kein Wechsel der Lernrate, von AdamW Weight Decay oder Warmup
innerhalb eines Paars. Neue Fits übernehmen den Trainingsablauf der eingefrorenen
Validierung unverändert: frische Basis und LoRA, gleiche Seeds, Reihenfolgen,
Fit-only Loss-Gewichte, drei Epochen, 16 Artikel pro Akkumulationsschritt.
`clip_grad_norm_(..., inf, error_if_nonfinite=True)` misst die Norm ohne endliche
Gradienten zu skalieren; nicht-endliche Gradienten brechen den Lauf ab.

```bash
.venv/bin/python -m src.studies.clipping_control initialize
# Code und Protokoll vor dem ersten Fit committen, GPU-Nutzer prüfen.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.clipping_control launch
.venv/bin/python -m src.studies.clipping_control collect
.venv/bin/python scripts/check_clipping_control.py --partial
```

Initialisierung nur einmal, keine automatischen Wiederholungen unvollständiger
Fits. Der Worker nutzt die gemeinsame GPU-Sperre und frische Kindprozesse. Ein
`runtime/clipping_control/STOP` beendet nach dem aktuellen Fit. Für kontrollierte
API-Pause kann nach Prozessprüfung eine ignorierte `api-pause-request.json` mit
der exakten PID hinterlegt werden; der Worker prüft deren Kommandozeile, beendet
nur diese API regulär und startet sie abschließend wieder. MLflow und Tunnel
werden nicht beendet. Kein Paketupdate auf dem Tower.

Ergebnisse unter `reports/clipping_control/`, Adapter und Logs nur lokal. Neue
MLflow-Runs verlinken ihren alten Clipping-Partner und enthalten nach Abschluss
die gesamte Paarliste und Auswertung als Artefakt. Der Worker prüft alle 60 Fits
und reproduziert die Bootstrap-Intervalle unabhängig über sklearn-Gewichte.
Er publiziert nicht selbst auf GitHub; dort ist der dokumentierte Stand eine
Momentaufnahme. Laufender Fortschritt steht in MLflow und `runtime/clipping_control/`.

CV-Auswertung getrennt pro Rezept; Folds vor F1 poolen, anschließend Seeds
mitteln. Vorzeichen immer **mit minus ohne Clipping**. Die 16 ursprünglichen
Faktorpaare separat ausweisen, keine Doppelzählung der vier überlappenden Paare.
Keine erneute Inferenz auf dem Abschlusstest, keine Serving-Promotion. Planung
erfolgt nach bekannten Ergebnissen und ist explorativ, keine unabhängige
Bestätigung. Volltrain-Adapter ohne zurückgehaltene Daten sind keine CV-Paare.
