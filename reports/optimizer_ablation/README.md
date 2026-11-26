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

## Aktueller Ausführungsstand (26. November 2026)

**0/32 Fits abgeschlossen.** Der erste Versuch ist nach 10 von 30
Optimierungsschritten mit CUDA-Fehler abgebrochen; gleichzeitig meldete der
Kernel NVIDIA **Xid 79: GPU has fallen off the bus**. Die GPU ist danach auch
für `nvidia-smi` nicht nutzbar. Das ist kein gemessener Hyperparametereffekt;
die eigentliche Hardware-/Treiber-/PCIe-Ursache ist nicht bestimmt.
[Abbruchnachweis](engineering-interruption-v1.json),
[Statusmomentaufnahme](summary-v1.json), [Ergebnistabelle](results-v1.md).
Der fehlgeschlagene MLflow-Versuch bleibt erhalten und zählt nicht als Fit.

Die lokale Benutzer-Unit `article-optimizer-ablation-resume.service` ist für
die nächste Anmeldung **nach einem tatsächlichen Rechnerneustart** aktiviert.
Sie prüft neue Boot-ID, unverändertes Protokoll, freien GPU-Speicher und CUDA-
Matrixmultiplikation. Danach startet sie die unveränderte Warteschlange einmal
neu. Sie deaktiviert ihren Autostart vor Trainingsbeginn; erneute Fehler lösen
keine Wiederholungsschleife aus. Kein Systemneustart, Treiberupdate oder
GPU-Reset wurde durch den Agenten durchgeführt. Die aktuell defekte GPU kann
hier ohne Administratorzugriff nicht wiederhergestellt werden.

`systemctl --user status article-optimizer-ablation-resume.service` zeigt nach
dem Neustart den Dienststatus. Die Steuerung liegt in
[`scripts/resume_optimizer_ablation.py`](../../scripts/resume_optimizer_ablation.py).
Mit `systemctl --user disable article-optimizer-ablation-resume.service` lässt
sich der einmalige Autostart vor dem Neustart widerrufen.
