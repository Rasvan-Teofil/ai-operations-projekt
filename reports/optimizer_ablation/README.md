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

## Infrastrukturabbruch vor Neustart (26. November 2026)

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

## Nach Neustart fortgesetzt (26. November 2026)

Der einmalige Autostart hat nach der Anmeldung funktioniert und sich selbst
deaktiviert. **Der erste Fit ist vollständig abgeschlossen und in MLflow als
FINISHED gespeichert; Lauf 2 trainiert.** Momentaufnahme, kein Live-Zähler:
[Wiederaufnahme- und Updateprüfung](post-reboot-check-v1.json). Laufender Stand
über CLI `status` bzw. das lokale MLflow-Experiment. Nachfolgende Ergebnisse
werden automatisch lokal ergänzt und anschließend geprüft auf `main` ergänzt.
Der erste fehlgeschlagene Versuch bleibt separat erhalten; keine Änderung am
32-Fit-Plan, Standardmodell oder endgültigen Test.

Treiber, geladenes Kernelmodul und Bibliotheken haben konsistent Version
595.99.02; dies ist weiterhin Ubuntus Empfehlung für die Karte. Frische isolierte
APT-Indizes bestätigen: Kernel 7.0.0-44 mit passendem NVIDIA-Modul ist verfügbar.
Eine Simulation ergibt 4 Aktualisierungen/9 neue Pakete/0 Entfernungen. Der aktive
Kernel 7.0.0-38 wird durch eine schon vorhandene GRUB-Konfiguration bewusst
bevorzugt. Keine Systempakete oder Boot-Einstellungen wurden verändert.

Die Journale belegen Xid 79/154 (GPU nicht mehr erreichbar, Neustart erforderlich),
aber keine genaue Hardware-/Treiber-/Stromversorgungsursache. Im neuen Boot sind
bis zur Prüfung keine neuen NVIDIA-Xids erfasst. Aktuelle Temperaturen lagen
während des Trainings um 61–63 °C; das beweist keine dauerhafte Stabilität und
rekonstruiert nicht die Temperatur beim ursprünglichen Ausfall.

Vor Systemwartung laufende Fits beenden lassen, aktuellen Zustand und
Rückfall-Boot erhalten, passenden Kernel samt NVIDIA-Modul gemeinsam vorbereiten
und Aktivierung separat prüfen. Ein Treiberzweigwechsel ist kein nachgewiesener
Fix: Ein ähnlicher [Erfahrungsbericht bei NVIDIA](https://forums.developer.nvidia.com/t/three-rtx-5060-ti-gb206-randomly-fallen-off-the-bus-xid-79-154-at-any-load-level-no-pcie-aer-precursor-open-kernel-module-595-99-02/384551)
meldet solche Ausfälle auch nach einem Treiberupdate. Daraus folgt keine
bestätigte Ursache auf diesem Rechner.
