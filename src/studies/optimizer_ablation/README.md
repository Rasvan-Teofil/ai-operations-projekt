# Optimizer ablation v1

Explorative Nachfolgestudie für Qwen-LoRA: **32 Fits**, vollständiges 2⁴-Design
mit zwei Seeds. Originales Modell, Originalberichte und Serving-Auswahl bleiben
unverändert. Keine erneute Vorhersage auf dem endgültigen Testdatensatz.

| Faktor | Niedrig | Hoch |
|---|---|---|
| Globale L2-Gradientengrenze | aus | 1,0 |
| Spitzenlernrate | 0,00005 | 0,0001 |
| AdamW Weight Decay | 0 | 0,01 |
| Warmup-Anteil | 0 | 0,1 |

AdamW selbst bleibt fest, ebenso Betas (0,9/0,999), Epsilon, drei Epochen,
LoRA-Struktur, Loss, Prompt, Kontext und Akkumulation (16 Artikel). Der dritte
Faktor untersucht AdamWs Regularisierung; daraus folgt kein Vergleich mit
anderen Optimierern. Die lineare Lernratenabnahme bleibt in beiden Warmup-Stufen.

**Daten:** derselbe bestehende gruppengetrennte Train-Fold 0 mit 159 Fit- und
81 Entwicklungsartikeln (DE/EN/JA). Fit-only Klassenhäufigkeiten bestimmen die
Loss-Gewichte. Der komplette 691er Test einschließlich Ausschlüssen bleibt
reserviert. Das Protokoll wird vor dem ersten Lauf mit Eingabe-, Code-,
Basisgewichts- und Split-Hashes eingefroren. Neue Adapter liegen ausschließlich
lokal unter `models/optimizer_ablation/`, keine Basisgewichtskopien.

**Messungen:** DE/EN-Macro-F1-Mittel als primäre Metrik, getrennte Sprachwerte,
Entwicklungs-NLL, gewichteter Trainingsloss, globale Gradientennorm vor Clipping,
Anteil beschnittener Schritte, tatsächliche Lernrate, L2-Norm der Adapteränderung,
Zeit und GPU-Spitzenbelegung. Clipping begrenzt Gradienten; AdamW kann daraus
anders skalierte Parameteränderungen erzeugen. Deshalb wird auch die tatsächliche
Adapteränderung gemessen. Die unbeschnittene Bedingung misst mit unendlicher
Normgrenze; nicht-endliche Gradienten führen in jeder Bedingung zum Abbruch.

**Auswertung:** pro Seed acht ansonsten identische Paare je Faktor, dann Mittel
über beide Seeds. Sechs Zweifaktor-Wechselwirkungen als gemittelte Differenz
von Differenzen; Sprachwerte und NLL analog. Effekte in absoluten F1-Prozentpunkten,
Seed-Effekte separat. Keine Auswahl neuer Produktionsgewichte. Keine P-Werte:
Paare teilen Daten und sind keine unabhängigen Replikate. Nur ein Split, zwei
Seeds, kleine Klassen; keine abgesicherte allgemeine Wirkungsgröße und kein
Beweis eines kausalen Lernmechanismus. Diese Studie folgt zeitlich auf den bereits
bekannten Abschlusstest und ist ausdrücklich explorativ.

Vom Repository aus mit der vorhandenen CUDA-Umgebung, **keine Pakete aktualisieren**:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_ablation initialize
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_ablation launch
.venv/bin/python -m src.studies.optimizer_ablation status
```

Initialisierung nur einmal. `launch` startet einen vom Chat unabhängigen lokalen
Prozess; der Rechner muss weiterlaufen. Exklusive Trainingssperre, frischer
Unterprozess pro Fit, Mindestprüfung freier GPU-Speicher, Fail-fast ohne
ungeprüfte automatische Wiederholung. Der Worker setzt beide Offline-Schalter.
`runtime/optimizer_ablation/worker.log`, `worker.json` und `current-step.json`
zeigen Fortschritt/Fehler. Eine Datei `runtime/optimizer_ablation/STOP` beendet
die Warteschlange nach dem aktuellen Lauf. Ein abgebrochener Fit wird vor einem
Neustart ausdrücklich untersucht; vorhandene Adapter/Evidenz nie überschreiben.

MLflow-Experiment: **`article-optimizer-ablation-v1`**, bestehende lokale SQLite-
Datenbank, neue Laufkennungen. Ergebnisse, Telemetrie, Vorhersagen ohne Rohtexte
und fertige Kontraste landen automatisch unter `reports/optimizer_ablation/`.
Der Worker publiziert nichts selbst auf GitHub. Gemessene Nachweise nach Abschluss
prüfen und auf `main` ergänzen. Bei laufenden Fits sind GitHub-Ergebnisse eine
Momentaufnahme; MLflow und lokale Statusdateien enthalten den neuesten Stand.

## Dokumentierter Infrastrukturabbruch

Der erste Versuch am 26. November 2026 scheiterte nach 10 Trainingsschritten
am zeitgleich protokollierten GPU-Xid-79-Ausfall, vor jeder Entwicklungs-
auswertung. [Aktueller Stand und einmalige Wiederaufnahme](../../../reports/optimizer_ablation/README.md#aktueller-ausführungsstand-26-november-2026).
Der Protokoll-/Quellcode-Hash bleibt unverändert: Die zusätzliche Wiederaufnahme-
steuerung in `scripts/` ändert keine Trainingsbedingung. Ein neuer MLflow-Versuch
ersetzt keinen fehlgeschlagenen Datensatz. Keine Wiederholungen bis zu einem
gewünschten Score; ausschließlich ein explizit dokumentierter Infrastruktur-Retry.
