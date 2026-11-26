# Gepaarte Prüfung des Optimizer-Kandidaten

**Protokoll vor neuen Fits festgeschrieben; Ergebnisse stehen aus.**

[Öffentliches MLflow](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments)
zeigt neue Trainingsläufe im Originalexperiment `article-optimizer-validation-v1`.
Sechs exakt passende bestehende Fits bleiben mit ihren ursprünglichen MLflow-
Kennungen verknüpft. Sie werden nicht als neue Trainingsläufe angelegt.

| Rezept | Clipping | Lernrate | AdamW-Decay | Warmup |
|---|---:|---:|---:|---:|
| Bisheriges Rezept | 1.0 | 1e-4 | 0.01 | 0.1 |
| Neuer Kandidat | 1.0 | 5e-5 | 0 | 0 |
| Clipping-Kontrolle | aus | 5e-5 | 0 | 0 |

Drei feste Seeds × drei gruppierte Train-Folds × drei Rezepte ergeben 27
Auswertungen: sechs exakt übernommene Fits und 21 neue Fits. Pro Fit bleiben
79–81 Artikel zurückgehalten. Pro Seed wird F1 aus den gepoolten Vorhersagen
für alle 240 Train-Artikel berechnet. Die zusätzliche Auswertung ohne den zur
Rezeptwahl verwendeten Fold 0 umfasst die beiden anderen Folds.

Der neue Kandidat gegen das bisherige Rezept misst den gemeinsamen Effekt der
Rezeptänderung. Beide nutzen bereits Clipping 1. Nur der Vergleich mit der
Clipping-Kontrolle isoliert den Clipping-Schalter im neuen Rezept. Die Richtung
der Ergebnisse ist offen; es gibt keine weitere Optimierung anhand dieser Werte.

- [Festes Protokoll einschließlich Gruppenmitgliedschaft, Seeds, Konfigurationen und Hashes](protocol-v1.json).
- [Implementierung und Auswertungsregeln](../../src/studies/optimizer_validation/README.md).
- [Bereits trainiertes Volltrain-Artefakt](../optimizer_candidate/README.md).
- [Ursprüngliche 32-Fit-Faktorstudie](../optimizer_ablation/README.md).

Diese Prüfung ist eine explorative Validierung der Trainingsrezepte auf
wiederverwendeten Train-Daten. Sie liefert keine unabhängige Testgüte für den
bereits gespeicherten 240-Artikel-Adapter. Der frühere Abschlusstest wird nicht
erneut verwendet, der API-Standard nicht automatisch geändert.
