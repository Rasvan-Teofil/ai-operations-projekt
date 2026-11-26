# Optimizer candidate v1

Eine neue, explorative Qwen-LoRA-Version aus der abgeschlossenen 32-Fit-Ablation.
Frischer Fit der unveränderten Basisgewichte mit frischem Adapter auf allen 240
Train-Artikeln: Clipping 1,0, Lernrate 0,00005, Weight Decay 0, kein Warmup.
Drei Epochen, Akkumulation 16 Artikel, 45 Optimierungsschritte, fester Seed
20261110. Keine Fortsetzung und kein Mittelwert bestehender Fold-Adapter.

Die Auswahl verwendet den mittleren DE/EN-Entwicklungswert über beide Seeds,
nicht den besten einzelnen Seed. Der Gleichstand beim Weight Decay wird vor dem
neuen Fit zugunsten von 0 aufgelöst. Japanisch ist keine nachträgliche Auswahlregel.
Der zweite Seed wird nicht wegen seines höheren Einzelwerts übernommen.

**Einordnung:** Die ursprünglichen 81 Entwicklungsartikel gehören nun zum Fit.
Der Entwicklungswert der Rezeptauswahl ist kein Qualitätswert dieses neuen
Artefakts. Der endgültige Test wird nicht geladen oder erneut ausgewertet.
Synthetische DE/EN/JA-Beispiele prüfen nur die technische Inferenz. Die bestehende
API-Standardauswahl und alle eingefrorenen Studien bleiben unverändert.

```bash
# Einmal: Quellen/Abhängigkeiten prüfen und Protokoll vor dem Fit einfrieren.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_candidate initialize
# Erst Protokoll prüfen und committen, dann genau einen Fit starten.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_candidate launch
.venv/bin/python -m src.studies.optimizer_candidate status
# Nach erfolgreichem Fit, ohne konkurrierende GPU-Arbeit:
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.studies.optimizer_candidate smoke
```

`initialize`, Fit und Smoke schreiben ihre Nachweise jeweils nur einmal.
Vorhandene oder unvollständige Artefakte werden nicht überschrieben oder automatisch
wiederholt. Trainingssperre: `runtime/gpu-training.lock`. Fortschritt lokal unter
`runtime/optimizer_candidate/`; unveränderliche Nachweise unter
`reports/optimizer_candidate/`. Neue Gewichte nur unter `models/optimizer_candidate/v1/`.
MLflow-Experiment `article-optimizer-candidate-v1` mit neuer Laufkennung.

Der Adapter verweist auf dieselben lokalen Basisgewichte und verwendet den
vorhandenen gemeinsamen Inferenzlader. Modellkennung:
`article-qwen-lora-optimizer-v1`; optionaler API-Alias:
`article_qwen_lora_optimizer_v1`. Kein Download oder Nachtraining beim API-Start.
Eine spätere Beförderung zum Standard braucht eine eigene begründete Entscheidung;
diese Erstellung behauptet keinen unabhängig belegten Qualitätsgewinn.
