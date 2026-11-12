# Quellcode-Navigation

| Rolle der Vorlage | Bestehende Implementierung |
|---|---|
| Daten laden und aufbereiten | `prepare.py`, `news_data.py`, `next_stage_collect.py`, `next_stage_prepare.py` |
| Features und Kontext | `tokenization.py`, `aggregation.py`, `inference.py`, `article.py` |
| Training und Tracking | `train.py`, `tracking.py`, `next_stage_experiment.py` |
| Bewertung | `evaluate.py`, `next_stage_final.py`, `next_stage_test_report.py` |
| Inferenz | `serving.py`, `inference.py`, `local_llm.py` |
| HTTP und UI | `../app/main.py`, `../app/schemas.py`, `../app/static/` |

`news_*` gehört zu den historischen v3–v5-Studien; `next_stage_*` zur
eingefrorenen Vollartikelstudie. Alte Präfixmodule bleiben wegen Import- und
Hashbindungen erhalten. Details und Laufbefehle stehen in den versionierten
Reports und in `../docs/reproduce-full-article.md`.

Neue Studien gehören als Paket nach `studies/<name>/`, nicht als weitere lose
Präfixreihe in dieses Verzeichnis. Gemeinsame Laufzeitmodule nur mit passenden
Vertragsprüfungen ändern; eingefrorene Messungen behalten ihre Code-Identität.
