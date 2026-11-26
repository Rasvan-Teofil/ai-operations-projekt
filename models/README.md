# Lokale Modellartefakte

Gewichte und Adapter sind ignoriert. Aktuelle Modellkarten: `../reports/next_stage/model-cards-v1.md`. Die vier eingefrorenen Kandidaten liegen lokal unter `next_stage/final/`. Keine weiteren Dateien in gehashte Artefaktordner schreiben. Standardauswahl in `serving_selected.json`, vorherige Auswahl gesichert. Neue Studien und Reproduktionen bekommen eigene Verzeichnisse.

Der aus der Optimizer-Ablation abgeleitete Kandidat liegt separat unter
`optimizer_candidate/v1/` (Adapter und Inferenzprofil) und teilt die Basisgewichte
in `local_llm/`. Modellkarte und Nachweise:
[Optimizer-Kandidat](../reports/optimizer_candidate/README.md). Er ersetzt
`serving_selected.json` nicht und hat keinen unabhängigen Qualitätswert.
