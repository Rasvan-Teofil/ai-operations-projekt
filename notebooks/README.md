# Lesende Exploration

`01_full_article_results.ipynb` zeigt gespeicherte Messwerte des gemeinsamen Tests. Es lädt keine Rohtexte oder Modelle, startet keine Inferenz und trainiert nichts. Die produktive Logik liegt unter `src/`. Ausgaben vor einem Commit leeren; Messnachweise liegen versioniert unter `reports/`.

`00_data_exploration.ipynb` explores the 240 train and 691 test articles from versioned metadata only (labels, languages, dates, sources, event groups, folds, annotation agreement, test exclusions, model errors); the rendered HTML and chart PNGs are in `reports/eda/`.
