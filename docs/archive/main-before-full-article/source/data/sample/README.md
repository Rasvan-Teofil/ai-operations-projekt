# Dummy sample

`sentiment_sample.csv` contains **hand-written example sentences**, not news and not a benchmark. The `origin` column is `dummy` everywhere.

Purpose: tests and CI (`SENTIMENT_DATA=dummy`). Six short sentences per class, German and English, labels `negative`, `neutral`, `positive`. The real run uses the prepared splits from `python -m src.prepare_data`, see `data/README.md`.

Do not fill this file with real articles. The test `test_sample_is_marked_as_dummy` expects `origin=dummy`.

`opinion_supplement.csv` is a second handwritten file (`origin=handwritten`). It is not the CI sample. `load_work_splits` appends it to the training split only, for both `SENTIMENT_DATA=dummy` and the prepared financial splits. Validation and test stay as they are. The sentences are original text written for this project, not a redistributed corpus.
