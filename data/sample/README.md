# Dummy-Stichprobe

`sentiment_sample.csv` sind **handgeschriebene Beispielsätze**, keine Nachrichten und kein Benchmark. Spalte `origin` ist überall `dummy`.

Zweck: Tests und CI (`SENTIMENT_DATA=dummy`). Pro Klasse sechs kurze Sätze, Deutsch und Englisch, Labels `negative`, `neutral`, `positive`. Der echte Lauf nimmt die aufbereiteten Splits aus `python -m src.prepare_data`, siehe `data/README.md`.

Diese Datei nicht mit echten Artikeln auffüllen. Der Test `test_sample_is_marked_as_dummy` erwartet `origin=dummy`.
