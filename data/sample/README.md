# Dummy-Stichprobe

`sentiment_sample.csv` sind **handgeschriebene Beispielsätze**, keine Nachrichten und kein Benchmark. Spalte `origin` ist überall `dummy`.

Zweck: Split, TF-IDF-Training, API und Tests laufen, bevor ein echter Datensatz gewählt ist. Pro Klasse sechs kurze Sätze, Deutsch und Englisch gemischt, Labels `negative`, `neutral`, `positive`.

Diese Datei nicht mit echten Artikeln auffüllen. Den echten gelabelten Nachrichten-Datensatz (DE/EN) für Meilenstein 1 separat ablegen, in `data/README.md` Herkunft und Lizenz notieren und `src/data.py` darauf umstellen. Der Test `test_sample_is_marked_as_dummy` erwartet bis dahin `origin=dummy`.
