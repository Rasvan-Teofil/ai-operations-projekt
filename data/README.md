# Daten

Trainings- und Testtexte für das Artikel-Sentiment. Eine Zeile ist ein Satz aus einer Finanznachricht, nicht ein ganzer Artikel. Deutsch und Englisch, drei Klassen: `negative`, `neutral`, `positive`.

Die Rohtexte und die aufbereiteten Splits liegen **nicht** im Git. Im Repository bleiben diese Erklärung, die Dummy-Stichprobe unter `sample/` und die Split-Logik in `src/prepare_data.py`.

## Gewählte Datensätze

| | Englisch | Deutsch |
| --- | --- | --- |
| Hugging Face | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) |
| Ausschnitt | `Sentences_75Agree.txt` (mindestens 75 % Annotator-Einigkeit) | nur `language=de` |
| Zeilen vor dem Filter | 3 453 | 5 023 |
| Lizenz | CC BY-NC-SA 3.0 | akademisch / nicht-kommerziell (siehe unten) |
| Labels | schon `negative` / `neutral` / `positive` | schon dieselben drei Namen |

Phrasebank-Zählung in der 75-%-Datei: neutral 2 146, positive 887, negative 420. Deutsch bei Kenpache: neutral 2 425, negative 1 392, positive 1 206.

Nach dem Filter (Stand der Prepare-Lauf, Snapshots `8d3fe0c` und `9d43950`): Englisch 3 440 Sätze (8 zu kurz, 5 doppelte Texte), Deutsch 5 023, zusammen 8 463. Kein Text kam in beiden Quellen vor.

| Split | Zeilen | de neg / neu / pos | en neg / neu / pos |
| --- | --- | --- | --- |
| Train | 5 923 | 974 / 1 697 / 844 | 294 / 1 493 / 621 |
| Val | 1 270 | 209 / 364 / 181 | 63 / 320 / 133 |
| Test | 1 270 | 209 / 364 / 181 | 63 / 320 / 133 |

Dieselben Zahlen stehen lokal in `data/processed/metadata.json`. Ein späterer Prepare-Lauf mit demselben Seed schreibt dieselben Anteile, solange die Upstream-Dateien gleich bleiben.

### Englisch: Financial PhraseBank

Quelle der Sätze: englische Finanznachrichten zu OMX-Helsinki-Firmen (LexisNexis). 16 Personen mit Finanzhintergrund haben aus Investorsicht annotiert: würde der Satz den Aktienkurs eher positiv, negativ oder gar nicht beeinflussen. Was ökonomisch nichts ändert, ist `neutral`. Das ist nicht dieselbe Frage wie „klingt der Artikel freundlich“.

Wir nehmen die Datei mit mindestens 75 % Einigkeit, nicht die kleinere All-Agree-Datei und nicht die lautere 50-%-Datei. `datasets.load_dataset("takala/financial_phrasebank", "sentences_75agree")` funktioniert an diesem Stand nicht: das Dataset-Repo enthält noch `financial_phrasebank.py`, und aktuelle `datasets`-Versionen führen solche Skripte nicht mehr aus. `src/prepare_data.py` lädt deshalb die veröffentlichte Zip `data/FinancialPhraseBank-v1.0.zip` und liest `Sentences_75Agree.txt` (Kodierung Latin-1, Label hinter dem letzten `@`).

Lizenz auf der Dataset-Karte und in `License.txt` der Zip: [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). Akademische, nicht-kommerzielle Nutzung ist erlaubt. Kommerziell: Pekka Malo und Ankur Sinha. Die Sätze werden hier nicht weitergegeben (ShareAlike würde eine Weitergabe der Sammlung betreffen). Zitieren: Malo, Sinha, Korhonen, Wallenius, Takala (2014), *Good debt or bad debt: Detecting semantic orientations in economic texts*, Journal of the Association for Information Science and Technology.

### Deutsch: Kenpache, multilingual financial sentiment

Kurze Sätze aus deutschsprachigen Finanzmedien (Börse.de, ntv, FAZ, Handelsblatt, Süddeutsche, Tagesschau, WiWo und weitere, Feld `source`). Spalten: `sentence`, `label`, `source`, `language`. Labels sind bereits die drei Klassennamen, keine Index-Abbildung nötig.

Die Dataset-Karte sagt zweierlei. Oben: nur akademische, nicht-kommerzielle Forschung, Text-und-Data-Mining nach EU-DSM-Richtlinie Art. 3, Urheberrecht bleibt bei den Verlagen, kommerziell brauche man Lizenzen der Quellen. Unten und im Badge: Apache-2.0. Für diese Arbeit gilt die engere Lesart: akademisch, Texte nicht ins Git, nicht kommerziell nutzen. Das englische Kenpache-Subset (6 887 Zeilen) mischen wir nicht dazu. Phrasebank ist die sauberer lizenzierte und besser annotierte englische Quelle; zwei englische Labelrichtlinien in einem Topf würden den Vergleich verschlechtern.

Stichproben zeigen brauchbare, aber nicht expertengeprüfte Labels: Glossarzeilen liegen bei `neutral`, manche Zeilen sind aus zwei Meldungen zusammengeklebt. Das ist eine echte Schwäche der deutschen Hälfte.

## Labelraum

Beide Quellen liefern schon `negative`, `neutral`, `positive`. Die Fünf-zu-drei-Abbildung in `src/labels.py` (Very Negative + Negative → `negative`, und so weiter) bleibt für die Zero-Shot-Modelle, nicht für diese Dateien.

## Was der Prepare-Schritt tut

```bash
python -m src.prepare_data
```

1. Phrasebank-Zip und Kenpache-CSV über die Hugging-Face-Hub-API laden (Cache, kein Commit).
2. Deutsch filtern. Leere Labels verwerfen. Texte unter 20 Zeichen verwerfen. Doppelte Texte innerhalb einer Quelle und danach über beide Quellen hinweg verwerfen (erster Treffer bleibt).
3. Split, nur hier, Seed `42`, Schicht `label|language`:
   - Test = 15 % (`train_test_split`, `stratify`).
   - Aus dem Rest Val mit Anteil `0.15/0.85`, wieder stratifiziert, derselbe Seed.
   - Train ≈ 70 %.
4. Schreiben: `data/processed/train.csv`, `val.csv`, `test.csv`, `metadata.json`. Spalten: `text`, `label`, `language`, `source`, `dataset`.

Training, Vergleich und Fine-Tuning lesen diese Dateien über `load_work_splits()` und mischen sie nicht neu. Val wird beim Fine-Tuning nicht zum Modellwählen benutzt; die Vergleichstabelle ist der Testsplit. Dieselbe Testmenge für alle fünf Modelle.

`SENTIMENT_DATA=dummy` (Tests und CI) ignoriert `data/processed` und nimmt `data/sample/sentiment_sample.csv` mit dem alten 80/20-Split nur nach Label.

## Grenzen

- Domäne ist Finanznachricht, nicht allgemeine Nachrichten. Phrasebank-`positive` heißt „gut für den Kurs“, nicht „freundlicher Ton“.
- Eine Zeile ist ein Satz. Die API bekommt später ganze Artikel. TF-IDF sieht den vollen Text, Transformer stückeln bei 512 Tokens.
- Es gibt keine Artikel-ID. Sätze desselben Stücks können in Train und Test liegen. Ein Split nach Medium oder Datum ist mit diesen Dateien nicht sauber möglich.
- Neutral ist in beiden Quellen die größte Klasse. Es wird nicht heruntergesampelt. Auswahlmaß ist macro-F1.
- Lizenzen: akademisch / nicht-kommerziell. Keine Rohtexte committen.
- Kenpache-Labels sind unruhiger als Phrasebank. Die englische und die deutsche Hälfte folgen nicht derselben Annotieranleitung.

## Was ins Git gehört

- diese Datei, `data/.gitkeep`, `data/sample/**`
- nicht: `data/processed/`, Zips, Crawls, Volltexte
