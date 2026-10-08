# Daten

Hier liegen die Texte für das Artikel-Sentiment.

## Was schon im Repository ist

`sample/sentiment_sample.csv` ist eine **kleine Dummy-Datei**: selbst geschriebene deutsche und englische Sätze, gelabelt mit `negative`, `neutral`, `positive`, Spalte `origin=dummy`. Sie hält die Pipeline am Laufen. Sie ist kein Nachrichtendatensatz und darf nicht als Evaluationsergebnis verkauft werden.

## Was noch gewählt werden muss (Meilenstein 1)

Ein echter gelabelter Nachrichten-Datensatz auf Deutsch und Englisch. Offen sind Quelle, Stand, Lizenz und ob die Labels schon drei Klassen sind oder aus fünf Stufen abgebildet werden müssen. Dieselbe Abbildung wie im Code: Very Negative und Negative werden `negative`, Neutral bleibt `neutral`, Positive und Very Positive werden `positive`.

Die Split-Logik steht ausschließlich in `src/data.py`. Heute ist das ein stratifizierter 80/20-Zufallssplit mit festem Seed, weil die Dummy-Sätze unabhängig sind. Für echte Artikel prüfen, ob Medium, Ereignis oder Datum zusammenbleiben müssen. Sonst steht derselbe Vorgang in Train und Test.

## Was ins Git gehört

- diese Erklärung, `.gitkeep` und die kleine Dummy-Datei unter `sample/`
- Herkunft, Stand und Lizenz, sobald der echte Datensatz feststeht

## Was lokal bleibt

Alles andere unter `data/` ist in `.gitignore` ausgeschlossen. Große Rohdaten, Crawls und Exporte nicht committen.

Nicht ablegen:

- personenbezogene oder nicht freigegebene Artikel im Volltext, wenn die Lizenz das verbietet
- Zugangsdaten
- Modellgewichte (die liegen unter `models/` und sind ebenfalls ignoriert)

Wenn die Datei zu groß für Git ist: Speicherort und Prüfsumme hier dokumentieren. Der Split bleibt in `src/data.py`.
