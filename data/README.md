# Daten

Hier liegen später die Rohdaten des gewählten Geschäftsproblems.

Der aktuelle Code trainiert noch **nicht** aus dieser Dateiablage. Bis der echte Datensatz feststeht, lädt `src/data.py` einen klar markierten Platzhalter: den Iris-Datensatz aus scikit-learn. Sobald ihr euch festgelegt habt, kommt die Datei hierher und der Lader wird umgestellt. Notebooks und Training sollen dieselbe Ladefunktion benutzen.

## Was ins Git gehört

- diese Erklärung und `.gitkeep`, damit der Ordner im Repository sichtbar bleibt
- kleine, unkritische Beispieldateien, wenn das Team das ausdrücklich will
- eine kurze Herkunftsangabe (Quelle, Stand, Lizenz), sobald der Datensatz gewählt ist

## Was lokal bleibt

Große Rohdaten sind in `.gitignore` ausgeschlossen (`data/*`). Sie werden nicht committed, damit das Repository schlank und reproduzierbar bleibt.

Nicht ablegen:

- personenbezogene oder vertrauliche Daten
- Zugangsdaten, Tokens, Exporte aus internen Systemen ohne Freigabe
- Modellartefakte (die gehören nach dem Training nach `models/` und sind ebenfalls ignoriert)

Wenn eine Datei zu groß für Git ist, aber das Team sie teilen muss: Speicherort und Prüfsumme hier dokumentieren (zum Beispiel Cloud-Laufwerk oder später DVC/Git LFS). Die Split-Logik bleibt davon unabhängig in `src/data.py`.
