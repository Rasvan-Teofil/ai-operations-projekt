# Stand von main vor der Vollartikelintegration

Die vollständigen 168 versionierten Dateien des bisherigen Hauptzweigs liegen
unter `source/`, unverändert in ihren damaligen relativen Verzeichnissen.
Ursprung: Commit `8668255ed7b94c24e16d52bec1d992f9ad507e87`.
Der Hash und die ursprüngliche Position jeder Datei stehen in `manifest.json`.

Dieser Bestand dokumentiert die erste Iteration und die frühe MiniLM-Iteration.
Die aktuelle Anwendung, DE/EN/JA-Vollartikelstudie und ihre Nachweise liegen jetzt
in den regulären Verzeichnissen auf `main`; zum Vorstellen ist kein Branchwechsel nötig.

Der damalige [README](source/README.md), [Pitch](source/docs/executive-pitch/)
und die [Vorlagendateien](source/docs/vorlagen/) bleiben vollständig zugänglich.
Archivierte Startbefehle beziehen sich auf `source/` als damalige Projektwurzel
und die damaligen Abhängigkeiten. Lokale Gewichte, Rohdaten und MLflow-Datenbank
waren nicht in Git und sind deshalb nicht Teil dieses Archivs. Bestehende lokale
Messdateien und Gewichte wurden durch die Integration nicht geändert.

Archivcode wird nicht durch die aktuelle CI trainiert oder ausgeführt.
`python scripts/check_main_integration.py` prüft die archivierten Bytes.
