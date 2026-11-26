# Vorführungsweg

Der vollständige aktuelle Stand liegt auf `main`; kein Branchwechsel für die Vorführung.
Japanisch ist in [der Sprachübersicht](languages/japanese.md) ausdrücklich dokumentiert.

1. README öffnen: Problem Medienbeobachtung, vollständiger Text oder eine URL,
   drei Klassen, native DE/EN und getrennt ausgewiesenes JA.
2. [Struktur](repository-structure.md) zeigen: Vorlesungsvorlage, Trennung von
   `src/`, HTTP-Vertrag in `app/schemas.py`, lokalen Daten/Modellen und Nachweisen.
3. Lokale Anwendung öffnen, einen eigenen kurzen Nachrichtentext eingeben.
   Modellversion und Scores zeigen; Scores sind keine kalibrierten Wahrscheinlichkeiten.
4. Compare: gleicher 691er Test für vier Kandidaten, DE273/EN118/JA300,
   240 Train. Qwen LoRA 0,8218, Zero 0,7796, MiniLM 0,5928, Frozen 0,5150
   im gleichgewichteten DE/EN-macro-F1. Auswahl erfolgte vor Test.
5. Gegenleistung benennen: MiniLM rund 3,66 ms/Artikel und 241 MiB;
   Qwen LoRA rund 184,26 ms und 7699 MiB einschließlich Basis.
6. Development: 27 Auswertungen innerhalb Train, keine Sättigung der kleinen
   Mengenkurve. 240 ist kein Datenoptimum; keine künstliche Testvergrößerung.
7. [Originalquellen und Labels](data/README.md) sowie den
   [projektseitigen Review](data/project-review.md) zeigen. Die ursprüngliche
   KI-unterstützte Erstellung ist als Herkunft dokumentiert. Eingeschränktes
   Archiv und mögliche Vortrainingsüberschneidung bleiben Grenzen.
   95%-Intervalle bedeuten nicht 95% Genauigkeit.
8. Reproduce und Data & licences: feste Hashes, neue MLflow-Läufe, genaue
   Quellenattribution; Rohtexte und Gewichte fehlen bewusst im Git-Repository.
9. [Gradient-Clipping-Ablation](../reports/clipping_control/interpretation-v1.md):
   gleicher Seed, gleicher Split, gleiches Rezept, nur Clipping 1,0 oder aus.
   30 verschiedene Paare; innerhalb der CV drei Folds und drei Seeds je Rezept.
   Mit minus ohne Clipping im DE/EN-Hauptwert: bisheriges Rezept +0,32 Punkte
   [−2,13; +2,61], neues Rezept −0,61 [−3,43; +1,75]. Beide bedingten Intervalle
   umfassen null. Kein robuster Gütevorteil nachgewiesen, kein Modellwechsel.
   Die Werte sind explorative Train-interne CV, keine Wiederholung des 691er Tests.

Die früheren Studien bleiben im ausklappbaren Archiv. Ihre unterschiedlichen
Testbestände nicht als Fortschrittskurve mit dem Vollartikeltest vergleichen.
Der Dienst ist lokal auf `127.0.0.1:8000`, kein öffentliches Deployment.

## Nächster Wochenblock: Qwen-Code auf Englisch

Für die Erklärung des besten Modells und der Qwen-Varianten dem
[englischen Code-Walkthrough](qwen-code-walkthrough.md) folgen. Er trennt
Trainingspfad, Kandidatenauswahl, Inferenz und FastAPI-Anbindung und nennt die
Funktionen, die tatsächlich gezeigt werden sollten.
