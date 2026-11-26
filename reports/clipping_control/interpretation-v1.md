# Was wir mit dem Clipping-Ergebnis machen

Die Clipping-Frage ist für den untersuchten Versuchsplan beantwortet: **kein
robuster positiver Qualitätsnachweis und kein Anlass, deshalb das bisherige
Standardmodell zu ersetzen.** Alle sieben zusätzlichen Fits sind erfolgreich;
die vollständige Zuordnung umfasst 30 verschiedene Paare / 60 Fits. Davon sind
53 bereits vorhandene, exakt passende Fits. Vier Paare gehören zu beiden
Auswertungsansichten und werden nicht als zusätzliche unabhängige Evidenz gezählt.

## Vergleich für die Präsentation

Innerhalb eines Paars bleiben Fit-/Entwicklungsdaten, Seed, Lernrate, AdamW
Weight Decay, Warmup, LoRA und drei Epochen identisch. Nur die globale
Clipping-Grenze wechselt zwischen 1,0 und aus. Die folgende CV vergleicht beide
Rezepte getrennt über drei gruppierte Folds und drei feste Seeds. Zunächst alle
240 zurückgehaltenen Vorhersagen pro Seed poolen, dann F1 berechnen und die
drei Seed-Werte mitteln. Hauptwert ist das gleichgewichtete DE/EN-Macro-F1-Mittel.

| Rezept | DE/EN mit Clipping | DE/EN ohne Clipping | Mit minus ohne (F1-pp) | Bedingtes 95%-Intervall (pp) |
|---|---:|---:|---:|---:|
| Bisheriges Rezept | 0,7483 | 0,7451 | +0,32 | [−2,13; +2,61] |
| Neues Rezept | 0,7310 | 0,7371 | −0,61 | [−3,43; +1,75] |

Beim bisherigen Rezept lauten die drei Seed-Unterschiede +0,00 / −0,23 / +1,19
Punkte. Der kleine positive Mittelwert ist damit nicht durchgehend positiv.
Beim neuen Rezept sind alle drei Unterschiede negativ, aber auch dort umfasst
das bedingte Intervall null. Die festgelegte Regel für einen positiven
Clipping-Nachweis ist in beiden Rezepten nicht erfüllt. Das ist kein Beweis
für Wirkungslosigkeit, Gleichwertigkeit oder allgemeine Schädlichkeit.

Sprachwerte nicht verschweigen: beim bisherigen Rezept DE −0,72, EN +1,36,
JA +0,99 Punkte; beim neuen DE +0,00, EN −1,22, JA −1,86. Auch diese Punktwerte
sind keine abgesicherten allgemeinen Spracheffekte. Die [vollständige Auswertung](README.md)
enthält alle Sprachintervalle in [summary-v1.json](summary-v1.json), jeden Seed,
die Kontrolle ohne Auswahlfold 0 und die ursprünglichen 16 Faktorpaare.

Für den Prof lässt sich das so formulieren:

> Wir haben Gradient Clipping durch ansonsten identische Trainingsläufe mit und
> ohne Clipping isoliert untersucht. Beim bisherigen Rezept beträgt der mittlere
> Unterschied +0,32, beim neuen −0,61 F1-Prozentpunkte. Beide bedingten
> Unsicherheitsintervalle umfassen null. Auf unseren Daten ist daher kein
> verlässlicher Qualitätsvorteil nachgewiesen; wir behalten das bisherige
> Standardmodell bei.

## Konsequenz für das Projekt

1. **Standardmodell behalten.** Der bisher ausgewählte Adapter und seine
   Clipping-Einstellung 1,0 bleiben bestehen. Das ist die Entscheidung, den
   vorhandenen Stand zu erhalten, kein Beweis, dass Clipping dafür nötig ist.
   Aus den kleinen explorativen Unterschieden folgt kein neuer Modellwechsel.
2. **Ablation als abgeschlossenes Experiment zeigen.** Protokoll, Paarliste,
   MLflow-Verläufe und die geprüften Ergebnisse belegen, was untersucht wurde.
   Der frühere mittlere Vorteil von +1,75 Punkten gilt für Fold 0 über acht
   Parametereinstellungen und zwei Seeds. Er darf nicht als allgemeiner
   Clipping-Vorteil oder als derselbe Vergleich wie die erweiterte CV ausgegeben werden.
3. **Keine weiteren Läufe allein zur Suche nach einem positiven Ergebnis.**
   Sollte später ein besseres Modell das Ziel sein, braucht es eine neue,
   begründete Hypothese und dafür getrennte Evaluation. Dieser Auftrag
   untersucht Clipping; ein weiteres Volltrain-Modell ist dafür nicht erforderlich.

Clipping begrenzt in diesen Läufen tatsächlich große Gradienten. Die Telemetrie
enthält aber auch bei den verglichenen Ohne-Clipping-Fits endliche Werte und
erfolgreiche Abschlüsse. Deshalb behaupten wir weder einen hier bewiesenen
Stabilitätsgewinn noch einen nachgewiesenen Schutz vor Trainingsabbrüchen.
Eine begrenzte Gradientennorm allein garantiert keine bessere Klassifikation.

## Nachweise und Grenzen

- [Vor den sieben neuen Fits eingefrorenes Protokoll](protocol-v1.json),
  [Ergebnisse](summary-v1.json), [30 eindeutige Paare als CSV](pairs-v1.csv),
  [unabhängige Verifikation aller 60 Fits und beider Bootstrap-Auswertungen](verification-v1.json).
- [Original-MLflow · sieben neue Kontrollen](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/5).
  Jeder Run enthält seinen Clipping-Partner und die komplette Auswertung als
  Artefakt unter `clipping-comparison-v1/`. Vorhandene Runs behalten ihre IDs.
- Nach bekannten Ergebnissen geplante explorative Erweiterung; dieselbe kleine
  Train-Stichprobe. Überlappende Fits und drei feste Seeds sind keine unabhängigen
  Stichproben. Die 2000 gepaarten semantischen Gruppen-Bootstrap-Ziehungen sind
  bedingt auf diesen Artikeln und Fits; sie korrigieren keine Auswahlverzerrung
  und bilden nicht die gesamte Trainingsunsicherheit ab.
- Kein erneuter Abschlusstest. Die Werte bewerten Trainingsrezepte in
  159/160/161-Artikel-Fits; sie sind kein unabhängiger Testwert eines gespeicherten
  Volltrain-Adapters. Gewichte, Serving-Auswahl und ursprüngliche Testberichte
  bleiben erhalten. Die Inferenz-API ist nach Abschluss wieder gestartet.

Diese redaktionelle Einordnung ergänzt den automatisch erzeugten Ergebnisbericht.
Der eingefrorene Trainings- und Auswertungscode wird dafür nicht geändert.
