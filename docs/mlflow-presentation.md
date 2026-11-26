# Öffentlicher MLflow-Zugang

**[MLflow öffnen](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments)**

Dieser Link öffnet die bestehende MLflow-Instanz auf dem Tower über einen
öffentlichen, lesenden Zugang. Es gibt keine zusätzliche Ergebnisdatenbank,
keine umsortierten Experimente und keine neu importierten Läufe. Originale
Laufkennungen, Startzeiten, Parameter, Metriken, Verläufe und Artefakte bleiben
identisch. Neue lokale Ergebnisse werden über denselben Zugang sichtbar.

| Originalexperiment | Öffentlicher Link |
|---|---|
| article-sentiment-second-path | [Experiment 1](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/1) |
| article-optimizer-ablation-v1 | [Experiment 2](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/2) |
| article-optimizer-candidate-v1 | [Experiment 3](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3) |
| article-optimizer-validation-v1 | [Experiment 4](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/4) |
| article-gradient-clipping-controls-v1 | [Experiment 5 · neue Gegenläufe ohne Clipping](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/5) |
| Neuer Volltrain-Kandidat | [Originalmodelllauf](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3/runs/5230309ca351479980869ef61fc239f0) |

Beim Abgleich vor Beginn der zusätzlichen Clipping-Kontrollen am 26.11.2026 enthielt die Originalinstanz fünf
Experimente einschließlich des leeren Default-Experiments und 187 gespeicherte
Läufe: 131 in Experiment 1, 33 in Experiment 2, einer in Experiment 3 und 22 in
Experiment 4. Die neue Validierung umfasst 21 neue Fits und sechs exakt
übernommene ältere Fits; die sechs behalten ihre alten Kennungen aus Experiment 2.
Ein zusätzlicher Setup-Fehlschlag in Experiment 4 hat null Trainingsschritte
und keine Vorhersagen. Er bleibt als Ausführungshistorie erhalten.
Die abgeschlossene Faktorstudie umfasst weiterhin 32 Fits.
[Abschluss und Modellentscheidung](../reports/optimizer_validation/README.md).

Die nachfolgend beauftragten [Clipping-Gegenläufe](../reports/clipping_control/README.md)
stehen in Experiment 5: sieben neue Fits ohne Clipping, jeweils mit Link zum
ursprünglichen Clipping-Partner und eigenen Vergleichsmetriken. Zusammen mit
53 vorhandenen Fits sind alle 30 verschiedenen Paare / 60 Fits abgeschlossen
und unabhängig geprüft. Die vollständige Paarliste, Auswertung und Verifikation
liegen an jedem der sieben neuen Runs unter `clipping-comparison-v1/` als Artefakte.
Keine vorhandenen Fits wurden als neue Trainingsläufe importiert. Der abschließende
Abgleich enthält sechs Experimente einschließlich Default und 194 gespeicherte
Runs: die oben genannten 187 plus sieben erfolgreiche Kontrollen in Experiment 5.
Die Inferenz-API ist wieder gestartet; das bisherige Standardmodell bleibt ausgewählt.

Clipping ergibt im mittleren DE/EN-OOF-Wert +0,32 F1-Prozentpunkte beim bisherigen
und −0,61 beim neuen Rezept. Beide bedingten 95%-Intervalle umfassen null.
[Ergebnis und Einordnung](../reports/clipping_control/interpretation-v1.md).

Die Oberfläche ist dieselbe MLflow-Version. Browserabhängige Einstellungen wie
Tabellenspalten, Filter und Sprache können auf anderen Geräten abweichen.
Öffentliches Anlegen, Ändern, Hochladen und Löschen sind gesperrt.

## Verfügbarkeit

Tower, lokales MLflow auf Port 5000, lesender Gateway auf Port 5002 und Tunnel
müssen laufen. Der kostenlose [Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
ist vorübergehend; sein Link kann sich nach einem Tunnelneustart ändern.
GitHub hostet die Datenbank nicht. Beim Betrachter ist keine Installation nötig.

[Startanleitung für den Zugang](../src/studies/mlflow_presentation/README.md).
Ohne laufenden Server bleiben die [Optimizer-Berichte](../reports/optimizer_ablation/README.md)
und die [Modellkarte](../reports/optimizer_candidate/README.md) auf GitHub lesbar.
